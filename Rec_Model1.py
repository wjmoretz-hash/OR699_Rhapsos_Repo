# -*- coding: utf-8 -*-
"""
Graph-Aware Action Recommender Engine
=====================================
Evaluates primary actions (Effect 1) on entities across ALL time stages,
calculates cascading secondary effects (Effect 2) on neighboring entities based
on meter-based distance bounds, and scores total strategic impact based on the 
fraction of completed nodes reachable in the directed battle_goals_matrix graph.
"""

import sqlite3
import pandas as pd
import numpy as np
import networkx as nx
from pyproj import Transformer

# Import prediction engine if dynamically generating predictions in memory
try:
    from world_model_prediction import run_prediction_pipeline
except ImportError:
    run_prediction_pipeline = None

DB_PATH = "C:/Users/wjmor/OneDrive/Documents/MS OR/OR699/GMU Project.db"

# Coordinate Transformer: WGS84 Lat/Lon (EPSG:4326) to UTM Zone 18N (EPSG:32618)
# Converts geographical degrees into metric planar Easting & Northing
GEO_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:32618", always_xy=True)

# Synonym mapping to align action terminology in effects tables with graph goal nodes
ACTION_GRAPH_SYNONYMS = {
    'flatten': 'destroy',
    'shrink': 'degrade',
    'suppress': 'disrupt',
    'disrupt': 'disrupt',
    'neutralize': 'destroy'
}

# ---------------------------------------------------------
# Helper: Database Schema Normalization
# ---------------------------------------------------------
def normalize_effects_dataframe(df):
    """
    Normalizes column headers from effects_description_array table into a 
    standardized schema required for spatial cascading lookups.
    """
    column_mapping = {
        'primary_action': 'action',
        'action_name': 'action',
        'action_type': 'action',
        'effect_type': 'action',
        'effect': 'action',
        'act': 'action',
        'effect_1': 'action',
        
        'min_dist': 'min_distance',
        'minimum_distance': 'min_distance',
        'min_distance_m': 'min_distance',
        'min_dist_m': 'min_distance',
        
        'max_dist': 'max_distance',
        'maximum_distance': 'max_distance',
        'max_distance_m': 'max_distance',
        'max_dist_m': 'max_distance',
        'radius_m': 'max_distance',
        
        'secondary_action': 'secondary_effect',
        'cascading_effect': 'secondary_effect',
        'splash_effect': 'secondary_effect',
        'collateral_effect': 'secondary_effect',
        'secondary_impact': 'secondary_effect',
        'sec_effect': 'secondary_effect',
        'effect_2': 'secondary_effect'
    }
    
    # Strip whitespace and normalize column names to lowercase
    df.columns = [str(col).strip().lower() for col in df.columns]
    df = df.rename(columns=column_mapping)
    
    if 'action' not in df.columns:
        raise KeyError(
            f"Could not locate primary action column in effects table. "
            f"Available database columns: {list(df.columns)}"
        )
        
    # Set standard defaults if distance or secondary effect columns are unpopulated
    if 'min_distance' not in df.columns:
        df['min_distance'] = 0.0
    if 'max_distance' not in df.columns:
        df['max_distance'] = 100.0  # Fallback if unpopulated in database
    if 'secondary_effect' not in df.columns:
        df['secondary_effect'] = 'None'
        
    return df

# ---------------------------------------------------------
# 1. Graph-Aware Evaluator & Spatial Cascading Engine
# ---------------------------------------------------------
class GraphAwareActionRecommender:
    def __init__(self, effects_df, battle_goals_df):
        self.effects_df = effects_df
        self.goals_df = battle_goals_df
        self.goal_graph = self._build_goal_graph()

    def _build_goal_graph(self):
        """
        Constructs a directed dependency graph from battle_goals_matrix.
        Edges represent strategic pathways toward primary operational goals.
        """
        G = nx.DiGraph()
        cols = self.goals_df.columns.tolist()
        
        parent_col = cols[0]
        child_col = cols[1]
        val_col = cols[2] if len(cols) > 2 else 'value'

        for _, row in self.goals_df.iterrows():
            parent_goal = str(row[parent_col]).strip()
            target_goal = str(row[child_col]).strip()
            rel_weight = row[val_col]
            
            # Add directed dependency edge: child goal -> parent goal
            G.add_edge(target_goal, parent_goal, relationship=rel_weight)
            
        return G

    def evaluate_goal_congruency(self, effect, primary_mission="Defend GMU from Adversary Shapes"):
        """
        Calculates utility as a function of reachable goals in the directed graph.
        Utility = (Reachable Goal Nodes from Effect) / (Total Goal Nodes in Graph)
        """
        mapped_effect = ACTION_GRAPH_SYNONYMS.get(str(effect).lower(), str(effect).lower())
        total_nodes = len(self.goal_graph.nodes)

        # Fallback if graph is empty
        if total_nodes == 0:
            return 1.0

        # Direct match with primary mission statement
        if mapped_effect in primary_mission.lower() or str(effect).lower() in primary_mission.lower():
            return 1.0

        # Find matching source nodes in the goal graph
        source_nodes = [
            n for n in self.goal_graph.nodes 
            if mapped_effect in n.lower() or str(effect).lower() in n.lower()
        ]

        if source_nodes:
            src = source_nodes[0]
            # Find all nodes reachable from this action node via directed edges
            reachable_nodes = nx.descendants(self.goal_graph, src)
            reachable_nodes.add(src)  # Include the node itself
            
            # Ratio of goal graph completed / contributed to by this action
            completion_ratio = len(reachable_nodes) / float(total_nodes)
            return completion_ratio

        # Default baseline if action isn't explicitly linked in the graph
        return 0.25

    def _calculate_distance_meters(self, lat1, lon1, lat2, lon2):
        """
        Converts Lat/Lon coordinates to projected planar meters and calculates
        Euclidean distance.
        """
        e1, n1 = GEO_TRANSFORMER.transform(lon1, lat1)
        e2, n2 = GEO_TRANSFORMER.transform(lon2, lat2)
        return np.sqrt((e1 - e2)**2 + (n1 - n2)**2)

    def _determine_secondary_effect(self, primary_action, distance_meters):
        """
        Queries effects_description_array to determine if Effect 1 produces an Effect 2
        on a secondary entity located at distance_meters away.
        """
        matches = self.effects_df[
            (self.effects_df['action'].astype(str).str.lower() == str(primary_action).lower()) & 
            (self.effects_df['min_distance'] <= distance_meters) & 
            (self.effects_df['max_distance'] >= distance_meters)
        ]
        if not matches.empty:
            return matches.iloc[0]['secondary_effect']
        return 'None'

    def recommend_best_action(self, state_predictions_df, state_transition):
        """
        Evaluates ALL candidate actions across target entities and friendly initiating units
        for the specified state transition, returning the FULL UNFILTERED list of options.
        """
        recommendations = []
        available_actions = self.effects_df['action'].dropna().unique()
        all_entities = state_predictions_df.to_dict('records')

        # Separate candidate targets (enemies) from potential initiating units (allies)
        enemy_entities = [e for e in all_entities if e.get('allied_probability', 0.5) < 0.5]
        friendly_entities = [e for e in all_entities if e.get('allied_probability', 0.5) >= 0.5]

        # Fallback if no friendly units are identified
        if not friendly_entities:
            friendly_entities = [{'unique_identifier': 'UNASSIGNED_ASSET'}]

        for e1 in enemy_entities:
            e1_id = e1['unique_identifier']
            p_allied_1 = e1.get('allied_probability', 0.5)
            e1_lat, e1_lon = e1['latitude'], e1['longitude']
            reliability_1 = e1.get('latitude_reliability', 1.0)

            for action in available_actions:
                # Retrieve action-specific max cascading distance from effects table
                action_rows = self.effects_df[self.effects_df['action'].astype(str).str.lower() == str(action).lower()]
                action_max_dist = action_rows['max_distance'].max() if not action_rows.empty else 100.0

                # 1. Primary target impact (derived from goal graph reachability)
                e1_goal_utility = self.evaluate_goal_congruency(action)
                primary_impact = e1_goal_utility * (1.0 - p_allied_1) * reliability_1
                
                total_battle_goal_impact = primary_impact
                cascading_events = []
                goals_completed_count = int(e1_goal_utility * len(self.goal_graph.nodes))

                # 2. Check all surrounding entities E2 for cascading Effect 2 up to action_max_dist
                for e2 in all_entities:
                    e2_id = e2['unique_identifier']
                    if e1_id == e2_id:
                        continue

                    # Compute distance between E1 and E2 in meters
                    dist_meters = self._calculate_distance_meters(e1_lat, e1_lon, e2['latitude'], e2['longitude'])

                    # Cascading bounds defined strictly by the effect's defined range
                    if dist_meters <= action_max_dist:
                        effect_2 = self._determine_secondary_effect(action, dist_meters)
                        
                        if effect_2 != 'None':
                            p_allied_2 = e2.get('allied_probability', 0.5)
                            e2_goal_utility = self.evaluate_goal_congruency(effect_2)
                            
                            # Positive impact for enemies; penalty for allies
                            net_e2_impact = e2_goal_utility * (1.0 - p_allied_2) - (0.9 * p_allied_2)
                            
                            # Proximity attenuation based on max action range
                            proximity_weight = 1.0 - (dist_meters / action_max_dist) if action_max_dist > 0 else 1.0
                            weighted_e2_impact = net_e2_impact * proximity_weight
                            
                            total_battle_goal_impact += weighted_e2_impact
                            
                            cascading_events.append({
                                'secondary_entity': e2_id,
                                'distance_m': round(dist_meters, 2),
                                'effect_2': effect_2,
                                'effect_2_goal_score': round(e2_goal_utility, 2),
                                'net_cascading_impact': round(weighted_e2_impact, 3)
                            })

                # Scale overall impact score
                total_efficacy_score = np.clip(total_battle_goal_impact * 100.0, -100.0, 100.0)

                # Assign recommendation across initiating friendly assets
                for unit in friendly_entities:
                    initiating_unit_id = unit['unique_identifier']

                    recommendations.append({
                        'evaluation_state': state_transition,
                        'initiating_unit': initiating_unit_id,
                        'target_entity': e1_id,
                        'recommended_action': action,
                        'goals_completed_estimate': goals_completed_count,
                        'effect_1_goal_score': round(e1_goal_utility, 3),
                        'predicted_side': 'Enemy',
                        'target_confidence': round(1.0 - p_allied_1, 3),
                        'total_battle_goal_impact_score': round(total_efficacy_score, 1),
                        'cascading_effects_detail': cascading_events
                    })

        # Rank all options by largest positive effect on battle goals
        df_recs = pd.DataFrame(recommendations).sort_values(
            by=['total_battle_goal_impact_score'], 
            ascending=False
        ).reset_index(drop=True)
        
        # Return complete list without truncation
        return df_recs

# ---------------------------------------------------------
# 2. Database Pipeline & Execution Runner (All Options Across All Stages)
# ---------------------------------------------------------
def generate_recommendations_from_db(db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    
    # 1. Load World State Predictions
    try:
        df_predictions = pd.read_sql_query("SELECT * FROM predicted_world_state;", conn)
    except Exception:
        print("[Notice] 'predicted_world_state' table not found. Generating predictions in memory...")
        conn.close()
        if run_prediction_pipeline is not None:
            df_predictions = run_prediction_pipeline(db_path, save_to_db=True)[0]
            conn = sqlite3.connect(db_path)
        else:
            raise RuntimeError("World model prediction pipeline is missing or not imported.")

    # 2. Load effects_description_array
    effects_tables = ['effects_description_array', 'effects_description', 'action_effects_matrix']
    effects_df = None

    for tbl in effects_tables:
        try:
            raw_df = pd.read_sql_query(f"SELECT * FROM {tbl};", conn)
            effects_df = normalize_effects_dataframe(raw_df)
            print(f"[Database Ingestion] Parsed effects matrix from table '{tbl}'")
            break
        except Exception:
            continue

    if effects_df is None or effects_df.empty:
        raise ValueError("Could not find or parse effects table from SQLite database.")

    # 3. Load battle_goals_matrix
    try:
        df_goals = pd.read_sql_query("SELECT * FROM battle_goals_matrix;", conn)
    except Exception:
        df_goals = pd.read_sql_query("SELECT * FROM battle_goals;", conn)
        
    conn.close()

    # 4. Iterate across ALL evaluation states / time stages
    unique_states = sorted(df_predictions['evaluation_state'].unique())
    print(f"[Execution Engine] Generating full option set across {len(unique_states)} time stages: {unique_states}")

    recommender = GraphAwareActionRecommender(effects_df, df_goals)
    all_recommendations = []

    for state in unique_states:
        state_preds = df_predictions[df_predictions['evaluation_state'] == state]
        recs_stage = recommender.recommend_best_action(
            state_preds, 
            state_transition=state
        )
        all_recommendations.append(recs_stage)

    # Combine full option sets across all stages into a single DataFrame
    master_recs_df = pd.concat(all_recommendations, ignore_index=True) if all_recommendations else pd.DataFrame()
    return master_recs_df


if __name__ == "__main__":
    recs_df = generate_recommendations_from_db(DB_PATH)
    
    print(f"\n================ FULL RECOMMENDATIONS LIST ({len(recs_df)} TOTAL OPTIONS) ================")
    if not recs_df.empty:
        for state, group in recs_df.groupby('evaluation_state'):
            print(f"\n--- TIME STAGE / STATE TRANSITION {state} ({len(group)} Options Available) ---")
            # Show top 5 print preview per stage in console (full data is in recs_df)
            for _, row in group.head(5).iterrows():
                print(f"\n* Unit [{row['initiating_unit']}] -> Action: [{row['recommended_action'].upper()}] against Target [{row['target_entity']}]")
                print(f"   - Graph Goal Completion Fraction: {row['effect_1_goal_score']} (~{row['goals_completed_estimate']} goals reachable)")
                print(f"   - Target Confidence (Enemy Prob): {row['target_confidence']}")
                print(f"   - Total Strategic Impact Score: {row['total_battle_goal_impact_score']}")
                if row['cascading_effects_detail']:
                    print("   - Cascading Secondary Effects (Effect 2):")
                    for casc in row['cascading_effects_detail']:
                        print(f"     -> Entity {casc['secondary_entity']} ({casc['distance_m']}m away) "
                              f"receives Effect 2 [{casc['effect_2']}] (Net Utility: {casc['net_cascading_impact']})")
                else:
                    print("   - Cascading Secondary Effects: None detected within range")
