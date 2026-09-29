# -*- coding: utf-8 -*-
"""
Graph-Aware Action Recommender Engine
=====================================
Evaluates primary actions (Effect 1) on entities, calculates cascading secondary
effects (Effect 2) on neighboring entities based on meter-based distances, 
and scores total strategic impact against top-level battle goals using a 
directed dependency graph.
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
# Used to convert degrees into metric planar Easting & Northing for distance calculations
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
        df['max_distance'] = 100.0  # Default 100 meters
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
        Scores how strongly an effect serves top-level battle goals using 
        shortest directed path distance within the goal network.
        """
        mapped_effect = ACTION_GRAPH_SYNONYMS.get(str(effect).lower(), str(effect).lower())

        # Direct match with primary mission statement
        if mapped_effect in primary_mission.lower() or str(effect).lower() in primary_mission.lower():
            return 1.0
        
        # Locate corresponding nodes in the goal graph
        source_nodes = [
            n for n in self.goal_graph.nodes 
            if mapped_effect in n.lower() or str(effect).lower() in n.lower()
        ]
        target_nodes = [n for n in self.goal_graph.nodes if primary_mission.lower() in n.lower()]

        if source_nodes and target_nodes:
            src = source_nodes[0]
            tgt = target_nodes[0]
            
            # Traversing path to top goal gives higher strategic utility
            if nx.has_path(self.goal_graph, src, tgt):
                path = nx.shortest_path(self.goal_graph, src, tgt)
                weight = 1.0
                for i in range(len(path) - 1):
                    rel = self.goal_graph.get_edge_data(path[i], path[i+1]).get('relationship', 1)
                    weight *= 0.85 if rel == 1 else 0.45
                return weight

        # Baseline value for general tactical actions not explicitly linked in graph
        return 0.3

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

    def recommend_best_action(self, state_predictions_df, state_transition, max_eval_radius=100.0, top_k=3):
        """
        Evaluates candidate actions across target entities, identifies cascading Effect 2 
        impacts on surrounding entities, and recommends the action yielding the 
        largest overall positive impact on battle goals.
        """
        recommendations = []
        available_actions = self.effects_df['action'].dropna().unique()
        entities = state_predictions_df.to_dict('records')

        for e1 in entities:
            e1_id = e1['unique_identifier']
            p_allied_1 = e1.get('allied_probability', 0.5)

            # Skip target selection if entity is perceived as Allied
            if p_allied_1 >= 0.5:
                continue

            e1_lat, e1_lon = e1['latitude'], e1['longitude']
            reliability_1 = e1.get('latitude_reliability', 1.0)

            for action in available_actions:
                # 1. Evaluate strategic impact of Effect 1 on primary target E1
                e1_goal_utility = self.evaluate_goal_congruency(action)
                primary_impact = e1_goal_utility * (1.0 - p_allied_1) * reliability_1
                
                total_battle_goal_impact = primary_impact
                cascading_events = []

                # 2. Check all surrounding entities E2 for cascading Effect 2
                for e2 in entities:
                    e2_id = e2['unique_identifier']
                    if e1_id == e2_id:
                        continue

                    # Compute distance between E1 and E2 in meters
                    dist_meters = self._calculate_distance_meters(e1_lat, e1_lon, e2['latitude'], e2['longitude'])

                    if dist_meters <= max_eval_radius:
                        # Lookup Effect 2 based on primary action and distance
                        effect_2 = self._determine_secondary_effect(action, dist_meters)
                        
                        if effect_2 != 'None':
                            p_allied_2 = e2.get('allied_probability', 0.5)
                            
                            # Evaluate strategic impact of Effect 2 on E2 against battle goals
                            e2_goal_utility = self.evaluate_goal_congruency(effect_2)
                            
                            # Positive value if E2 is enemy; penalty if E2 is friendly
                            net_e2_impact = e2_goal_utility * (1.0 - p_allied_2) - (0.9 * p_allied_2)
                            
                            # Distance attenuation factor
                            proximity_weight = 1.0 - (dist_meters / max_eval_radius)
                            weighted_e2_impact = net_e2_impact * proximity_weight
                            
                            total_battle_goal_impact += weighted_e2_impact
                            
                            cascading_events.append({
                                'secondary_entity': e2_id,
                                'distance_m': round(dist_meters, 2),
                                'effect_2': effect_2,
                                'effect_2_goal_score': round(e2_goal_utility, 2),
                                'net_cascading_impact': round(weighted_e2_impact, 3)
                            })

                # Scale overall impact to an intuitive score
                total_efficacy_score = np.clip(total_battle_goal_impact * 100.0, -100.0, 100.0)

                recommendations.append({
                    'evaluation_state': state_transition,
                    'target_entity': e1_id,
                    'recommended_action': action,
                    'effect_1_goal_score': round(e1_goal_utility, 2),
                    'predicted_side': 'Enemy',
                    'target_confidence': round(1.0 - p_allied_1, 3),
                    'total_battle_goal_impact_score': round(total_efficacy_score, 1),
                    'cascading_effects_detail': cascading_events
                })

        # Rank actions by largest positive effect on battle goals
        df_recs = pd.DataFrame(recommendations).sort_values(
            by=['total_battle_goal_impact_score'], 
            ascending=False
        ).reset_index(drop=True)
        
        return df_recs.head(top_k)

# ---------------------------------------------------------
# 2. Database Pipeline & Execution Runner
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

    # 4. Generate recommendations across evaluation states
    recommender = GraphAwareActionRecommender(effects_df, df_goals)
    unique_states = sorted(df_predictions['evaluation_state'].unique())
    all_recommendations = []

    for state in unique_states:
        if state > 1:
            state_preds = df_predictions[df_predictions['evaluation_state'] == state]
            recs = recommender.recommend_best_action(
                state_preds, 
                state_transition=state, 
                max_eval_radius=100.0, 
                top_k=3
            )
            all_recommendations.append(recs)

    master_recs_df = pd.concat(all_recommendations, ignore_index=True) if all_recommendations else pd.DataFrame()
    return master_recs_df


if __name__ == "__main__":
    recs_df = generate_recommendations_from_db(DB_PATH)
    
    print("\n================ RECOMMENDED ACTIONS (LARGEST EFFECT ON BATTLE GOALS) ================")
    if not recs_df.empty:
        for state, group in recs_df.groupby('evaluation_state'):
            print(f"\n--- STATE TRANSITION {state} ---")
            for _, row in group.iterrows():
                print(f"\n* Action: [{row['recommended_action'].upper()}] against Entity [{row['target_entity']}]")
                print(f"  - Primary Effect 1 Goal Utility: {row['effect_1_goal_score']}")
                print(f"  - Total Strategic Impact Score: {row['total_battle_goal_impact_score']}")
                if row['cascading_effects_detail']:
                    print("  - Cascading Secondary Effects (Effect 2):")
                    for casc in row['cascading_effects_detail']:
                        print(f"    -> Entity {casc['secondary_entity']} ({casc['distance_m']}m away) "
                              f"receives Effect 2 [{casc['effect_2']}] (Net Utility: {casc['net_cascading_impact']})")
                else:
                    print("  - Cascading Secondary Effects: None detected within radius")