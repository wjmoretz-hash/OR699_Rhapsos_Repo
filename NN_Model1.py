# -*- coding: utf-8 -*-
"""
World Model Prediction Engine
=============================
Handles DB ingestion, target entity synthesis, learned-at time decay,
relational loss neural training, state-by-state inference, and 
cumulative entity alignment tracking across state transitions.
"""

import sqlite3
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

DB_PATH = "C:/Users/wjmor/OneDrive/Documents/MS OR/OR699/GMU Project.db"

# ---------------------------------------------------------
# 1. Dynamic Data Loading & Implicit Target Synthesis
# ---------------------------------------------------------
def load_data(db_path):
    conn = sqlite3.connect(db_path)
    # Pull all available columns dynamically to handle schema updates
    df = pd.read_sql_query("SELECT * FROM commander_world_model;", conn)
    conn.close()
    return df

def synthesize_targeted_entities(df):
    """
    Scans active reports for units referenced in 'entity_target'. 
    If a target unit does not have a direct record in the current state_transition,
    creates a synthesized record so the neural net can track it.
    """
    synthesized_rows = []

    for state, group in df.groupby('state_transition'):
        existing_ids = set(group['unique_identifier'].dropna().unique())

        for _, row in group.iterrows():
            target_id = row.get('entity_target')

            if pd.notna(target_id) and str(target_id).strip() != '' and target_id not in existing_ids:
                implicit_row = {
                    'unique_identifier': str(target_id).strip(),
                    'state_transition': state,
                    'simulated_time': row.get('simulated_time'),
                    'side': np.nan,
                    'entity_type': 'Unknown',
                    'latitude': row.get('latitude'),
                    'longitude': row.get('longitude'),
                    'entity_goal_effect': 'Targeted',
                    'entity_target': np.nan
                }

                # Copy remaining columns dynamically if they exist
                for col in df.columns:
                    if col not in implicit_row:
                        implicit_row[col] = np.nan

                synthesized_rows.append(implicit_row)
                existing_ids.add(target_id)

    if synthesized_rows:
        synth_df = pd.DataFrame(synthesized_rows)
        df = pd.concat([df, synth_df], ignore_index=True)
        df = df.sort_values(by=['state_transition', 'unique_identifier']).reset_index(drop=True)
        print(f"[Entity Discovery] Synthesized {len(synth_df)} implicit target entity records across states.")

    return df

# ---------------------------------------------------------
# 2. Informational Weight Decay & Feature Preprocessing
# ---------------------------------------------------------
def apply_learned_at_decay(df, current_time_col='simulated_time', decay_gamma=0.05):
    """
    Scales numerical features toward zero based on elapsed time 
    from their respective '..._learned_at' timestamps using a gamma of 0.05.
    """
    df_decayed = df.copy()
    
    # Identify all timestamp columns ending with '_learned_at' or named 'learned_at'
    learned_at_cols = [c for c in df.columns if c.endswith('_learned_at') or c == 'learned_at']
    
    for l_col in learned_at_cols:
        base_feature = l_col.replace('_learned_at', '')
        
        if base_feature in df_decayed.columns:
            # Calculate elapsed time and exponential decay factor
            elapsed_time = (df_decayed[current_time_col] - df_decayed[l_col]).fillna(0).clip(lower=0)
            decay_factor = np.exp(-decay_gamma * elapsed_time)
            
            # Attenuate numerical feature toward 0
            if pd.api.types.is_numeric_dtype(df_decayed[base_feature]):
                df_decayed[base_feature] = df_decayed[base_feature] * decay_factor
            
            # Append explicit reliability metric feature
            df_decayed[f'{base_feature}_reliability'] = decay_factor

    return df_decayed

def detect_and_build_preprocessor(df, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = ['unique_identifier', 'side', 'target']

    candidate_cols = [c for c in df.columns if c not in exclude_cols]

    num_cols = df[candidate_cols].select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()
    cat_cols = df[candidate_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, num_cols),
            ('cat', cat_pipeline, cat_cols)
        ]
    )
    return preprocessor, num_cols + cat_cols

# ---------------------------------------------------------
# 3. Neural Network Architecture & PyTorch Helpers
# ---------------------------------------------------------
class WorldModelDataset(Dataset):
    def __init__(self, features, labels=None):
        self.X = torch.tensor(features, dtype=torch.float32)
        if labels is not None:
            self.y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class SideCategorizerNet(nn.Module):
    def __init__(self, input_dim):
        super(SideCategorizerNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)

# ---------------------------------------------------------
# 4. Model Training & Relational Loss Execution
# ---------------------------------------------------------
def train_and_predict_subset(historical_df, current_state_df, epochs=40, lr=0.001, decay_gamma=0.05, relational_weight=0.5):
    def parse_side(val):
        if pd.isna(val) or str(val).strip() == '':
            return np.nan
        return 1.0 if 'allied' in str(val).lower() else 0.0

    train_data = historical_df.copy().reset_index(drop=True)
    eval_data = current_state_df.copy().reset_index(drop=True)

    # Apply 0.05 decay to feature data based on learned_at timestamps
    train_data = apply_learned_at_decay(train_data, current_time_col='simulated_time', decay_gamma=decay_gamma)
    eval_data = apply_learned_at_decay(eval_data, current_time_col='simulated_time', decay_gamma=decay_gamma)

    train_data['target'] = train_data['side'].apply(parse_side)
    labeled_mask = train_data['target'].notna().values

    # Exclude metadata, identifiers, targets, and raw timestamp columns from NN input
    exclude_cols = ['unique_identifier', 'side', 'target'] + [c for c in train_data.columns if 'learned_at' in c]
    preprocessor, feature_cols = detect_and_build_preprocessor(train_data, exclude_cols=exclude_cols)
    
    X_train_all = preprocessor.fit_transform(train_data[feature_cols])
    y_train_all = train_data['target'].values

    X_tensor = torch.tensor(X_train_all, dtype=torch.float32)
    y_tensor = torch.tensor(np.nan_to_num(y_train_all), dtype=torch.float32).unsqueeze(1)

    # Extract adversarial interaction pairs for relational loss penalty
    id_to_idx = {uid: idx for idx, uid in enumerate(train_data['unique_identifier'])}
    interaction_pairs = []
    
    for src_idx, row in train_data.iterrows():
        tgt_id = row.get('entity_target')
        effect = str(row.get('entity_goal_effect', '')).lower()
        if pd.notna(tgt_id) and tgt_id in id_to_idx and 'support' not in effect and 'heal' not in effect:
            tgt_idx = id_to_idx[tgt_id]
            interaction_pairs.append((src_idx, tgt_idx))

    # Initialize model dynamically matching processed feature dimensions
    model = SideCategorizerNet(input_dim=X_train_all.shape[1])
    bce_loss = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        all_probs = model(X_tensor)

        # 1. Standard Supervised Loss on ground truth labels
        if labeled_mask.sum() > 0:
            loss_sup = bce_loss(all_probs[labeled_mask], y_tensor[labeled_mask])
        else:
            loss_sup = torch.tensor(0.0)

        # 2. Adversarial Relational Loss on interacting pairs
        if len(interaction_pairs) > 0:
            src_indices = [p[0] for p in interaction_pairs]
            tgt_indices = [p[1] for p in interaction_pairs]
            
            p_src = all_probs[src_indices]
            p_tgt = all_probs[tgt_indices]
            
            loss_rel = torch.mean(p_src * p_tgt + (1 - p_src) * (1 - p_tgt))
        else:
            loss_rel = torch.tensor(0.0)

        total_loss = loss_sup + (relational_weight * loss_rel)
        total_loss.backward()
        optimizer.step()

    # Inference on current state entities
    X_eval = preprocessor.transform(eval_data[feature_cols])
    eval_tensor = torch.tensor(X_eval, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        probabilities = model(eval_tensor).numpy().flatten()

    eval_data['allied_probability'] = probabilities
    eval_data['predicted_side'] = ['Allied' if p >= 0.5 else 'Enemy' for p in probabilities]
    
    return eval_data

# ---------------------------------------------------------
# 5. Pipeline Orchestrator & Cumulative Tracking Table
# ---------------------------------------------------------
def run_prediction_pipeline(db_path=DB_PATH, save_to_db=True):
    df = load_data(db_path)
    df = synthesize_targeted_entities(df)
    
    unique_states = sorted(df['state_transition'].unique())
    all_results = []

    print(f"Running World Model Inference across {len(unique_states)} state transitions...\n")

    for current_state in unique_states:
        historical_df = df[df['state_transition'] <= current_state]
        
        # Grab the most recent record for EVERY known entity up to current_state
        current_state_df = (
            historical_df.sort_values('state_transition')
            .groupby('unique_identifier')
            .last()
            .reset_index()
        )
        current_state_df['state_transition'] = current_state
        
        predicted_df = train_and_predict_subset(historical_df, current_state_df, decay_gamma=0.05)
        predicted_df['evaluation_state'] = current_state
        all_results.append(predicted_df)

    master_log_df = pd.concat(all_results, ignore_index=True)

    # Build Cumulative Knowledge Tracker (Entity x State Matrix)
    tracking_df = master_log_df.pivot_table(
        index='unique_identifier',
        columns='evaluation_state',
        values=['predicted_side', 'allied_probability'],
        aggfunc='first'
    )

    # Attach ground truth metadata
    metadata_cols = [col for col in ['side', 'entity_type', 'unit_name'] if col in master_log_df.columns]
    if metadata_cols:
        metadata = master_log_df.groupby('unique_identifier')[metadata_cols].last()
        metadata.columns = pd.MultiIndex.from_tuples([('metadata', col) for col in metadata_cols])
        tracking_df = metadata.join(tracking_df)

    if save_to_db:
        conn = sqlite3.connect(db_path)
        master_log_df.to_sql('predicted_world_state', conn, if_exists='replace', index=False)
        
        # Flatten multi-index columns for database persistence
        flat_tracking_df = tracking_df.copy()
        flat_tracking_df.columns = ['_'.join([str(c) for c in col if str(c) != '']).strip('_') for col in flat_tracking_df.columns]
        flat_tracking_df.to_sql('entity_tracking_matrix', conn, if_exists='replace', index=True)
        
        conn.close()
        print("[Database] Successfully saved predictions to 'predicted_world_state' and tracking matrix to 'entity_tracking_matrix'.")

    return master_log_df, tracking_df

# ---------------------------------------------------------
# Standalone Execution Block
# ---------------------------------------------------------
if __name__ == "__main__":
    master_log_df, tracking_df = run_prediction_pipeline(DB_PATH, save_to_db=True)
    
    print("\n--- 1. MASTER TIMELINE LOG (Sample) ---")
    print(master_log_df[['evaluation_state', 'unique_identifier', 'side', 'predicted_side', 'allied_probability']].head(15))
    
    print("\n--- 2. ENTITY CUMULATIVE KNOWLEDGE TRACKER (Sample) ---")
    print(tracking_df.head(10))