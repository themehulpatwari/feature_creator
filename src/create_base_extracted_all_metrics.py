import pandas as pd
from pathlib import Path

print("=" * 70)
print("Creating Base + Extracted + All Metrics (without LLM names)")
print("=" * 70)

# Read base.csv
base_path = Path(__file__).resolve().parent.parent / 'output' / 'base.csv'
base_df = pd.read_csv(base_path)
print(f"\nBase CSV shape: {base_df.shape}")

# Read extracted_features.csv
extracted_path = Path(__file__).resolve().parent.parent / 'input' / 'extracted_features.csv'
extracted_df = pd.read_csv(extracted_path)
print(f"Extracted features CSV shape: {extracted_df.shape}")

# Read all_metrics.csv
all_metrics_path = Path(__file__).resolve().parent.parent / 'input' / 'all_metrics.csv'
all_metrics_df = pd.read_csv(all_metrics_path)
print(f"All Metrics CSV shape: {all_metrics_df.shape}")

# Rename base columns to match others (lowercase)
base_df_renamed = base_df.rename(columns={'query_ID': 'query_id'})

# Step 1: Merge base with extracted_features
print("\n1. Merging base with extracted features...")
# Get columns from extracted that are not in base and not llm_name columns
base_cols = set(base_df_renamed.columns.str.lower())
extracted_cols_to_add = []
for col in extracted_df.columns:
    if col.lower() not in base_cols and col not in ['llm_name_1', 'llm_name_2', 'query_id', 'user_id', 'task_id']:
        extracted_cols_to_add.append(col)

extracted_subset = extracted_df[['query_id', 'user_id', 'task_id'] + extracted_cols_to_add]
merged_df = base_df_renamed.merge(extracted_subset, on=['query_id', 'user_id', 'task_id'], how='inner')
print(f"   After merging with extracted: {merged_df.shape}")

# Step 2: Merge with all_metrics
print("\n2. Merging with all metrics...")
# Get columns from all_metrics that are not already in merged
merged_cols = set(merged_df.columns.str.lower())
metrics_cols_to_add = []
for col in all_metrics_df.columns:
    if col.lower() not in merged_cols and col not in ['query_id', 'user_id', 'task_id']:
        metrics_cols_to_add.append(col)

all_metrics_subset = all_metrics_df[['query_id', 'user_id', 'task_id'] + metrics_cols_to_add]
merged_df = merged_df.merge(all_metrics_subset, on=['query_id', 'user_id', 'task_id'], how='inner')
print(f"   After merging with all_metrics: {merged_df.shape}")

# Step 3: Drop LLM name columns (one-hot encoded)
print("\n3. Dropping LLM name columns...")
llm_cols = [col for col in merged_df.columns if col.startswith('llm_1_') or col.startswith('llm_2_')]
print(f"   Dropping {len(llm_cols)} LLM columns: {llm_cols}")
merged_df = merged_df.drop(columns=llm_cols)
print(f"   After dropping LLM columns: {merged_df.shape}")

# Save result
output_path = Path(__file__).resolve().parent.parent / 'output' / 'base+extracted+all_metrics.csv'
merged_df.to_csv(output_path, index=False)

print(f"\n4. Saved to: {output_path}")
print(f"   Final shape: {merged_df.shape}")

print("\n" + "=" * 70)
print("Base + Extracted + All Metrics created successfully!")
print("=" * 70)
