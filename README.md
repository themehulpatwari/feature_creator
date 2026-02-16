# Feature Creator Pipeline

## Input Files

- **query_logs_table.csv**: Raw query logs from SQL database
- **extracted_features.csv**: Eye-tracking and mouse features from NLP gazing repo
- **all_metrics.csv**: Additional metrics (Jeff's features)

## Scripts

### 1. exclude_bad_workers.py
- **Input**: `extracted_features.csv`
- **Output**: Modifies `extracted_features.csv` in-place
- **Purpose**: Removes rows from bad workers (24 worker IDs)

### 2. create_base.py
- **Input**: `query_logs_table.csv`
- **Output**: `base.csv`
- **Purpose**: Creates base dataset with:
  - All query log data
  - One-hot encoded LLM names (llm_1_* and llm_2_*)
  - comparison_type column (pairwise/pointwise)
  - Filters out bad workers

### 3. create_base_user_specific.py
- **Input**: `base.csv`
- **Output**: `base+user_specific.csv`
- **Purpose**: Adds one-hot encoded user features
- **Config**: MODE = 'top_n' (top N users) or 'all' (all users), TOP_N = 5

### 4. create_base_extracted_features.py
- **Input**: `base.csv`, `extracted_features.csv`
- **Output**: `base+extracted_features.csv`
- **Purpose**: Merges base with all extracted features (intersection on query_id + user_id)

### 5. create_base_relative_features.py
- **Input**: `base.csv`, `extracted_features.csv`
- **Output**: `base+relative_features.csv`
- **Purpose**: Creates derived features from extracted features:
  - Difference features (response_A - response_B)
  - Ratio features (response_A / response_B)
  - Window differences (200 features: 100 gaze + 100 mouse)
  - Aggregated window stats (mean, std, max, min)

### 6. create_base_all_metrics.py
- **Input**: `base.csv`, `all_metrics.csv`
- **Output**: `base+all_metrics.csv`
- **Purpose**: Merges base with all_metrics (intersection on query_id + user_id + task_id)

### 7. create_base_extracted_all_metrics.py
- **Input**: `base.csv`, `extracted_features.csv`, `all_metrics.csv`
- **Output**: `base+extracted+all_metrics.csv`
- **Purpose**: Combines all features, drops LLM name columns (intersection on all merge keys)

## Pipeline Execution Order

```bash
# 1. Filter bad workers from extracted features
python src/exclude_bad_workers.py

# 2. Create base dataset
python src/create_base.py

# 3. Generate feature variations
python src/create_base_user_specific.py
python src/create_base_extracted_features.py
python src/create_base_relative_features.py
python src/create_base_all_metrics.py
python src/create_base_extracted_all_metrics.py
```

## Output Files Summary

| File | Description |
|------|-------------|
| base.csv | Base features + one-hot LLM names |
| base+user_specific.csv | Base + user one-hot encoding |
| base+extracted_features.csv | Base + all extracted features |
| base+relative_features.csv | Base + derived features only |
| base+all_metrics.csv | Base + Jeff's metrics |
| base+extracted+all_metrics.csv | All features, no LLM names |

## Notes

- All merges use intersection (inner join) - only keeps rows present in both datasets
- Bad workers list: 24 worker IDs filtered from all outputs
- One-hot encoded columns are dropped after encoding (llm_name_1, llm_name_2, user_id)
- comparison_type: determined by presence of llm_response_2 and llm_name_2

## TODO

7. think about running the models (rf, logistic) on these files and also about how to run them with/without window features or engineered window feature
