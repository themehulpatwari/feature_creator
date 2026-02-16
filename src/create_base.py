import pandas as pd
from pathlib import Path

# List of bad workers to exclude
bad_workers = ['A1AYL5V5GI9HC1', 'A1CMJNN1QQKLU8', 'A1DLZK8TZJ7ESF', 'A1HQGU5SA9UORE', 
               'A1SMWIBTQ2AESU', 'A1SUMUAP2HD6I1', 'A24UWUL93BCA60', 'A27WZ1GYA5H2JZ', 
               'A2D07SVRO4UFVL', 'A2IXADH4RS1WEZ', 'A2M3DR0V45HNUL', 'A2TG71MJ93AECP', 
               'A2VBD0D63OY36V', 'A34O453D7VWWUK', 'A34ZJFQ9UCP1CR', 'A3AA5G6HENO6VJ', 
               'A3EFKBV6W73WTX', 'A3I8V1SR4ZGLCI', 'A3PJ51GS2AKBO6', 'A3U042Q64BVD6G', 
               'A3UWNTYN8750BT', 'AAGUHBMCOYB8J', 'AM0JKZVOEOTMA', 'AXRWAXX9EXR0Y']

input_path = Path(__file__).resolve().parent.parent / 'input' / 'query_logs_table.csv'
df = pd.read_csv(input_path)

print(f"Initial shape: {df.shape}")

df = df[~df['user_id'].isin(bad_workers)]
print(f"After filtering bad workers: {df.shape}")

# Clean llm_name columns - remove newlines, carriage returns, and extra whitespace
df['llm_name_1'] = df['llm_name_1'].apply(lambda x: x.replace('\r\n', '').replace('\n', '').replace('\r', '').strip() if pd.notna(x) else x)
df['llm_name_2'] = df['llm_name_2'].apply(lambda x: x.replace('\r\n', '').replace('\n', '').replace('\r', '').strip() if pd.notna(x) else x)
print("Cleaned llm_name columns")

df['comparison_type'] = df.apply(
    lambda row: 'pointwise' if pd.isna(row['llm_response_2']) and pd.isna(row['llm_name_2']) else 'pairwise',
    axis=1
)

all_llm = pd.concat([
    df['llm_name_1'].dropna(),
    df['llm_name_2'].dropna()
]).unique()

print(f"\nUnique LLMs found: {sorted(all_llm)}")

# One-hot encode llm_name_1
for llm in all_llm:
    df[f'llm_1_{llm}'] = (df['llm_name_1'] == llm).astype(int)

# One-hot encode llm_name_2
for llm in all_llm:
    df[f'llm_2_{llm}'] = (df['llm_name_2'] == llm).astype(int)

# Drop the original llm_name columns
df = df.drop(columns=['llm_name_1', 'llm_name_2'])
print("\nDropped original llm_name_1 and llm_name_2 columns")

# Reorder columns to put comparison_type at the front
cols = df.columns.tolist()
cols.remove('comparison_type')
df = df[['comparison_type'] + cols]

# Save to base.csv
output_path = Path(__file__).resolve().parent.parent / 'output' / 'base.csv'
df.to_csv(output_path, index=False)

print(f"\nBase CSV created with shape: {df.shape}")
print(f"Saved to: {output_path}")
print(f"\nComparison type distribution:")
print(df['comparison_type'].value_counts())
