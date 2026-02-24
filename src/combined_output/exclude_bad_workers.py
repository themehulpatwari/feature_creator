import pandas as pd
from pathlib import Path

bad_workers = ['A1AYL5V5GI9HC1', 'A1CMJNN1QQKLU8', 'A1DLZK8TZJ7ESF', 'A1HQGU5SA9UORE', 'A1SMWIBTQ2AESU', 'A1SUMUAP2HD6I1', 'A24UWUL93BCA60', 'A27WZ1GYA5H2JZ', 'A2D07SVRO4UFVL', 'A2IXADH4RS1WEZ', 'A2M3DR0V45HNUL', 'A2TG71MJ93AECP', 'A2VBD0D63OY36V', 'A34O453D7VWWUK', 'A34ZJFQ9UCP1CR', 'A3AA5G6HENO6VJ', 'A3EFKBV6W73WTX', 'A3I8V1SR4ZGLCI', 'A3PJ51GS2AKBO6', 'A3U042Q64BVD6G', 'A3UWNTYN8750BT', 'AAGUHBMCOYB8J', 'AM0JKZVOEOTMA', 'AXRWAXX9EXR0Y']

input_dir = Path(__file__).resolve().parent.parent.parent / 'input'

# List of CSV files to process
csv_files = [
    'extracted_features.csv',
    'all_metrics.csv',
    'query_logs_table.csv'
]

for csv_file in csv_files:
    csv_path = input_dir / csv_file
    
    if not csv_path.exists():
        print(f"\n⚠️  Skipping {csv_file} - file not found")
        continue
    
    print(f"\n{'='*60}")
    print(f"Processing: {csv_file}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    
    print(f"Initial shape: {df.shape}")
    print(f"Initial rows: {len(df)}")
    
    df_filtered = df[~df['user_id'].isin(bad_workers)]
    
    print(f"Filtered shape: {df_filtered.shape}")
    print(f"Filtered rows: {len(df_filtered)}")
    print(f"Rows removed: {len(df) - len(df_filtered)}")
    
    df_filtered.to_csv(csv_path, index=False)
    print(f"✓ Filtered data saved to {csv_path}")

print(f"\n{'='*60}")
print(f"All files processed successfully!")
print(f"{'='*60}")

