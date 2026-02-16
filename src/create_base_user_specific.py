import pandas as pd
from pathlib import Path

# Configuration
MODE = 'top_n'  # Options: 'top_n' or 'all'
TOP_N = 5       # Only used when MODE = 'top_n'

# Read base.csv
base_path = Path(__file__).resolve().parent.parent / 'output' / 'base.csv'
df = pd.read_csv(base_path)

print(f"Base CSV shape: {df.shape}")
print(f"Mode: {MODE}")

# Get user counts
user_counts = df['user_id'].value_counts()
print(f"\nTotal unique users: {len(user_counts)}")

if MODE == 'top_n':
    # Get top N users
    top_users = user_counts.head(TOP_N).index.tolist()
    print(f"\nTop {TOP_N} users by frequency:")
    for user, count in user_counts.head(TOP_N).items():
        print(f"  {user}: {count} rows")
    
    # One-hot encode only top N users
    for user in top_users:
        df[f'user_{user}'] = (df['user_id'] == user).astype(int)
    
    print(f"\nCreated {TOP_N} one-hot encoded user columns")

elif MODE == 'all':
    # One-hot encode all users
    all_users = user_counts.index.tolist()
    
    for user in all_users:
        df[f'user_{user}'] = (df['user_id'] == user).astype(int)
    
    print(f"\nCreated {len(all_users)} one-hot encoded user columns")

else:
    raise ValueError(f"Invalid MODE: {MODE}. Must be 'top_n' or 'all'")

# Drop the original user_id column
df = df.drop(columns=['user_id'])
print(f"\nDropped original user_id column")

# Save to base+user_specific.csv
output_path = Path(__file__).resolve().parent.parent / 'output' / 'base+user_specific.csv'
df.to_csv(output_path, index=False)

print(f"\nBase+User Specific CSV created with shape: {df.shape}")
print(f"Saved to: {output_path}")
