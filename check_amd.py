import pandas as pd
df = pd.read_parquet('/app/shared/data/cache/AMD_1m.parquet')
if df.index.tz is None:
    df.index = df.index.tz_localize('UTC')
df = df.tz_convert('America/New_York')
sub = df.loc['2026-03-23 09:30':'2026-03-23 09:40']
print(sub)
