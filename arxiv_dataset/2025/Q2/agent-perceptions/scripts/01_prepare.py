# IRB Exemption: No personally identifiable information (PII) is present.
# This dataset is exempt from IRB review.
import pandas as pd
import json
from pathlib import Path
import numpy as np

np.random.seed(42)

RAW = 'data/raw/survey_responses_rows_20250601.csv'  # Update path if needed
data_dir = Path('data')
data_dir.mkdir(exist_ok=True)

raw = pd.read_csv(RAW)

# Drop 'id' column if present
drop_cols = [col for col in ['id'] if col in raw.columns]
if drop_cols:
    raw = raw.drop(columns=drop_cols)

# Jitter 'created_at' by ±5 min if present
if 'created_at' in raw.columns:
    raw['created_at'] = pd.to_datetime(raw['created_at'], errors='coerce')
    jitter = np.random.randint(-5, 6, size=len(raw))  # minutes
    raw['created_at'] = raw['created_at'] + pd.to_timedelta(jitter, unit='m')

# Expand JSON answer blob into columns Q1..Q11
answers = raw['answers'].apply(json.loads).apply(pd.Series)


# Region (coarse) from metadata if present
def extract_region(metadata):
    try:
        return json.loads(metadata).get('timeZone', 'Unknown').split('/')[0]
    except Exception:
        return 'Unknown'


raw['region'] = raw['metadata'].apply(extract_region)

df = pd.concat([raw[['created_at', 'region']], answers], axis=1)

# Fix Q3: merge old and new regulatory/compliance answers
if '3' in df.columns:
    df['3'] = df['3'].replace(
        'Regulatory/compliance concerns',
        'Regulatory/compliance or technical readiness concerns'
    )

# Fix Q5: merge 'Emergent consciousness' with 'No opinion'
if '5' in df.columns:
    df['5'] = df['5'].replace('Emergent consciousness', 'No opinion')

# Drop duplicates, strip whitespace
df = df.drop_duplicates().reset_index(drop=True)
for q in range(1, 11):
    df[str(q)] = pd.Categorical(df[str(q)].str.strip(), ordered=False)

# Save tidy data
out_parquet = data_dir / 'clean_survey.parquet'
df.to_parquet(out_parquet, index=False)

# Save Q11 free-text
q11 = df['11'].dropna().astype(str).str.strip()
q11_out = data_dir / 'q11.txt'
q11.to_csv(q11_out, index=False, header=False)

print(f"Saved: {out_parquet}, {q11_out}")