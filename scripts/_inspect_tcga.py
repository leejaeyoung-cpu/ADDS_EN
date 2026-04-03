import pandas as pd
df = pd.read_csv("data/analysis/prpc_validation/open_data/real/tcga_all_cancers_prnp_real.csv")
print(f"Shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head(3).to_string())
if "cancer_type" in df.columns:
    print(f"\nCancer types: {df['cancer_type'].value_counts().to_dict()}")
