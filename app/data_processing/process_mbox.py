
import sys
from pathlib import Path

# Add parent directory to path to allow absolute imports when running as script
if __name__ == "__main__":
    root_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(root_dir))

from app.data_processing.feature_extraction import FeatureExtraction
import os
import polars as pl
fe = FeatureExtraction()

dfs = []
for i in range(2024, 2012, -1):
    mbox_path = f"data/mbox_final_phish/phishing-{i}.mbox"
    df = fe.process_mbox(mbox_path, f"phishing-{i}", phishing=True)
    print(df)
    dfs.append(df)
final_df = pl.concat(dfs)
fe.write_to_file(final_df, "emails_phishing_jose.parquet")
csv_configs = [
    {"input_csv": "data/csv_misc/TREC-05.csv", "output_parquet": "trec_05.parquet"},
    {"input_csv": "data/csv_misc/TREC-06.csv", "output_parquet": "trec_06.parquet"},
    {"input_csv": "data/csv_misc/Enron.csv", "output_parquet": "enron.parquet"},
    {"input_csv": "data/csv_misc/TREC-07.csv", "output_parquet": "trec_07.parquet"},
    {"input_csv": "data/csv_misc/CEAS-08.csv", "output_parquet": "ceas_08.parquet"},
    {"input_csv": "data/csv_misc/Ling.csv", "output_parquet": "ling.parquet"},
    {"input_csv": "data/csv_misc/Assassin.csv", "output_parquet": "assassin.parquet"}
]
results = fe.process_multiple_csvs(csv_configs)
final_df = pl.concat(results)
fe.write_to_file(final_df, "emails_csv.parquet")


dfs = []
directory = os.fsencode("data/mbox_final_leg")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    if filename.endswith(".mbox"): 
        df = fe.process_mbox(f"data/mbox_final_leg/{filename}", f"{filename}", phishing=False)
        dfs.append(df)
    else:
        continue
final_df = pl.concat(dfs)
fe.write_to_file(final_df, "emails_gmail.parquet")

df = fe.process_mbox("data/mbox_final_phish/fradulent_emails.mbox", f"phishing-nigerian", phishing=True)
print(df)
fe.write_to_file(df, f"phishing_nigerian.parquet")
