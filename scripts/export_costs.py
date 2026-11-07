"""Summarise every Output/GeneratedDMs/**/costs.txt into Output/costs_TPC_C.csv.

Run after the pipeline: python -m scripts.export_costs
"""
from pathlib import Path

from lib.costs import COLUMNS, COSTS_FILE
from lib.parse_costs import build_costs_dataframe

df = build_costs_dataframe(Path("Output"))
cols = ["xml_file", "data_volume", "data_model_size", "servers", *COLUMNS]
df[cols].sort_values(["xml_file", "data_volume", "servers"]).to_csv(COSTS_FILE, index=False)
print(f"{df.xml_file.nunique()} models, {len(df)} rows -> {COSTS_FILE}")
