"""Total costs per model, read from the CostCalculation.jar output."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from lib.signatures import Model

COSTS_FILE = Path("Output/costs_TPC_C.csv")
COLUMNS = {
    "total_time_cost": "Time",
    "total_environmental_cost": "Carbon",
    "total_financial_cost": "Financial",
}


def model_key(model: Model) -> str:
    """Same id as lib.parse_costs: folder path under GeneratedDMs, or the base XML for M0."""
    rel = model.path.split("GeneratedDMs/", 1)[-1].strip("/")
    return rel or "TPC_C.xml"


def load_costs(path: Path | str = COSTS_FILE) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    return pd.read_csv(path).rename(columns=COLUMNS)


def costs_at(df: pd.DataFrame, volume: float, servers: float) -> pd.DataFrame:
    """One row per model for a given volume and number of servers, indexed by model key."""
    sub = df[(df.data_volume == volume) & (df.servers == servers)]
    return sub.set_index("xml_file")[list(COLUMNS.values())]
