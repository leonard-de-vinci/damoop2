"""Scenario creation, filtering, ranking and flip detection."""
from __future__ import annotations

import json as _json
from datetime import datetime
from typing import Any

import pandas as pd
import streamlit as st

from .constants import DEFAULT_QUERY_FREQUENCIES, METRIC_OPTIONS, SCENARIOS_FILE
from .data_loading import get_query_cost_stems, make_label_map, parse_query_frequencies
from .equations import (
    eq14_add_composite_score,
    eq14_normalize_metrics,
    eq14_prepare_scored_frame,
    eq17_build_stability_scores,
)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_scenarios(scenarios: dict) -> None:
    """Persist the scenarios dict to disk so they survive page refreshes."""
    try:
        with open(SCENARIOS_FILE, "w", encoding="utf-8") as _f:
            _json.dump(scenarios, _f, indent=2, default=str)
    except Exception:
        pass


def load_scenarios() -> dict | None:
    """Load scenarios from disk. Returns None if the file doesn't exist or is invalid."""
    if not SCENARIOS_FILE.exists():
        return None
    try:
        with open(SCENARIOS_FILE, "r", encoding="utf-8") as _f:
            return _json.load(_f)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# CRUD helpers (operate on st.session_state["scenarios"])
# ---------------------------------------------------------------------------
def create_scenario(scenario_id: str, label: str, base_scenario: dict, modifications: dict) -> None:
    new_scenario = base_scenario.copy()
    new_scenario["label"] = label
    new_scenario.update(modifications)
    new_scenario["created_at"] = datetime.now().isoformat()
    st.session_state["scenarios"][scenario_id] = new_scenario
    save_scenarios(st.session_state["scenarios"])


def delete_scenario(scenario_id: str) -> bool:
    if scenario_id == "baseline":
        return False
    if scenario_id in st.session_state["scenarios"]:
        del st.session_state["scenarios"][scenario_id]
        save_scenarios(st.session_state["scenarios"])
        return True
    return False


def reweight_costs(
    df: pd.DataFrame,
    new_freqs: dict[str, int],
    original_freqs: dict[str, int],
) -> pd.DataFrame:
    """Recompute total_* cost columns using new per-query frequencies.

    Formula per cost dimension:
      - Time:       new_total = Σ(q_i_unit_cost × new_freq_i)
      - Env / Fin:  new_total = Σ(q_i_unit_cost × new_freq_i) + fixed_overhead

    The fixed_overhead (storage cost for env, server cost for fin) is recovered as:
      fixed = original_total - Σ(q_i_unit_cost × original_freq_i)

    This preserves infrastructure overheads that don't scale with query frequency.
    Partial overrides are supported — missing keys in new_freqs fall back to original_freqs.
    """
    result = df.copy()

    cost_dims = [
        ("total_time_cost",          "_time_cost"),
        ("total_environmental_cost", "_env_cost"),
        ("total_financial_cost",     "_fin_cost"),
    ]

    for total_col, unit_suffix in cost_dims:
        if total_col not in result.columns:
            continue

        orig_variable = pd.Series(0.0, index=result.index)
        new_variable  = pd.Series(0.0, index=result.index)
        any_col_found = False

        for stem in get_query_cost_stems():
            unit_col = f"{stem}{unit_suffix}"
            if unit_col not in result.columns:
                continue
            any_col_found = True
            unit_vals = pd.to_numeric(result[unit_col], errors="coerce").fillna(0.0)
            orig_freq = original_freqs.get(stem, DEFAULT_QUERY_FREQUENCIES.get(stem, 0))
            new_freq  = new_freqs.get(stem, orig_freq)
            orig_variable += unit_vals * orig_freq
            new_variable  += unit_vals * new_freq

        if not any_col_found:
            continue

        orig_total = pd.to_numeric(result[total_col], errors="coerce").fillna(0.0)
        fixed_overhead = orig_total - orig_variable
        result[total_col] = new_variable + fixed_overhead

    return result


def apply_scenario_to_df(df: pd.DataFrame, scenario: dict) -> pd.DataFrame:
    df_scenario = df.copy()
    if "volumes" in scenario and scenario["volumes"]:
        df_scenario = df_scenario[df_scenario["data_volume"].isin(scenario["volumes"])]
    if "servers" in scenario and scenario["servers"]:
        df_scenario = df_scenario[df_scenario["servers"].isin(scenario["servers"])]

    freq_adj: dict = scenario.get("query_freq_adjustments", {})
    if freq_adj:
        original_freqs = parse_query_frequencies()
        effective_freqs = {**original_freqs, **freq_adj}
        df_scenario = reweight_costs(df_scenario, effective_freqs, original_freqs)

    return df_scenario


def rank_models_in_scenario(df_scenario: pd.DataFrame, metric: str = "total_time_cost") -> pd.Series:
    if df_scenario.empty:
        return pd.Series(dtype=int)
    model_avg = df_scenario.groupby("xml_file")[metric].mean().sort_values()
    ranks = pd.Series(range(1, len(model_avg) + 1), index=model_avg.index)
    return ranks


def get_scenario_summary(scenario_id: str) -> dict:
    scenario = st.session_state["scenarios"].get(scenario_id, {})
    return {
        "label": scenario.get("label", scenario_id),
        "num_volumes": len(scenario.get("volumes", [])),
        "num_servers": len(scenario.get("servers", [])),
        "description": scenario.get("description", ""),
    }


# ---------------------------------------------------------------------------
# Ranking / flip helpers (used by Scenario Analysis page)
# ---------------------------------------------------------------------------
def sc_compute_rankings(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Full per-model ranking using Eq. 14 normalisation + composite Eq. 16-17 D̃."""
    if df.empty:
        return pd.DataFrame()
    lmap = make_label_map(df["xml_file"])
    agg  = df.groupby("xml_file", as_index=False)[METRIC_OPTIONS].mean()
    agg["label"] = agg["xml_file"].map(lmap).fillna(agg["xml_file"])
    agg = eq14_normalize_metrics(agg)
    agg = eq14_add_composite_score(agg, score_col="composite")
    agg["rank_time"]      = agg["total_time_cost_norm"].rank(method="min").astype(int)
    agg["rank_env"]       = agg["total_environmental_cost_norm"].rank(method="min").astype(int)
    agg["rank_fin"]       = agg["total_financial_cost_norm"].rank(method="min").astype(int)
    agg["rank_composite"] = agg["composite"].rank(method="min").astype(int)
    agg["scenario_label"] = label
    df_c = eq14_prepare_scored_frame(df.copy(), score_col="C_sc")
    df_stab = eq17_build_stability_scores(df_c, score_col="C_sc", avg_col="avg_c")
    if not df_stab.empty:
        agg = agg.merge(df_stab[["xml_file", "d_tilde"]], on="xml_file", how="left")
        agg["rank_dtilde"] = agg["d_tilde"].rank(method="min").astype(int)
    else:
        agg["d_tilde"]     = 0.0
        agg["rank_dtilde"] = 0
    return agg.sort_values("rank_composite").reset_index(drop=True)


def sc_detect_flips(all_rankings: dict) -> pd.DataFrame:
    """Find models whose rank changed ≥ 3 places between any two scenarios."""
    rows: list[dict[str, Any]] = []
    ids  = list(all_rankings.keys())
    rank_cols = {
        "rank_time":      "Time (s)",
        "rank_env":       "Env (kg CO₂e)",
        "rank_fin":       "Financial (€)",
        "rank_composite": "Composite",
    }
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a_id, b_id = ids[i], ids[j]
            a_name = all_rankings[a_id]["scenario_label"].iloc[0]
            b_name = all_rankings[b_id]["scenario_label"].iloc[0]
            a_idx  = all_rankings[a_id].set_index("label")
            b_idx  = all_rankings[b_id].set_index("label")
            for rc, mn in rank_cols.items():
                if rc not in a_idx.columns or rc not in b_idx.columns:
                    continue
                for mdl in a_idx.index:
                    if mdl not in b_idx.index:
                        continue
                    ra, rb = int(a_idx.loc[mdl, rc]), int(b_idx.loc[mdl, rc])
                    chg = rb - ra
                    if abs(chg) >= 3:
                        rows.append({
                            "Model":      mdl,
                            "Metric":     mn,
                            "Scenario A": a_name,
                            "Scenario B": b_name,
                            "Rank in A":  ra,
                            "Rank in B":  rb,
                            "Δ Rank":     chg,
                            "Direction":  "↑ Improved" if chg < 0 else "↓ Dropped",
                        })
    return pd.DataFrame(rows) if rows else pd.DataFrame()
