"""Predict page — cost prediction and best-model recommendation.

Two modes:
- **Predict one model** — legacy cost-evolution bundles: schema config + workload +
  infrastructure + a chosen data model -> instant predicted cost, with optional
  validation against the real JAR pipeline (scripts.simulate_one).
- **Recommend best model** — unit-cost ranker bundles (scripts/train_ranker.py):
  schema config + workload + infrastructure -> predicted unit costs for ALL 135
  models, workload applied via the verified closed form
  (time=U@f; env/fin=U@f+overhead), eq14-style composite, ranked table with the
  best model highlighted. "Validate" runs the real pipeline once and shows where
  the recommendation actually landed (actual rank + actual regret).
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from lib.constants import GREEN, PROJECT_ROOT, fmt_metric

REGISTRY = PROJECT_ROOT / "Output" / "models"
LEGACY = PROJECT_ROOT / "Output" / "model_cost_evolution.pkl"
COLS8 = ["w_name", "w_city", "w_tax", "c_last", "balance", "c_credit", "o_carrier_id", "o_ol_cnt"]
VOLUMES = [1000, 10000, 100000, 1000000, 10000000, 100000000]
SERVERS = [10, 50, 100, 500, 1000, 5000, 10000]
SEL_BASE = {"w_name": 1, "w_city": 2, "w_tax": 10, "c_last": 10,
            "balance": 3, "c_credit": 2, "o_carrier_id": 100, "o_ol_cnt": 15}
TARGETS = ["total_time_cost", "total_environmental_cost", "total_financial_cost"]
STEMS = [f"q{i}" for i in range(1, 21)]
UNIT_T = [f"{s}_time_cost" for s in STEMS]
UNIT_E = [f"{s}_env_cost" for s in STEMS]
UNIT_F = [f"{s}_fin_cost" for s in STEMS]


def _list_models(rankers: bool) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if REGISTRY.is_dir():
        # Discover both plain `.pkl` and committed `.pkl.xz` bundles. A fresh clone
        # ships only the compressed `.xz` (the ranker is 366 MB uncompressed); a
        # plain `.pkl` twin (decompressed cache) is preferred when present.
        seen: dict[str, Path] = {}
        for p in list(REGISTRY.glob("*.pkl")) + list(REGISTRY.glob("*.pkl.xz")):
            stem = p.name[:-7] if p.name.endswith(".pkl.xz") else p.stem
            seen.setdefault(stem, p)          # `.pkl` (listed first) wins over `.pkl.xz`
        for stem in sorted(seen):
            if ("ranker" in stem.lower()) == rankers:
                out[stem] = seen[stem]
    if not rankers and LEGACY.exists():
        out.setdefault("cost_evolution (legacy)", LEGACY)
    return out


@st.cache_resource
def _load(path_str: str):
    from lib.model_io import load_bundle      # decompresses `.pkl.xz` on first use
    return load_bundle(path_str)


def _sel_ranges(bundle: dict) -> dict:
    if "selectivity_ranges" in bundle:
        return bundle["selectivity_ranges"]
    from scripts.sweep_profiles import SELECTIVITY_RANGES
    return dict(SELECTIVITY_RANGES)


def _count_ranges(bundle: dict) -> tuple[tuple, tuple]:
    if "customer_range" in bundle:
        return bundle["customer_range"], bundle["order_range"]
    from scripts.sweep_profiles import CUSTOMER_RANGE, ORDER_RANGE
    return CUSTOMER_RANGE, ORDER_RANGE


def _schema_inputs(bundle: dict) -> tuple[int, int, dict, dict]:
    st.markdown('<div class="section-heading">Schema Configuration</div>', unsafe_allow_html=True)
    (cust_lo, cust_hi), (ord_lo, ord_hi) = _count_ranges(bundle)
    sel_ranges = _sel_ranges(bundle)
    sc1, sc2 = st.columns(2)
    customer_count = sc1.number_input("customer_count", int(cust_lo), int(cust_hi), 30_000_000, step=100_000)
    order_count = sc2.number_input("order_count", int(ord_lo), int(ord_hi), 60_000_000, step=100_000)
    sels, idxs = {}, {}
    cols = st.columns(4)
    for i, c in enumerate(COLS8):
        with cols[i % 4]:
            lo, hi = sel_ranges[c]
            sels[c] = st.slider(f"sel · {c}", int(lo), int(hi), int(SEL_BASE[c]))
            idxs[c] = st.checkbox(f"idx · {c}", value=True, key=f"pidx_{c}")
    return customer_count, order_count, sels, idxs


def _workload_inputs(default_freqs: dict) -> dict:
    st.markdown('<div class="section-heading">Workload — query frequencies</div>', unsafe_allow_html=True)
    freqs = {}
    with st.expander("Edit 20 query frequencies", expanded=False):
        fcols = st.columns(5)
        for i in range(1, 21):
            with fcols[(i - 1) % 5]:
                freqs[f"q{i}"] = st.number_input(f"q{i}", 0, value=int(default_freqs[f"q{i}"]),
                                                 step=10, key=f"pfreq_q{i}")
    for i in range(1, 21):
        freqs.setdefault(f"q{i}", int(default_freqs[f"q{i}"]))
    return freqs


def _model_labels(model_ids: list[str]) -> dict[str, str]:
    ordered = (["TPC_C.xml"] if "TPC_C.xml" in model_ids else []) + \
              [m for m in sorted(model_ids) if m != "TPC_C.xml"]
    return {xf: f"M{i:02d} · {xf.split('/')[-1].replace('.xml', '')}"
            for i, xf in enumerate(ordered)}


def render() -> None:
    st.markdown('<div class="section-heading">Mode</div>', unsafe_allow_html=True)
    mode = st.radio("Mode", ["Recommend best model", "Predict one model"],
                    horizontal=True, label_visibility="collapsed", key="pred_mode")
    recommend = mode == "Recommend best model"

    models = _list_models(rankers=recommend)
    if not models:
        if recommend:
            st.warning("No ranker bundle found. Train one (CLI):\n\n"
                       "`python -m scripts.train_ranker` — needs `Output/sweep_ml.parquet` "
                       "(Profile Sweep → Assemble).")
        else:
            st.warning("No trained model found. Train one (CLI):\n\n"
                       "`python -m scripts.train_predictor --name cost_evolution`")
        return

    st.markdown('<div class="section-heading">Model</div>', unsafe_allow_html=True)
    name = st.selectbox("Trained model", options=list(models.keys()), key="pred_model_" + ("r" if recommend else "p"))
    bundle = _load(str(models[name]))

    if recommend:
        if bundle.get("kind") != "unit_cost_ranker":
            st.error("This bundle is not a unit-cost ranker — pick a `model_ranker_*` bundle "
                     "or train one with `python -m scripts.train_ranker`.")
            return
        _render_recommend(bundle)
    else:
        _render_predict(bundle)


# ═══════════════════════════════════════════════════════════════════════════
# Mode 1 — Recommend best model (unit-cost ranker + closed-form workload)
# ═══════════════════════════════════════════════════════════════════════════
def _render_recommend(bundle: dict) -> None:
    meta = bundle.get("meta", {})
    mm = meta.get("metrics", {})
    if mm:
        st.caption(
            f"Ranker `{meta.get('name', '?')}` (objective **{bundle.get('objective')}**) · "
            f"trained {meta.get('created', '?')} on {meta.get('train_rows', 0):,} rows · "
            f"held-out ranking: median regret {mm.get('regret_median', 0):.1%}, "
            f"top-1 {mm.get('top1_acc', 0):.0%}, top-3 {mm.get('top3_recall', 0):.0%}, "
            f"Spearman {mm.get('spearman_mean', 0):.2f}"
        )

    st.markdown('<div class="section-heading">Infrastructure</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    volume = c1.select_slider("Data volume (warehouses)", options=VOLUMES, value=1000000,
                              format_func=lambda v: f"{v:,}")
    servers = c2.select_slider("Servers", options=SERVERS, value=100, format_func=lambda v: f"{v:,}")

    customer_count, order_count, sels, idxs = _schema_inputs(bundle)
    freqs = _workload_inputs(bundle["default_freqs"])
    f = np.array([freqs[s] for s in STEMS], float)

    code_map: dict[str, int] = bundle["model_code_map"]
    labels = _model_labels(list(code_map.keys()))

    def _rank_frame() -> pd.DataFrame:
        """Predict unit costs for every model, apply the workload analytically, rank."""
        model_ids = sorted(code_map, key=code_map.get)
        base = {"customer_count": customer_count, "order_count": order_count,
                "data_volume": volume, "servers": servers}
        base |= {f"sel_{c}": sels[c] for c in COLS8}
        X = pd.DataFrame([base] * len(model_ids))
        for c in bundle["log_feats"]:
            X[f"lg_{c}"] = np.log10(X[c].astype(float)) if c in X else np.nan
        for c in COLS8:
            X[f"idx_{c}"] = int(idxs[c])
        X["model_code"] = [code_map[m] for m in model_ids]
        X = X[bundle["features"]]

        objective, scales = bundle["objective"], bundle.get("target_scales", {})
        pred = {}
        for t, m in bundle["models"].items():
            p = m.predict(X)
            if objective == "gamma":
                p = p * scales.get(t, 1.0)
            elif objective == "log":
                p = np.expm1(p)
            pred[t] = np.clip(np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)

        Ut = np.column_stack([pred[c] for c in UNIT_T])
        Ue = np.column_stack([pred[c] for c in UNIT_E])
        Uf = np.column_stack([pred[c] for c in UNIT_F])
        tt = Ut @ f
        ee = Ue @ f + pred["env_overhead"]
        ff = Uf @ f + pred["fin_overhead"]

        def _n01(v):
            v = np.log1p(np.maximum(v, 0.0)); s = v.max() - v.min()
            return (v - v.min()) / s if s > 0 else np.zeros_like(v)

        comp = (_n01(tt) + _n01(ee) + _n01(ff)) / 3
        out = pd.DataFrame({
            "model": [labels[m] for m in model_ids], "xml_file": model_ids,
            "total_time_cost": tt, "total_environmental_cost": ee,
            "total_financial_cost": ff, "composite": comp,
        }).sort_values("composite").reset_index(drop=True)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out

    st.markdown("---")
    b1, b2 = st.columns(2)
    if b1.button("Recommend", type="primary", key="rec_btn"):
        st.session_state["rec_result"] = _rank_frame()
    do_validate = b2.button("Validate recommendation (real JAR, ~10 s)", key="rec_val_btn",
                            disabled="rec_result" not in st.session_state,
                            help="Runs the actual pipeline for this config and shows where the "
                                 "recommendation really lands among all 135 models.")

    ranked = st.session_state.get("rec_result")
    if ranked is None:
        st.info("Set the config and press **Recommend** to rank all 135 data models.")
        return

    best = ranked.iloc[0]
    margin = float(ranked.iloc[1]["composite"] - best["composite"]) if len(ranked) > 1 else 0.0
    st.markdown('<div class="section-heading">Recommended data model</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Best model", best["model"].split(" · ")[0], help=best["xml_file"])
    k2.metric(fmt_metric("total_time_cost"), f"{best['total_time_cost']:,.4g}")
    k3.metric(fmt_metric("total_environmental_cost"), f"{best['total_environmental_cost']:,.4g}")
    k4.metric(fmt_metric("total_financial_cost"), f"{best['total_financial_cost']:,.4g}")
    st.caption(f"`{best['xml_file']}` · composite margin to #2: {margin:.4f} "
               f"(small margins mean #2 is nearly as good)")

    show = ranked.copy()
    for t in TARGETS:
        show[fmt_metric(t)] = show[t].map(lambda v: f"{v:,.4g}")
    show["composite"] = show["composite"].round(4)
    cols = ["rank", "model", *[fmt_metric(t) for t in TARGETS], "composite"]
    st.markdown('<div class="section-heading">Top 10 (predicted)</div>', unsafe_allow_html=True)
    st.dataframe(show.head(10)[cols], width="stretch", hide_index=True)
    with st.expander(f"Full ranking ({len(show)} models)"):
        st.dataframe(show[cols + ["xml_file"]], width="stretch", hide_index=True, height=500)

    if do_validate:
        cfg = {"selectivities": sels, "index_flags": idxs,
               "customer_count": customer_count, "order_count": order_count,
               "data_volume": volume, "servers": servers, "freqs": freqs}
        from scripts.simulate_one import simulate_frame
        with st.spinner("Running the real DMG + Cost pipeline for this config (~10 s)…"):
            try:
                act = simulate_frame(cfg)
            except Exception as e:  # noqa: BLE001
                st.error(f"Simulation failed: {e}")
                return
        for t in TARGETS:
            act[t] = pd.to_numeric(act[t], errors="coerce")

        def _n01(v):
            v = np.log1p(np.maximum(v, 0.0)); s = v.max() - v.min()
            return (v - v.min()) / s if s > 0 else np.zeros_like(v)

        a = act[["xml_file"] + TARGETS].copy()
        a["composite"] = (_n01(a[TARGETS[0]].to_numpy(float))
                          + _n01(a[TARGETS[1]].to_numpy(float))
                          + _n01(a[TARGETS[2]].to_numpy(float))) / 3
        a = a.sort_values("composite").reset_index(drop=True)
        a.insert(0, "actual_rank", range(1, len(a) + 1))

        pick = best["xml_file"]
        pick_rows = a[a["xml_file"] == pick]
        st.markdown('<div class="section-heading">Validation — real pipeline</div>', unsafe_allow_html=True)
        if pick_rows.empty:
            st.warning(f"Recommended model `{pick}` not found in the simulated output.")
            return
        pr = pick_rows.iloc[0]
        true_best = a.iloc[0]
        regret = float(np.mean([(pr[t] - true_best[t]) / max(true_best[t], 1e-9) for t in TARGETS]))
        v1, v2, v3 = st.columns(3)
        v1.metric("Actual rank of recommendation", f"#{int(pr['actual_rank'])} / {len(a)}")
        v2.metric("Actual regret vs true best", f"{regret:.2%}",
                  help="Mean % extra cost across time/env/financial if you use the recommended "
                       "model instead of the actual cheapest.")
        v3.metric("Actual best model", labels.get(true_best["xml_file"], true_best["xml_file"]).split(" · ")[0],
                  help=true_best["xml_file"])
        if int(pr["actual_rank"]) == 1:
            st.markdown(f'<span style="color:{GREEN};font-weight:600">The recommendation IS the '
                        'actual cheapest model for this config.</span>', unsafe_allow_html=True)
        av = a.head(5).copy()
        av["model"] = av["xml_file"].map(lambda x: labels.get(x, x))
        for t in TARGETS:
            av[fmt_metric(t)] = av[t].map(lambda v: f"{v:,.4g}")
        st.caption("Actual top 5 (real JAR, this exact config + workload):")
        st.dataframe(av[["actual_rank", "model", *[fmt_metric(t) for t in TARGETS]]],
                     width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# Mode 2 — Predict one model (legacy cost-evolution bundle)
# ═══════════════════════════════════════════════════════════════════════════
def _render_predict(bundle: dict) -> None:
    meta = bundle.get("meta", {})
    if meta:
        hm = meta.get("held_out_metrics", {})
        r2s = " · ".join(f"{t.replace('total_', '').replace('_cost', '')}: R²={hm[t]['r2']}"
                         for t in TARGETS if t in hm)
        st.caption(f"**{meta.get('algorithm', '?')}** · trained {meta.get('created', '?')} · "
                   f"{meta.get('train_rows', '?'):,} rows from `{meta.get('dataset', '?')}` · {r2s}")
    else:
        st.caption("Legacy model (no metadata).")

    models_d = bundle["models"]
    features = bundle["features"]
    default_freqs = bundle["default_freqs"]
    model_map = bundle["model_code_map"]

    st.markdown('<div class="section-heading">Infrastructure & Data Model</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    volume = c1.select_slider("Data volume (warehouses)", options=VOLUMES, value=1000000,
                              format_func=lambda v: f"{v:,}")
    servers = c2.select_slider("Servers", options=SERVERS, value=100, format_func=lambda v: f"{v:,}")
    model_ids = sorted(model_map.keys())
    if "TPC_C.xml" in model_ids:
        model_ids = ["TPC_C.xml"] + [m for m in model_ids if m != "TPC_C.xml"]
    mlabel = {xf: f"M{i:02d} · {xf.split('/')[-1].replace('.xml', '')}"
              for i, xf in enumerate(model_ids)}
    model_id = c3.selectbox("Data model", options=model_ids,
                            format_func=lambda xf: mlabel[xf], index=0)

    customer_count, order_count, sels, idxs = _schema_inputs(bundle)
    freqs = _workload_inputs(default_freqs)

    def _build_X():
        row = {"customer_count": customer_count, "order_count": order_count,
               "data_volume": volume, "servers": servers, "model_code": model_map.get(model_id, 0)}
        for c in COLS8:
            row[f"sel_{c}"] = sels[c]; row[f"idx_{c}"] = int(idxs[c])
        for i in range(1, 21):
            row[f"freq_q{i}"] = freqs[f"q{i}"]
        return pd.DataFrame([[row[f] for f in features]], columns=features)

    normalized = bundle.get("normalized", False)
    base_tables = bundle.get("base_tables", {})

    def _predict():
        X = _build_X()
        out = {}
        for t in TARGETS:
            log_pred = float(models_d[t].predict(X)[0])
            if normalized:
                # model predicts the residual; add back the (volume, servers, model) base
                base = base_tables.get(t, {}).get((int(volume), int(servers), model_id), 0.0)
                log_pred += base
            out[t] = float(np.expm1(log_pred))
        return out

    st.markdown("---")
    b1, b2 = st.columns(2)
    do_predict = b1.button("Predict cost", type="primary", key="pred_btn")
    do_validate = b2.button("Validate against real cost", key="pred_validate_btn",
                            help="Runs the actual JAR pipeline for this config (~10 s) and compares.")

    if do_predict:
        st.session_state["pred_result"] = _predict()
    if "pred_result" in st.session_state and not do_validate:
        st.markdown('<div class="section-heading">Predicted cost</div>', unsafe_allow_html=True)
        mc = st.columns(3)
        for col, t in zip(mc, TARGETS):
            col.metric(fmt_metric(t), f"{st.session_state['pred_result'][t]:,.4g}")
        st.caption("Predicted in log space, exp-transformed to real units. Typical error ≈ model MAE.")

    if do_validate:
        pred = _predict()
        cfg = {
            "selectivities": sels, "index_flags": idxs,
            "customer_count": customer_count, "order_count": order_count,
            "model": model_id, "data_volume": volume, "servers": servers, "freqs": freqs,
        }
        # Call inline (not a fresh subprocess) — avoids re-importing the whole stack
        # (~3-5 s) on every click. The JAR runs are still subprocesses inside simulate().
        from scripts.simulate_one import simulate
        with st.spinner("Running the real DMG + Cost pipeline for this config (~10 s)…"):
            try:
                actual = simulate(cfg)
            except Exception as e:  # noqa: BLE001
                st.error(f"Simulation failed: {e}")
                return
        st.markdown('<div class="section-heading">Predicted vs. actual (real pipeline)</div>',
                    unsafe_allow_html=True)
        rows = []
        for t in TARGETS:
            p, a = pred[t], actual[t]
            err = abs(p - a) / a * 100 if a else float("nan")
            rows.append({"metric": fmt_metric(t), "predicted": f"{p:,.4g}",
                         "actual": f"{a:,.4g}", "error %": f"{err:.1f}%",
                         "ratio": f"{p / a:.2f}x" if a else "—"})
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption("Actual = real DMgeneration + CostCalculation pipeline for this exact config, "
                   "reweighted by the workload. The model never ran the JAR.")
