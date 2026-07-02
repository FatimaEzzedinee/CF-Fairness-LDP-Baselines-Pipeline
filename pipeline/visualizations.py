"""
visualizations.py — All figures for the fairness / privacy pipeline.

Figures produced
----------------
Analysis 1 — Fairness
  F1  Group fairness bar chart (SPD, EOD, AOD, EqOdds, DI, PP) per model/scenario
  F2  Radar (spider) chart of group-fairness metrics across all scenarios
  F3  Per-group positive-rate comparison (Male vs Female)
  F4  TPR / FPR per group heatmap
  F5  Individual fairness: consistency + CF-fairness bar chart
  F6  Theil Index comparison

Analysis 2 — MIA
  M1  AUC-ROC bar chart — all attackers, both input types, all scenarios
  M2  ROC curves overlay — best attacker per scenario/input-type
  M3  MIA advantage (TPR − FPR) grouped bar chart
  M4  Precision / Recall scatter
  M5  Privacy gain heatmap (scenario × model-family)

Analysis 3 — NiCE CF Quality
  C1  Proximity comparison across scenarios
  C2  Plausibility comparison
  C3  Sparsity comparison
  C4  CF quality radar chart (proximity, plausibility, sparsity)

Combined
  X1  Full comparison dashboard (fairness + MIA + CF quality side-by-side)

All figures are saved to  OUTPUT_DIR/figures/  as PNG files.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")    # non-interactive backend — safe in all environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.config import OUTPUT_DIR, FIGURE_DPI, FIGURE_EXT, GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------

PALETTE = {
    "baseline":                    "#555555",   # dark grey
    "augmented":                   "#2ca02c",   # green  (legacy — kept for backward compat)
    "augmented_SCM":               "#2ca02c",   # green
    "augmented_update_labels":     "#e377c2",   # pink/magenta
    "augmented_add_comparators":   "#9467bd",   # purple
    "ldp":                         "#d62728",   # red
}
SCENARIO_LABELS = {
    "baseline":                    "Baseline",
    "augmented":                   "Augmented\n(+MM-CFs)",
    "augmented_SCM":               "Aug\n(SCM)",
    "augmented_update_labels":     "Aug\n(Upd. Labels)",
    "augmented_add_comparators":   "Aug\n(Add Comp.)",
    "ldp":                         "LDP-Aug.",
}

# Fixed family colours — used wherever bars are grouped by model family.
# Deliberately separate from PALETTE (which is scenario-based) so that
# family bars never accidentally inherit a scenario colour.
FAMILY_COLORS = ["#4C72B0", "#DD8452", "#55A868"]   # blue, orange, green  (LR / RF / XGB)
FAMILY_MARKERS = {
    "logistic_regression": "o",
    "random_forest":       "s",
    "xgboost":             "^",
}
FAMILY_LABELS = {
    "logistic_regression": "LR",
    "random_forest":       "RF",
    "xgboost":             "XGB",
}

_FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def _save(fig: plt.Figure, name: str, tight: bool = True) -> str:
    """Save figure to OUTPUT_DIR/figures/<name>.<FIGURE_EXT>."""
    os.makedirs(_FIG_DIR, exist_ok=True)
    path = os.path.join(_FIG_DIR, f"{name}.{FIGURE_EXT}")
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved -> {path}")
    return path


def _legend_if_present(ax: plt.Axes, **kwargs) -> None:
    """Add a legend only when the axis has labeled artists."""
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, **kwargs)


def _scenario_color(sc: str) -> str:
    # Exact match first
    if sc in PALETTE:
        return PALETTE[sc]
    # Prefix match for augmented variants
    if sc.startswith("augmented_"):
        # Map unknown augmented variants to the generic augmented color
        return PALETTE.get(sc, "#4CAF50")
    # Prefix match for ldp variants
    if sc.startswith("ldp_"):
        return PALETTE.get("ldp", "#FF9800")
    return "#999999"


# ---------------------------------------------------------------------------
# Analysis 1 — Fairness
# ---------------------------------------------------------------------------

def plot_group_fairness_bars(
    fairness_results: Dict,
    metrics: List[str] = ("SPD", "EOD", "AOD", "EqOdds", "PP"),
) -> str:
    """F1 — Grouped bar chart of group-fairness metrics.

    One cluster per metric; bars coloured by scenario; three bars per cluster
    (one per model family).  Horizontal reference line at 0 marks perfect fairness.
    """
    scenarios  = list(fairness_results.keys())
    families   = list(next(iter(fairness_results.values())).keys())
    n_metrics  = len(metrics)
    n_families = len(families)
    n_scenarios = len(scenarios)

    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        x_base = np.arange(n_scenarios * n_families)
        ticks, tick_labels, colors, values = [], [], [], []

        for si, sc in enumerate(scenarios):
            for fi, fam in enumerate(families):
                idx = si * n_families + fi
                fr  = fairness_results[sc][fam]
                val = fr.group.get(metric, 0.0)
                ticks.append(idx)
                tick_labels.append(f"{SCENARIO_LABELS.get(sc, sc)[:3]}\n{FAMILY_LABELS.get(fam, fam)}")
                colors.append(_scenario_color(sc))
                values.append(val)

        bars = ax.bar(ticks, values, color=colors, edgecolor="white", linewidth=0.5, width=0.7)
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(ticks)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel("Value")

        # Value labels on bars
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if val >= 0 else -0.015),
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # Legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=_scenario_color(sc),
                             label=SCENARIO_LABELS.get(sc, sc))
                      for sc in scenarios]
    fig.legend(handles=legend_handles, loc="lower center",
               ncol=len(scenarios), bbox_to_anchor=(0.5, -0.05),
               frameon=False)

    fig.suptitle("Group Fairness Metrics (ideal = 0)", fontsize=13, fontweight="bold", y=1.02)
    return _save(fig, "F1_group_fairness_bars")


def plot_fairness_radar(
    fairness_results: Dict,
    metrics: List[str] = ("SPD", "EOD", "AOD", "EqOdds", "DI", "PP"),
) -> str:
    """F2 — Radar chart of absolute group-fairness metric values per scenario.

    DI is remapped to |DI − 1| so that 0 = perfect for all axes.
    """
    scenarios = list(fairness_results.keys())
    families  = list(next(iter(fairness_results.values())).keys())
    n_metrics = len(metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5),
                              subplot_kw=dict(polar=True))
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontweight="bold", pad=15)
        ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=8)

        for sc in scenarios:
            fr = fairness_results[sc][fam]
            vals = []
            for m in metrics:
                v = fr.group.get(m, 0.0)
                if m == "DI":
                    v = abs(v - 1.0)   # remap to deviation from 1
                else:
                    v = abs(v)          # use absolute magnitude
                vals.append(v)
            vals += vals[:1]
            ax.plot(angles, vals, color=_scenario_color(sc), linewidth=2,
                    label=SCENARIO_LABELS.get(sc, sc))
            ax.fill(angles, vals, color=_scenario_color(sc), alpha=0.1)

        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)

    fig.suptitle("Group Fairness Radar (lower = fairer)", fontsize=13, fontweight="bold")
    return _save(fig, "F2_fairness_radar")


def plot_group_positive_rates(fairness_results: Dict) -> str:
    """F3 — Positive-prediction rate for Male vs Female across scenarios and families."""
    scenarios = list(fairness_results.keys())
    families  = list(next(iter(fairness_results.values())).keys())

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4), sharey=True)
    if len(families) == 1:
        axes = [axes]

    x = np.arange(len(scenarios))
    width = 0.35

    for ax, fam in zip(axes, families):
        priv_rates   = [fairness_results[sc][fam].group.get(f"PosRate_{GROUP_PRIV_LABEL}",   0) for sc in scenarios]
        unpriv_rates = [fairness_results[sc][fam].group.get(f"PosRate_{GROUP_UNPRIV_LABEL}", 0) for sc in scenarios]

        ax.bar(x - width/2, priv_rates,   width, label=GROUP_PRIV_LABEL,   color="#1E88E5", alpha=0.85)
        ax.bar(x + width/2, unpriv_rates, width, label=GROUP_UNPRIV_LABEL, color="#E91E63", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=9)
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontweight="bold")
        ax.set_ylabel("Positive Prediction Rate")
        ax.set_ylim(0, 1)
        ax.legend()

    fig.suptitle(f"Positive Prediction Rate: {GROUP_PRIV_LABEL} vs {GROUP_UNPRIV_LABEL}", fontsize=13, fontweight="bold")
    return _save(fig, "F3_positive_rates_by_gender")


def plot_tpr_fpr_heatmap(fairness_results: Dict) -> str:
    """F4 — TPR and FPR per group as a heatmap (scenario × group)."""
    scenarios = list(fairness_results.keys())
    families  = list(next(iter(fairness_results.values())).keys())
    groups    = [GROUP_PRIV_LABEL, GROUP_UNPRIV_LABEL]
    metrics   = ["TPR", "FPR"]

    # One figure per family (rows = metric, columns = group)
    fig, axes = plt.subplots(
        len(families), 1,
        figsize=(7, 3 * len(families)),
    )
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        data_matrix = np.zeros((len(scenarios), len(groups) * len(metrics)))
        col_labels = [f"{g} {m}" for m in metrics for g in groups]

        for si, sc in enumerate(scenarios):
            fr = fairness_results[sc][fam]
            data_matrix[si] = [
                fr.group.get(f"TPR_{GROUP_PRIV_LABEL}", 0),
                fr.group.get(f"TPR_{GROUP_UNPRIV_LABEL}", 0),
                fr.group.get(f"FPR_{GROUP_PRIV_LABEL}", 0),
                fr.group.get(f"FPR_{GROUP_UNPRIV_LABEL}", 0),
            ]

        im = ax.imshow(data_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([SCENARIO_LABELS.get(s, s) for s in scenarios], fontsize=9)
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontweight="bold")
        for i in range(len(scenarios)):
            for j in range(len(col_labels)):
                ax.text(j, i, f"{data_matrix[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"TPR / FPR by Race ({GROUP_PRIV_LABEL} vs {GROUP_UNPRIV_LABEL}) and Scenario", fontsize=13, fontweight="bold")
    return _save(fig, "F4_tpr_fpr_heatmap")


def plot_individual_fairness(fairness_results: Dict) -> str:
    """F5 — Consistency and Counterfactual Fairness bar chart."""
    scenarios = list(fairness_results.keys())
    families  = list(next(iter(fairness_results.values())).keys())
    ind_metrics = ["consistency", "cf_fairness"]
    labels_map  = {"consistency": "k-NN Consistency", "cf_fairness": "CF Fairness"}

    fig, axes = plt.subplots(1, len(ind_metrics), figsize=(5 * len(ind_metrics), 4))
    if len(ind_metrics) == 1:
        axes = [axes]

    x = np.arange(len(scenarios))
    width = 0.25

    for ax, metric in zip(axes, ind_metrics):
        for fi, fam in enumerate(families):
            vals = [fairness_results[sc][fam].individual.get(metric, 0) for sc in scenarios]
            offset = (fi - len(families) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=FAMILY_LABELS.get(fam, fam),
                   color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                   alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=9)
        ax.set_ylabel("Score [0–1]")
        ax.set_ylim(0, 1.05)
        ax.axhline(1, color="gray", linestyle="--", linewidth=0.8)
        ax.set_title(labels_map.get(metric, metric), fontweight="bold")
        ax.legend()

    fig.suptitle("Individual Fairness Metrics", fontsize=13, fontweight="bold")
    return _save(fig, "F5_individual_fairness")


def plot_theil_index(fairness_results: Dict) -> str:
    """F6 — Theil Index comparison across scenarios and model families."""
    scenarios = list(fairness_results.keys())
    families  = list(next(iter(fairness_results.values())).keys())

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(scenarios))
    width = 0.25

    for fi, fam in enumerate(families):
        vals = [fairness_results[sc][fam].group.get("Theil", 0) for sc in scenarios]
        offset = (fi - len(families) / 2 + 0.5) * width
        ax.bar(x + offset, vals, width,
               label=FAMILY_LABELS.get(fam, fam),
               color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
               alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios])
    ax.set_ylabel("Theil Index")
    ax.set_title("Theil Index of Individual Benefit", fontweight="bold")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    return _save(fig, "F6_theil_index")


# ---------------------------------------------------------------------------
# Analysis 2 — MIA
# ---------------------------------------------------------------------------

def plot_mia_auc_bars(mia_df: pd.DataFrame, input_type: str = "mm_cf") -> str:
    """M1 — AUC-ROC bar chart for all attackers and scenarios."""
    df = mia_df[mia_df["input_type"] == input_type].copy()
    if df.empty:
        print(f"  [fig] No MIA data for input_type={input_type}, skipping M1.")
        return ""

    scenarios = df["scenario"].unique()
    families  = df["family"].unique()
    # Show top attackers by mean AUC
    top_attackers = (df.groupby("attacker")["auc_roc"]
                       .mean()
                       .nlargest(12)
                       .index.tolist())
    df = df[df["attacker"].isin(top_attackers)]

    fig, axes = plt.subplots(1, len(families), figsize=(6 * len(families), 5), sharey=True)
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        sub  = df[df["family"] == fam]
        atts = top_attackers
        x    = np.arange(len(atts))
        width = 0.8 / max(len(scenarios), 1)

        for si, sc in enumerate(scenarios):
            sub2 = sub[sub["scenario"] == sc]
            vals = [float(sub2[sub2["attacker"] == a]["auc_roc"].mean()
                          if not sub2[sub2["attacker"] == a].empty else 0.5)
                    for a in atts]
            offset = (si - len(scenarios) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=SCENARIO_LABELS.get(sc, sc),
                   color=_scenario_color(sc),
                   alpha=0.85)

        ax.axhline(0.5, color="black", linestyle="--", linewidth=0.8, label="Random (0.5)")
        ax.set_xticks(x)
        ax.set_xticklabels(atts, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC-ROC")
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle(f"MIA AUC-ROC per Attacker ({input_type.upper()})",
                 fontsize=13, fontweight="bold")
    return _save(fig, f"M1_mia_auc_bars_{input_type}")


def plot_mia_roc_curves(
    all_results: Dict,
    input_type: str = "mm_cf",
    attacker_name: str = "ensemble_mean",
) -> str:
    """M2 — ROC curves overlay for the chosen attacker across all scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    families = None

    for ax, sc_name in zip(axes, ["baseline", "augmented", "ldp"]):
        if sc_name not in all_results:
            ax.set_visible(False)
            continue
        sc_results = all_results[sc_name]
        if families is None:
            families = list(sc_results.keys())

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(SCENARIO_LABELS.get(sc_name, sc_name), fontweight="bold")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        for fam, inputs in sc_results.items():
            mia_res = inputs.get(input_type)
            if mia_res is None:
                continue
            att = mia_res.attacks.get(attacker_name)
            if att is None:
                # try first available
                if mia_res.attacks:
                    att = next(iter(mia_res.attacks.values()))
                else:
                    continue
            fpr = att.fpr_curve
            tpr = att.tpr_curve
            auc = att.metrics.get("auc_roc", 0)
            ax.plot(fpr, tpr,
                    marker=FAMILY_MARKERS.get(fam, "o"),
                    markevery=max(1, len(fpr) // 10),
                    linewidth=2,
                    label=f"{FAMILY_LABELS.get(fam, fam)} (AUC={auc:.2f})")
        ax.legend(fontsize=8)

    fig.suptitle(f"ROC Curves — {attacker_name} ({input_type.upper()})",
                 fontsize=13, fontweight="bold")
    return _save(fig, f"M2_roc_curves_{input_type}_{attacker_name}")


def plot_mia_advantage(mia_df: pd.DataFrame) -> str:
    """M3 — MIA advantage (TPR − FPR) grouped bar chart for both input types."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, itype in zip(axes, ["mm_cf", "nice_cf"]):
        df = mia_df[(mia_df["input_type"] == itype) &
                    (mia_df["attacker"] == "ensemble_mean")]
        if df.empty:
            df = mia_df[mia_df["input_type"] == itype]
            if df.empty:
                ax.set_title(f"No data ({itype})")
                continue
            # Use best attacker per scenario/family
            df = (df.sort_values("mia_advantage", ascending=False)
                    .drop_duplicates(subset=["scenario", "family"]))

        scenarios = df["scenario"].unique()
        families  = df["family"].unique()
        x = np.arange(len(scenarios))
        width = 0.8 / max(len(families), 1)

        for fi, fam in enumerate(families):
            vals = []
            for sc in scenarios:
                v = df[(df["scenario"] == sc) & (df["family"] == fam)]["mia_advantage"]
                vals.append(float(v.mean()) if not v.empty else 0.0)
            offset = (fi - len(families) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=FAMILY_LABELS.get(fam, fam),
                   color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                   alpha=0.85)

        ax.axhline(0, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios])
        ax.set_ylabel("MIA Advantage (TPR − FPR)")
        ax.set_title(itype.replace("_", " ").upper(), fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("MIA Advantage — Higher = Greater Privacy Leak",
                 fontsize=13, fontweight="bold")
    return _save(fig, "M3_mia_advantage")


def plot_mia_precision_recall(mia_df: pd.DataFrame) -> str:
    """M4 — Precision vs Recall scatter coloured by scenario."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, itype in zip(axes, ["mm_cf", "nice_cf"]):
        df = mia_df[mia_df["input_type"] == itype]
        if df.empty:
            continue
        for sc in df["scenario"].unique():
            sub = df[df["scenario"] == sc]
            ax.scatter(sub["recall"], sub["precision"],
                       c=_scenario_color(sc), s=30, alpha=0.6,
                       label=SCENARIO_LABELS.get(sc, sc), edgecolors="none")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.5)
        ax.set_title(itype.replace("_", " ").upper(), fontweight="bold")
        ax.legend(fontsize=8)

    fig.suptitle("MIA Precision vs Recall", fontsize=13, fontweight="bold")
    return _save(fig, "M4_mia_precision_recall")


def plot_privacy_gain_heatmap(mia_df: pd.DataFrame) -> str:
    """M5 — Privacy gain heatmap (scenario × model family) for both input types."""
    # MIA uses "privacy_gain"; AIA uses "attribute_privacy_gain"
    privacy_col = next(
        (c for c in ("privacy_gain", "attribute_privacy_gain") if c in mia_df.columns),
        None,
    )
    if privacy_col is None:
        return ""

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, itype in zip(axes, ["mm_cf", "nice_cf"]):
        df = mia_df[mia_df["input_type"] == itype]
        # Average privacy_gain across all attackers per (scenario, family)
        pivot = (df.groupby(["scenario", "family"])[privacy_col]
                   .mean()
                   .unstack(fill_value=0.5))
        if pivot.empty:
            continue
        im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([FAMILY_LABELS.get(c, c) for c in pivot.columns], fontsize=9)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([SCENARIO_LABELS.get(s, s) for s in pivot.index], fontsize=9)
        ax.set_title(itype.replace("_", " ").upper(), fontweight="bold")
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                ax.text(j, i, f"{pivot.values[i,j]:.2f}",
                        ha="center", va="center", fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Privacy Gain")

    fig.suptitle("Privacy Gain (1 − MIA Advantage)  [Higher = More Private]",
                 fontsize=13, fontweight="bold")
    return _save(fig, "M5_privacy_gain_heatmap")


# ---------------------------------------------------------------------------
# Analysis 3 — NiCE CF Quality
# ---------------------------------------------------------------------------

def _collect_nice_metrics(nice_results: Dict) -> pd.DataFrame:
    """Flatten NiCE CF results into a tidy DataFrame."""
    rows = []
    for sc, families in nice_results.items():
        for fam, nice_res in families.items():
            m = nice_res.metrics
            rows.append({
                "scenario":    sc,
                "family":      fam,
                "proximity":   m.get("proximity",    np.nan),
                "l2_proximity":m.get("l2_proximity",  np.nan),
                "plausibility":m.get("plausibility",  np.nan),
                "sparsity":    m.get("sparsity",      np.nan),
            })
    return pd.DataFrame(rows)


def plot_cf_quality_bars(nice_results: Dict) -> str:
    """C1–C3 — Proximity, Plausibility, and Sparsity comparison bars."""
    df = _collect_nice_metrics(nice_results)
    if df.empty:
        print("  [fig] No NiCE results, skipping CF quality bars.")
        return ""

    metrics_info = [
        ("proximity",    "Proximity (L1↓)",    "C1_cf_proximity"),
        ("plausibility", "Plausibility (↑)",   "C2_cf_plausibility"),
        ("sparsity",     "Sparsity (↑)",        "C3_cf_sparsity"),
    ]
    paths = []
    scenarios = df["scenario"].unique()
    families  = df["family"].unique()

    for metric, ylabel, fname in metrics_info:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(scenarios))
        width = 0.25

        for fi, fam in enumerate(families):
            vals = []
            for sc in scenarios:
                v = df[(df["scenario"] == sc) & (df["family"] == fam)][metric]
                vals.append(float(v.mean()) if not v.empty else 0.0)
            offset = (fi - len(families) / 2 + 0.5) * width
            ax.bar(x + offset, vals, width,
                   label=FAMILY_LABELS.get(fam, fam),
                   color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                   alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios])
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.legend()
        paths.append(_save(fig, fname))

    return paths[0] if paths else ""


def plot_cf_quality_radar(nice_results: Dict) -> str:
    """C4 — Radar chart of CF quality metrics across scenarios."""
    df = _collect_nice_metrics(nice_results)
    if df.empty:
        return ""

    metrics  = ["plausibility", "sparsity"]
    # proximity → inverted (1 - normalised) so that "higher = better" for all axes
    families  = df["family"].unique()
    scenarios = df["scenario"].unique()

    n_metrics = 3
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    labels = ["Plausibility ↑", "Sparsity ↑", "1−Proximity ↑"]

    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 5),
                              subplot_kw=dict(polar=True))
    if len(families) == 1:
        axes = [axes]

    for ax, fam in zip(axes, families):
        ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=8)
        ax.set_title(FAMILY_LABELS.get(fam, fam), fontweight="bold", pad=15)

        # Normalise proximity across scenarios for radar scaling
        prox_vals = df[df["family"] == fam]["proximity"]
        max_prox = prox_vals.max() if prox_vals.max() > 0 else 1.0

        for sc in scenarios:
            row = df[(df["family"] == fam) & (df["scenario"] == sc)]
            if row.empty:
                continue
            pla  = float(row["plausibility"].mean())
            spa  = float(row["sparsity"].mean())
            prox = float(row["proximity"].mean())
            vals = [pla, spa, 1 - prox / max_prox] + [pla]  # close polygon
            ax.plot(angles, vals, color=_scenario_color(sc), linewidth=2,
                    label=SCENARIO_LABELS.get(sc, sc))
            ax.fill(angles, vals, color=_scenario_color(sc), alpha=0.1)
        ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=8)

    fig.suptitle("NiCE CF Quality Radar", fontsize=13, fontweight="bold")
    return _save(fig, "C4_cf_quality_radar")


# ---------------------------------------------------------------------------
# Combined comparison dashboard
# ---------------------------------------------------------------------------

def plot_combined_dashboard(
    fairness_results: Dict,
    mia_df: pd.DataFrame,
    nice_results: Dict,
) -> str:
    """X1 — 3-panel summary dashboard: fairness | MIA | CF quality."""
    scenarios = list(fairness_results.keys())
    families  = list(next(iter(fairness_results.values())).keys())

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Panel 1: SPD per scenario/family ---
    ax1 = fig.add_subplot(gs[0, 0])
    x = np.arange(len(scenarios))
    width = 0.25
    for fi, fam in enumerate(families):
        vals = [fairness_results[sc][fam].group.get("SPD", 0) for sc in scenarios]
        offset = (fi - len(families) / 2 + 0.5) * width
        ax1.bar(x + offset, vals, width,
                label=FAMILY_LABELS.get(fam, fam),
                color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                alpha=0.85)
    ax1.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=8)
    ax1.set_ylabel("SPD")
    ax1.set_title("Statistical Parity Difference", fontweight="bold", fontsize=10)
    ax1.legend(fontsize=7)

    # --- Panel 2: EOD per scenario/family ---
    ax2 = fig.add_subplot(gs[0, 1])
    for fi, fam in enumerate(families):
        vals = [fairness_results[sc][fam].group.get("EOD", 0) for sc in scenarios]
        offset = (fi - len(families) / 2 + 0.5) * width
        ax2.bar(x + offset, vals, width,
                label=FAMILY_LABELS.get(fam, fam),
                color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                alpha=0.85)
    ax2.axhline(0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=8)
    ax2.set_ylabel("EOD")
    ax2.set_title("Equal Opportunity Difference", fontweight="bold", fontsize=10)
    ax2.legend(fontsize=7)

    # --- Panel 3: CF Fairness ---
    ax3 = fig.add_subplot(gs[0, 2])
    for fi, fam in enumerate(families):
        vals = [fairness_results[sc][fam].individual.get("cf_fairness", 0) for sc in scenarios]
        offset = (fi - len(families) / 2 + 0.5) * width
        ax3.bar(x + offset, vals, width,
                label=FAMILY_LABELS.get(fam, fam),
                color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                alpha=0.85)
    ax3.axhline(1, color="gray", linestyle="--", linewidth=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=8)
    ax3.set_ylabel("CF Fairness Score")
    ax3.set_ylim(0, 1.05)
    ax3.set_title("Counterfactual Fairness", fontweight="bold", fontsize=10)
    _legend_if_present(ax3, fontsize=7)

    # --- Panel 4: MIA AUC (mm_cf, ensemble_mean) ---
    ax4 = fig.add_subplot(gs[1, 0])
    df_mm = mia_df[(mia_df["input_type"] == "mm_cf") &
                   (mia_df["attacker"] == "ensemble_mean")]
    if df_mm.empty:
        df_mm = mia_df[mia_df["input_type"] == "mm_cf"].head(len(scenarios) * len(families))
    if not df_mm.empty:
        for fi, fam in enumerate(families):
            vals = []
            for sc in scenarios:
                v = df_mm[(df_mm["scenario"] == sc) & (df_mm["family"] == fam)]["auc_roc"]
                vals.append(float(v.mean()) if not v.empty else 0.5)
            offset = (fi - len(families) / 2 + 0.5) * width
            ax4.bar(x + offset, vals, width,
                    label=FAMILY_LABELS.get(fam, fam),
                    color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                    alpha=0.85)
    ax4.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=8)
    ax4.set_ylabel("AUC-ROC")
    ax4.set_ylim(0.4, 1.0)
    ax4.set_title("MIA AUC (MM-CFs)", fontweight="bold", fontsize=10)
    _legend_if_present(ax4, fontsize=7)

    # --- Panel 5: MIA AUC (nice_cf, ensemble_mean) ---
    ax5 = fig.add_subplot(gs[1, 1])
    df_nice = mia_df[(mia_df["input_type"] == "nice_cf") &
                     (mia_df["attacker"] == "ensemble_mean")]
    if df_nice.empty:
        df_nice = mia_df[mia_df["input_type"] == "nice_cf"].head(len(scenarios) * len(families))
    if not df_nice.empty:
        for fi, fam in enumerate(families):
            vals = []
            for sc in scenarios:
                v = df_nice[(df_nice["scenario"] == sc) & (df_nice["family"] == fam)]["auc_roc"]
                vals.append(float(v.mean()) if not v.empty else 0.5)
            offset = (fi - len(families) / 2 + 0.5) * width
            ax5.bar(x + offset, vals, width,
                    label=FAMILY_LABELS.get(fam, fam),
                    color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                    alpha=0.85)
    ax5.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    ax5.set_xticks(x)
    ax5.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=8)
    ax5.set_ylabel("AUC-ROC")
    ax5.set_ylim(0.4, 1.0)
    ax5.set_title("MIA AUC (NiCE CFs)", fontweight="bold", fontsize=10)
    _legend_if_present(ax5, fontsize=7)

    # --- Panel 6: CF Proximity ---
    ax6 = fig.add_subplot(gs[1, 2])
    nice_df = _collect_nice_metrics(nice_results)
    if not nice_df.empty:
        for fi, fam in enumerate(families):
            vals = []
            for sc in scenarios:
                v = nice_df[(nice_df["scenario"] == sc) & (nice_df["family"] == fam)]["proximity"]
                vals.append(float(v.mean()) if not v.empty else 0.0)
            offset = (fi - len(families) / 2 + 0.5) * width
            ax6.bar(x + offset, vals, width,
                    label=FAMILY_LABELS.get(fam, fam),
                    color=FAMILY_COLORS[fi % len(FAMILY_COLORS)],
                    alpha=0.85)
    ax6.set_xticks(x)
    ax6.set_xticklabels([SCENARIO_LABELS.get(sc, sc) for sc in scenarios], fontsize=8)
    ax6.set_ylabel("L1 Proximity (↓ better)")
    ax6.set_title("CF Proximity (NiCE)", fontweight="bold", fontsize=10)
    _legend_if_present(ax6, fontsize=7)

    fig.suptitle("Comprehensive Comparison Dashboard\n"
                 "(Baseline vs Augmented vs LDP-Augmented)",
                 fontsize=14, fontweight="bold")
    return _save(fig, "X1_combined_dashboard", tight=False)


# ---------------------------------------------------------------------------
# Main entry-point
# ---------------------------------------------------------------------------

def generate_all_figures(
    fairness_results: Dict,
    all_mia_results:  Dict,
    nice_cf_results:  Dict,
    verbose: bool = True,
) -> List[str]:
    """Generate and save all figures.

    Parameters
    ----------
    fairness_results : {scenario: {family: FairnessResult}}
    all_mia_results  : {scenario: {family: {input_type: MIAResult}}}
    nice_cf_results  : {scenario: {family: NiCEResult}}
    verbose          : Print saved paths.

    Returns
    -------
    List of saved file paths.
    """
    from pipeline.mia_analysis import mia_results_to_dataframe

    if verbose:
        print("\n" + "="*60)
        print("  GENERATING FIGURES")
        print("="*60)

    mia_df = mia_results_to_dataframe(all_mia_results)

    saved_paths = []

    # Analysis 1 — Fairness
    if fairness_results:
        saved_paths.append(plot_group_fairness_bars(fairness_results))
        saved_paths.append(plot_fairness_radar(fairness_results))
        saved_paths.append(plot_group_positive_rates(fairness_results))
        saved_paths.append(plot_tpr_fpr_heatmap(fairness_results))
        saved_paths.append(plot_individual_fairness(fairness_results))
        saved_paths.append(plot_theil_index(fairness_results))

    # Analysis 2 — MIA
    if not mia_df.empty:
        saved_paths.append(plot_mia_auc_bars(mia_df, "mm_cf"))
        saved_paths.append(plot_mia_auc_bars(mia_df, "nice_cf"))
        saved_paths.append(plot_mia_roc_curves(all_mia_results, "mm_cf"))
        saved_paths.append(plot_mia_roc_curves(all_mia_results, "nice_cf"))
        saved_paths.append(plot_mia_advantage(mia_df))
        saved_paths.append(plot_mia_precision_recall(mia_df))
        saved_paths.append(plot_privacy_gain_heatmap(mia_df))

    # Analysis 3 — NiCE CF Quality
    if nice_cf_results:
        saved_paths.append(plot_cf_quality_bars(nice_cf_results))
        saved_paths.append(plot_cf_quality_radar(nice_cf_results))

    # Combined dashboard
    if fairness_results and nice_cf_results:
        saved_paths.append(plot_combined_dashboard(
            fairness_results, mia_df, nice_cf_results
        ))

    saved_paths = [p for p in saved_paths if p]   # filter empty strings
    if verbose:
        print(f"\n[figures] {len(saved_paths)} figures saved to {_FIG_DIR}")

    return saved_paths
