"""
visualizations_structured.py -- Four-folder structured visualizations.

Folder 1  viz/1_baseline_vs_augmented/
    F1a  Fairness metrics: baseline vs MM-CF augmented
    F1b  MIA AUC-ROC comparison
    F1c  NiCE CF quality metrics
    F1d  MIA multi-metric paired bars (Baseline vs Augmented, 4 metrics)
    F1e  CF Fairness: Baseline vs MM-CF Augmented (per family)

Folder 2  viz/2_augmented_vs_fairlearn/
    F2a  Fairness SPD/EOD: baseline + augmented vs fairlearn
    F2b  MIA AUC: augmented vs fairlearn (3 family panels)
    F2c  Privacy-Fairness scatter (MIA AUC vs |SPD|)
    F2d  NiCE flip-rate with interpretation guide
    F2e  MIA multi-metric — 4 separate vertical figures (landscape)
    F2f  MIA AUC — all variants: augmented + LDP + LDP+Fairlearn (per family)
    F2g  CF Fairness: Augmented vs LDP+Fairlearn variants (per family)

Folder 3  viz/3_ldp_fairness/
    F3a  LDP epsilon sweep
    F3b  LDP+Fairlearn SPD/EOD heatmaps
    F3c  LDP+Fairlearn fairness vs baseline benchmark
    F3d  All-scenarios SPD with best-epsilon annotation
    F3e  MIA overview — 3 separate figures, one per model family
    F3f  Privacy-Fairness SPD+MIA — 3 separate figures, one per family
    F3g  LDP+Fairlearn: all 6 models, 4 fairness metrics
    F3h  MIA multi-metric for LDP sweep — 4 separate vertical figures

Folder 4  viz/4_mia_methods/
    F4a  Attacker comparison heatmap (attacker × scenario, per family)
    F4b  Attacker group ranking (DCR / GEN-LRA / DPI / Statistical / Ensemble)
    F4c  Ensemble (majority_vote) analysis across scenarios
    F4d  Input proxy comparison: mm_cf vs nice_cf
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.3, "axes.axisbelow": True,
})
_DPI = 150; _SAVE = dict(dpi=_DPI, bbox_inches="tight")
_TF = 15; _LF = 13; _TK = 11; _LG = 11; _AN = 9

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR     = os.path.dirname(os.path.abspath(__file__))
_BASE_DIR     = os.path.dirname(_THIS_DIR)
_MAIN_RES     = os.path.join(_BASE_DIR, "pipeline_outputs",           "results")
_FL_RES       = os.path.join(_BASE_DIR, "pipeline_outputs_fairlearn", "results")
_LF_RES       = os.path.join(_BASE_DIR, "pipeline_outputs", "ldp_fair", "results")
_VIZ_ROOT     = os.path.join(_BASE_DIR, "viz")
_DIR1 = os.path.join(_VIZ_ROOT, "1_baseline_vs_augmented")
_DIR2 = os.path.join(_VIZ_ROOT, "2_augmented_vs_fairlearn")
_DIR3 = os.path.join(_VIZ_ROOT, "3_ldp_fairness")
_DIR4 = os.path.join(_VIZ_ROOT, "4_mia_methods")

# ---------------------------------------------------------------------------
# Colour / family constants  — consistent across ALL figures
# ---------------------------------------------------------------------------
_C_LR   = "#4C72B0"   # blue
_C_RF   = "#DD8452"   # orange
_C_XGB  = "#55A868"   # green
_C_LDPF = "#8172B2"   # purple  (LDP+Fairlearn)

_FAM_ORDER = ["logistic_regression", "random_forest", "xgboost"]
_FAM_TO_FULL = {"lr":"logistic_regression","rf":"random_forest","xgb":"xgboost"}
_FAM_COLORS = {
    "logistic_regression":_C_LR,"random_forest":_C_RF,"xgboost":_C_XGB,
    "lr":_C_LR,"rf":_C_RF,"xgb":_C_XGB,
}
_FAM_DISP = {
    "logistic_regression":"LR","random_forest":"RF","xgboost":"XGB",
    "lr":"LR","rf":"RF","xgb":"XGB",
}
_HATCH_BASE = ""
_HATCH_AUG  = "///"

_MIA_METRICS = [
    ("auc_roc",        "AUC-ROC",           0.50, "(0.50 = random baseline; lower = more private)"),
    ("mia_advantage",  "MIA Advantage",      0.00, "(TPR - FPR;  0 = perfect privacy)"),
    ("tpr_at_fpr_0.1", "TPR @ FPR = 10%",   0.10, "(matches random @ 0.10; lower = more private)"),
    ("pr_auc",         "PR-AUC",            None, "(higher = attacker has more signal = worse)"),
]

# Attacker groups for Folder 4
_ATTACKER_GROUPS = {
    "DCR":        ["dcr_l1","dcr_l2","dcr_diff_l1","dcr_diff_l2"],
    "GEN-LRA":    ["gen_lra_k1","gen_lra_k5","gen_lra_k10","gen_lra_k20","gen_lra_k50"],
    "DPI":        ["dpi_l2_k5","dpi_l2_k10","dpi_l2_k15","dpi_l2_k20",
                   "dpi_l1_k5","dpi_l1_k10","dpi_l1_k15","dpi_l1_k20"],
    "Statistical":["logan","domias","mc","classifier","mean","median",
                   "q25","q75","iqr","max","min","range"],
    "Ensemble":   ["majority_vote_median","majority_vote_90","majority_vote_91",
                   "majority_vote_92","majority_vote_93","majority_vote_94",
                   "majority_vote_95","majority_vote_96","majority_vote_97",
                   "majority_vote_98"],
}
_ATK_GRP_COLORS = {
    "DCR":"#4C72B0","GEN-LRA":"#DD8452","DPI":"#55A868",
    "Statistical":"#C44E52","Ensemble":"#8172B2",
}

# Scenario colours — semantic by group, shade by epsilon
# Group palettes: [lightest→darkest] for epsilon ordering
_GRP_PALETTES = {
    "baseline":                   ["#555555"],   # dark grey
    "augmented":                  ["#2ca02c"],   # green  (legacy)
    "augmented_SCM":              ["#2ca02c"],   # green
    "augmented_update_labels":    ["#e377c2"],   # pink/magenta  -- clearly distinct
    "augmented_add_comparators":  ["#9467bd"],   # purple
    "fairlearn":  ["#1f77b4","#4393c3","#74add1","#abd9e9","#6baed6","#9ecae1","#c6dbef","#2171b5"],
    "ldp_baseline":  ["#fcbba1","#fc9272","#de2d26","#a50f15","#67000d"],
    "ldp_augmented": ["#fdd0a2","#fdae6b","#f16913","#d94801","#8c2d04"],
    "ldp_fair":      ["#dadaeb","#bcbddc","#9e9ac8","#756bb1","#4a1486"],
}
_EPS_ORDER = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

def _scen_group(scenario):
    s = str(scenario)
    if s == "baseline":                          return "baseline"
    if s == "augmented":                         return "augmented"           # legacy
    if s == "augmented_SCM":                     return "augmented_SCM"
    if s == "augmented_update_labels":           return "augmented_update_labels"
    if s == "augmented_add_comparators":         return "augmented_add_comparators"
    if s.startswith("augmented_"):               return "augmented_SCM"       # unknown augmented → SCM fallback
    if s.startswith("ldp_fair_"):                return "ldp_fair"
    if s.startswith("ldp_augmented_eps"):        return "ldp_augmented"       # legacy key
    if s.startswith("ldp_baseline_"):            return "ldp_baseline"
    if s.startswith("ldp_eps"):                  return "ldp_baseline"        # legacy key
    # unified_analysis.py naming: ldp_{method}_eps{e}
    if s.startswith("ldp_SCM_"):                 return "ldp_augmented"
    if s.startswith("ldp_update_labels_"):       return "ldp_augmented"
    if s.startswith("ldp_add_comparators_"):     return "ldp_augmented"
    if s.startswith("ldp_"):                     return "ldp_augmented"       # catch any other ldp+aug variant
    return "fairlearn"

def _scen_eps(scenario):
    import re
    m = re.search(r"eps([\d.]+)", str(scenario))
    return float(m.group(1)) if m else None

def _scenario_color(scenario):
    grp = _scen_group(scenario)
    pal = _GRP_PALETTES.get(grp, ["#888888"])
    eps = _scen_eps(scenario)
    if eps is None or len(pal) == 1:
        return pal[0]
    try:
        idx = _EPS_ORDER.index(eps)
    except ValueError:
        import bisect
        idx = bisect.bisect_left(_EPS_ORDER, eps)
    idx = min(idx, len(pal) - 1)
    return pal[idx]

_GRP_ORDER  = [
    "baseline",
    "augmented",                 # legacy
    "augmented_SCM",
    "augmented_update_labels",
    "augmented_add_comparators",
    "fairlearn",
    "ldp_baseline",
    "ldp_augmented",
    "ldp_fair",
]
_GRP_LABELS = {
    "baseline":                  "Baseline",
    "augmented":                 "MM-CF Augmented",      # legacy
    "augmented_SCM":             "SCM Augmented",
    "augmented_update_labels":   "MM Update Labels",
    "augmented_add_comparators": "MM Add Comparators",
    "fairlearn":                 "Fairlearn",
    "ldp_baseline":              "LDP Only",
    "ldp_augmented":             "LDP + Augmented",
    "ldp_fair":                  "LDP + Fairlearn",
}

def _sort_scenarios(labels):
    def _key(s):
        g = _scen_group(s)
        gi = _GRP_ORDER.index(g) if g in _GRP_ORDER else 99
        e = _scen_eps(s) or 0
        return (gi, e, s)
    return sorted(labels, key=_key)

# Legacy flat dict for backward-compatible lookups (F3e etc.)
_SCEN_COLORS = {
    "baseline":                  "#555555",
    "augmented":                 "#2ca02c",  # legacy
    "augmented_SCM":             "#2ca02c",
    "augmented_update_labels":   "#e377c2",  # pink/magenta
    "augmented_add_comparators": "#9467bd",
    "ldp_eps0.1":        "#fee5d9",
    "ldp_eps0.5":        "#fcae91",
    "ldp_eps1.0":        "#fb6a4a",
    "ldp_eps2.0":        "#de2d26",
    "ldp_eps5.0":        "#a50f15",
    "ldp_eps10.0":       "#67000d",
    "ldp_baseline_eps0.1":  "#fcbba1",
    "ldp_baseline_eps0.5":  "#fc9272",
    "ldp_baseline_eps1.0":  "#fb6a4a",
    "ldp_baseline_eps2.0":  "#de2d26",
    "ldp_baseline_eps5.0":  "#a50f15",
    "ldp_baseline_eps10.0": "#67000d",
    "ldp_augmented_eps0.1": "#fdd0a2",
    "ldp_augmented_eps0.5": "#fdae6b",
    "ldp_augmented_eps1.0": "#f16913",
    "ldp_augmented_eps2.0": "#d94801",
    "ldp_augmented_eps5.0": "#8c2d04",
    # unified_analysis.py naming: ldp_{method}_eps{e}
    "ldp_SCM_eps0.1":              "#fdd0a2",
    "ldp_SCM_eps0.5":              "#fdae6b",
    "ldp_SCM_eps1.0":              "#f16913",
    "ldp_SCM_eps2.0":              "#d94801",
    "ldp_SCM_eps5.0":              "#8c2d04",
    "ldp_update_labels_eps0.1":    "#fdd0a2",
    "ldp_update_labels_eps0.5":    "#fdae6b",
    "ldp_update_labels_eps1.0":    "#f16913",
    "ldp_update_labels_eps2.0":    "#d94801",
    "ldp_update_labels_eps5.0":    "#8c2d04",
    "ldp_add_comparators_eps0.1":  "#fdd0a2",
    "ldp_add_comparators_eps0.5":  "#fdae6b",
    "ldp_add_comparators_eps1.0":  "#f16913",
    "ldp_add_comparators_eps2.0":  "#d94801",
    "ldp_add_comparators_eps5.0":  "#8c2d04",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(path, label=""):
    if not os.path.exists(path):
        if label: print("  [viz] Not found:", os.path.basename(path))
        return pd.DataFrame()
    try:
        if os.path.getsize(path) <= 5: return pd.DataFrame()
        df = pd.read_csv(path)
        if label: print("  [viz] Loaded {} ({} rows)".format(label, len(df)))
        return df
    except Exception as e:
        print("  [viz] Error reading {}: {}".format(os.path.basename(path), e))
        return pd.DataFrame()


def _save_fig(fig, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, **_SAVE); plt.close(fig)
    print("  [viz] Saved ->", os.path.basename(path))


def _fc(fam):   return _FAM_COLORS.get(str(fam), "#888888")
def _fd(fam):   return _FAM_DISP.get(str(fam), str(fam).upper())
def _wrap(t, w=18):
    if len(t) <= w: return t
    words = t.split(); lines, cur = [], ""
    for wd in words:
        if len(cur)+len(wd)+1 > w and cur: lines.append(cur); cur = wd
        else: cur = (cur+" "+wd).strip()
    if cur: lines.append(cur)
    return "\n".join(lines)


def _bar_labels(ax, bars, fmt="{:.3f}", fs=9, pad=0.003):
    for b in bars:
        h = b.get_height()
        if np.isnan(h): continue
        ax.text(b.get_x()+b.get_width()/2, h+(pad if h>=0 else -pad),
                fmt.format(h), ha="center", va="bottom" if h>=0 else "top",
                fontsize=fs, clip_on=True)


def _hbar_labels(ax, bars, fmt="{:.4f}", fs=9, pad=0.001):
    for b in bars:
        w = b.get_width()
        if np.isnan(w): continue
        ax.text(w+(pad if w>=0 else -pad), b.get_y()+b.get_height()/2,
                fmt.format(w), ha="left" if w>=0 else "right",
                va="center", fontsize=fs, clip_on=True)


def _normalize_family(df):
    """Map short family tags (lr/rf/xgb) to full names in-place copy."""
    if df.empty or "family" not in df.columns: return df
    df = df.copy()
    df["family"] = df["family"].map(lambda f: _FAM_TO_FULL.get(str(f), str(f)))
    return df


def _fl_norm(df):
    """Normalise fairlearn fairness CSV: add group_* columns + scenario + display."""
    if df.empty: return df
    df = _normalize_family(df.copy())
    for c in ["SPD","DI","EOD","AOD","EqOdds","PP","Theil"]:
        if c in df.columns and "group_"+c not in df.columns:
            df["group_"+c] = df[c]
    for c in ["consistency","cf_fairness"]:
        if c in df.columns and "ind_"+c not in df.columns:
            df["ind_"+c] = df[c]
    if "model"  in df.columns and "scenario" not in df.columns: df["scenario"] = df["model"]
    if "label"  in df.columns and "display"  not in df.columns: df["display"]  = df["label"]
    df["source"] = "fairlearn"
    return df


def _fmt_ldp_label(scenario, family=""):
    """Human-readable label for a scenario key."""
    sc = str(scenario)
    import re
    if sc == "baseline":
        s = "Baseline"
    elif sc == "augmented":
        s = "Augmented"
    elif sc == "augmented_SCM":
        s = "SCM Augmented"
    elif sc == "augmented_update_labels":
        s = "MM Update Labels"
    elif sc == "augmented_add_comparators":
        s = "MM Add Comparators"
    elif sc.startswith("ldp_baseline_eps"):
        eps = sc.replace("ldp_baseline_eps", "")
        s = "LDP-Only\neps={}".format(eps)
    elif sc.startswith("ldp_augmented_eps"):          # legacy key
        eps = sc.replace("ldp_augmented_eps", "")
        s = "LDP+Aug\neps={}".format(eps)
    # unified_analysis.py: ldp_{method}_eps{e}
    elif sc.startswith("ldp_SCM_eps"):
        eps = re.sub(r"ldp_SCM_eps", "", sc)
        s = "LDP+SCM\neps={}".format(eps)
    elif sc.startswith("ldp_update_labels_eps"):
        eps = re.sub(r"ldp_update_labels_eps", "", sc)
        s = "LDP+UpdLbl\neps={}".format(eps)
    elif sc.startswith("ldp_add_comparators_eps"):
        eps = re.sub(r"ldp_add_comparators_eps", "", sc)
        s = "LDP+AddComp\neps={}".format(eps)
    elif sc.startswith("ldp_fair_eps"):               # legacy
        m = re.search(r"eps([\d.]+)_(.*?)_(lr|rf|xgb|logistic_regression|random_forest|xgboost)$", sc)
        if m:
            s = "LDP+FL\neps={} {}".format(m.group(1), m.group(2))
        else:
            s = _wrap(sc, 16)
    elif sc.startswith("ldp_eps"):                    # legacy
        eps = sc.replace("ldp_eps", "")
        s = "LDP eps={}".format(eps)
    # unified_analysis.py fairlearn scenarios: fl_{ctag} and fl_ldp_{eps}_{ctag}
    elif sc.startswith("fl_ldp_"):
        m = re.search(r"fl_ldp_eps([\d.]+)_(.*)", sc)
        if m:
            s = "FL+LDP\neps={} ({})".format(m.group(1), m.group(2).upper())
        else:
            s = _wrap(sc, 16)
    elif sc.startswith("fl_"):
        ctag = sc[3:]
        _fl_names = {
            "baseline": "FL Baseline",
            "eg_dp": "FL EG-DP", "eg_eo": "FL EG-EO", "eg_tpr": "FL EG-TPR",
            "gs_dp": "FL GS-DP", "gs_eo": "FL GS-EO",
            "to_dp": "FL TO-DP", "to_eo": "FL TO-EO",
        }
        s = _fl_names.get(ctag, "FL " + ctag.upper())
    else:
        s = _wrap(sc, 14)
    return "{}\n{}".format(s, _fd(family)) if family else s


def _load_fl_fair():
    return _fl_norm(_load(os.path.join(_FL_RES, "fairlearn_fairness.csv"), "fl fairness"))


# ===========================================================================
#  SHARED  multi-metric MIA  — compact 4-panel (reused in F1d, F3g)
# ===========================================================================

def _viz_mia_4panel(df, label_col, family_col, color_fn, title, out_path,
                    input_type_filter="nice_cf", ref_legend_outside=False):
    """4 vertical-bar panels: AUC, Advantage, TPR@FPR=10%, PR-AUC."""
    if df.empty: return
    sub = df
    if "input_type" in df.columns and input_type_filter:
        s2 = df[df["input_type"]==input_type_filter]
        sub = s2 if not s2.empty else df

    avail = [(c,t,r,d) for c,t,r,d in _MIA_METRICS if c in sub.columns]
    if not avail: return
    grps = [c for c in [label_col, family_col] if c in sub.columns]
    if not grps: return

    rec = {}
    for c,*_ in avail:
        for _, row in sub.groupby(grps)[c].max().reset_index().iterrows():
            key = tuple(str(row[g]) for g in grps)
            rec.setdefault(key, {})[c] = row[c]
    if not rec: return

    keys = sorted(rec, key=lambda k:(k[1] if len(k)>1 else "", k[0]))
    labels = ["{}\n({})".format(_wrap(k[0],16), _fd(k[1])) if len(k)>1 else _wrap(k[0],16)
              for k in keys]
    colors = [color_fn(k) for k in keys]
    x = np.arange(len(keys))

    n = len(avail)
    fig, axes = plt.subplots(1, n, figsize=(5.5*n, max(6,len(keys)*0.5+3)))
    if n==1: axes=[axes]
    fig.suptitle(title, fontsize=_TF+1, fontweight="bold", y=1.02)

    for ax, (col, panel_t, ref_val, note) in zip(axes, avail):
        vals = [rec[k].get(col, float("nan")) for k in keys]
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", width=0.7)
        _bar_labels(ax, bars, fmt="{:.4f}", fs=8)
        if ref_val is not None:
            ax.axhline(ref_val, color="red", ls="--", lw=1.5)
        ax.set_title("{}\n{}".format(panel_t, note), fontsize=_LF, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=_TK-1)
        ax.set_ylabel(panel_t, fontsize=_LF)

        if ref_val is not None:
            ref_handle = Line2D([0],[0], color="red", ls="--", lw=1.5,
                                label="Ref. ({})".format(ref_val))
            if ref_legend_outside:
                ax.legend(handles=[ref_handle], fontsize=_LG-1,
                          bbox_to_anchor=(1.01,1), loc="upper left",
                          framealpha=0.9)
            else:
                ax.legend(handles=[ref_handle], fontsize=_LG-1,
                          bbox_to_anchor=(1.01, 1.0), loc="upper left",
                          framealpha=0.9)

    plt.tight_layout()
    _save_fig(fig, out_path)


# ===========================================================================
#  SHARED  multi-metric MIA  — 4 SEPARATE HORIZONTAL figures
# ===========================================================================

def _viz_mia_4sep_vertical(df, label_col, family_col, color_fn,
                            title_prefix, out_prefix,
                            input_type_filter="nice_cf"):
    """Generate 4 separate landscape vertical-bar figures, one per MIA metric."""
    if df.empty: return
    sub = df
    if "input_type" in df.columns and input_type_filter:
        s2 = df[df["input_type"]==input_type_filter]
        sub = s2 if not s2.empty else df

    avail = [(c,t,r,d) for c,t,r,d in _MIA_METRICS if c in sub.columns]
    if not avail: return
    grps = [c for c in [label_col, family_col] if c in sub.columns]
    if not grps: return

    rec = {}
    for c,*_ in avail:
        for _, row in sub.groupby(grps)[c].max().reset_index().iterrows():
            key = tuple(str(row[g]) for g in grps)
            rec.setdefault(key, {})[c] = row[c]
    if not rec: return

    keys   = sorted(rec, key=lambda k:(k[1] if len(k)>1 else "", k[0]))
    labels = ["{}\n({})".format(_fmt_ldp_label(k[0]).replace("\n"," "),
                                 _fd(k[1])) if len(k)>1
              else _fmt_ldp_label(k[0]).replace("\n"," ")
              for k in keys]
    colors = [color_fn(k) for k in keys]
    x = np.arange(len(keys))

    for idx, (col, panel_t, ref_val, note) in enumerate(avail):
        vals = [rec[k].get(col, float("nan")) for k in keys]
        fig, ax = plt.subplots(figsize=(max(14, len(keys)*0.85 + 3), 7))

        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white", width=0.7)
        _bar_labels(ax, bars, fmt="{:.4f}", fs=8)
        if ref_val is not None:
            ax.axhline(ref_val, color="red", ls="--", lw=1.8,
                       label="Reference ({:.2f})".format(ref_val))

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=_TK-1)
        ax.set_ylabel("{} — {}".format(panel_t, note), fontsize=_LF)
        ax.set_title("{}: {}\n{}".format(title_prefix, panel_t, note),
                     fontsize=_TF, fontweight="bold")

        # Legend
        legend_els = []
        if ref_val is not None:
            legend_els.append(Line2D([0],[0], color="red", ls="--", lw=1.8,
                                     label="Reference ({})".format(ref_val)))
        fams_seen = list(dict.fromkeys(str(k[1]) if len(k)>1 else "" for k in keys))
        for fam in fams_seen:
            if fam:
                legend_els.append(mpatches.Patch(facecolor=_fc(fam),
                                                  label=_fd(fam)+" base"))
        if legend_els:
            ax.legend(handles=legend_els, fontsize=_LG, framealpha=0.9,
                      bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        _save_fig(fig, "{}_metric{:02d}_{}.png".format(out_prefix, idx+1,
                                                        col.replace(".","p")))


# ===========================================================================
# FOLDER 1 — Baseline vs MM-CF Augmented
# ===========================================================================

def _viz1_fairness(fairness, out_dir):
    """F1a: One figure per fairness metric — horizontal bars, baseline vs augmented.
    Each bar is labeled by scenario+family; semantic colors (gray=baseline, green=augmented)."""
    if fairness.empty: return
    fams = [f for f in _FAM_ORDER if f in fairness["family"].unique()]
    metrics = [
        ("group_SPD",    "SPD",    "Statistical Parity Difference  (0 = perfectly fair)"),
        ("group_EOD",    "EOD",    "Equal Opportunity Difference   (0 = perfectly fair)"),
        ("group_DI",     "DI",     "Disparate Impact                (1 = perfectly fair)"),
        ("group_EqOdds", "EqOdds", "Equalized Odds                  (0 = perfectly fair)"),
    ]
    avail = [(c, s, t) for c, s, t in metrics if c in fairness.columns]
    if not avail: return

    _AUG_VARIANTS = ["augmented_SCM", "augmented_update_labels", "augmented_add_comparators",
                     "augmented"]  # augmented last for legacy backward compat
    _SCEN_ORDER = ["baseline"] + _AUG_VARIANTS
    scens = [sc for sc in _SCEN_ORDER if sc in fairness["scenario"].unique()]

    for col, short, title in avail:
        rows = []
        for scen in scens:
            for fam in fams:
                sub = fairness[(fairness["scenario"] == scen) & (fairness["family"] == fam)]
                if sub.empty or col not in sub.columns:
                    continue
                val = float(sub[col].iloc[0])
                rows.append({
                    "label":  "{} — {}".format(_fmt_ldp_label(scen), _fd(fam)),
                    "value":  val,
                    "color":  _scenario_color(scen),
                    "family": fam,
                    "scen":   scen,
                })
        if not rows:
            continue

        # Sort: baseline first, then augmented variants; within each group by family order
        rows.sort(key=lambda r: (scens.index(r["scen"]) if r["scen"] in scens else 99,
                                  _FAM_ORDER.index(r["family"])
                                  if r["family"] in _FAM_ORDER else 99))

        labels = [r["label"] for r in rows]
        values = [r["value"] for r in rows]
        colors = [r["color"] for r in rows]
        ideal  = 1.0 if short == "DI" else 0.0

        fig_h = max(5, len(rows) * 0.55 + 2.5)
        fig, ax = plt.subplots(figsize=(11, fig_h))

        y = np.arange(len(rows))
        bars = ax.barh(y, values, color=colors, alpha=0.88, edgecolor="white", height=0.6)

        # Annotate values on bars
        for b in bars:
            w = b.get_width()
            if np.isnan(w):
                continue
            pad = (ax.get_xlim()[1] - ax.get_xlim()[0]) * 0.005 if ax.get_xlim()[1] != ax.get_xlim()[0] else 0.002
            ax.text(w + (0.003 if w >= 0 else -0.003),
                    b.get_y() + b.get_height() / 2,
                    "{:.4f}".format(w),
                    ha="left" if w >= 0 else "right",
                    va="center", fontsize=9, clip_on=True)

        ax.axvline(ideal, color="black", lw=1.5, ls="--", alpha=0.7,
                   label="Ideal ({})".format(ideal))

        # Group separators between scenario groups
        prev_grp = None
        for ri, r in enumerate(rows):
            grp = _scen_group(r["scen"])
            if prev_grp is not None and grp != prev_grp:
                ax.axhline(ri - 0.5, color="#aaaaaa", lw=1.2, ls=":")
            prev_grp = grp

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 1)
        ax.invert_yaxis()
        ax.set_xlabel(short, fontsize=_LF)
        ax.set_title("Baseline vs. MM-CF Augmented\n{}".format(title),
                     fontsize=_TF, fontweight="bold")

        legend_els = []
        for sc in scens:
            legend_els.append(mpatches.Patch(
                facecolor=_scenario_color(sc),
                label=_GRP_LABELS.get(_scen_group(sc), _fmt_ldp_label(sc))))
        legend_els.append(Line2D([0], [0], color="black", ls="--", lw=1.5,
                                  label="Ideal ({})".format(ideal)))
        ax.legend(handles=legend_els, fontsize=_LG, framealpha=0.9,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        slug = col.lower().replace("group_", "")
        _save_fig(fig, os.path.join(out_dir, "F1a_fairness_{}.png".format(slug)))


def _viz1_mia(mia, out_dir):
    """F1b: MIA AUC, baseline vs all augmented variants, mm_cf + nice_cf."""
    if mia.empty or "auc_roc" not in mia.columns: return
    fams = [f for f in _FAM_ORDER if f in mia["family"].unique()]
    itypes = [t for t in ["mm_cf","nice_cf"]
              if "input_type" in mia.columns and t in mia["input_type"].unique()]
    if not fams or not itypes: return

    _AUG_VARIANTS = ["augmented_SCM", "augmented_update_labels",
                     "augmented_add_comparators", "augmented"]
    _SCEN_ORDER_MIA = ["baseline"] + _AUG_VARIANTS
    scens_present = [sc for sc in _SCEN_ORDER_MIA if sc in mia["scenario"].unique()]

    best = mia[mia["scenario"].isin(scens_present)].groupby(
        ["scenario","family","input_type"])["auc_roc"].max().reset_index()
    n = len(itypes)
    fig, axes = plt.subplots(1, n, figsize=(7*n, 6.5), sharey=True)
    if n==1: axes=[axes]
    fig.suptitle("MIA AUC-ROC: Baseline vs. Augmented Methods (best attacker)",
                 fontsize=_TF+1, fontweight="bold", y=1.02)
    n_scens = len(scens_present)
    w_total = 0.8
    w = w_total / max(n_scens, 1)
    x = np.arange(len(fams))
    for ax, itype in zip(axes, itypes):
        sub = best[best["input_type"]==itype]
        for i, scen in enumerate(scens_present):
            sv   = sub[sub["scenario"]==scen]
            vals = [float(sv[sv["family"]==f]["auc_roc"].iloc[0])
                    if not sv[sv["family"]==f].empty else float("nan") for f in fams]
            h    = _HATCH_AUG if scen != "baseline" else _HATCH_BASE
            offset = (-w_total/2 + i*w + w/2)
            bars = ax.bar(x+offset, vals, w, color=[_fc(f) for f in fams],
                          alpha=0.85, hatch=h,
                          edgecolor="black" if h else "white", linewidth=0.6)
            _bar_labels(ax, bars, fmt="{:.4f}", fs=8)
        ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Random (0.50)")
        ax.set_title("Input: {}".format(itype.replace("_","-").upper()), fontsize=_LF)
        ax.set_xticks(x); ax.set_xticklabels([_fd(f) for f in fams], fontsize=_TK+1)
        ax.set_ylabel("MIA AUC-ROC", fontsize=_LF)
        vv = [v for v in best["auc_roc"].tolist() if not np.isnan(v)]
        ax.set_ylim(max(0.44, min(vv)-0.01) if vv else 0.44,
                    min(0.60, max(vv)+0.02) if vv else 0.60)
    els = ([mpatches.Patch(facecolor=_scenario_color(sc), edgecolor="black",
                            label=_GRP_LABELS.get(_scen_group(sc), sc))
            for sc in scens_present] +
           [Line2D([0],[0], color="red", ls="--", lw=1.5, label="Random (0.50)")] +
           [mpatches.Patch(facecolor=_fc(f), label=_fd(f)) for f in fams])
    fig.legend(handles=els, loc="lower center", ncol=len(els), fontsize=_LG,
               bbox_to_anchor=(0.5,-0.04), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F1b_mia_baseline_vs_augmented.png"))


def _viz1_nice(nice, out_dir):
    """F1c: NiCE CF quality, baseline vs all augmented variants."""
    if nice.empty: return
    _AUG_VARIANTS = ["augmented_SCM", "augmented_update_labels",
                     "augmented_add_comparators", "augmented"]
    _SCEN_ORDER_NICE = ["baseline"] + _AUG_VARIANTS
    scens_present = [sc for sc in _SCEN_ORDER_NICE if sc in nice["scenario"].unique()]
    sub  = nice[nice["scenario"].isin(scens_present)].copy()
    if sub.empty: return
    fams = [f for f in _FAM_ORDER if f in sub["family"].unique()]
    mets = [c for c in ["flip_rate","proximity","plausibility","sparsity"] if c in sub.columns]
    if not mets: return
    n = len(mets)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 6.5))
    if n==1: axes=[axes]
    fig.suptitle("NiCE CF Quality: Baseline vs. Augmented Methods",
                 fontsize=_TF+1, fontweight="bold", y=1.02)
    n_scens = len(scens_present)
    w_total = 0.8
    w = w_total / max(n_scens, 1)
    x = np.arange(len(fams))
    for ax, col in zip(axes, mets):
        for i, scen in enumerate(scens_present):
            sv   = sub[sub["scenario"]==scen]
            vals = [float(sv[sv["family"]==f][col].iloc[0])
                    if not sv[sv["family"]==f].empty else float("nan") for f in fams]
            h    = _HATCH_AUG if scen != "baseline" else _HATCH_BASE
            offset = (-w_total/2 + i*w + w/2)
            bars = ax.bar(x+offset, vals, w, color=[_fc(f) for f in fams],
                          alpha=0.85, hatch=h,
                          edgecolor="black" if h else "white", linewidth=0.6)
            _bar_labels(ax, bars, fmt="{:.3f}", fs=8)
        ax.set_title(col.replace("_"," ").title(), fontsize=_LF)
        ax.set_xticks(x); ax.set_xticklabels([_fd(f) for f in fams], fontsize=_TK+1)
        ax.set_ylabel(col.replace("_"," ").title(), fontsize=_LF)
    els = ([mpatches.Patch(facecolor=_scenario_color(sc), edgecolor="black",
                            label=_GRP_LABELS.get(_scen_group(sc), sc))
            for sc in scens_present] +
           [mpatches.Patch(facecolor=_fc(f), label=_fd(f)) for f in fams])
    fig.legend(handles=els, loc="lower center", ncol=len(els), fontsize=_LG,
               bbox_to_anchor=(0.5,-0.04), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F1c_nice_quality.png"))


def _viz_cf_fairness_baseline_vs_aug(fair, out_dir):
    """F1e: CF Fairness — Baseline vs all augmented variants, one figure per family.

    Higher = fairer (fraction of test points whose CF does not flip prediction).
    Files: F1e_cf_fairness_<family>.png
    """
    if fair.empty or "ind_cf_fairness" not in fair.columns:
        return

    _AUG_VARIANTS = ["augmented_SCM", "augmented_update_labels",
                     "augmented_add_comparators", "augmented"]
    _SCEN_ORDER_CF = ["baseline"] + _AUG_VARIANTS
    scens_present = [sc for sc in _SCEN_ORDER_CF if sc in fair["scenario"].unique()]
    sub = fair[fair["scenario"].isin(scens_present)].copy()
    if sub.empty:
        return

    for fam in _FAM_ORDER:
        fsub = sub[sub["family"] == fam]
        if fsub.empty:
            continue

        vals   = []
        colors = []
        labels = []
        for sc in scens_present:
            row = fsub[fsub["scenario"] == sc]
            if row.empty:
                continue
            vals.append(float(row["ind_cf_fairness"].iloc[0]))
            colors.append(_scenario_color(sc))
            labels.append(_GRP_LABELS.get(_scen_group(sc), _fmt_ldp_label(sc)))

        if not vals:
            continue

        fig_h = max(4, len(vals) * 0.6 + 2)
        fig, ax = plt.subplots(figsize=(9, fig_h))
        fig.suptitle(
            "CF Fairness: Baseline vs Augmented Methods  [{}]\n"
            "Higher = fairer  (fraction of test points whose CF does not flip prediction)"
            .format(_fd(fam)),
            fontsize=_TF, fontweight="bold")

        y = np.arange(len(labels))
        bars = ax.barh(y, vals, color=colors, alpha=0.88,
                       edgecolor="white", linewidth=0.4, height=0.5)
        for i, v in enumerate(vals):
            ax.text(v + 0.005, y[i], "{:.4f}".format(v),
                    va="center", ha="left", fontsize=10, fontweight="bold")

        ax.axvline(1.0, color="green",  ls="--", lw=1.5, label="Perfect (1.0)")
        ax.axvline(0.5, color="orange", ls=":",  lw=1.2, label="Random (0.5)")
        ax.set_xlim(0, 1.12)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 2)
        ax.invert_yaxis()
        ax.set_xlabel("CF Fairness  (higher = fairer)", fontsize=_LF)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.2, axis="x")
        ax.legend(fontsize=_LG, bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  framealpha=0.9)
        plt.tight_layout()
        tag = {"logistic_regression": "lr", "random_forest": "rf",
               "xgboost": "xgb"}.get(fam, fam)
        _save_fig(fig, os.path.join(out_dir, "F1e_cf_fairness_{}.png".format(tag)))


def _viz_cf_fairness_aug_vs_fairlearn(fair, ldp_fair_f, out_dir):
    """F2g: CF Fairness — Augmented vs LDP+Fairlearn variants, one figure per family.

    Rows: augmented (green) then all ldp_fair_eps* variants (purple shades),
    sorted by epsilon then constraint.
    Files: F2g_cf_fairness_<family>.png
    """
    col_main = "ind_cf_fairness"
    col_lf   = "cf_fairness"

    rows = []

    # All augmented variants from main fairness results
    _AUG_VARIANTS_CF2G = ["augmented_SCM", "augmented_update_labels",
                           "augmented_add_comparators", "augmented"]
    if not fair.empty and col_main in fair.columns:
        aug = fair[fair["scenario"].isin(_AUG_VARIANTS_CF2G)]
        for _, r in aug.iterrows():
            sc_r = str(r["scenario"])
            rows.append({
                "scenario": sc_r,
                "family":   str(r["family"]),
                "label":    _GRP_LABELS.get(_scen_group(sc_r), _fmt_ldp_label(sc_r)),
                "cf_fair":  float(r[col_main]),
            })

    # LDP+Fairlearn variants
    if not ldp_fair_f.empty and col_lf in ldp_fair_f.columns:
        df = ldp_fair_f.copy()
        if "epsilon" in df.columns:
            df["epsilon"] = df["epsilon"].astype(float)
        sort_cols = [c for c in ["epsilon", "constraint", "family"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        for _, r in df.iterrows():
            sc  = str(r.get("model", "ldp_fair"))
            fam_tag = str(r.get("family", ""))
            # map short tag to full name
            fam = _FAM_TO_FULL.get(fam_tag, fam_tag)
            eps  = r.get("epsilon", "?")
            c    = r.get("constraint", "")
            rows.append({
                "scenario": sc,
                "family":   fam,
                "label":    "LDP+FL ε={} {}".format(eps, c),
                "cf_fair":  float(r[col_lf]),
            })

    if not rows:
        return

    df_all = pd.DataFrame(rows)

    for fam in _FAM_ORDER:
        fsub = df_all[df_all["family"] == fam].copy()
        if fsub.empty:
            continue

        labels = fsub["label"].tolist()
        vals   = fsub["cf_fair"].tolist()
        colors = [_scenario_color(s) for s in fsub["scenario"].tolist()]
        n      = len(labels)
        height = max(5, n * 0.65 + 2.5)

        fig, ax = plt.subplots(figsize=(13, height))
        fig.suptitle(
            "CF Fairness: Augmented vs LDP+Fairlearn  [{}]\n"
            "Higher = fairer  (fraction of test points whose CF does not flip prediction)"
            .format(_fd(fam)),
            fontsize=_TF, fontweight="bold")

        y = np.arange(n)
        ax.barh(y, vals, color=colors, alpha=0.88,
                edgecolor="white", linewidth=0.4, height=0.68)

        for i, v in enumerate(vals):
            ax.text(v + 0.004, y[i], "{:.4f}".format(v),
                    va="center", ha="left", fontsize=9)

        # Separator between augmented and ldp_fair rows
        n_aug = len(fsub[fsub["scenario"].isin(_AUG_VARIANTS_CF2G)])
        if n_aug > 0 and n_aug < n:
            ax.axhline(n_aug - 0.5, color="#cccccc", lw=1.2, ls="--", zorder=0)

        ax.axvline(1.0, color="green",  ls="--", lw=1.5, label="Perfect (1.0)")
        ax.axvline(0.5, color="orange", ls=":",  lw=1.2, label="Random (0.5)")
        ax.set_xlim(0, 1.12)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 1)
        ax.invert_yaxis()
        ax.set_xlabel("CF Fairness  (higher = fairer)", fontsize=_LF)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.2, axis="x")

        aug_scens_present = [sc for sc in _AUG_VARIANTS_CF2G
                              if sc in fsub["scenario"].values]
        leg_els = ([mpatches.Patch(facecolor=_scenario_color(sc),
                                   label=_GRP_LABELS.get(_scen_group(sc), sc))
                    for sc in aug_scens_present] +
                   [mpatches.Patch(facecolor=_GRP_PALETTES["ldp_fair"][2],
                                   label="LDP + Fairlearn"),
                    Line2D([0], [0], color="green",  ls="--", lw=1.5, label="Perfect (1.0)"),
                    Line2D([0], [0], color="orange", ls=":",  lw=1.2, label="Random (0.5)")])
        leg_els = leg_els
        ax.legend(handles=leg_els, fontsize=_LG, bbox_to_anchor=(1.01, 1.0),
                  loc="upper left", framealpha=0.9)
        plt.tight_layout()
        tag = {"logistic_regression": "lr", "random_forest": "rf",
               "xgboost": "xgb"}.get(fam, fam)
        _save_fig(fig, os.path.join(out_dir, "F2g_cf_fairness_{}.png".format(tag)))


def generate_folder1(out_dir=_DIR1):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 1: Baseline vs. Augmented\n" + "="*60)
    # Prefer unified CSVs (multi-method) over single-run CSVs
    _u_fair = os.path.join(_UNIFIED_RES, "unified_fairness.csv")
    _u_mia  = os.path.join(_UNIFIED_RES, "unified_mia.csv")
    _u_nice = os.path.join(_UNIFIED_RES, "unified_nice_quality.csv")
    fair = (_load(_u_fair, "unified_fairness")   if os.path.exists(_u_fair)
            else _load(os.path.join(_MAIN_RES, "fairness_results.csv"), "fairness"))
    mia  = (_load(_u_mia,  "unified_mia")        if os.path.exists(_u_mia)
            else _load(os.path.join(_MAIN_RES, "mia_results.csv"),      "mia"))
    nice = (_load(_u_nice, "unified_nice")       if os.path.exists(_u_nice)
            else _load(os.path.join(_MAIN_RES, "nice_cf_quality.csv"),  "nice"))

    _viz1_fairness(fair, out_dir)
    _viz1_mia(mia, out_dir)
    _viz1_nice(nice, out_dir)
    _viz_cf_fairness_baseline_vs_aug(fair, out_dir)   # F1e

    # F1d: Paired bars — Baseline (solid) vs Augmented variants (hatched) for 4 MIA metrics
    if not mia.empty:
        _AUG_F1D = ["augmented_SCM", "augmented_update_labels",
                    "augmented_add_comparators", "augmented"]
        _SCEN_F1D = ["baseline"] + _AUG_F1D
        scens_f1d = [sc for sc in _SCEN_F1D if sc in mia["scenario"].unique()]
        sub = mia[mia["scenario"].isin(scens_f1d)].copy()
        if "input_type" in sub.columns:
            nc = sub[sub["input_type"]=="nice_cf"]
            sub = nc if not nc.empty else sub
        avail_m = [(c,t,r,d) for c,t,r,d in _MIA_METRICS if c in sub.columns]
        if avail_m:
            fams = [f for f in _FAM_ORDER if f in sub["family"].unique()]
            n = len(avail_m)
            fig, axes = plt.subplots(1, n, figsize=(5.5*n, 6.5))
            if n==1: axes=[axes]
            fig.suptitle("MIA Multi-Metric: Baseline vs. Augmented Methods\n"
                         "(NiCE-CF proxy — best attacker per model; solid = Baseline, hatched = Augmented)",
                         fontsize=_TF, fontweight="bold", y=1.02)
            n_scens_f1d = len(scens_f1d)
            w_tot_f1d = 0.8
            w_f1d = w_tot_f1d / max(n_scens_f1d, 1)
            x = np.arange(len(fams))
            for ax, (col, panel_t, ref_val, note) in zip(axes, avail_m):
                for i, scen in enumerate(scens_f1d):
                    ssub = sub[sub["scenario"]==scen]
                    vals = []
                    for f in fams:
                        fsub = ssub[ssub["family"]==f]
                        vals.append(float(fsub[col].max()) if not fsub.empty else float("nan"))
                    hatch = _HATCH_AUG if scen != "baseline" else _HATCH_BASE
                    offset = (-w_tot_f1d/2 + i*w_f1d + w_f1d/2)
                    bars = ax.bar(x+offset, vals, w_f1d,
                                  color=[_fc(f) for f in fams], alpha=0.85,
                                  hatch=hatch, edgecolor="black" if hatch else "white",
                                  linewidth=0.5)
                    _bar_labels(ax, bars, fmt="{:.4f}", fs=7.5)
                if ref_val is not None:
                    ax.axhline(ref_val, color="red", ls="--", lw=1.5,
                               label="Ref. ({})".format(ref_val))
                    ax.legend(fontsize=_LG-1, bbox_to_anchor=(1.01, 1.0),
                              loc="upper left", framealpha=0.9)
                ax.set_title("{}\n{}".format(panel_t, note), fontsize=_LF-1, pad=6)
                ax.set_xticks(x)
                ax.set_xticklabels([_fd(f) for f in fams], fontsize=_TK+1)
                ax.set_ylabel(panel_t, fontsize=_LF)
            els = ([mpatches.Patch(facecolor=_scenario_color(sc), edgecolor="black",
                                   label=_GRP_LABELS.get(_scen_group(sc), sc))
                    for sc in scens_f1d] +
                   [mpatches.Patch(facecolor=_fc(f), label=_fd(f)) for f in fams])
            fig.legend(handles=els, loc="lower center", ncol=len(els), fontsize=_LG,
                       bbox_to_anchor=(0.5,-0.04), framealpha=0.9)
            plt.tight_layout()
            _save_fig(fig, os.path.join(out_dir, "F1d_mia_multi_metrics.png"))
    print("  [Folder 1] Done ->", out_dir)


# ===========================================================================
# FOLDER 2 — Augmented vs Fairlearn
# ===========================================================================

def _viz2_fairness(main_fair, fl_fair, out_dir):
    """F2a: One horizontal-bar figure per fairness metric.
    Groups: Baseline → MM-CF Augmented → Fairlearn.
    Color = semantic scenario group; bar label = scenario + family."""
    if main_fair.empty and fl_fair.empty: return

    fams = [f for f in _FAM_ORDER if not main_fair.empty
            and f in main_fair["family"].unique()]
    fl   = fl_fair[~fl_fair["scenario"].str.startswith("baseline")].copy() \
           if not fl_fair.empty else pd.DataFrame()

    metrics = [
        ("group_SPD", "SPD", "Statistical Parity Difference  (0 = perfectly fair)"),
        ("group_EOD", "EOD", "Equal Opportunity Difference   (0 = perfectly fair)"),
    ]

    for col, short, title in metrics:
        rows = []

        # Baseline + Augmented variants (main pipeline)
        _AUG_F2A = ["augmented_SCM", "augmented_update_labels",
                    "augmented_add_comparators", "augmented"]
        _SCEN_F2A = ["baseline"] + _AUG_F2A
        scens_f2a = [sc for sc in _SCEN_F2A if not main_fair.empty
                     and sc in main_fair["scenario"].unique()]
        for scen in scens_f2a:
            for fam in fams:
                if main_fair.empty: continue
                row = main_fair[(main_fair["scenario"] == scen) &
                                (main_fair["family"]   == fam)]
                if row.empty or col not in row.columns: continue
                rows.append({
                    "label": "{} — {}".format(_fmt_ldp_label(scen), _fd(fam)),
                    "value": float(row[col].iloc[0]),
                    "color": _scenario_color(scen),
                    "group": scen,
                })

        n_main = len(rows)

        # Fairlearn models
        if not fl.empty and col in fl.columns:
            for _, row in fl.sort_values(["family", "scenario"]).iterrows():
                lbl = row.get("display", row.get("label", row.get("scenario", "?")))
                v   = row[col]
                if pd.isna(v): continue
                rows.append({
                    "label": _wrap(str(lbl), 22),
                    "value": float(v),
                    "color": _GRP_PALETTES["fairlearn"][2],
                    "group": "fairlearn",
                })

        if not rows: continue

        labels = [r["label"] for r in rows]
        values = [r["value"] for r in rows]
        colors = [r["color"] for r in rows]

        fig_h = max(5, len(rows) * 0.55 + 2.5)
        fig, ax = plt.subplots(figsize=(12, fig_h))

        y    = np.arange(len(rows))
        bars = ax.barh(y, values, color=colors, alpha=0.88, edgecolor="white", height=0.62)

        # Value annotations
        _hbar_labels(ax, bars, fmt="{:.4f}", fs=9)

        ax.axvline(0, color="black", lw=1.5, ls="--", alpha=0.7, label="Ideal (0)")

        # Group separator between baseline/augmented block and fairlearn block
        if n_main > 0 and n_main < len(rows):
            ax.axhline(n_main - 0.5, color="#aaaaaa", lw=1.2, ls=":")
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -0.01,
                    n_main / 2 - 0.3,
                    "Baseline / Aug", ha="left", va="center",
                    fontsize=_TK - 1, color="#555", style="italic")
            ax.text(ax.get_xlim()[0] if ax.get_xlim()[0] != 0 else -0.01,
                    n_main + (len(rows) - n_main) / 2 - 0.3,
                    "Fairlearn", ha="left", va="center",
                    fontsize=_TK - 1, color="#555", style="italic")

        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 1)
        ax.invert_yaxis()
        ax.set_xlabel(short, fontsize=_LF)
        ax.set_title(
            "Baseline, MM-CF Augmented and Fairlearn\n{}".format(title),
            fontsize=_TF, fontweight="bold")

        aug_lgd_scens = [sc for sc in scens_f2a if sc != "baseline"]
        legend_els = ([mpatches.Patch(facecolor=_scenario_color("baseline"), label="Baseline")] +
                      [mpatches.Patch(facecolor=_scenario_color(sc),
                                      label=_GRP_LABELS.get(_scen_group(sc), sc))
                       for sc in aug_lgd_scens] +
                      [mpatches.Patch(facecolor=_GRP_PALETTES["fairlearn"][2], label="Fairlearn"),
                       Line2D([0], [0], color="black", ls="--", lw=1.5, label="Ideal (0)")])
        ax.legend(handles=legend_els, fontsize=_LG, framealpha=0.9,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")
        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir, "F2a_fairness_{}.png".format(short.lower())))


def _viz2_mia(aug_mia, fl_mia, out_dir):
    """F2b: MIA AUC — augmented vs fairlearn, 3 panels by family."""
    if aug_mia.empty and fl_mia.empty: return

    # Normalise both datasets to full family names
    aug_mia = _normalize_family(aug_mia.copy()) if not aug_mia.empty else aug_mia
    fl_mia  = _normalize_family(fl_mia.copy())  if not fl_mia.empty  else fl_mia

    fams = [f for f in _FAM_ORDER
            if (not aug_mia.empty and f in aug_mia.get("family", pd.Series()).unique())
            or (not fl_mia.empty  and f in fl_mia.get("family",  pd.Series()).unique())]
    if not fams: return

    # Augmented: best nice_cf AUC per family (all augmented variants)
    _AUG_F2B = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    aug_sub = pd.DataFrame()
    if not aug_mia.empty:
        a = aug_mia[aug_mia["scenario"].isin(_AUG_F2B)].copy()
        if "input_type" in a.columns:
            a2 = a[a["input_type"]=="nice_cf"]
            a  = a2 if not a2.empty else a
        if not a.empty:
            aug_sub = a.groupby(["scenario","family"])["auc_roc"].max().reset_index()

    # Fairlearn: best AUC per model × family × constraint (skip baselines)
    fl_sub = pd.DataFrame()
    if not fl_mia.empty:
        f2 = fl_mia.copy()
        if "constraint" in f2.columns:
            f2 = f2[f2["constraint"] != "baseline"]
        grp_cols = [c for c in ["model","family","constraint"] if c in f2.columns]
        if grp_cols:
            fl_sub = f2.groupby(grp_cols)["auc_roc"].max().reset_index()

    n_fam = len(fams)
    fig, axes = plt.subplots(1, n_fam, figsize=(8*n_fam, 8), sharey=True)
    if n_fam==1: axes=[axes]
    fig.suptitle("MIA AUC-ROC: MM-CF Augmented vs. Fairlearn Models\n"
                 "(NiCE-CF proxy — best attacker; each panel = one base estimator)",
                 fontsize=_TF+1, fontweight="bold", y=1.02)

    all_vals = []
    for ax, fam in zip(axes, fams):
        labels, vals, colors, hatches = [], [], [], []

        # Augmented bar (hatched, family colour)
        if not aug_sub.empty:
            row = aug_sub[aug_sub["family"]==fam]
            if not row.empty:
                labels.append("Aug-{}".format(_fd(fam)))
                v = float(row["auc_roc"].iloc[0])
                vals.append(v); all_vals.append(v)
                colors.append(_fc(fam)); hatches.append(_HATCH_AUG)

        # Fairlearn bars for this family
        if not fl_sub.empty and "family" in fl_sub.columns:
            sf = fl_sub[fl_sub["family"]==fam].sort_values("constraint")
            for _, row in sf.iterrows():
                labels.append(_wrap(str(row.get("constraint","?")), 12))
                v = float(row["auc_roc"]); vals.append(v); all_vals.append(v)
                colors.append(_fc(fam)); hatches.append(_HATCH_BASE)

        if not labels:
            ax.set_title("{} — no data".format(_fd(fam))); continue

        x = np.arange(len(labels))
        for i, (v, c, h) in enumerate(zip(vals, colors, hatches)):
            bar = ax.bar([x[i]], [v], color=c, alpha=0.85, hatch=h,
                         edgecolor="black" if h else "white",
                         linewidth=0.5, width=0.72)
            _bar_labels(ax, bar, fmt="{:.4f}", fs=8)

        ax.axhline(0.5, color="red", ls="--", lw=1.5)
        ax.set_title("{} base estimator".format(_fd(fam)),
                     fontsize=_LF+1, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=_TK)
        ax.set_ylabel("MIA AUC-ROC", fontsize=_LF)

        # Divider after augmented bar
        if labels and hatches[0] == _HATCH_AUG and len(labels) > 1:
            ax.axvline(0.5, color="gray", ls=":", lw=1.5, alpha=0.7)

    ylo = max(0.44, min(all_vals)-0.005) if all_vals else 0.44
    yhi = min(0.62, max(all_vals)+0.02)  if all_vals else 0.62
    for ax in axes: ax.set_ylim(ylo, yhi)

    els = [mpatches.Patch(facecolor="#bbb", hatch=_HATCH_AUG, edgecolor="black",
                           label="MM-CF Augmented"),
           mpatches.Patch(facecolor="#bbb", edgecolor="white", label="Fairlearn"),
           Line2D([0],[0], color="red", ls="--", lw=1.5, label="Random (0.50)")]
    fig.legend(handles=els, loc="lower center", ncol=3, fontsize=_LG,
               bbox_to_anchor=(0.5,-0.03), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F2b_mia_aug_vs_fairlearn.png"))


def _viz2_mia_all_variants(aug_mia, ldp_fair_mia, out_dir):
    """F2f: MIA AUC-ROC — all augmented variants vs all fairlearn variants.

    Horizontal bar chart per family showing:
      Group 1  Baseline / Augmented
      Group 2  LDP-Only (ldp_baseline_eps*)
      Group 3  LDP + Augmented (ldp_augmented_eps*)
      Group 4  LDP + Fairlearn (ldp_fair_eps* × constraint)

    Uses best-attacker AUC per scenario/family.
    Files: F2f_mia_all_lr.png / _rf.png / _xgb.png
    """
    if aug_mia.empty and ldp_fair_mia.empty:
        return

    # ── Build best-AUC rows from mia_results.csv ─────────────────────────
    rows = []
    if not aug_mia.empty and "auc_roc" in aug_mia.columns:
        grp_cols = [c for c in ["scenario", "family", "input_type"] if c in aug_mia.columns]
        for _, r in aug_mia.groupby(grp_cols)["auc_roc"].max().reset_index().iterrows():
            sc  = str(r["scenario"])
            fam = str(r.get("family", ""))
            itp = str(r.get("input_type", "any"))
            rows.append({"scenario": sc, "family": fam, "input_type": itp,
                         "label": None, "auc": float(r["auc_roc"])})

    # Keep best AUC per (scenario, family): prefer nice_cf over mm_cf
    from collections import defaultdict
    best = {}   # (sc, fam) -> dict
    for row in rows:
        k = (row["scenario"], row["family"])
        itp = row["input_type"]
        prev = best.get(k)
        if prev is None:
            best[k] = row
        else:
            # prefer nice_cf; if same type, take higher
            if itp == "nice_cf" and prev["input_type"] != "nice_cf":
                best[k] = row
            elif itp == prev["input_type"] and row["auc"] > prev["auc"]:
                best[k] = row
    aug_rows = list(best.values())

    # ── Build rows from ldp_fair_mia.csv ─────────────────────────────────
    lf_rows = []
    if not ldp_fair_mia.empty and "auc_roc" in ldp_fair_mia.columns:
        gc = [c for c in ["model", "family", "constraint", "epsilon"]
              if c in ldp_fair_mia.columns]
        for _, r in ldp_fair_mia.groupby(gc)["auc_roc"].max().reset_index().iterrows():
            fam  = str(r.get("family", ""))
            eps  = r.get("epsilon", "?")
            c    = r.get("constraint", "")
            sc   = str(r.get("model", "ldp_fair_eps{}_{}".format(eps, c)))
            lf_rows.append({"scenario": sc, "family": fam,
                             "input_type": "any", "label": None,
                             "auc": float(r["auc_roc"])})

    # ── One figure per family ────────────────────────────────────────────
    for fam in _FAM_ORDER:
        fam_aug = [r for r in aug_rows if r["family"] == fam]
        fam_lf  = [r for r in lf_rows  if r["family"] == fam]
        if not fam_aug and not fam_lf:
            continue

        # Sort aug_rows by group then eps
        def _sort_key(r):
            sc = r["scenario"]
            g  = _scen_group(sc)
            gi = _GRP_ORDER.index(g) if g in _GRP_ORDER else 99
            e  = _scen_eps(sc) or 0.0
            return (gi, e, sc)

        fam_aug_s = sorted(fam_aug, key=_sort_key)
        # Sort lf rows by epsilon then constraint
        fam_lf_s  = sorted(fam_lf, key=lambda r: (
            _scen_eps(r["scenario"]) or 0.0,
            r["scenario"]))

        combined = fam_aug_s + fam_lf_s

        # Build display labels
        def _label(r):
            sc = r["scenario"]
            if sc == "baseline":   return "Baseline"
            if sc in ("augmented", "augmented_SCM"):              return "SCM Augmented"
            if sc == "augmented_update_labels":                   return "MM Update Labels"
            if sc == "augmented_add_comparators":                 return "MM Add Comparators"
            g = _scen_group(sc)
            eps = _scen_eps(sc)
            eps_s = "ε={}".format(eps) if eps is not None else ""
            if g == "ldp_baseline":  return "LDP-Only {}".format(eps_s)
            if g == "ldp_augmented": return "LDP+Aug {}".format(eps_s)
            if g == "ldp_fair":
                # parse constraint from model name
                parts = sc.split("_")
                # ldp_fair_eps0.5_eg_dp_lr → constraint = eg_dp or to_dp
                c_part = ""
                for part in ["eg_dp", "to_dp", "eg_eo", "to_eo"]:
                    if part in sc:
                        c_part = part; break
                return "LDP+FL {} {}".format(eps_s, c_part)
            return sc

        labels = [_label(r) for r in combined]
        aucs   = [r["auc"] for r in combined]
        colors = [_scenario_color(r["scenario"]) for r in combined]
        n      = len(labels)

        height = max(6, n * 0.62 + 2.5)
        fig, ax = plt.subplots(figsize=(14, height))
        fig.suptitle(
            "MIA AUC-ROC — All Variants: Augmented vs. Fairlearn  [{}]"
            .format(_fd(fam)),
            fontsize=_TF + 1, fontweight="bold")

        y = np.arange(n)
        ax.barh(y, aucs, color=colors, alpha=0.88,
                edgecolor="white", linewidth=0.4, height=0.68)

        # value annotations
        finite_aucs = [v for v in aucs if np.isfinite(v)]
        lo = max(0.44, min(finite_aucs) - 0.01) if finite_aucs else 0.44
        hi = max(0.65, max(finite_aucs) + 0.06) if finite_aucs else 0.65
        for i, v in enumerate(aucs):
            if np.isfinite(v):
                ax.text(v + (hi - lo) * 0.01, y[i], "{:.4f}".format(v),
                        va="center", ha="left", fontsize=9, color="#222")

        # Group separators
        seen_groups = []
        for i, r in enumerate(combined):
            g = _scen_group(r["scenario"])
            if not seen_groups or g != seen_groups[-1]:
                if seen_groups:
                    ax.axhline(i - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
                seen_groups.append(g)

        ax.axvline(0.5, color="red", ls="--", lw=1.8, label="Random baseline (0.50)")
        ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.set_xlabel("MIA AUC-ROC  (lower = more private)", fontsize=_LF)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 1)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis="x")

        # Legend — scenario groups
        present_grps = list(dict.fromkeys(
            _scen_group(r["scenario"]) for r in combined))
        leg_els = [
            mpatches.Patch(
                facecolor=_GRP_PALETTES[g][min(1, len(_GRP_PALETTES[g]) - 1)],
                edgecolor="#555", label=_GRP_LABELS.get(g, g))
            for g in _GRP_ORDER if g in present_grps
        ]
        leg_els.append(
            Line2D([0], [0], color="red", ls="--", lw=1.8,
                   label="Random MIA (0.50)"))
        ax.legend(handles=leg_els, fontsize=_LG, framealpha=0.9,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        tag = {"logistic_regression": "lr",
               "random_forest":       "rf",
               "xgboost":             "xgb"}.get(fam, fam)
        _save_fig(fig, os.path.join(out_dir, "F2f_mia_all_{}.png".format(tag)))


def _viz2_scatter(main_fair, aug_mia, fl_fair, fl_mia, out_dir):
    """F2c: Privacy-Fairness scatter — MIA AUC (x) vs |SPD| (y).
    Augmented = diamond (◆), Fairlearn = circle (●).
    Both axes derived from available CSVs (no accuracy needed)."""

    # Build MIA lookup: best nice_cf AUC per (scenario/model, family)
    _AUG_F2C = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    mia_lookup = {}
    if not aug_mia.empty and "auc_roc" in aug_mia.columns:
        a = aug_mia[aug_mia["scenario"].isin(_AUG_F2C)]
        if "input_type" in a.columns:
            a2 = a[a["input_type"]=="nice_cf"]
            a  = a2 if not a2.empty else a
        for _, r in a.groupby(["scenario","family"])["auc_roc"].max().reset_index().iterrows():
            mia_lookup[(str(r["scenario"]), str(r["family"]))] = float(r["auc_roc"])

    fl_mia_n = _normalize_family(fl_mia.copy()) if not fl_mia.empty else fl_mia
    if not fl_mia_n.empty and "auc_roc" in fl_mia_n.columns:
        if "constraint" in fl_mia_n.columns:
            fl_mia_n = fl_mia_n[fl_mia_n["constraint"]!="baseline"]
        if "model" in fl_mia_n.columns:
            for _, r in fl_mia_n.groupby(["model","family"])["auc_roc"].max().reset_index().iterrows():
                mia_lookup[(str(r["model"]), str(r["family"]))] = float(r["auc_roc"])

    rows = []
    # Augmented points (all variants)
    if not main_fair.empty and "group_SPD" in main_fair.columns:
        for _, r in main_fair[main_fair["scenario"].isin(_AUG_F2C)].iterrows():
            sc, fam = str(r["scenario"]), str(r["family"])
            mia_v = mia_lookup.get((sc, fam), float("nan"))
            rows.append({"label": "{}-{}".format(_fmt_ldp_label(sc), _fd(fam)),
                         "spd_abs": abs(float(r["group_SPD"])),
                         "mia_auc": mia_v, "group": _scen_group(sc), "family": fam})

    # Fairlearn points (skip baselines)
    fl_n = _fl_norm(fl_fair)
    if not fl_n.empty and "group_SPD" in fl_n.columns:
        fl_nb = fl_n[~fl_n["scenario"].str.startswith("baseline")]
        for _, r in fl_nb.iterrows():
            sc, fam = str(r["scenario"]), str(r["family"])
            mia_v = mia_lookup.get((sc, fam), float("nan"))
            rows.append({"label": _wrap(str(r.get("display", r.get("label", sc))), 18),
                         "spd_abs": abs(float(r["group_SPD"])),
                         "mia_auc": mia_v, "group": "fairlearn", "family": fam})

    df = pd.DataFrame(rows).dropna(subset=["spd_abs"])
    if df.empty: return

    # Figure: plot area + legend panel
    fig, (ax, ax_leg) = plt.subplots(1, 2, figsize=(15, 8),
                                      gridspec_kw={"width_ratios":[3.5,1]})
    ax_leg.axis("off")

    for _, row in df.iterrows():
        c = _fc(row["family"])
        _AUG_GROUPS_SC = {"augmented", "augmented_SCM", "augmented_update_labels",
                          "augmented_add_comparators"}
        m = "D" if row["group"] in _AUG_GROUPS_SC else "o"
        mia_v = row.get("mia_auc", float("nan"))
        if not np.isnan(mia_v):
            ax.scatter(mia_v, row["spd_abs"], color=c, marker=m,
                       s=160, alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
            ax.annotate(row["label"], (mia_v, row["spd_abs"]),
                        textcoords="offset points", xytext=(6,5), fontsize=9, alpha=0.88)
        else:
            # No MIA data: plot on right margin as stripped
            ax.scatter(0.505, row["spd_abs"], color=c, marker=m,
                       s=90, alpha=0.35, edgecolors=c, linewidths=1.2,
                       zorder=2, linestyle=":")

    ax.axvline(0.5, color="red", ls="--", lw=1.2, alpha=0.7, label="Random MIA (0.50)")
    ax.set_xlabel("MIA AUC-ROC   (lower = more private  ←)", fontsize=_LF)
    ax.set_ylabel("|SPD|   (lower = fairer  ↓)", fontsize=_LF)
    ax.set_title("Privacy–Fairness Trade-off: MM-CF Augmented vs. Fairlearn\n"
                 "Ideal region: bottom-left corner (low MIA + low |SPD|)",
                 fontsize=_TF, fontweight="bold")

    els = [
        mpatches.Patch(facecolor=_C_LR,  label="LR base"),
        mpatches.Patch(facecolor=_C_RF,  label="RF base"),
        mpatches.Patch(facecolor=_C_XGB, label="XGB base"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#555",
               ms=10, label="MM-CF Augmented (◆)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#555",
               ms=10, label="Fairlearn models (●)"),
        Line2D([0],[0], color="red", ls="--", lw=1.5, label="Random MIA (0.50)"),
    ]
    ax_leg.legend(handles=els, loc="center", fontsize=_LG+1, framealpha=0.9,
                  title="Legend", title_fontsize=_LG+1)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F2c_privacy_fairness_scatter.png"))


def _viz2_flip_rate(aug_nice, fl_nice, out_dir):
    """F2d: NiCE flip-rate with clear interpretation subtitle."""
    _AUG_F2D = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    rows_a, rows_f = [], []
    if not aug_nice.empty and "flip_rate" in aug_nice.columns:
        for _, r in aug_nice[aug_nice["scenario"].isin(_AUG_F2D)].iterrows():
            sc_r = str(r["scenario"])
            rows_a.append({"label": "{}-{}".format(_fmt_ldp_label(sc_r), _fd(r["family"])),
                           "flip_rate": float(r["flip_rate"]),
                           "family": r["family"], "group": sc_r})

    fl_n = _normalize_family(fl_nice.copy()) if not fl_nice.empty else fl_nice
    if not fl_n.empty and "flip_rate" in fl_n.columns:
        mask = True
        if "model" in fl_n.columns:
            mask = ~fl_n["model"].str.startswith("baseline")
        elif "scenario" in fl_n.columns:
            mask = ~fl_n["scenario"].str.startswith("baseline")
        for _, r in fl_n[mask].iterrows():
            lbl = r.get("label", r.get("model","?"))
            rows_f.append({"label": _wrap(str(lbl),18),
                           "flip_rate": float(r["flip_rate"]),
                           "family": r.get("family",""), "group":"fairlearn"})

    if not rows_a and not rows_f: return

    _AUG_GROUPS_F2D = {"augmented", "augmented_SCM", "augmented_update_labels",
                       "augmented_add_comparators"}
    df = pd.DataFrame(rows_a + rows_f)
    x  = np.arange(len(df))
    colors  = [_scenario_color(str(r["group"])) if r["group"] in _AUG_GROUPS_F2D
               else _fc(r["family"]) for _, r in df.iterrows()]
    hatches = [_HATCH_AUG if r["group"] in _AUG_GROUPS_F2D else _HATCH_BASE
               for _, r in df.iterrows()]

    fig, ax = plt.subplots(figsize=(max(14, len(df)*0.78+2), 8))

    for i, (v, c, h) in enumerate(zip(df["flip_rate"], colors, hatches)):
        bar = ax.bar([x[i]], [v], color=c, alpha=0.85, hatch=h,
                     edgecolor="black" if h else "white",
                     linewidth=0.5, width=0.72)
        _bar_labels(ax, bar, fmt="{:.2f}", fs=9)

    ax.axhline(1.0, color="gray", ls=":", lw=1.2, alpha=0.7, label="Flip rate = 1.0 (all flip)")
    ax.set_title(
        "NiCE CF Flip-Rate: MM-CF Augmented vs. Fairlearn Models\n"
        "How to read: Flip Rate = fraction of test inputs where the closest counterfactual\n"
        "changes the model's prediction.   HIGH = easy to flip = model relies on sensitive features.\n"
        "LOW = hard to flip = model is robust = better privacy against CF-based attacks.",
        fontsize=_TF-1, fontweight="bold", linespacing=1.5)
    ax.set_xticks(x)
    ax.set_xticklabels([_wrap(l,20) for l in df["label"].tolist()],
                       rotation=40, ha="right", fontsize=_TK-1)
    ax.set_ylabel("NiCE Flip Rate  (lower = harder to flip = more robust)", fontsize=_LF)
    ax.set_ylim(0, 1.18)

    n_aug = len(rows_a)
    if n_aug > 0 and rows_f:
        ax.axvline(n_aug-0.5, color="gray", ls=":", lw=1.5)

    els = [mpatches.Patch(facecolor="#bbb", hatch=_HATCH_AUG, edgecolor="black",
                           label="MM-CF Augmented"),
           mpatches.Patch(facecolor="#bbb", edgecolor="white", label="Fairlearn"),
           mpatches.Patch(facecolor=_C_LR, label="LR"),
           mpatches.Patch(facecolor=_C_RF, label="RF"),
           mpatches.Patch(facecolor=_C_XGB, label="XGB")]
    fig.legend(handles=els, loc="lower center", ncol=5, fontsize=_LG,
               bbox_to_anchor=(0.5,-0.03), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F2d_nice_flip_rate.png"))


def generate_folder2(out_dir=_DIR2):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 2: Augmented vs. Fairlearn\n" + "="*60)
    _u_fair2 = os.path.join(_UNIFIED_RES, "unified_fairness.csv")
    _u_mia2  = os.path.join(_UNIFIED_RES, "unified_mia.csv")
    _u_nice2 = os.path.join(_UNIFIED_RES, "unified_nice_quality.csv")
    main_fair    = (_load(_u_fair2, "unified_fairness_f2") if os.path.exists(_u_fair2)
                    else _load(os.path.join(_MAIN_RES, "fairness_results.csv"), "main fairness"))
    aug_mia      = (_load(_u_mia2, "unified_mia_f2")       if os.path.exists(_u_mia2)
                    else _load(os.path.join(_MAIN_RES, "mia_results.csv"),      "aug mia"))
    aug_nice     = (_load(_u_nice2, "unified_nice_f2")     if os.path.exists(_u_nice2)
                    else _load(os.path.join(_MAIN_RES, "nice_cf_quality.csv"),  "aug nice"))
    ldp_fair_mia = _load(os.path.join(_LF_RES,   "ldp_fair_mia.csv"),         "ldp_fair mia")
    ldp_fair_f   = _load(os.path.join(_LF_RES,   "ldp_fair_fairness.csv"),    "ldp_fair fairness")
    fl_fair      = _load_fl_fair()
    fl_mia       = _load(os.path.join(_FL_RES, "fairlearn_mia.csv"),           "fl mia")
    fl_metrics   = _load(os.path.join(_FL_RES, "fairlearn_model_metrics.csv"), "fl metrics")
    fl_nice      = _load(os.path.join(_FL_RES, "fairlearn_nice_quality.csv"),  "fl nice")

    _viz2_fairness(main_fair, fl_fair, out_dir)
    _viz2_mia(aug_mia, fl_mia, out_dir)
    _viz2_mia_all_variants(aug_mia, ldp_fair_mia, out_dir)      # F2f
    _viz_cf_fairness_aug_vs_fairlearn(main_fair, ldp_fair_f, out_dir)  # F2g
    _viz2_scatter(main_fair, aug_mia, fl_fair, fl_mia, out_dir)
    _viz2_flip_rate(aug_nice, fl_nice, out_dir)

    # F2e: 4 separate landscape vertical figures for fairlearn MIA
    if not fl_mia.empty and "auc_roc" in fl_mia.columns:
        f2 = _normalize_family(fl_mia.copy())
        f2["input_type"] = "nice_cf"
        if "constraint" in f2.columns:
            f2 = f2[f2["constraint"]!="baseline"]
        lbl_col = "model_label" if "model_label" in f2.columns else \
                  "label"       if "label"       in f2.columns else "model"
        f2["_lbl"] = f2[lbl_col]
        _viz_mia_4sep_vertical(
            df=f2, label_col="_lbl", family_col="family",
            color_fn=lambda k: _fc(k[1] if len(k)>1 else ""),
            title_prefix="Fairlearn MIA",
            out_prefix=os.path.join(out_dir, "F2e"),
            input_type_filter=None,
        )
    print("  [Folder 2] Done ->", out_dir)


# ===========================================================================
# FOLDER 3 — LDP and Fairness
# ===========================================================================

def _viz3_ldp_sweep(ldp, out_dir):
    """F3a: LDP epsilon sweep — all epsilon ticks forced."""
    if ldp.empty or "epsilon" not in ldp.columns: return
    ldp = ldp.copy(); ldp["epsilon"] = ldp["epsilon"].astype(float)
    fams     = [f for f in _FAM_ORDER if f in ldp["family"].unique()]
    eps_vals = sorted(ldp["epsilon"].unique())
    if not fams or not eps_vals: return

    panels = [(c,y,ab,note) for c,y,ab,note in [
        ("group_SPD",    "|SPD|",       True,  "lower = fairer"),
        ("group_EOD",    "|EOD|",       True,  "lower = fairer"),
        ("group_EqOdds", "EqOdds",      False, "lower = fairer"),
        ("ind_cf_fairness","CF-Fair.",  False, "higher = fairer"),
    ] if c in ldp.columns]
    if not panels: return

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(6.5*n, 6.5))
    if n==1: axes=[axes]
    fig.suptitle("LDP Epsilon Sweep: Privacy–Fairness Trade-off (COMPAS)\n"
                 "Higher ε = less noise on race attribute = closer to original",
                 fontsize=_TF+1, fontweight="bold", y=1.02)

    for ax, (col, ylabel, abs_val, note) in zip(axes, panels):
        for fam in fams:
            sub = ldp[ldp["family"]==fam].sort_values("epsilon")
            if sub.empty or col not in sub.columns: continue
            y = sub[col].abs() if abs_val else sub[col]
            ax.plot(sub["epsilon"], y, marker="o", ms=9, lw=2.5,
                    color=_fc(fam), label=_fd(fam))
            for ex, ey in zip(sub["epsilon"].tolist(), y.tolist()):
                ax.annotate("{:.3f}".format(ey), (ex, ey),
                            textcoords="offset points", xytext=(4,5), fontsize=8)

        ax.set_xscale("log")
        ax.set_xlim(eps_vals[0] * 0.65, eps_vals[-1] * 1.55)
        ax.set_xticks(eps_vals)
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(e) for e in eps_vals]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.tick_params(axis="x", labelsize=_TK, rotation=0)
        ax.set_xlabel("LDP  ε", fontsize=_LF)
        ax.set_ylabel(ylabel, fontsize=_LF)
        ax.set_title("{} — {}".format(ylabel, note), fontsize=_LF, pad=8)
        ax.legend(fontsize=_LG, bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  framealpha=0.9)
    plt.tight_layout(rect=[0, 0, 0.88, 1])
    _save_fig(fig, os.path.join(out_dir, "F3a_ldp_epsilon_sweep.png"))


def _viz3_ldp_fair_heatmap(ldp_fair_f, out_dir):
    """F3b: SPD/EOD heatmaps for LDP+Fairlearn."""
    if ldp_fair_f.empty: return
    for orig, title_s in [("SPD","closer to 0 = fairer"),("EOD","closer to 0 = fairer")]:
        src = "group_"+orig if "group_"+orig in ldp_fair_f.columns \
              else orig if orig in ldp_fair_f.columns else None
        if src is None or "epsilon" not in ldp_fair_f.columns: continue
        df = ldp_fair_f.copy(); df["epsilon"] = df["epsilon"].astype(float)
        eps_vals = sorted(df["epsilon"].unique())

        # Build readable row labels from model key (avoid unicode epsilon in CSV)
        def _row_lbl(row):
            eps = row.get("epsilon", "?")
            c   = row.get("constraint", "?")
            f   = _fd(row.get("family",""))
            return "eps={} {} ({})".format(eps, c, f)

        df["_row"] = df.apply(_row_lbl, axis=1)
        row_labels = df.drop_duplicates("_row")["_row"].tolist()
        piv = []
        for lbl in row_labels:
            sub = df[df["_row"]==lbl]
            piv.append([float(sub[np.isclose(sub["epsilon"].astype(float), e)][src].iloc[0])
                        if not sub[np.isclose(sub["epsilon"].astype(float), e)].empty
                        else float("nan") for e in eps_vals])

        mat = np.array(piv)
        fig, ax = plt.subplots(figsize=(max(8, len(eps_vals)*2+2),
                                        max(5, len(row_labels)*0.6+2)))
        im = ax.imshow(mat, cmap="RdYlGn_r", aspect="auto", vmin=-0.5, vmax=0.5)
        cb = plt.colorbar(im, ax=ax, fraction=0.025)
        cb.set_label("{} value".format(orig), fontsize=_LF)
        ax.set_xticks(range(len(eps_vals)))
        ax.set_xticklabels(["eps={}".format(e) for e in eps_vals], fontsize=_TK+1)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=_TK)
        ax.set_xlabel("LDP Epsilon", fontsize=_LF)
        ax.set_title("LDP + Fairlearn: {} — {}".format(orig, title_s),
                     fontsize=_TF, fontweight="bold")
        for i in range(len(row_labels)):
            for j in range(len(eps_vals)):
                v = mat[i,j]
                if not np.isnan(v):
                    ax.text(j, i, "{:.3f}".format(v), ha="center", va="center",
                            fontsize=_AN+1,
                            color="black" if abs(v)<0.25 else "white")
        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir,"F3b_ldpfair_{}_heatmap.png".format(orig.lower())))


def _viz3_ldp_fair_fairness_bars(main_fair, ldp_fair_f, out_dir):
    """F3c: LDP + Fairlearn fairness — one separate horizontal-bar figure per metric.

    Each figure shows all LDP+Fair model combinations (epsilon × constraint × family)
    sorted by epsilon then constraint.  Reference lines mark baseline and augmented LR
    values so the fairness gain/loss is immediately visible.
    Files: F3c_ldpfair_<metric>.png
    """
    if ldp_fair_f.empty: return

    metrics = [
        ("group_SPD"    if "group_SPD"    in ldp_fair_f.columns else "SPD",
         "SPD",    "Statistical Parity Difference  (0 = perfectly fair)",   0),
        ("group_EOD"    if "group_EOD"    in ldp_fair_f.columns else "EOD",
         "EOD",    "Equal Opportunity Difference    (0 = perfectly fair)",   0),
        ("group_EqOdds" if "group_EqOdds" in ldp_fair_f.columns else "EqOdds",
         "EqOdds", "Equalized Odds                  (0 = perfectly fair)",   0),
        ("ind_cf_fairness" if "ind_cf_fairness" in ldp_fair_f.columns else "cf_fairness",
         "CF-Fair","CF Fairness                     (higher = fairer)",     None),
    ]
    metrics = [(c, s, t, i) for c, s, t, i in metrics if c in ldp_fair_f.columns]
    if not metrics: return

    # Reference values from main pipeline — use ALL families, not just LR
    _REF_SCENS_F3C = ["baseline", "augmented_SCM", "augmented_update_labels",
                      "augmented_add_comparators", "augmented"]
    ref = {}   # (scen, col, fam) -> value
    if not main_fair.empty:
        ref_scens = [sc for sc in _REF_SCENS_F3C
                     if sc in main_fair["scenario"].unique()]
        for scen in ref_scens:
            for fam in _FAM_ORDER:
                row = main_fair[(main_fair["scenario"] == scen) &
                                (main_fair["family"]   == fam)]
                if row.empty: continue
                for col, *_ in metrics:
                    if col in row.columns:
                        ref[(scen, col, fam)] = float(row[col].iloc[0])

    # Sort rows: by epsilon asc, then constraint, then family
    df = ldp_fair_f.copy()
    if "epsilon" in df.columns:
        df["epsilon"] = df["epsilon"].astype(float)
        df = df.sort_values(["epsilon", "constraint", "family"]
                             if "constraint" in df.columns
                             else ["epsilon", "family"]).reset_index(drop=True)

    def _row_label(row):
        eps = row.get("epsilon", "?")
        c   = row.get("constraint", "")
        f   = _fd(row.get("family", ""))
        return "ε={} {} [{}]".format(eps, c, f)

    labels = [_row_label(r) for _, r in df.iterrows()]
    colors = [_scenario_color("ldp_fair_eps{}".format(
                  r.get("epsilon", 1.0))) for _, r in df.iterrows()]

    n_bars = len(labels)
    height = max(5, n_bars * 0.62 + 2.5)

    for col, short, full_title, ideal in metrics:
        vals = [float(r[col]) if not pd.isna(r[col]) else float("nan")
                for _, r in df.iterrows()]

        fig, ax = plt.subplots(figsize=(13, height))

        y    = np.arange(n_bars)
        bars = ax.barh(y, vals, color=colors, alpha=0.88,
                       edgecolor="white", linewidth=0.4, height=0.65)

        # Value annotations
        _hbar_labels(ax, bars, fmt="{:.4f}", fs=9)

        # Ideal (0 or None)
        if ideal is not None:
            ax.axvline(ideal, color="black", lw=1.3, ls=":", alpha=0.6,
                       label="Ideal ({})".format(ideal))

        # Reference lines — one per family and scenario combination
        ref_styles = {
            "baseline":                  ("red",     "--", 1.8),
            "augmented":                 ("#2ca02c", "-.", 1.8),   # legacy
            "augmented_SCM":             ("#2ca02c", "-.", 1.8),
            "augmented_update_labels":   ("#17becf", "-.", 1.8),
            "augmented_add_comparators": ("#9467bd", "-.", 1.8),
        }
        for (scen, rcol, fam), rval in ref.items():
            if rcol != col: continue
            color, ls, lw = ref_styles.get(scen, ("#888", ":", 1.2))
            ax.axvline(rval, color=color, ls=ls, lw=lw, alpha=0.75,
                       label="{} {} ({:.3f})".format(
                           scen.capitalize(), _fd(fam), rval))

        # Epsilon group separators
        if "epsilon" in df.columns:
            prev_eps = None
            for i, (_, row) in enumerate(df.iterrows()):
                eps = row.get("epsilon")
                if prev_eps is not None and eps != prev_eps:
                    ax.axhline(i - 0.5, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
                    ax.text(-0.005, i - 0.5,
                            "ε = {}".format(eps),
                            ha="right", va="center",
                            fontsize=_TK - 1, color="#555", style="italic",
                            transform=ax.get_yaxis_transform())
                prev_eps = eps

        # Axes
        finite = [v for v in vals if np.isfinite(v)]
        if finite:
            span = max(max(finite) - min(finite), 1e-9)
            lo   = min(min(finite), 0) - span * 0.02
            hi   = max(max(finite), 0) + span * 0.22   # right padding for labels
            ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: "{:.3f}".format(x)))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 1)
        ax.invert_yaxis()
        ax.set_xlabel(short, fontsize=_LF)
        ax.set_title("LDP + Fairlearn: {}\n{}".format(short, full_title),
                     fontsize=_TF, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.2, axis="x")

        # Legend — deduplicate reference lines
        handles, seen = [], set()
        if ideal is not None:
            handles.append(Line2D([0], [0], color="black", ls=":", lw=1.3,
                                  label="Ideal ({})".format(ideal)))
        for scen, (color, ls, lw) in ref_styles.items():
            label = "{} (reference)".format(scen.capitalize())
            if label not in seen:
                handles.append(Line2D([0], [0], color=color, ls=ls, lw=lw,
                                      label=label))
                seen.add(label)
        if handles:
            ax.legend(handles=handles, fontsize=_LG - 1, framealpha=0.9,
                      bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        slug = short.lower().replace("-", "").replace(" ", "_")
        _save_fig(fig, os.path.join(out_dir,
                                    "F3c_ldpfair_{}.png".format(slug)))


def _viz3_all_scenarios(main_fair, ldp, fl_fair, ldp_fair_f, out_dir):
    """F3d: SPD across all approaches — best model per approach.
    Best epsilon value is annotated ON each LDP/LDP+Fair bar."""
    _AUG_F3D = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    _SCEN_F3D = ["baseline"] + _AUG_F3D
    rows = []
    if not main_fair.empty and "group_SPD" in main_fair.columns:
        scens_f3d = [sc for sc in _SCEN_F3D if sc in main_fair["scenario"].unique()]
        for _, r in main_fair[main_fair["scenario"].isin(scens_f3d)].iterrows():
            rows.append({"label":"{}\n{}".format(_fmt_ldp_label(r["scenario"]),_fd(r["family"])),
                         "SPD":r["group_SPD"],"group":_scen_group(r["scenario"]),
                         "family":r["family"],"best_eps":None})

    if not ldp.empty and "group_SPD" in ldp.columns:
        ldp2 = ldp.copy(); ldp2["epsilon"] = ldp2["epsilon"].astype(float)
        for fam in _FAM_ORDER:
            sub = ldp2[ldp2["family"]==fam]
            if sub.empty: continue
            bi = sub["group_SPD"].abs().idxmin()
            br = sub.loc[bi]
            rows.append({"label":"LDP best\n{}".format(_fd(fam)),
                         "SPD":br["group_SPD"],"group":"ldp",
                         "family":fam,"best_eps":float(br["epsilon"])})

    fl_n = _fl_norm(fl_fair) if not fl_fair.empty else fl_fair
    if not fl_n.empty and "group_SPD" in fl_n.columns:
        fl_nb = fl_n[~fl_n["scenario"].str.startswith("baseline")]
        if "family" in fl_nb.columns:
            for fam in _FAM_ORDER:
                sub = fl_nb[fl_nb["family"]==fam]
                if sub.empty: continue
                bi = sub["group_SPD"].abs().idxmin()
                br = sub.loc[bi]
                rows.append({"label":"FL best\n{}".format(_fd(fam)),
                             "SPD":br["group_SPD"],"group":"fairlearn",
                             "family":fam,"best_eps":None})

    spd_col = "group_SPD" if "group_SPD" in ldp_fair_f.columns \
              else "SPD" if "SPD" in ldp_fair_f.columns else None
    if not ldp_fair_f.empty and spd_col:
        df2 = ldp_fair_f.copy()
        if "epsilon" in df2.columns: df2["epsilon"] = df2["epsilon"].astype(float)
        for fam in _FAM_ORDER:
            if "family" not in df2.columns or fam not in df2["family"].unique(): continue
            sub = df2[df2["family"]==fam]
            if sub.empty: continue
            bi = sub[spd_col].abs().idxmin()
            br = sub.loc[bi]
            best_eps = float(br["epsilon"]) if "epsilon" in br.index else None
            rows.append({"label":"LDP+FL best\n{}".format(_fd(fam)),
                         "SPD":br[spd_col],"group":"ldp_fair",
                         "family":fam,"best_eps":best_eps})

    if not rows: return
    df = pd.DataFrame(rows)

    _AUG_GROUPS_F3D = {"baseline", "augmented", "augmented_SCM",
                        "augmented_update_labels", "augmented_add_comparators"}
    group_color = {"baseline":"#aaaaaa","augmented":_C_XGB,
                   "augmented_SCM":_C_XGB,"augmented_update_labels":"#17becf",
                   "augmented_add_comparators":"#9467bd",
                   "ldp":_C_RF,"fairlearn":_C_LR,"ldp_fair":_C_LDPF}
    colors  = [_scenario_color(r["group"]) if r["group"] in _AUG_GROUPS_F3D
               else group_color.get(r["group"],"#888") for _, r in df.iterrows()]
    hatches = [_HATCH_AUG if r["group"] in _AUG_GROUPS_F3D - {"baseline"}
               else _HATCH_BASE for _, r in df.iterrows()]

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(max(18, len(df)*1.1+2), 9))

    for i, (v, c, h) in enumerate(zip(df["SPD"], colors, hatches)):
        bar = ax.bar([x[i]], [v], color=c, alpha=0.85, hatch=h,
                     edgecolor="black" if h else "white",
                     linewidth=0.5, width=0.78)
        _bar_labels(ax, bar, fmt="{:.3f}", fs=9)

    # Best epsilon annotation — placed inside the bar for LDP/LDP+Fair groups
    for i, (_, row) in enumerate(df.iterrows()):
        eps_v = row["best_eps"]
        if eps_v is not None and not pd.isna(eps_v):
            y_pos = 0.012 if df["SPD"].iloc[i] >= 0 else -0.012
            va    = "bottom" if df["SPD"].iloc[i] >= 0 else "top"
            ax.text(x[i], y_pos,
                    "Best\neps={}".format(eps_v),
                    ha="center", va=va, fontsize=8.5, fontweight="bold",
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.3", fc=colors[i],
                              ec="black", linewidth=0.9, alpha=0.95))

    ax.axhline(0, color="black", lw=1.5, ls="--", alpha=0.7)
    ax.set_title(
        "SPD Comparison: Best Model per Approach and Family\n"
        "Colored badges show the best epsilon found for LDP-based methods",
        fontsize=_TF+1, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"].tolist(), rotation=40, ha="right", fontsize=_TK)
    ax.set_ylabel("Statistical Parity Difference (SPD)", fontsize=_LF)

    els = [mpatches.Patch(facecolor="#aaa",  edgecolor="black", label="Baseline"),
           mpatches.Patch(facecolor=_C_XGB,  hatch=_HATCH_AUG, edgecolor="black",
                          label="MM-CF Augmented"),
           mpatches.Patch(facecolor=_C_RF,   label="LDP only (best eps)"),
           mpatches.Patch(facecolor=_C_LR,   label="Fairlearn (best constraint)"),
           mpatches.Patch(facecolor=_C_LDPF, label="LDP+Fairlearn (best eps)")]
    ax.legend(handles=els, fontsize=_LG, framealpha=0.9,
              bbox_to_anchor=(1.01, 1.0), loc="upper left")
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F3d_all_scenarios_spd.png"))


def _viz3_mia_overview(main_mia, fl_mia, ldp_fair_mia, out_dir):
    """F3e: MIA AUC overview — 3 separate figures, one per model family.
    Color = scenario type (baseline=gray, augmented=green, LDP=red gradient,
    fairlearn=purple, LDP+FL=dark blue).
    """
    # Collect rows per family
    all_rows = []

    def _add_mia(df, sc_col, fam_col, itype=None, group="main"):
        if df.empty or "auc_roc" not in df.columns: return
        sub = df
        if itype and "input_type" in df.columns:
            s2 = df[df["input_type"]==itype]; sub = s2 if not s2.empty else df
        grp = [c for c in [sc_col, fam_col] if c in sub.columns]
        if not grp: return
        for _, r in sub.groupby(grp)["auc_roc"].max().reset_index().iterrows():
            sc  = str(r.get(sc_col,""))
            fam = str(r.get(fam_col,""))
            all_rows.append({"scenario":sc, "family":fam,
                             "auc_roc": float(r["auc_roc"]), "group":group})

    # Use itype=None to catch LDP scenarios that only have mm_cf data
    _add_mia(main_mia, "scenario","family", itype=None, group="main")
    fl2 = _normalize_family(fl_mia.copy()) if not fl_mia.empty else fl_mia
    if not fl2.empty:
        if "constraint" in fl2.columns: fl2 = fl2[fl2["constraint"]!="baseline"]
        _add_mia(fl2, "model","family", group="fairlearn")
    _add_mia(ldp_fair_mia, "model","family", group="ldp_fair")

    if not all_rows: return
    df_all = pd.DataFrame(all_rows)

    def _scen_color(sc, group):
        if group == "fairlearn":  return _C_LDPF
        if group == "ldp_fair":   return "#2255aa"
        return _SCEN_COLORS.get(sc, "#888888")

    def _scen_label(sc, group):
        if group == "fairlearn":  return "FL:{}".format(sc[:12])
        if group == "ldp_fair":   return "LDP+FL\n{}".format(sc[:10])
        return _fmt_ldp_label(sc)

    for fam in _FAM_ORDER:
        sub = df_all[df_all["family"]==fam]
        # Normalize family for fairlearn rows (they might use short tags)
        sub_extra = df_all[df_all["family"]==_FAM_DISP.get(fam,fam)]
        sub = pd.concat([sub, sub_extra]).drop_duplicates()
        if sub.empty: continue

        labels = [_scen_label(r["scenario"], r["group"])
                  for _, r in sub.iterrows()]
        vals   = sub["auc_roc"].tolist()
        colors = [_scen_color(r["scenario"], r["group"])
                  for _, r in sub.iterrows()]
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(max(12, len(labels)*0.85+3), 7))
        bars = ax.bar(x, vals, color=colors, alpha=0.88, edgecolor="white", width=0.72)
        _bar_labels(ax, bars, fmt="{:.4f}", fs=8)
        ax.axhline(0.5, color="red", ls="--", lw=1.8, label="Random (0.50)")
        ax.set_title("MIA AUC-ROC: All Scenarios — {} base\n"
                     "(NiCE-CF proxy; best attacker per model)".format(_fd(fam)),
                     fontsize=_TF+1, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=_TK)
        ax.set_ylabel("MIA AUC-ROC", fontsize=_LF)
        vv = [v for v in vals if not np.isnan(v)]
        if vv: ax.set_ylim(max(0.44,min(vv)-0.01), min(0.65,max(vv)+0.03))

        # Scenario color legend
        legend_els = [
            mpatches.Patch(facecolor=_SCEN_COLORS["baseline"],  label="Baseline"),
            mpatches.Patch(facecolor=_SCEN_COLORS["augmented"], label="MM-CF Augmented"),
        ]
        for eps_key in ["ldp_eps0.1","ldp_eps0.5","ldp_eps1.0","ldp_eps2.0",
                         "ldp_eps5.0","ldp_eps10.0"]:
            eps_val = eps_key.replace("ldp_eps","")
            legend_els.append(mpatches.Patch(facecolor=_SCEN_COLORS[eps_key],
                                              label="LDP eps={}".format(eps_val)))
        legend_els += [
            mpatches.Patch(facecolor=_C_LDPF, label="Fairlearn"),
            mpatches.Patch(facecolor="#2255aa", label="LDP+Fairlearn"),
            Line2D([0],[0], color="red", ls="--", lw=1.8, label="Random (0.50)"),
        ]
        ax.legend(handles=legend_els, fontsize=_LG-1,
                  bbox_to_anchor=(1.01,1), loc="upper left", framealpha=0.9)
        plt.tight_layout()
        tag = {"logistic_regression":"lr","random_forest":"rf","xgboost":"xgb"}.get(fam,fam)
        _save_fig(fig, os.path.join(out_dir, "F3e_{}.png".format(tag)))


def _viz3_spd_mia_sidebyside(main_fair, main_mia, ldp, fl_fair, fl_mia,
                               ldp_fair_f, ldp_fair_mia, out_dir):
    """F3f: |SPD| + MIA AUC side-by-side horizontal bars — one figure per family.

    Each figure has two columns (subplots): left = |SPD|, right = MIA AUC-ROC.
    Rows = scenarios sorted by group then epsilon.  Missing MIA data shows an
    explicit N/A stub so no row silently disappears.
    Files: F3f_<family>.png
    """

    # ── Build unified MIA lookup: (scenario, family, input_type) → best AUC ──
    mia_lkp = {}
    for src_mia, src_name in [(main_mia, "main"), (fl_mia, "fl"), (ldp_fair_mia, "lf")]:
        if src_mia.empty or "auc_roc" not in src_mia.columns:
            continue
        src_n = _normalize_family(src_mia.copy()) if src_name != "main" else src_mia
        # Key: use "scenario" column if present, else "model" column
        sc_col = "scenario" if "scenario" in src_n.columns else \
                 "model"    if "model"    in src_n.columns else None
        if sc_col is None:
            continue
        it_col = "input_type" if "input_type" in src_n.columns else None
        grp = [c for c in [sc_col, "family", it_col] if c]
        for _, r in src_n.groupby(grp)["auc_roc"].max().reset_index().iterrows():
            sc  = str(r[sc_col])
            fam = str(r.get("family", ""))
            itp = str(r.get("input_type", "any")) if it_col else "any"
            # prefer nice_cf, fall back to mm_cf / any
            key = (sc, fam, itp)
            if key not in mia_lkp or itp == "nice_cf":
                mia_lkp[key] = float(r["auc_roc"])

    def _best_mia(sc, fam):
        for itp in ("nice_cf", "mm_cf", "any"):
            v = mia_lkp.get((sc, fam, itp))
            if v is not None:
                return v
        return float("nan")

    # ── Build records ──────────────────────────────────────────────────────────
    records = []

    # Baseline + Augmented variants
    _AUG_F3F = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    _SCEN_F3F = ["baseline"] + _AUG_F3F
    if not main_fair.empty and "group_SPD" in main_fair.columns:
        scens_f3f = [sc for sc in _SCEN_F3F if sc in main_fair["scenario"].unique()]
        for _, r in main_fair[main_fair["scenario"].isin(scens_f3f)].iterrows():
            sc, fam = str(r["scenario"]), str(r["family"])
            records.append({
                "scenario": sc, "family": fam,
                "label":    "{} [{}]".format(_fmt_ldp_label(sc), _fd(fam)),
                "SPD":      abs(float(r["group_SPD"])),
                "mia_auc":  _best_mia(sc, fam),
            })

    # LDP sweep  (ldp_sweep_results.csv has the scenario column with full names)
    if not ldp.empty and "group_SPD" in ldp.columns:
        ldp2 = ldp.copy()
        if "epsilon" in ldp2.columns:
            ldp2["epsilon"] = ldp2["epsilon"].astype(float)
        for _, r in ldp2.iterrows():
            sc  = str(r.get("scenario", ""))
            fam = str(r.get("family", ""))
            eps = r.get("epsilon", "?")
            sc_type = r.get("sc_type", "")
            label = "{}  ε={}  [{}]".format(
                "LDP-Only" if "baseline" in str(sc_type) else "LDP+Aug",
                eps, _fd(fam))
            records.append({
                "scenario": sc, "family": fam,
                "label":    label,
                "SPD":      abs(float(r["group_SPD"])),
                "mia_auc":  _best_mia(sc, fam),
            })

    # Fairlearn
    fl_n = _fl_norm(fl_fair) if not fl_fair.empty else fl_fair
    if not fl_n.empty and "group_SPD" in fl_n.columns:
        for _, r in fl_n[~fl_n["scenario"].str.startswith("baseline")].iterrows():
            sc, fam = str(r["scenario"]), str(r["family"])
            lbl = str(r.get("display", r.get("label", sc)))
            records.append({
                "scenario": sc, "family": fam,
                "label":    "{} [{}]".format(_wrap(lbl, 22), _fd(fam)),
                "SPD":      abs(float(r["group_SPD"])),
                "mia_auc":  _best_mia(sc, fam),
            })

    # LDP + Fairlearn
    spd_col = "group_SPD" if "group_SPD" in ldp_fair_f.columns \
              else "SPD"   if "SPD"       in ldp_fair_f.columns else None
    if not ldp_fair_f.empty and spd_col:
        for _, r in ldp_fair_f.iterrows():
            sc  = str(r.get("model", "?"))
            fam = str(r.get("family", ""))
            eps = r.get("epsilon", "?")
            c   = r.get("constraint", "")
            records.append({
                "scenario": sc, "family": fam,
                "label":    "LDP+FL ε={} {} [{}]".format(eps, c, _fd(fam)),
                "SPD":      abs(float(r[spd_col])),
                "mia_auc":  _best_mia(sc, fam),
            })

    if not records:
        return
    df_all = pd.DataFrame(records)

    _NA_COLOR = "#dddddd"
    _NA_HATCH = "////"
    _NA_MIA_STUB = 0.501   # just above the 0.50 random line — visible but clearly wrong

    # ── One figure per family ─────────────────────────────────────────────────
    for fam in _FAM_ORDER:
        sub = df_all[df_all["family"] == fam].copy()
        if sub.empty:
            continue

        # Sort by scenario group then epsilon
        sub["_gi"]  = sub["scenario"].map(
            lambda s: _GRP_ORDER.index(_scen_group(s)) if _scen_group(s) in _GRP_ORDER else 99)
        sub["_eps"] = sub["scenario"].map(lambda s: _scen_eps(s) or 0.0)
        sub = sub.sort_values(["_gi", "_eps", "scenario"]).reset_index(drop=True)

        labels   = sub["label"].tolist()
        spd_vals = sub["SPD"].tolist()
        mia_vals = sub["mia_auc"].tolist()
        colors   = [_scenario_color(str(s)) for s in sub["scenario"]]
        n        = len(labels)
        height   = max(6, n * 0.65 + 2.5)

        fig, (ax_spd, ax_mia) = plt.subplots(
            1, 2, figsize=(20, height), sharey=True)
        fig.suptitle(
            "Privacy–Fairness Trade-off — {} base estimator  (COMPAS)\n"
            "Left: |SPD| (lower = fairer)   Right: MIA AUC-ROC (lower = more private)"
            .format(_fd(fam)),
            fontsize=_TF + 1, fontweight="bold")

        y = np.arange(n)

        # ── Left panel: |SPD| ──────────────────────────────────────────────
        ax_spd.barh(y, spd_vals, color=colors, alpha=0.88,
                    edgecolor="white", linewidth=0.4, height=0.68)
        _hbar_labels(ax_spd, ax_spd.patches, fmt="{:.4f}", fs=9)
        ax_spd.axvline(0, color="black", lw=1.2, ls="--", alpha=0.5,
                       label="Ideal (0)")
        # group separators
        prev_grp = None
        for i, sc in enumerate(sub["scenario"]):
            grp = _scen_group(str(sc))
            if prev_grp is not None and grp != prev_grp:
                ax_spd.axhline(i - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
            prev_grp = grp
        spd_finite = [v for v in spd_vals if np.isfinite(v)]
        if spd_finite:
            span = max(max(spd_finite), 1e-9)
            ax_spd.set_xlim(0, span * 1.25)
        ax_spd.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: "{:.3f}".format(x)))
        ax_spd.set_xlabel("|SPD|  (lower = fairer)", fontsize=_LF)
        ax_spd.set_title("|SPD| — Statistical Parity Difference",
                         fontsize=_LF, fontweight="bold")
        ax_spd.set_yticks(y)
        ax_spd.set_yticklabels(labels, fontsize=_TK + 1)
        ax_spd.invert_yaxis()
        ax_spd.grid(True, alpha=0.2, axis="x")

        # ── Right panel: MIA AUC ───────────────────────────────────────────
        for i, (mia_v, c) in enumerate(zip(mia_vals, colors)):
            if np.isfinite(mia_v):
                ax_mia.barh(y[i], mia_v, color=c, alpha=0.88,
                            edgecolor="white", linewidth=0.4, height=0.68)
                ax_mia.text(mia_v + 0.002, y[i], "{:.4f}".format(mia_v),
                            va="center", ha="left", fontsize=9, color="#333")
            else:
                ax_mia.barh(y[i], _NA_MIA_STUB, color=_NA_COLOR, alpha=0.7,
                            edgecolor="#aaa", linewidth=0.6,
                            hatch=_NA_HATCH, height=0.68)
                ax_mia.text(_NA_MIA_STUB + 0.002, y[i], "N/A",
                            va="center", ha="left", fontsize=9,
                            color="#888", style="italic")
        ax_mia.axvline(0.5, color="red", ls="--", lw=1.8,
                       label="Random (0.50)")
        # group separators (same positions)
        prev_grp = None
        for i, sc in enumerate(sub["scenario"]):
            grp = _scen_group(str(sc))
            if prev_grp is not None and grp != prev_grp:
                ax_mia.axhline(i - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
            prev_grp = grp
        mia_finite = [v for v in mia_vals if np.isfinite(v)]
        lo_m = max(0.44, min(mia_finite) - 0.01) if mia_finite else 0.44
        hi_m = min(0.65, max(mia_finite) + 0.06) if mia_finite else 0.65
        ax_mia.set_xlim(lo_m, hi_m)
        ax_mia.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax_mia.set_xlabel("MIA AUC-ROC  (lower = more private)", fontsize=_LF)
        ax_mia.set_title("MIA AUC-ROC  (striped = not available)",
                         fontsize=_LF, fontweight="bold")
        ax_mia.grid(True, alpha=0.2, axis="x")

        # ── Shared legend ──────────────────────────────────────────────────
        present_grps = list(dict.fromkeys(
            _scen_group(s) for s in sub["scenario"].tolist()))
        leg_els = [
            mpatches.Patch(
                facecolor=_GRP_PALETTES[g][min(1, len(_GRP_PALETTES[g]) - 1)],
                edgecolor="#555",
                label=_GRP_LABELS.get(g, g))
            for g in _GRP_ORDER if g in present_grps
        ]
        leg_els += [
            Line2D([0], [0], color="red",   ls="--", lw=1.8, label="Random MIA (0.50)"),
            Line2D([0], [0], color="black", ls="--", lw=1.2, label="Ideal SPD (0)"),
            mpatches.Patch(facecolor=_NA_COLOR, hatch=_NA_HATCH,
                           edgecolor="#aaa", label="N/A — MIA not available"),
        ]
        fig.legend(handles=leg_els, loc="lower center",
                   ncol=min(len(leg_els), 5), fontsize=_LG,
                   bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

        plt.tight_layout()
        tag = {"logistic_regression": "lr",
               "random_forest":       "rf",
               "xgboost":             "xgb"}.get(fam, fam)
        _save_fig(fig, os.path.join(out_dir, "F3f_{}.png".format(tag)))


def _viz3_ldp_fair_all_metrics(main_fair, ldp_fair_f, ldp_fair_mia, out_dir):
    """F3g: LDP + Fairlearn — one horizontal-bar figure per fairness metric.
    MIA AUC-ROC is shown as a secondary annotation '[MIA=X.XXX]' on each bar.
    Files: F3g_ldpfair_<metric>.png
    """
    if ldp_fair_f.empty: return

    metrics = []
    for c, short, full in [
        ("SPD",         "SPD",      "Statistical Parity Difference  (0 = perfectly fair)"),
        ("EOD",         "EOD",      "Equal Opportunity Difference   (0 = perfectly fair)"),
        ("EqOdds",      "EqOdds",   "Equalized Odds                 (0 = perfectly fair)"),
        ("cf_fairness", "CF-Fair",  "CF Fairness                    (higher = fairer)"),
    ]:
        src = "group_"+c if "group_"+c in ldp_fair_f.columns \
              else c      if c          in ldp_fair_f.columns else None
        if src: metrics.append((src, short, full))
    if not metrics: return

    # MIA lookup: best AUC per (model, family)
    mia_lkp = {}
    if not ldp_fair_mia.empty and "auc_roc" in ldp_fair_mia.columns:
        grp_cols = [c for c in ["model", "family"] if c in ldp_fair_mia.columns]
        if grp_cols:
            for _, r in ldp_fair_mia.groupby(grp_cols)["auc_roc"].max()\
                                     .reset_index().iterrows():
                key = tuple(str(r[c]) for c in grp_cols)
                mia_lkp[key] = float(r["auc_roc"])

    # Reference values from main pipeline — all families
    _REF_SCENS_F3G = ["baseline", "augmented_SCM", "augmented_update_labels",
                      "augmented_add_comparators", "augmented"]
    ref = {}  # (scen, col, fam) -> value
    if not main_fair.empty:
        ref_scens_g = [sc for sc in _REF_SCENS_F3G
                       if sc in main_fair["scenario"].unique()]
        for scen in ref_scens_g:
            for fam in _FAM_ORDER:
                row = main_fair[(main_fair["scenario"] == scen) &
                                (main_fair["family"]   == fam)]
                if row.empty: continue
                for src, *_ in metrics:
                    if src in row.columns:
                        ref[(scen, src, fam)] = float(row[src].iloc[0])

    # Sort: epsilon asc, then constraint, then family
    df = ldp_fair_f.copy()
    if "epsilon" in df.columns:
        df["epsilon"] = df["epsilon"].astype(float)
        sort_cols = [c for c in ["epsilon", "constraint", "family"] if c in df.columns]
        df = df.sort_values(sort_cols).reset_index(drop=True)

    def _row_label(row):
        eps = row.get("epsilon", "?")
        c   = row.get("constraint", "")
        f   = _fd(row.get("family", ""))
        return "ε={} {} [{}]".format(eps, c, f)

    labels = [_row_label(r) for _, r in df.iterrows()]
    models = df.get("model", pd.Series(["?"] * len(df))).tolist()
    fams   = df.get("family", pd.Series([""]  * len(df))).tolist()
    colors = [_scenario_color("ldp_fair_eps{}".format(
                  r.get("epsilon", 1.0))) for _, r in df.iterrows()]

    n_bars = len(labels)
    height = max(5, n_bars * 0.65 + 2.5)

    ref_styles = {
        "baseline":                  ("red",     "--", 1.8),
        "augmented":                 ("#2ca02c", "-.", 1.8),   # legacy
        "augmented_SCM":             ("#2ca02c", "-.", 1.8),
        "augmented_update_labels":   ("#17becf", "-.", 1.8),
        "augmented_add_comparators": ("#9467bd", "-.", 1.8),
    }

    for src, short, full_title in metrics:
        vals = [float(df[src].iloc[i]) if not pd.isna(df[src].iloc[i])
                else float("nan") for i in range(len(df))]

        # Collect MIA values for annotation
        mia_vals = []
        for i, (m, fam) in enumerate(zip(models, fams)):
            key1 = (str(m), str(fam))
            key2 = (str(m),)
            mia_vals.append(mia_lkp.get(key1, mia_lkp.get(key2)))

        fig, ax = plt.subplots(figsize=(14, height))
        y    = np.arange(n_bars)
        bars = ax.barh(y, vals, color=colors, alpha=0.88,
                       edgecolor="white", linewidth=0.4, height=0.65)

        # Fairness value annotations
        _hbar_labels(ax, bars, fmt="{:.4f}", fs=9)

        # MIA AUC as secondary annotation to the right of the fairness value
        finite = [v for v in vals if np.isfinite(v)]
        right_bound = (max(finite) if finite else 0) * 1.22
        for i, mia_v in enumerate(mia_vals):
            if mia_v is not None:
                ax.text(right_bound, i,
                        "MIA={:.3f}".format(mia_v),
                        ha="left", va="center", fontsize=8,
                        color="#1a5fa8", fontweight="bold")

        # Ideal reference
        ax.axvline(0, color="black", lw=1.2, ls=":", alpha=0.5,
                   label="Ideal (0)")

        # Reference lines — baseline and augmented per family
        for (scen, rcol, fam), rval in ref.items():
            if rcol != src: continue
            color, ls, lw = ref_styles.get(scen, ("#888", ":", 1.2))
            ax.axvline(rval, color=color, ls=ls, lw=lw, alpha=0.7,
                       label="{} {} ({:.3f})".format(
                           scen.capitalize(), _fd(fam), rval))

        # Epsilon group separators
        if "epsilon" in df.columns:
            prev_eps = None
            for i, (_, row) in enumerate(df.iterrows()):
                eps = row.get("epsilon")
                if prev_eps is not None and eps != prev_eps:
                    ax.axhline(i - 0.5, color="#bbbbbb", lw=1.0, ls="--", zorder=0)
                    ax.text(-0.003, i - 0.5,
                            "ε = {}".format(eps),
                            ha="right", va="center",
                            fontsize=_TK - 1, color="#555", style="italic",
                            transform=ax.get_yaxis_transform())
                prev_eps = eps

        # X-axis limits with right padding for both annotations
        if finite:
            span = max(max(finite) - min(finite), 1e-9)
            lo   = min(min(finite), 0) - span * 0.02
            hi   = max(max(finite), 0) + span * 0.45   # extra room for MIA labels
            ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: "{:.3f}".format(x)))
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=_TK + 1)
        ax.invert_yaxis()
        ax.set_xlabel(short, fontsize=_LF)
        ax.set_title("LDP + Fairlearn: {}\n{}\n"
                     "Blue text = MIA AUC-ROC (lower = more private)".format(
                         short, full_title),
                     fontsize=_TF, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.2, axis="x")

        # Legend — deduplicate reference lines
        handles, seen = [], set()
        handles.append(Line2D([0], [0], color="black", ls=":", lw=1.2,
                               label="Ideal (0)"))
        for scen, (color, ls, lw) in ref_styles.items():
            lbl = "{} (reference)".format(scen.capitalize())
            if lbl not in seen:
                handles.append(Line2D([0], [0], color=color, ls=ls, lw=lw, label=lbl))
                seen.add(lbl)
        if any(v is not None for v in mia_vals):
            handles.append(Line2D([0], [0], color="#1a5fa8", lw=0,
                                   marker="$M$", markersize=9,
                                   label="MIA AUC annotation"))
        ax.legend(handles=handles, fontsize=_LG - 1, framealpha=0.9,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        slug = short.lower().replace("-", "").replace(" ", "_")
        _save_fig(fig, os.path.join(out_dir,
                                    "F3g_ldpfair_{}.png".format(slug)))


def generate_folder3(out_dir=_DIR3):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 3: LDP and Fairness\n" + "="*60)
    _u_fair3 = os.path.join(_UNIFIED_RES, "unified_fairness.csv")
    _u_mia3  = os.path.join(_UNIFIED_RES, "unified_mia.csv")
    _u_ldp3  = os.path.join(_UNIFIED_RES, "unified_ldp_sweep.csv")
    main_fair    = (_load(_u_fair3, "unified_fairness_f3") if os.path.exists(_u_fair3)
                    else _load(os.path.join(_MAIN_RES, "fairness_results.csv"), "main fairness"))
    main_mia     = (_load(_u_mia3, "unified_mia_f3")       if os.path.exists(_u_mia3)
                    else _load(os.path.join(_MAIN_RES, "mia_results.csv"),      "main mia"))
    ldp_sweep    = (_load(_u_ldp3, "unified_ldp_sweep_f3") if os.path.exists(_u_ldp3)
                    else _load(os.path.join(_MAIN_RES, "ldp_sweep_results.csv"), "ldp sweep"))
    fl_fair      = _load_fl_fair()
    fl_mia       = _load(os.path.join(_FL_RES, "fairlearn_mia.csv"),           "fl mia")
    ldp_fair_f   = _load(os.path.join(_LF_RES, "ldp_fair_fairness.csv"),       "ldp_fair fairness")
    ldp_fair_mia = _load(os.path.join(_LF_RES, "ldp_fair_mia.csv"),            "ldp_fair mia")

    # Normalise
    if not fl_mia.empty:
        fl_mia = _normalize_family(fl_mia.copy())
        if "input_type" not in fl_mia.columns: fl_mia["input_type"] = "nice_cf"
    if not ldp_fair_f.empty:
        for c in ["SPD","DI","EOD","AOD","EqOdds","PP","Theil"]:
            if c in ldp_fair_f.columns and "group_"+c not in ldp_fair_f.columns:
                ldp_fair_f["group_"+c] = ldp_fair_f[c]

    _viz3_ldp_sweep(ldp_sweep, out_dir)
    _viz3_ldp_fair_heatmap(ldp_fair_f, out_dir)
    _viz3_ldp_fair_fairness_bars(main_fair, ldp_fair_f, out_dir)
    _viz3_all_scenarios(main_fair, ldp_sweep, fl_fair, ldp_fair_f, out_dir)
    _viz3_mia_overview(main_mia, fl_mia, ldp_fair_mia, out_dir)
    _viz3_spd_mia_sidebyside(main_fair, main_mia, ldp_sweep, fl_fair, fl_mia,
                              ldp_fair_f, ldp_fair_mia, out_dir)
    _viz3_ldp_fair_all_metrics(main_fair, ldp_fair_f, ldp_fair_mia, out_dir)

    # F3h: LDP sweep MIA — 4 separate horizontal figures
    if not main_mia.empty:
        ldp_sub = main_mia[main_mia["scenario"].str.startswith("ldp")].copy()
        if not ldp_sub.empty:
            if "input_type" in ldp_sub.columns:
                mm = ldp_sub[ldp_sub["input_type"]=="mm_cf"]
                ldp_sub = mm if not mm.empty else ldp_sub
            # Clean labels: ldp_eps0.5 -> LDP eps=0.5
            ldp_sub["_lbl"] = ldp_sub["scenario"].str.replace("ldp_eps","LDP eps=",regex=False)
            _viz_mia_4sep_vertical(
                df=ldp_sub, label_col="_lbl", family_col="family",
                color_fn=lambda k: _fc(k[1] if len(k)>1 else ""),
                title_prefix="LDP Sweep MIA",
                out_prefix=os.path.join(out_dir, "F3h"),
                input_type_filter=None,
            )

    print("  [Folder 3] Done ->", out_dir)


# ===========================================================================
# FOLDER 4 — MIA Methods Comparison
# ===========================================================================

def _viz4_attacker_heatmap(mia, out_dir):
    """F4a: Heatmap of attacker AUC-ROC × scenario, one panel per family.
    Rows = attackers, Columns = scenarios. Color = AUC-ROC value.
    Shows which attacker is most dangerous for each model/scenario."""
    if mia.empty or "auc_roc" not in mia.columns: return
    if "attacker" not in mia.columns or "scenario" not in mia.columns: return

    fams = [f for f in _FAM_ORDER if f in mia["family"].unique()]
    _AUG_F4A = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    scenarios = [s for s in ["baseline"] + _AUG_F4A + sorted(
                     [s for s in mia["scenario"].unique() if s.startswith("ldp_eps")])]
    scenarios = [s for s in scenarios if s in mia["scenario"].unique()]

    for fam in fams:
        sub = mia[mia["family"]==fam].copy()
        if sub.empty: continue
        if "input_type" in sub.columns:
            nc = sub[sub["input_type"]=="nice_cf"]
            sub = nc if not nc.empty else sub

        # Pivot: rows=attacker, cols=scenario
        try:
            piv = sub.groupby(["attacker","scenario"])["auc_roc"].max().unstack(fill_value=float("nan"))
        except Exception: continue
        piv = piv.reindex(columns=[s for s in scenarios if s in piv.columns])
        if piv.empty: continue

        # Sort attackers by their max AUC across scenarios (descending)
        piv["_max"] = piv.max(axis=1)
        piv = piv.sort_values("_max", ascending=False).drop(columns=["_max"])

        # Attacker group colour for row labels
        atk_grp_map = {}
        for g, atks in _ATTACKER_GROUPS.items():
            for a in atks: atk_grp_map[a] = g

        n_rows, n_cols = piv.shape
        fig_h = max(8, n_rows * 0.32 + 2)
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.2 + 3), fig_h))

        arr = piv.values.astype(float)
        vmin, vmax = 0.49, min(0.6, float(np.nanmax(arr)) + 0.01)
        im = ax.imshow(arr, aspect="auto", cmap="Reds", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax, label="AUC-ROC", fraction=0.025, pad=0.02)

        # Annotate each cell
        for i in range(n_rows):
            for j in range(n_cols):
                v = arr[i,j]
                if not np.isnan(v):
                    c = "white" if (v-vmin)/(vmax-vmin) > 0.6 else "black"
                    ax.text(j, i, "{:.3f}".format(v), ha="center", va="center",
                            fontsize=7, color=c)

        # Y-axis: attacker names with group colour prefix
        ylabels = []
        for atk in piv.index:
            grp = atk_grp_map.get(atk,"?")
            ylabels.append("[{}] {}".format(grp[:3], atk))
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(ylabels, fontsize=7.5)

        # Colour y-tick labels by attacker group
        for tick, atk in zip(ax.get_yticklabels(), piv.index):
            grp = atk_grp_map.get(atk,"Statistical")
            tick.set_color(_ATK_GRP_COLORS.get(grp,"#333"))

        col_labels = [s.replace("ldp_eps","LDP eps=").replace("baseline","Baseline")
                       .replace("augmented","Augmented") for s in piv.columns]
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
        ax.set_title("MIA Attacker Heatmap — {} base\n"
                     "AUC-ROC per attacker × scenario  (red = higher attack success)"
                     .format(_fd(fam)), fontsize=_TF, fontweight="bold")

        # Legend for attacker groups
        grp_patches = [mpatches.Patch(facecolor=_ATK_GRP_COLORS[g], label=g)
                       for g in _ATTACKER_GROUPS]
        ax.legend(handles=grp_patches, fontsize=_LG-1,
                  bbox_to_anchor=(1.18, 1), loc="upper left", title="Method family")
        plt.tight_layout()
        tag = {"logistic_regression":"lr","random_forest":"rf","xgboost":"xgb"}.get(fam,fam)
        _save_fig(fig, os.path.join(out_dir, "F4a_attacker_heatmap_{}.png".format(tag)))


def _viz4_attacker_groups(mia, out_dir):
    """F4b: Best attacker per method group across scenarios.
    For each scenario × family, shows the max AUC achieved by each attacker group.
    Answers: which method family is the strongest attacker?"""
    if mia.empty or "auc_roc" not in mia.columns: return

    atk_grp_map = {}
    for g, atks in _ATTACKER_GROUPS.items():
        for a in atks: atk_grp_map[a] = g

    df = mia.copy()
    if "attacker" not in df.columns: return
    df["atk_group"] = df["attacker"].map(lambda a: atk_grp_map.get(a, "Other"))

    # For each (scenario, family, group) → max AUC
    grp_cols = [c for c in ["scenario","family","atk_group"] if c in df.columns]
    agg = df.groupby(grp_cols)["auc_roc"].max().reset_index()

    fams = [f for f in _FAM_ORDER if f in agg["family"].unique()]
    groups = list(_ATTACKER_GROUPS.keys())
    _AUG_F4B = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    scenarios = [s for s in ["baseline"] + _AUG_F4B +
                 sorted([s for s in agg["scenario"].unique() if s.startswith("ldp_eps")])]
    scenarios = [s for s in scenarios if s in agg["scenario"].unique()]

    fig, axes = plt.subplots(1, len(fams), figsize=(7*len(fams), 7), sharey=False)
    if len(fams)==1: axes=[axes]
    fig.suptitle("MIA Attacker Group Comparison: Best AUC per Method Family\n"
                 "(DCR / GEN-LRA / DPI / Statistical / Ensemble)",
                 fontsize=_TF+1, fontweight="bold", y=1.02)

    w = 0.15; x = np.arange(len(scenarios))
    for ax, fam in zip(axes, fams):
        sub = agg[agg["family"]==fam]
        for gi, grp in enumerate(groups):
            gsub = sub[sub["atk_group"]==grp]
            vals = []
            for sc in scenarios:
                row = gsub[gsub["scenario"]==sc]
                vals.append(float(row["auc_roc"].iloc[0]) if not row.empty else float("nan"))
            offset = (gi - len(groups)/2 + 0.5)*w
            bars = ax.bar(x+offset, vals, w,
                          color=_ATK_GRP_COLORS.get(grp,"#888"),
                          alpha=0.85, label=grp, edgecolor="white")
        ax.axhline(0.5, color="red", ls="--", lw=1.5, alpha=0.8)
        ax.set_title("{} base".format(_fd(fam)), fontsize=_LF, fontweight="bold")
        ax.set_xticks(x)
        sc_labels = [s.replace("ldp_eps","LDP eps=").replace("baseline","Base")
                      .replace("augmented","Aug") for s in scenarios]
        ax.set_xticklabels(sc_labels, rotation=35, ha="right", fontsize=_TK-1)
        ax.set_ylabel("Best AUC-ROC in group", fontsize=_LF)
        ax.legend(fontsize=_LG-1, bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  framealpha=0.9)
        vv = [v for v in ax.get_children() if hasattr(v,"get_height")]
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F4b_attacker_group_ranking.png"))


def _viz4_ensemble(mia, out_dir):
    """F4c: Ensemble (majority_vote) attacker analysis.
    Shows all vote thresholds compared to best individual attacker and random baseline."""
    if mia.empty or "auc_roc" not in mia.columns: return

    ensemble_atks = _ATTACKER_GROUPS["Ensemble"]
    if "attacker" not in mia.columns: return

    ens_sub = mia[mia["attacker"].isin(ensemble_atks)].copy()
    best_ind = mia[~mia["attacker"].isin(ensemble_atks)].copy()
    if ens_sub.empty: return

    fams = [f for f in _FAM_ORDER if f in ens_sub["family"].unique()]
    _AUG_F4C = ["augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    scenarios = [s for s in ["baseline"] + _AUG_F4C +
                 sorted([s for s in ens_sub["scenario"].unique() if s.startswith("ldp_eps")])]
    scenarios = [s for s in scenarios if s in ens_sub["scenario"].unique()]

    for fam in fams:
        esub = ens_sub[ens_sub["family"]==fam]
        if esub.empty: continue
        # No input_type filter — include all proxy types so LDP scenarios appear

        atks_present = [a for a in ensemble_atks if a in esub["attacker"].unique()]
        if not atks_present: continue

        n_sc = len(scenarios)
        fig, ax = plt.subplots(figsize=(max(12, n_sc*1.5+3), 7))

        w = 0.8/max(len(atks_present),1)
        x = np.arange(n_sc)
        cmap = plt.cm.get_cmap("plasma", len(atks_present))

        for ai, atk in enumerate(atks_present):
            asub = esub[esub["attacker"]==atk]
            vals = []
            for sc in scenarios:
                row = asub[asub["scenario"]==sc]
                vals.append(float(row["auc_roc"].iloc[0]) if not row.empty else float("nan"))
            offset = (ai - len(atks_present)/2 + 0.5)*w
            lbl = atk.replace("majority_vote_","vote_")
            ax.bar(x+offset, vals, w, color=cmap(ai), alpha=0.85, label=lbl, edgecolor="white")

        # Best individual attacker per scenario (all proxy types)
        bind = best_ind[best_ind["family"]==fam]
        best_vals = []
        for sc in scenarios:
            brow = bind[bind["scenario"]==sc]
            best_vals.append(float(brow["auc_roc"].max()) if not brow.empty else float("nan"))
        ax.plot(x, best_vals, "k^--", ms=9, lw=2, label="Best individual attacker")
        ax.axhline(0.5, color="red", ls="--", lw=1.5, label="Random (0.50)")

        sc_labels = [s.replace("ldp_eps","LDP eps=").replace("baseline","Baseline")
                      .replace("augmented","Augmented") for s in scenarios]
        ax.set_xticks(x)
        ax.set_xticklabels(sc_labels, rotation=30, ha="right", fontsize=_TK)
        ax.set_ylabel("AUC-ROC", fontsize=_LF)
        ax.set_title("Ensemble (majority_vote) Attacker Analysis — {} base\n"
                     "How different vote thresholds compare to best individual attacker"
                     .format(_fd(fam)), fontsize=_TF, fontweight="bold")
        ax.legend(fontsize=_LG-2, ncol=1, bbox_to_anchor=(1.01, 1.0),
                  loc="upper left", framealpha=0.9)
        plt.tight_layout()
        tag = {"logistic_regression":"lr","random_forest":"rf","xgboost":"xgb"}.get(fam,fam)
        _save_fig(fig, os.path.join(out_dir, "F4c_ensemble_{}.png".format(tag)))


def _viz4_proxy_comparison(mia, out_dir):
    """F4d: Input proxy comparison — mm_cf vs nice_cf (one figure per family).

    Missing bars (e.g. NiCE CF not generated for a scenario/family) are shown as
    a near-zero stub labelled 'N/A' so the absence is explicit, not silent.
    Files: F4d_proxy_<family>.png
    """
    if mia.empty or "auc_roc" not in mia.columns: return
    if "input_type" not in mia.columns: return

    fams = [f for f in _FAM_ORDER if f in mia["family"].unique()]

    # All scenarios present in the data, sorted by group then epsilon
    all_scens = _sort_scenarios(mia["scenario"].unique().tolist())

    # Best attacker AUC per (scenario, family, input_type)
    agg = mia.groupby(["scenario","family","input_type"])["auc_roc"].max().reset_index()

    _NA_COLOR   = "#dddddd"   # light grey stub for missing bars
    _NA_HATCH   = "////"
    _PROXY_COLORS = {"mm_cf": "#888888", "nice_cf": None}  # None → use family color

    for fam in fams:
        sub     = agg[agg["family"] == fam]
        n_scens = len(all_scens)
        height  = max(5, n_scens * 0.75 + 2.5)

        fig, ax = plt.subplots(figsize=(13, height))

        y = np.arange(n_scens)
        w = 0.38

        for ii, itype in enumerate(["mm_cf", "nice_cf"]):
            isub   = sub[sub["input_type"] == itype]
            offset = (-w / 2 + ii * w)
            color  = _PROXY_COLORS[itype] if _PROXY_COLORS[itype] else _fc(fam)

            for si, sc in enumerate(all_scens):
                row = isub[isub["scenario"] == sc]
                yi  = y[si] + offset

                if not row.empty and np.isfinite(float(row["auc_roc"].iloc[0])):
                    val = float(row["auc_roc"].iloc[0])
                    ax.barh(yi, val, w, color=color, alpha=0.88,
                            edgecolor="white", linewidth=0.4)
                    ax.text(val + 0.002, yi, "{:.4f}".format(val),
                            va="center", ha="left", fontsize=8.5, color="#333")
                else:
                    # Explicit N/A stub — show absence rather than silence
                    stub = 0.501   # sits just above the 0.5 reference line
                    ax.barh(yi, stub, w, color=_NA_COLOR, alpha=0.7,
                            edgecolor="#aaa", linewidth=0.6,
                            hatch=_NA_HATCH)
                    ax.text(stub + 0.002, yi, "N/A",
                            va="center", ha="left", fontsize=8.5,
                            color="#888", style="italic")

        ax.axvline(0.5, color="red", ls="--", lw=1.5, label="Random baseline (0.50)")

        # Scenario group separators
        prev_grp = None
        for si, sc in enumerate(all_scens):
            grp = _scen_group(sc)
            if prev_grp is not None and grp != prev_grp:
                ax.axhline(si - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
            prev_grp = grp

        # Y-axis labels and inversion
        sc_labels = [_fmt_ldp_label(s).replace("\n", " ") for s in all_scens]
        ax.set_yticks(y)
        ax.set_yticklabels(sc_labels, fontsize=_TK + 1)
        ax.invert_yaxis()

        # X-axis
        ax.set_xlim(0.44, max(0.62, agg["auc_roc"].max() + 0.06
                               if not agg.empty else 0.62))
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x", labelsize=_TK)

        ax.set_xlabel("Best AUC-ROC  (max over all attackers)", fontsize=_LF)
        ax.set_title("MIA Proxy Comparison: mm_cf vs nice_cf — {} base estimator\n"
                     "(Striped grey bar = proxy not generated for this scenario)".format(
                         _fd(fam)),
                     fontsize=_TF, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.2, axis="x")

        legend_els = [
            mpatches.Patch(facecolor="#888888", label="mm_cf proxy"),
            mpatches.Patch(facecolor=_fc(fam),  label="nice_cf proxy"),
            mpatches.Patch(facecolor=_NA_COLOR, hatch=_NA_HATCH,
                           edgecolor="#aaa",    label="N/A (not generated)"),
            Line2D([0], [0], color="red", ls="--", lw=1.5,
                   label="Random baseline (0.50)"),
        ]
        ax.legend(handles=legend_els, fontsize=_LG, framealpha=0.9,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        slug = fam.replace(" ", "_")
        _save_fig(fig, os.path.join(out_dir, "F4d_proxy_{}.png".format(slug)))


def generate_folder4(out_dir=_DIR4):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 4: MIA Methods Comparison\n" + "="*60)
    _u_mia4 = os.path.join(_UNIFIED_RES, "unified_mia.csv")
    mia = (_load(_u_mia4, "unified_mia_f4") if os.path.exists(_u_mia4)
           else _load(os.path.join(_MAIN_RES, "mia_results.csv"), "main mia"))
    if mia.empty:
        print("  [Folder 4] No MIA data — skipping")
        return
    _viz4_attacker_heatmap(mia, out_dir)
    _viz4_attacker_groups(mia, out_dir)
    _viz4_ensemble(mia, out_dir)
    _viz4_proxy_comparison(mia, out_dir)
    print("  [Folder 4] Done ->", out_dir)


# ===========================================================================
# FOLDER 5 — Unified cross-pipeline comparison  (reads unified CSVs)
# ===========================================================================

_UNIFIED_RES = os.path.join(_BASE_DIR, "pipeline_outputs", "unified", "results")
_DIR5        = os.path.join(_VIZ_ROOT, "5_unified")


def _smart_fmt(x):
    """Human-readable number — avoids scientific notation."""
    if not np.isfinite(x): return "N/A"
    if abs(x) == 0:        return "0"
    if abs(x) >= 100:      return "{:.1f}".format(x)
    if abs(x) >= 1:        return "{:.3f}".format(x)
    if abs(x) >= 0.001:    return "{:.4f}".format(x)
    return "{:.2e}".format(x)


def _smart_formatter():
    return mticker.FuncFormatter(lambda x, _: _smart_fmt(x))


def _annotate_barh_u(ax, vals, fontsize=9):
    xlim = ax.get_xlim()
    span = (xlim[1] - xlim[0]) or 1e-9
    for i, v in enumerate(vals):
        if pd.isna(v) or not np.isfinite(float(v)): continue
        ax.text(float(v) + span * 0.01, i, _smart_fmt(float(v)),
                va="center", ha="left", fontsize=fontsize, color="#333333")


def _hbar_xlim_u(vals):
    finite = [v for v in vals if pd.notna(v) and np.isfinite(float(v))]
    if not finite: return None, None
    lo = min(min(finite), 0); hi = max(max(finite), 0)
    span = max(hi - lo, 1e-9)
    return lo - span * 0.02, hi + span * 0.18


def _u1_ldp_sweep(out_dir):
    """U1: LDP sweep — one figure per metric, line plot, log-epsilon x-axis."""
    ldp_csv = os.path.join(_UNIFIED_RES, "ldp_sweep_results.csv")
    if not os.path.exists(ldp_csv):
        # fallback: try main results path
        ldp_csv = os.path.join(_BASE_DIR, "pipeline_outputs", "results",
                               "ldp_sweep_results.csv")
    df = _load(ldp_csv, "ldp_sweep")
    if df.empty or "epsilon" not in df.columns: return

    metrics = [
        ("group_SPD",     "SPD",           "|SPD|  (0 = perfectly fair)",           False),
        ("group_EOD",     "EOD",           "|EOD|  (0 = perfectly fair)",           False),
        ("group_EqOdds",  "EqOdds",        "Equalized Odds  (0 = perfectly fair)",  False),
        ("mia_auc_roc",   "MIA_AUC",       "MIA AUC-ROC  (0.50 = random / private)",True),
        ("mia_advantage", "MIA_Advantage", "MIA Advantage  (0 = perfect privacy)",  False),
    ]
    fams   = [f for f in _FAM_ORDER if f in df["family"].unique()]
    colors = plt.cm.tab10(np.linspace(0, 0.9, max(len(fams), 1)))

    for col, slug, ylabel, add_ref in metrics:
        if col not in df.columns: continue
        fig, ax = plt.subplots(figsize=(9, 5))
        for fam, c in zip(fams, colors):
            sub = df[df["family"] == fam].sort_values("epsilon")
            if sub.empty: continue
            yv = sub[col].abs() if col in ("group_SPD","group_EOD") else sub[col]
            ax.plot(sub["epsilon"], yv, marker="o", lw=2, label=_fd(fam), color=c)
            for ex, ey in zip(sub["epsilon"], yv):
                ax.annotate(_smart_fmt(float(ey)), xy=(ex, float(ey)),
                            xytext=(4,4), textcoords="offset points",
                            fontsize=8, color=c)
        if add_ref:
            ax.axhline(0.5, color="red", ls="--", lw=1.4, label="random (0.50)")
        ax.set_xlabel("LDP  ε  (lower = more noise)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title("LDP Sweep — {}  (COMPAS)".format(slug.replace("_"," ")),
                     fontsize=13)
        eps_vals_u = sorted(df["epsilon"].unique())
        ax.set_xscale("log")
        ax.set_xlim(eps_vals_u[0] * 0.65, eps_vals_u[-1] * 1.55)
        ax.set_xticks(eps_vals_u)
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(e) for e in eps_vals_u]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.yaxis.set_major_formatter(_smart_formatter())
        ax.tick_params(axis="both", labelsize=11)
        ax.legend(fontsize=10, framealpha=0.9, bbox_to_anchor=(1.01, 1.0),
                  loc="upper left")
        ax.grid(True, alpha=0.3)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        _save_fig(fig, os.path.join(out_dir, "U1_ldp_sweep_{}.png".format(slug)))


def _u2_all_methods_fairness(out_dir):
    """U2: All-methods fairness — one horizontal-bar figure per metric.
    Reads unified_comparison_summary.csv produced by unified_analysis.py."""
    summary_csv = os.path.join(_UNIFIED_RES, "unified_comparison_summary.csv")
    summary = _load(summary_csv, "unified_summary")
    if summary.empty: return

    sub = summary.copy()
    sub["_grp"] = sub["scenario"].astype(str).map(_scen_group)
    sub["_eps"] = sub["scenario"].astype(str).map(_scen_eps).fillna(0.0)
    sub["_gi"]  = sub["_grp"].map(lambda g: _GRP_ORDER.index(g)
                                   if g in _GRP_ORDER else 99)
    sub = sub.sort_values(["_gi","_eps","scenario","family"]).reset_index(drop=True)
    sub["label"] = sub["scenario"].astype(str) + "  [" + sub["family"].astype(str) + "]"

    metrics = [
        ("SPD",    "Statistical Parity Difference (SPD)",  0),
        ("EOD",    "Equal Opportunity Difference (EOD)",   0),
        ("EqOdds", "Equalized Odds",                       0),
        ("AOD",    "Average Odds Difference (AOD)",        0),
        ("DI",     "Disparate Impact (DI)",                1),
    ]
    present_grps = [g for g in _GRP_ORDER if g in sub["_grp"].values]

    for col, title, ideal in metrics:
        if col not in sub.columns: continue
        df_plot = sub[["label","scenario",col]].dropna(subset=[col])
        if df_plot.empty: continue

        labels = df_plot["label"].tolist()
        vals   = [float(v) for v in df_plot[col].tolist()]
        colors = [_scenario_color(str(s)) for s in df_plot["scenario"]]
        height = max(5, len(labels) * 0.58)

        fig, ax = plt.subplots(figsize=(13, height))
        ax.barh(range(len(labels)), vals, color=colors,
                edgecolor="white", linewidth=0.4, height=0.72)
        ax.axvline(ideal, color="black", lw=1.5, ls="--", alpha=0.7,
                   label="ideal = {}".format(ideal))

        # Group separators
        prev_grp = None
        for i, (_, row) in enumerate(df_plot.iterrows()):
            grp = _scen_group(str(row["scenario"]))
            if prev_grp is not None and grp != prev_grp:
                ax.axhline(i - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
            prev_grp = grp

        _annotate_barh_u(ax, vals)

        lo, hi = _hbar_xlim_u(vals)
        if lo is not None: ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(_smart_formatter())
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(title, fontsize=12)
        ax.set_title("All Methods — {}  (COMPAS)".format(title), fontsize=13, pad=10)
        ax.grid(True, alpha=0.2, axis="x")

        legend_patches = []
        for g in present_grps:
            c = _GRP_PALETTES[g][min(1, len(_GRP_PALETTES[g]) - 1)]
            legend_patches.append(mpatches.Patch(
                facecolor=c, edgecolor="#555",
                label=_GRP_LABELS.get(g, g)))
        ax.legend(handles=legend_patches, fontsize=9, framealpha=0.92,
                  title="Scenario groups", title_fontsize=9,
                  bbox_to_anchor=(1.01, 1.0), loc="upper left")
        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir, "U2_fairness_{}.png".format(col)))


def _u3_nice_mia_comparison(out_dir):
    """U3: MIA AUC-ROC — one horizontal-bar figure per model family.
    Reads unified_mia.csv produced by unified_analysis.py."""
    mia_csv = os.path.join(_UNIFIED_RES, "unified_mia.csv")
    unified_mia = _load(mia_csv, "unified_mia")
    if unified_mia.empty or "auc_roc" not in unified_mia.columns: return
    if "input_type" not in unified_mia.columns: return

    best = (unified_mia.groupby(["scenario","family","input_type"])
            ["auc_roc"].max().reset_index())
    families = [f for f in _FAM_ORDER if f in best["family"].unique()]

    for fam in families:
        sub_fam  = best[best["family"] == fam]
        nice_sub = sub_fam[sub_fam["input_type"] == "nice_cf"].set_index("scenario")
        sc_list  = _sort_scenarios(nice_sub.index.tolist())
        if not sc_list: continue

        vals   = [float(nice_sub.loc[s,"auc_roc"]) if s in nice_sub.index
                  else float("nan") for s in sc_list]
        height = max(5, len(sc_list) * 0.55)

        fig, ax = plt.subplots(figsize=(12, height))
        bar_colors = [_scenario_color(s) for s in sc_list]
        ax.barh(range(len(sc_list)), vals, color=bar_colors,
                edgecolor="white", linewidth=0.4, height=0.7)
        ax.axvline(0.5, color="red", ls="--", lw=1.5, label="random baseline (0.50)")
        _annotate_barh_u(ax, vals)
        finite_vals = [v for v in vals if np.isfinite(v)]
        if finite_vals:
            lo = min(min(finite_vals) - 0.02, 0.45)
            hi = max(max(finite_vals) + 0.05, 0.55)
            ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_yticks(range(len(sc_list)))
        ax.set_yticklabels(sc_list, fontsize=11)
        ax.set_xlabel("MIA AUC-ROC  (NiCE-CF proxy,  0.50 = random / private)",
                      fontsize=12)
        ax.set_title("MIA AUC-ROC — {}  (COMPAS)".format(_fd(fam)),
                     fontsize=13, pad=10)
        ax.legend(fontsize=10, framealpha=0.9, bbox_to_anchor=(1.01, 1.0),
                  loc="upper left")
        ax.grid(True, alpha=0.25, axis="x")
        plt.tight_layout()
        slug = fam.replace(" ","_")
        _save_fig(fig, os.path.join(out_dir, "U3_mia_{}.png".format(slug)))


def _u4_fairness_heatmap(out_dir):
    """U4: Unified fairness heatmap — scenarios × metrics.
    Reads unified_comparison_summary.csv."""
    summary_csv = os.path.join(_UNIFIED_RES, "unified_comparison_summary.csv")
    summary = _load(summary_csv, "unified_summary_heatmap")
    if summary.empty: return

    heat_cols = ["SPD","DI","EOD","AOD","EqOdds","PP","Theil","cf_fairness","consistency"]
    avail = [c for c in heat_cols if c in summary.columns]
    if not avail: return

    labels    = (summary["scenario"].astype(str) + "  ["
                 + summary["family"].astype(str) + "]").tolist()
    heat_data = summary[avail].values.astype(float)

    height = max(6, len(labels) * 0.6 + 2)
    width  = max(8, len(avail) * 1.6 + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    im  = ax.imshow(heat_data, aspect="auto", cmap="RdYlGn_r")
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xticks(range(len(avail)))
    ax.set_xticklabels(avail, fontsize=12, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_title("Unified Fairness Heatmap — All Methods (COMPAS)",
                 fontsize=14, pad=12)
    cell_fs = max(7, min(10, int(200 / max(len(labels), 1))))
    for i in range(len(labels)):
        for j in range(len(avail)):
            v = heat_data[i, j]
            if np.isfinite(v):
                norm_v = (v - np.nanmin(heat_data)) / (
                    np.nanmax(heat_data) - np.nanmin(heat_data) + 1e-12)
                txt_c = "white" if norm_v > 0.75 or norm_v < 0.25 else "black"
                ax.text(j, i, _smart_fmt(v), ha="center", va="center",
                        fontsize=cell_fs, color=txt_c)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "U4_fairness_heatmap.png"))


def _u5_nice_cf_quality(out_dir):
    """U5: NiCE CF quality — one horizontal-bar figure per metric.
    Reads unified_nice_quality.csv."""
    nice_csv = os.path.join(_UNIFIED_RES, "unified_nice_quality.csv")
    unified_nice = _load(nice_csv, "unified_nice")
    if unified_nice.empty: return

    metrics = [c for c in ["flip_rate","proximity","plausibility","sparsity"]
               if c in unified_nice.columns]
    if not metrics: return

    sc_col  = unified_nice["scenario"].astype(str) if "scenario" in unified_nice.columns \
              else pd.Series(["?"] * len(unified_nice))
    fam_col = unified_nice["family"].astype(str)   if "family"   in unified_nice.columns \
              else pd.Series(["?"] * len(unified_nice))
    labels  = (sc_col + "  [" + fam_col + "]").tolist()
    colors  = [_scenario_color(str(s)) for s in sc_col]
    height  = max(5, len(labels) * 0.55)

    meta = {
        "flip_rate":    ("Flip Rate",
                         "Fraction of CFs that change the prediction  "
                         "(higher = CFs are effective)"),
        "proximity":    ("Proximity",
                         "Average distance from original input  "
                         "(lower = more actionable)"),
        "plausibility": ("Plausibility",
                         "Likelihood under training distribution  "
                         "(higher = more realistic)"),
        "sparsity":     ("Sparsity",
                         "Fraction of features unchanged  "
                         "(higher = fewer features modified)"),
    }

    for col in metrics:
        vals  = [float(v) if pd.notna(v) else float("nan")
                 for v in unified_nice[col].tolist()]
        short, xlabel = meta.get(col, (col, col))

        fig, ax = plt.subplots(figsize=(12, height))
        ax.barh(range(len(labels)), vals, color=colors,
                edgecolor="white", linewidth=0.4, height=0.7)
        _annotate_barh_u(ax, vals)
        lo, hi = _hbar_xlim_u(vals)
        if lo is not None: ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(_smart_formatter())
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_title("NiCE CF Quality — {}  (COMPAS)".format(short),
                     fontsize=13, pad=10)
        ax.grid(True, alpha=0.25, axis="x")
        plt.tight_layout()
        _save_fig(fig, os.path.join(out_dir, "U5_nice_{}.png".format(col)))


def generate_folder5(out_dir=_DIR5):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 5: Unified Cross-Pipeline\n" + "="*60)
    _u1_ldp_sweep(out_dir)
    _u2_all_methods_fairness(out_dir)
    _u3_nice_mia_comparison(out_dir)
    _u4_fairness_heatmap(out_dir)
    _u5_nice_cf_quality(out_dir)
    print("  [Folder 5] Done ->", out_dir)


# ===========================================================================
# FOLDER 6 — Model performance comparison (accuracy / precision / recall / F1 / AUC)
# ===========================================================================

_MAIN_METRICS_CSV    = os.path.join(_BASE_DIR, "pipeline_outputs",
                                     "results", "model_metrics.csv")
_FL_METRICS_CSV      = os.path.join(_BASE_DIR, "pipeline_outputs_fairlearn",
                                     "results", "fairlearn_model_metrics.csv")
_LDP_FAIR_METRICS_CSV= os.path.join(_BASE_DIR, "pipeline_outputs",
                                     "ldp_fair", "results", "ldp_fair_model_metrics.csv")
_DIR6                = os.path.join(_VIZ_ROOT, "6_model_performance")

_PERF_METRICS = [
    ("accuracy",  "Accuracy",           "(higher is better)",  None),
    ("auc_roc",   "AUC-ROC",            "(higher is better)",  0.5),
    ("f1",        "F1 Score",           "(higher is better)",  None),
    ("precision", "Precision",          "(higher is better)",  None),
    ("recall",    "Recall / Sensitivity","(higher is better)", None),
]


def _load_all_model_metrics():
    """Combine model_metrics.csv + fairlearn_model_metrics.csv + ldp_fair_model_metrics.csv
    into a single normalised DataFrame with columns:
        scenario, family, label, accuracy, auc_roc, f1, precision, recall
    """
    frames = []

    # ── Main pipeline (baseline, augmented, ldp_baseline_*, ldp_augmented_*) ──
    df_main = _load(_MAIN_METRICS_CSV, "main model_metrics")
    if not df_main.empty:
        df_main["source"] = "main"
        if "scenario" not in df_main.columns and "model" in df_main.columns:
            df_main["scenario"] = df_main["model"]
        if "label" not in df_main.columns:
            df_main["label"] = df_main["scenario"].astype(str) + "  [" + \
                               df_main["family"].astype(str) + "]"
        frames.append(df_main)

    # ── Fairlearn (fairness constraints, no LDP) ──
    df_fl = _load(_FL_METRICS_CSV, "fairlearn model_metrics")
    if not df_fl.empty:
        df_fl = _normalize_family(df_fl)
        df_fl["source"] = "fairlearn"
        # fairlearn CSVs use "model" as the scenario key
        if "scenario" not in df_fl.columns:
            df_fl["scenario"] = df_fl.get("model", pd.Series(["fairlearn"] * len(df_fl)))
        if "label" not in df_fl.columns:
            df_fl["label"] = df_fl.get("label", df_fl["scenario"].astype(str))
        # rename f1 column if needed
        if "f1_score" in df_fl.columns and "f1" not in df_fl.columns:
            df_fl = df_fl.rename(columns={"f1_score": "f1"})
        frames.append(df_fl)

    # ── LDP + Fairlearn ──
    df_lf = _load(_LDP_FAIR_METRICS_CSV, "ldp_fair model_metrics")
    if not df_lf.empty:
        df_lf = _normalize_family(df_lf)
        df_lf["source"] = "ldp_fair"
        if "scenario" not in df_lf.columns:
            df_lf["scenario"] = df_lf.get("model", pd.Series(["ldp_fair"] * len(df_lf)))
        if "label" not in df_lf.columns:
            df_lf["label"] = df_lf.get("label", df_lf["scenario"].astype(str))
        if "f1_score" in df_lf.columns and "f1" not in df_lf.columns:
            df_lf = df_lf.rename(columns={"f1_score": "f1"})
        frames.append(df_lf)

    if not frames:
        return pd.DataFrame()

    cols_keep = ["scenario", "family", "label", "source",
                 "accuracy", "auc_roc", "f1", "precision", "recall"]
    combined = pd.concat(frames, ignore_index=True, sort=False)
    cols_present = [c for c in cols_keep if c in combined.columns]
    return combined[cols_present].copy()


def _p6_one_metric(df, col, title, subtitle, ref_val, out_dir):
    """One horizontal-bar figure for a single performance metric.

    Bars sorted by scenario group then epsilon; semantic colors.
    Panels are split by model family so each family is its own figure.
    """
    if col not in df.columns:
        return

    fams = [f for f in _FAM_ORDER if f in df["family"].unique()]
    if not fams:
        # No family column — single combined figure
        fams = [None]

    for fam in fams:
        sub = df[df["family"] == fam] if fam else df
        if sub.empty:
            continue

        # Sort rows by scenario group then epsilon
        sub = sub.copy()
        sub["_gi"]  = sub["scenario"].map(lambda s: _GRP_ORDER.index(_scen_group(s))
                                           if _scen_group(s) in _GRP_ORDER else 99)
        sub["_eps"] = sub["scenario"].map(lambda s: _scen_eps(s) or 0.0)
        sub = sub.sort_values(["_gi", "_eps", "scenario"]).reset_index(drop=True)

        labels = sub["label"].tolist()
        vals   = [float(v) if pd.notna(v) else float("nan")
                  for v in sub[col].tolist()]
        colors = [_scenario_color(str(s)) for s in sub["scenario"]]

        height = max(5, len(labels) * 0.58)
        fig, ax = plt.subplots(figsize=(13, height))

        ax.barh(range(len(labels)), vals, color=colors,
                edgecolor="white", linewidth=0.4, height=0.72)

        # Value annotations
        _annotate_barh_u(ax, vals)

        # Reference line (e.g. 0.5 for AUC)
        if ref_val is not None:
            ax.axvline(ref_val, color="red", ls="--", lw=1.5,
                       label="reference ({})".format(ref_val))

        # Group separators
        prev_grp = None
        for i, row_s in enumerate(sub["scenario"]):
            grp = _scen_group(str(row_s))
            if prev_grp is not None and grp != prev_grp:
                ax.axhline(i - 0.5, color="#cccccc", lw=1.0, ls="--", zorder=0)
            prev_grp = grp

        lo, hi = _hbar_xlim_u(vals)
        if lo is not None:
            ax.set_xlim(lo, hi)
        ax.xaxis.set_major_formatter(_smart_formatter())
        ax.tick_params(axis="x", labelsize=11)
        ax.tick_params(axis="y", labelsize=11)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=11)
        ax.set_xlabel("{} — {}".format(title, subtitle), fontsize=12)
        fam_tag = "  ({})".format(_fd(fam)) if fam else ""
        ax.set_title("Model Performance: {}{}  (COMPAS)".format(title, fam_tag),
                     fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, alpha=0.2, axis="x")

        # Legend: one patch per scenario group present
        present_grps = list(dict.fromkeys(_scen_group(s)
                                          for s in sub["scenario"].tolist()))
        legend_patches = []
        for g in _GRP_ORDER:
            if g not in present_grps:
                continue
            c = _GRP_PALETTES[g][min(1, len(_GRP_PALETTES[g]) - 1)]
            legend_patches.append(mpatches.Patch(
                facecolor=c, edgecolor="#555",
                label=_GRP_LABELS.get(g, g)))
        if ref_val is not None:
            legend_patches.append(
                Line2D([0], [0], color="red", ls="--", lw=1.5,
                       label="reference ({})".format(ref_val)))
        if legend_patches:
            ax.legend(handles=legend_patches, fontsize=9, framealpha=0.92,
                      title="Scenario groups", title_fontsize=9,
                      bbox_to_anchor=(1.01, 1.0), loc="upper left")

        plt.tight_layout()
        slug_col = col.replace("_", "")
        slug_fam = ("_" + fam.replace(" ", "_")) if fam else ""
        _save_fig(fig, os.path.join(out_dir,
                                    "P6_{}{}.png".format(slug_col, slug_fam)))


def _p6_overview(df, out_dir):
    """Summary scatter: AUC-ROC vs F1 per scenario, colored by group.
    One figure — quick overview of the performance vs. trade-off space."""
    needed = {"auc_roc", "f1", "scenario", "family"}
    if not needed.issubset(df.columns):
        return
    sub = df.dropna(subset=["auc_roc", "f1"])
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(12, 8))
    for _, row in sub.iterrows():
        c = _scenario_color(str(row["scenario"]))
        ax.scatter(float(row["auc_roc"]), float(row["f1"]),
                   color=c, s=90, alpha=0.85, edgecolors="white", lw=0.6, zorder=3)
        lbl = "{} [{}]".format(str(row["scenario"]), _fd(str(row["family"])))
        ax.annotate(_wrap(lbl, 20), (float(row["auc_roc"]), float(row["f1"])),
                    textcoords="offset points", xytext=(5, 3), fontsize=7.5,
                    alpha=0.85)

    ax.set_xlabel("AUC-ROC  (higher = better discriminator)", fontsize=12)
    ax.set_ylabel("F1 Score  (higher = better balance precision/recall)", fontsize=12)
    ax.set_title("Model Performance Overview: AUC-ROC vs F1  (COMPAS)\n"
                 "Ideal region: top-right corner", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.25)

    present_grps = list(dict.fromkeys(_scen_group(s) for s in sub["scenario"]))
    legend_patches = [
        mpatches.Patch(facecolor=_GRP_PALETTES[g][min(1, len(_GRP_PALETTES[g]) - 1)],
                       edgecolor="#555", label=_GRP_LABELS.get(g, g))
        for g in _GRP_ORDER if g in present_grps
    ]
    ax.legend(handles=legend_patches, fontsize=9, framealpha=0.92,
              title="Scenario groups", title_fontsize=9,
              bbox_to_anchor=(1.01, 1.0), loc="upper left")
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "P6_overview_auc_vs_f1.png"))


def generate_folder6(out_dir=_DIR6):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 6: Model Performance\n" + "="*60)
    df = _load_all_model_metrics()
    if df.empty:
        print("  [Folder 6] No model metrics data found — skipping.")
        print("  (Run the pipeline first to generate model_metrics.csv)")
        return

    print("  [Folder 6] {} rows loaded across {} scenarios".format(
        len(df), df["scenario"].nunique() if "scenario" in df.columns else "?"))

    # One figure per metric per family
    for col, title, subtitle, ref in _PERF_METRICS:
        _p6_one_metric(df, col, title, subtitle, ref, out_dir)

    # Overview scatter
    _p6_overview(df, out_dir)

    print("  [Folder 6] Done ->", out_dir)


# ===========================================================================
# FOLDER 7 — Augmentation Methods Cross-Comparison  (reads unified CSVs)
# ===========================================================================

_DIR7 = os.path.join(_VIZ_ROOT, "7_augmentation_methods")

_AUG_METHOD_COLORS = {
    "SCM":               "#2ca02c",   # green
    "update_labels":     "#e377c2",   # pink/magenta  -- matches _GRP_PALETTES
    "add_comparators":   "#9467bd",   # purple
}
_AUG_METHOD_LABELS = {
    "SCM":               "SCM",
    "update_labels":     "MM Update Labels",
    "add_comparators":   "MM Add Comparators",
}
_AUG_SCEN_TO_METHOD = {
    "augmented_SCM":             "SCM",
    "augmented_update_labels":   "update_labels",
    "augmented_add_comparators": "add_comparators",
    "augmented":                 "SCM",   # legacy fallback
}


def _f7a_fairness_by_method(out_dir):
    """F7a: For each fairness metric, grouped bar chart: x = family, groups = augmentation method.
    Baseline shown as a reference line. One subplot per metric."""
    fair_csv = os.path.join(_UNIFIED_RES, "unified_fairness.csv")
    fair = _load(fair_csv, "unified_fairness_f7a")
    if fair.empty:
        print("  [F7a] No unified_fairness.csv found, skipping.")
        return

    fair = _normalize_family(fair)
    fams = [f for f in _FAM_ORDER if f in fair["family"].unique()]
    if not fams:
        return

    metrics = [
        ("group_SPD",    "SPD",    "Statistical Parity Difference  (0 = fair)"),
        ("group_EOD",    "EOD",    "Equal Opportunity Difference   (0 = fair)"),
        ("group_DI",     "DI",     "Disparate Impact               (1 = fair)"),
        ("group_EqOdds", "EqOdds", "Equalized Odds                 (0 = fair)"),
    ]
    avail = [(c, s, t) for c, s, t in metrics if c in fair.columns]
    if not avail:
        return

    aug_scens = [sc for sc in ["augmented_SCM", "augmented_update_labels",
                                "augmented_add_comparators", "augmented"]
                 if sc in fair["scenario"].unique()]
    if not aug_scens:
        return

    n_met = len(avail)
    fig, axes = plt.subplots(1, n_met, figsize=(6 * n_met, 7), sharey=False)
    if n_met == 1:
        axes = [axes]
    fig.suptitle("Fairness by Augmentation Method  (COMPAS)\n"
                 "Dashed line = baseline reference; bars = augmented scenarios per family",
                 fontsize=_TF + 1, fontweight="bold", y=1.02)

    n_methods = len(aug_scens)
    w_total = 0.75
    w = w_total / max(n_methods, 1)
    x = np.arange(len(fams))

    for ax, (col, short, title_t) in zip(axes, avail):
        ideal = 1.0 if short == "DI" else 0.0

        # Baseline reference lines per family
        for fi, fam in enumerate(fams):
            base_row = fair[(fair["scenario"] == "baseline") & (fair["family"] == fam)]
            if not base_row.empty and col in base_row.columns:
                base_val = float(base_row[col].iloc[0])
                ax.plot([fi - w_total / 2 - 0.05, fi + w_total / 2 + 0.05],
                        [base_val, base_val],
                        color="#555555", lw=2, ls="--", alpha=0.7,
                        label="Baseline" if fi == 0 else "")

        # Augmented bars
        for mi, sc in enumerate(aug_scens):
            method = _AUG_SCEN_TO_METHOD.get(sc, sc)
            color  = _AUG_METHOD_COLORS.get(method, "#888888")
            label  = _AUG_METHOD_LABELS.get(method, method)
            offset = (-w_total / 2 + mi * w + w / 2)
            vals = []
            for fam in fams:
                row = fair[(fair["scenario"] == sc) & (fair["family"] == fam)]
                vals.append(float(row[col].iloc[0]) if not row.empty and col in row.columns
                            else float("nan"))
            bars = ax.bar(x + offset, vals, w, color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.5, label=label)
            _bar_labels(ax, bars, fmt="{:.3f}", fs=8)

        ax.axhline(ideal, color="black", lw=1.2, ls=":", alpha=0.5,
                   label="Ideal ({})".format(ideal))
        ax.set_xticks(x)
        ax.set_xticklabels([_fd(f) for f in fams], fontsize=_TK + 1)
        ax.set_ylabel(short, fontsize=_LF)
        ax.set_title("{}\n{}".format(short, title_t), fontsize=_LF, pad=8)

    # Shared legend (deduplicate)
    handles, seen = [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); seen.add(l)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=_LG, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F7a_fairness_by_method.png"))


def _f7b_performance_by_method(out_dir):
    """F7b: Accuracy, F1, AUC-ROC — comparing augmented scenario across methods."""
    metrics_csv = os.path.join(_UNIFIED_RES, "unified_model_metrics.csv")
    metrics = _load(metrics_csv, "unified_model_metrics_f7b")
    if metrics.empty:
        print("  [F7b] No unified_model_metrics.csv found, skipping.")
        return

    metrics = _normalize_family(metrics)
    fams = [f for f in _FAM_ORDER if f in metrics["family"].unique()]
    if not fams:
        return

    perf_cols = [c for c in ["accuracy", "f1", "auc_roc"] if c in metrics.columns]
    if not perf_cols:
        return

    aug_scens = [sc for sc in ["augmented_SCM", "augmented_update_labels",
                                "augmented_add_comparators", "augmented"]
                 if sc in metrics["scenario"].unique()]
    if not aug_scens:
        return

    n_met = len(perf_cols)
    fig, axes = plt.subplots(1, n_met, figsize=(6 * n_met, 7), sharey=False)
    if n_met == 1:
        axes = [axes]
    fig.suptitle("Model Performance by Augmentation Method  (COMPAS)\n"
                 "Dashed line = baseline; bars = augmented scenarios per family",
                 fontsize=_TF + 1, fontweight="bold", y=1.02)

    n_methods = len(aug_scens)
    w_total = 0.75
    w = w_total / max(n_methods, 1)
    x = np.arange(len(fams))

    for ax, col in zip(axes, perf_cols):
        # Baseline reference
        for fi, fam in enumerate(fams):
            base_row = metrics[(metrics["scenario"] == "baseline") & (metrics["family"] == fam)]
            if not base_row.empty and col in base_row.columns:
                base_val = float(base_row[col].iloc[0])
                ax.plot([fi - w_total / 2 - 0.05, fi + w_total / 2 + 0.05],
                        [base_val, base_val],
                        color="#555555", lw=2, ls="--", alpha=0.7,
                        label="Baseline" if fi == 0 else "")

        for mi, sc in enumerate(aug_scens):
            method = _AUG_SCEN_TO_METHOD.get(sc, sc)
            color  = _AUG_METHOD_COLORS.get(method, "#888888")
            label  = _AUG_METHOD_LABELS.get(method, method)
            offset = (-w_total / 2 + mi * w + w / 2)
            vals = []
            for fam in fams:
                row = metrics[(metrics["scenario"] == sc) & (metrics["family"] == fam)]
                vals.append(float(row[col].iloc[0]) if not row.empty and col in row.columns
                            else float("nan"))
            bars = ax.bar(x + offset, vals, w, color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.5, label=label)
            _bar_labels(ax, bars, fmt="{:.4f}", fs=8)

        ax.set_xticks(x)
        ax.set_xticklabels([_fd(f) for f in fams], fontsize=_TK + 1)
        ax.set_ylabel(col.replace("_", " ").upper(), fontsize=_LF)
        ax.set_title(col.replace("_", " ").title(), fontsize=_LF, pad=8)

    handles, seen = [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); seen.add(l)
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               fontsize=_LG, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F7b_performance_by_method.png"))


def _f7c_cf_fairness_by_method(out_dir):
    """F7c: CF fairness and consistency — bar chart comparing all three augmented methods + baseline."""
    fair_csv = os.path.join(_UNIFIED_RES, "unified_fairness.csv")
    fair = _load(fair_csv, "unified_fairness_f7c")
    if fair.empty:
        print("  [F7c] No unified_fairness.csv found, skipping.")
        return

    fair = _normalize_family(fair)
    ind_cols = [c for c in ["ind_cf_fairness", "ind_consistency"] if c in fair.columns]
    if not ind_cols:
        return

    fams = [f for f in _FAM_ORDER if f in fair["family"].unique()]
    if not fams:
        return

    _SCEN_CF = ["baseline", "augmented_SCM", "augmented_update_labels",
                "augmented_add_comparators", "augmented"]
    scens_present = [sc for sc in _SCEN_CF if sc in fair["scenario"].unique()]

    n_met = len(ind_cols)
    fig, axes = plt.subplots(1, n_met, figsize=(7 * n_met, 7), sharey=False)
    if n_met == 1:
        axes = [axes]
    fig.suptitle("Individual Fairness by Augmentation Method  (COMPAS)\n"
                 "Higher = fairer",
                 fontsize=_TF + 1, fontweight="bold", y=1.02)

    n_scens = len(scens_present)
    w_total = 0.8
    w = w_total / max(n_scens, 1)
    x = np.arange(len(fams))

    for ax, col in zip(axes, ind_cols):
        for si, sc in enumerate(scens_present):
            color = _scenario_color(sc)
            label = _GRP_LABELS.get(_scen_group(sc), _fmt_ldp_label(sc))
            offset = (-w_total / 2 + si * w + w / 2)
            vals = []
            for fam in fams:
                row = fair[(fair["scenario"] == sc) & (fair["family"] == fam)]
                vals.append(float(row[col].iloc[0]) if not row.empty and col in row.columns
                            else float("nan"))
            bars = ax.bar(x + offset, vals, w, color=color, alpha=0.85,
                          edgecolor="white", linewidth=0.5, label=label)
            _bar_labels(ax, bars, fmt="{:.4f}", fs=8)

        ax.axhline(1.0, color="green", ls="--", lw=1.2, alpha=0.6, label="Perfect (1.0)")
        ax.axhline(0.5, color="orange", ls=":", lw=1.0, alpha=0.6, label="Random (0.5)")
        ax.set_ylim(0, 1.12)
        ax.set_xticks(x)
        ax.set_xticklabels([_fd(f) for f in fams], fontsize=_TK + 1)
        ax.set_ylabel(col.replace("ind_", "").replace("_", " ").title(), fontsize=_LF)
        ax.set_title(col.replace("ind_", "").replace("_", " ").title(), fontsize=_LF, pad=8)

    handles, seen = [], set()
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                handles.append(h); seen.add(l)
    fig.legend(handles=handles, loc="lower center", ncol=min(len(handles), 5),
               fontsize=_LG, bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
    plt.tight_layout()
    _save_fig(fig, os.path.join(out_dir, "F7c_cf_fairness_by_method.png"))


def _f7d_ldp_fairness_tradeoff(out_dir):
    """F7d: For each method, line plot of |SPD| vs epsilon, one line per augmentation method.
    Compares how augmentation affects the privacy-fairness tradeoff under LDP."""
    ldp_csv = os.path.join(_UNIFIED_RES, "unified_ldp_sweep.csv")
    if not os.path.exists(ldp_csv):
        # fallback: try main results ldp sweep
        ldp_csv = os.path.join(_MAIN_RES, "ldp_sweep_results.csv")
    ldp = _load(ldp_csv, "ldp_sweep_f7d")
    if ldp.empty or "epsilon" not in ldp.columns:
        print("  [F7d] No LDP sweep data found, skipping.")
        return

    ldp["epsilon"] = ldp["epsilon"].astype(float)
    fams = [f for f in _FAM_ORDER if f in ldp["family"].unique()]
    if not fams:
        return

    # Determine which scenario column to use for augmentation method
    aug_col = None
    if "augmentation_method" in ldp.columns:
        aug_col = "augmentation_method"
    elif "scenario" in ldp.columns:
        # scenario names like ldp_augmented_eps0.5 — can't differentiate methods here
        # Just plot the data we have
        pass

    spd_col = "group_SPD" if "group_SPD" in ldp.columns else \
              "SPD"       if "SPD"       in ldp.columns else None
    if spd_col is None:
        return

    eps_vals = sorted(ldp["epsilon"].unique())
    n_fams = len(fams)
    fig, axes = plt.subplots(1, n_fams, figsize=(7 * n_fams, 6), sharey=True)
    if n_fams == 1:
        axes = [axes]
    fig.suptitle("LDP Privacy-Fairness Tradeoff by Augmentation Method  (COMPAS)\n"
                 "Lower |SPD| = fairer; lower epsilon = more privacy noise",
                 fontsize=_TF + 1, fontweight="bold", y=1.02)

    if aug_col:
        aug_methods = ldp[aug_col].unique().tolist()
        method_colors = {m: list(_AUG_METHOD_COLORS.values())[i % 3]
                         for i, m in enumerate(aug_methods)}
    else:
        aug_methods = None

    for ax, fam in zip(axes, fams):
        sub_fam = ldp[ldp["family"] == fam].copy()
        if sub_fam.empty:
            continue

        if aug_methods and aug_col:
            for method in aug_methods:
                sub_m = sub_fam[sub_fam[aug_col] == method].sort_values("epsilon")
                if sub_m.empty:
                    continue
                yv = sub_m[spd_col].abs()
                color = method_colors.get(method, "#888888")
                ax.plot(sub_m["epsilon"], yv, marker="o", ms=7, lw=2,
                        color=color, label=method)
                for ex, ey in zip(sub_m["epsilon"], yv):
                    ax.annotate("{:.3f}".format(float(ey)), (ex, float(ey)),
                                textcoords="offset points", xytext=(4, 4), fontsize=8)
        else:
            # No method distinction — plot all LDP scenarios
            if "scenario" in sub_fam.columns:
                for sc in sorted(sub_fam["scenario"].unique()):
                    row = sub_fam[sub_fam["scenario"] == sc].sort_values("epsilon")
                    if row.empty:
                        continue
                    yv = row[spd_col].abs()
                    ax.plot(row["epsilon"], yv, marker="o", ms=7, lw=2,
                            color=_scenario_color(sc), label=_fmt_ldp_label(sc))
            else:
                sub_fam_s = sub_fam.sort_values("epsilon")
                yv = sub_fam_s[spd_col].abs()
                ax.plot(sub_fam_s["epsilon"], yv, marker="o", ms=7, lw=2,
                        color=_fc(fam), label=_fd(fam))

        ax.set_xscale("log")
        ax.set_xlim(eps_vals[0] * 0.65, eps_vals[-1] * 1.55)
        ax.set_xticks(eps_vals)
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(e) for e in eps_vals]))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel("LDP ε  (lower = more noise)", fontsize=_LF)
        ax.set_ylabel("|SPD|  (lower = fairer)", fontsize=_LF)
        ax.set_title("{} base estimator".format(_fd(fam)), fontsize=_LF, fontweight="bold")
        ax.legend(fontsize=_LG - 1, bbox_to_anchor=(1.01, 1.0), loc="upper left",
                  framealpha=0.9)

    plt.tight_layout(rect=[0, 0, 0.88, 1])
    _save_fig(fig, os.path.join(out_dir, "F7d_ldp_fairness_tradeoff.png"))


def generate_folder7(out_dir=_DIR7):
    os.makedirs(out_dir, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 7: Augmentation Methods Comparison\n" + "="*60)
    _f7a_fairness_by_method(out_dir)
    _f7b_performance_by_method(out_dir)
    _f7c_cf_fairness_by_method(out_dir)
    _f7d_ldp_fairness_tradeoff(out_dir)
    print("  [Folder 7] Done ->", out_dir)


# ===========================================================================
# Entry point
# ===========================================================================

def generate_all_structured_visualizations(output_root=_VIZ_ROOT):
    global _DIR1, _DIR2, _DIR3, _DIR4, _DIR5, _DIR6, _DIR7, _UNIFIED_RES
    _DIR1        = os.path.join(output_root, "1_baseline_vs_augmented")
    _DIR2        = os.path.join(output_root, "2_augmented_vs_fairlearn")
    _DIR3        = os.path.join(output_root, "3_ldp_fairness")
    _DIR4        = os.path.join(output_root, "4_mia_methods")
    _DIR5        = os.path.join(output_root, "5_unified")
    _DIR6        = os.path.join(output_root, "6_model_performance")
    _DIR7        = os.path.join(output_root, "7_augmentation_methods")
    _UNIFIED_RES = os.path.join(_BASE_DIR, "pipeline_outputs", "unified", "results")

    # For folders 1-4: if unified fairness CSV exists, prefer it over per-method CSV
    # so that all augmentation methods are visible in the same plots.
    unified_fair_csv = os.path.join(_UNIFIED_RES, "unified_fairness.csv")
    unified_mia_csv  = os.path.join(_UNIFIED_RES, "unified_mia.csv")
    unified_nice_csv = os.path.join(_UNIFIED_RES, "unified_nice_quality.csv")

    os.makedirs(output_root, exist_ok=True)
    print("\n" + "="*60 + "\n  STRUCTURED VISUALIZATIONS\n"
          + "  Root: {}\n".format(output_root) + "="*60)

    # Folders 1-4: try unified CSVs first, fall back to per-method
    def _fair_for_f1():
        if os.path.exists(unified_fair_csv):
            print("  [generate] Using unified_fairness.csv for Folder 1")
            return _load(unified_fair_csv, "unified_fairness")
        return _load(os.path.join(_MAIN_RES, "fairness_results.csv"), "fairness")

    def _mia_for_f1():
        if os.path.exists(unified_mia_csv):
            print("  [generate] Using unified_mia.csv for Folder 1")
            return _load(unified_mia_csv, "unified_mia")
        return _load(os.path.join(_MAIN_RES, "mia_results.csv"), "mia")

    def _nice_for_f1():
        if os.path.exists(unified_nice_csv):
            print("  [generate] Using unified_nice_quality.csv for Folder 1")
            return _load(unified_nice_csv, "unified_nice")
        return _load(os.path.join(_MAIN_RES, "nice_cf_quality.csv"), "nice")

    # Folder 1 — pass pre-loaded data if unified CSVs exist
    os.makedirs(_DIR1, exist_ok=True)
    print("\n" + "="*60 + "\n  FOLDER 1: Baseline vs. Augmented\n" + "="*60)
    fair1 = _fair_for_f1()
    mia1  = _mia_for_f1()
    nice1 = _nice_for_f1()
    if not fair1.empty:
        _viz1_fairness(fair1, _DIR1)
    if not mia1.empty:
        _viz1_mia(mia1, _DIR1)
    if not nice1.empty:
        _viz1_nice(nice1, _DIR1)
    if not fair1.empty:
        _viz_cf_fairness_baseline_vs_aug(fair1, _DIR1)
    # F1d
    if not mia1.empty:
        _AUG_F1DGEN = ["augmented_SCM", "augmented_update_labels",
                       "augmented_add_comparators", "augmented"]
        _SCEN_F1DGEN = ["baseline"] + _AUG_F1DGEN
        scens_f1dgen = [sc for sc in _SCEN_F1DGEN if sc in mia1["scenario"].unique()]
        sub_f1d = mia1[mia1["scenario"].isin(scens_f1dgen)].copy()
        if "input_type" in sub_f1d.columns:
            nc = sub_f1d[sub_f1d["input_type"] == "nice_cf"]
            sub_f1d = nc if not nc.empty else sub_f1d
        avail_mgen = [(c, t, r, d) for c, t, r, d in _MIA_METRICS if c in sub_f1d.columns]
        if avail_mgen:
            fams_f1d = [f for f in _FAM_ORDER if f in sub_f1d["family"].unique()]
            n_f1d = len(avail_mgen)
            fig, axes_f1d = plt.subplots(1, n_f1d, figsize=(5.5 * n_f1d, 6.5))
            if n_f1d == 1: axes_f1d = [axes_f1d]
            fig.suptitle("MIA Multi-Metric: Baseline vs. Augmented Methods\n"
                         "(NiCE-CF proxy — best attacker per model)",
                         fontsize=_TF, fontweight="bold", y=1.02)
            n_sc_f1d = len(scens_f1dgen)
            w_tot_gen = 0.8
            w_gen = w_tot_gen / max(n_sc_f1d, 1)
            xg = np.arange(len(fams_f1d))
            for axg, (col, panel_t, ref_val, note) in zip(axes_f1d, avail_mgen):
                for ig, scen in enumerate(scens_f1dgen):
                    ssub = sub_f1d[sub_f1d["scenario"] == scen]
                    vals_g = [float(ssub[ssub["family"] == f][col].max())
                               if not ssub[ssub["family"] == f].empty else float("nan")
                               for f in fams_f1d]
                    hatch_g = _HATCH_AUG if scen != "baseline" else _HATCH_BASE
                    offset_g = (-w_tot_gen / 2 + ig * w_gen + w_gen / 2)
                    bars_g = axg.bar(xg + offset_g, vals_g, w_gen,
                                     color=[_fc(f) for f in fams_f1d], alpha=0.85,
                                     hatch=hatch_g,
                                     edgecolor="black" if hatch_g else "white",
                                     linewidth=0.5)
                    _bar_labels(axg, bars_g, fmt="{:.4f}", fs=7.5)
                if ref_val is not None:
                    axg.axhline(ref_val, color="red", ls="--", lw=1.5,
                                label="Ref. ({})".format(ref_val))
                    axg.legend(fontsize=_LG - 1, bbox_to_anchor=(1.01, 1.0),
                               loc="upper left", framealpha=0.9)
                axg.set_title("{}\n{}".format(panel_t, note), fontsize=_LF - 1, pad=6)
                axg.set_xticks(xg)
                axg.set_xticklabels([_fd(f) for f in fams_f1d], fontsize=_TK + 1)
                axg.set_ylabel(panel_t, fontsize=_LF)
            els_g = ([mpatches.Patch(facecolor=_scenario_color(sc), edgecolor="black",
                                     label=_GRP_LABELS.get(_scen_group(sc), sc))
                      for sc in scens_f1dgen] +
                     [mpatches.Patch(facecolor=_fc(f), label=_fd(f)) for f in fams_f1d])
            fig.legend(handles=els_g, loc="lower center", ncol=len(els_g), fontsize=_LG,
                       bbox_to_anchor=(0.5, -0.04), framealpha=0.9)
            plt.tight_layout()
            _save_fig(fig, os.path.join(_DIR1, "F1d_mia_multi_metrics.png"))
    print("  [Folder 1] Done ->", _DIR1)

    generate_folder2(_DIR2)
    generate_folder3(_DIR3)
    generate_folder4(_DIR4)
    generate_folder5(_DIR5)
    generate_folder6(_DIR6)
    generate_folder7(_DIR7)
    print("\n" + "="*60 + "\n  All done.\n" + "="*60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--output-root", default=_VIZ_ROOT)
    p.add_argument("--folder", choices=["1","2","3","4","5","6","7","all"], default="all")
    a = p.parse_args()
    r = a.output_root
    if a.folder in ("all","1"): generate_folder1(os.path.join(r,"1_baseline_vs_augmented"))
    if a.folder in ("all","2"): generate_folder2(os.path.join(r,"2_augmented_vs_fairlearn"))
    if a.folder in ("all","3"): generate_folder3(os.path.join(r,"3_ldp_fairness"))
    if a.folder in ("all","4"): generate_folder4(os.path.join(r,"4_mia_methods"))
    if a.folder in ("all","5"): generate_folder5(os.path.join(r,"5_unified"))
    if a.folder in ("all","6"): generate_folder6(os.path.join(r,"6_model_performance"))
    if a.folder in ("all","7"): generate_folder7(os.path.join(r,"7_augmentation_methods"))
