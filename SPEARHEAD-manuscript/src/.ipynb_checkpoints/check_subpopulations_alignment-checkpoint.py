import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from beeswarm import summary_legacy as summary_plot_mod

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib
import re
import os

from typing import Iterable, Optional, List

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    brier_score_loss
)
from sklearn.metrics import (
    roc_curve, auc, f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, precision_recall_curve, average_precision_score, log_loss,
)
from sklearn.calibration import calibration_curve

import matplotlib

from data_processing import pipeline_func_UKBB
from ut import *
import sklearn
print(sklearn.__version__)
print("Matplotlib version:", matplotlib.__version__)


# -----------------------------
is_ukbb = False
model_to_train = "RF"
# -----------------------------

TARGETS_CHOSEN = [
    "urine_antibiogram_amoxicillin___clavulansäure",
    "urine_antibiogram_cefuroxim", # too much imbalance for e coli subset
    "urine_antibiogram_cotrimoxazol",  # Trimethoprim/Sulfamethoxazo (TMP/SMX)
    "urine_antibiogram_fosfomycin_trometamol",
    "urine_antibiogram_nitrofurantoin",
    "urine_antibiogram_norfloxacin",
    "urine_antibiogram_ciprofloxacin",
    "urine_antibiogram_ceftriaxon",
    "urine_antibiogram_piperacillin___tazobactam",
]

# TARGETS_CHOSEN = [
#     "urine_antibiogram_amoxicillin___clavulansäure",
#     "urine_antibiogram_nitrofurantoin",
#     "urine_antibiogram_ciprofloxacin",
# ]



# ---------- Data prep ----------

def generate_dt_roc(preds_df: pd.DataFrame | pd.Series) -> pd.DataFrame:
    """
    Melt a predictions dataframe to long format for plotting.

    Expected columns in `preds_df`:
      id_vars:  "target", "true_class", "repetition", "model", "indices"
      plus one or more prediction columns (e.g., "pred" or "pred_RF", "pred_XGB"...)

    Returns long frame with:
      ["target", "true_class", "repetition", "model", "indices", "Model", "Pred"]
    """
    df = pd.DataFrame(preds_df).copy()
    df = df.drop(columns="fold")
    id_vars = ["target", "true_class", "repetition", "model", "indices"]
    for col in id_vars:
        if col not in df.columns:
            raise KeyError(f"Missing required column in preds_df: {col}")

    value_vars = [c for c in df.columns if c not in id_vars]
    if not value_vars:
        # common single-model layout is just a "pred" column
        raise ValueError("No prediction columns found (columns other than id_vars).")

    melted = pd.melt(
        df, id_vars=id_vars, value_vars=value_vars,
        var_name="Model", value_name="Pred"
    )

    # coerce & tidy
    melted["true_class"] = pd.to_numeric(melted["true_class"], errors="coerce").fillna(0).astype(int)
    melted["Pred"] = pd.to_numeric(melted["Pred"], errors="coerce")
    # Make single-model label nice (strip 'pred' if that's the only column)
    if set(value_vars) == {"pred"}:
        melted["Model"] = "Model"
    else:
        melted["Model"] = melted["Model"].astype(str)

    return melted


# ---------- Plot helpers (one target at a time) ----------

def _short_name(name: str, maxlen: int = 28) -> str:
    s = str(name)
    return s if len(s) <= maxlen else "..." + s[-(maxlen-1):]


def overview_internal_plot_roc(dt_melted: pd.DataFrame,
                               ax_list: List[plt.Axes],
                               idx: int,
                               tar: str) -> List[float]:
    """Plot ROC curves for a single target on ax_list[idx], one curve per repetition."""
    ax = ax_list[idx]
    ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=2, label="No Skill")

    df = dt_melted.loc[dt_melted["target"] == tar].copy()
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return []

    scores = []
    # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for repetitions
    colors = sns.color_palette("mako", n_colors=5)
    
    for (model, repetition), g in df.groupby(["Model", "repetition"]):
        
        y = g["true_class"].to_numpy(dtype=int)
        p = g["Pred"].to_numpy(dtype=float)
        if y.size == 0 or y.min() == y.max():
            log(ERROR, f"y size, y min., y max.: {y.size, y.min(), y.max()}")
            log(INFO, "Due to error above, not calculating auroc")
            log(INFO, "Skipping iteration...")
            continue
        auroc = roc_auc_score(y, p)
        
        fpr, tpr, _ = roc_curve(y, p)
        scores.append(auroc)
        
        # # Use different colors for different repetitions
        # color_idx = int(repetition) % len(colors)
        # rep_num = int(df_group['repetition'].mean())
        color = colors[int(repetition) % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"Run {repetition}", alpha=0.7, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate" if idx % 2 == 0 else "")
    ax.set_xticklabels([])
    # ax.set_aspect("equal")
    return scores


def overview_internal_plot_pr(dt_melted: pd.DataFrame,
                              ax_list: List[plt.Axes],
                              idx: int,
                              tar: str) -> List[float]:
    """Plot PR curves for a single target on ax_list[idx], one curve per repetition."""
    ax = ax_list[idx]
    df = dt_melted.loc[dt_melted["target"] == tar].copy()
    if df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return []

    y_all = df["true_class"].to_numpy(dtype=int)
    pos_rate = float((y_all == 1).mean()) if y_all.size else 0.0
    ax.plot([0, 1], [pos_rate, pos_rate], ls="--", color="gray", lw=2, label="No Skill")

    scores = []
    # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for repetitions
    colors = sns.color_palette("mako", n_colors=5)
    
    for (model, repetition), g in df.groupby(["Model", "repetition"]):
        y = g["true_class"].to_numpy(dtype=int)
        p = g["Pred"].to_numpy(dtype=float)
        if y.size == 0 or y.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        scores.append(auc(rec, prec))
        
        # Use different colors for different repetitions
        color = colors[int(repetition) % len(colors)]
        ax.plot(rec, prec, lw=2, color=color, 
                label=f"Run {repetition}", alpha=0.7, zorder=3)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision" if idx % 2 == 0 else "")
    ax.set_xticklabels([])
    # ax.set_aspect("equal")
    return scores


def overview_internal_plot_calib(dt_melted: pd.DataFrame,
                                 ax_pairs: List[tuple[plt.Axes, plt.Axes]],
                                 idx: int,
                                 tar: str,
                                 n_bins: int = 10,
                                 strategy: str = "quantile") -> float | None:
    """
    Plot Calibration (reliability) curve + probability density for a single target,
    one curve per repetition.
    Returns average Brier score (float) or None if not plottable.
    """
    ax_main, ax_den = ax_pairs[idx]

    df = dt_melted.loc[dt_melted["target"] == tar].copy()
    if df.empty:
        ax_main.text(0.5, 0.5, "No data", ha="center", va="center")
        ax_den.text(0.5, 0.5, "No data", ha="center", va="center")
        return None

    brier_vals = []
    any_curve = False
    # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for repetitions
    colors = sns.color_palette("mako", n_colors=5)

    

    for (model, repetition), g in df.groupby(["Model", "repetition"]):
        y = g["true_class"].to_numpy(dtype=int)
        p = g["Pred"].to_numpy(dtype=float)

        if y.size == 0:
            continue

        # Brier score (lower is better)
        bs = brier_score_loss(y, p)
        brier_vals.append(bs)

        # Calibration curve
        try:
            prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
        except Exception:
            prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")

        if not any_curve:  # Draw reference line only once
            ax_main.plot([0, 1], [0, 1], ls="--", color="gray", lw=2)
            
        color = colors[int(repetition) % len(colors)]
        ax_main.plot(prob_pred, prob_true, marker="o", lw=2, 
                    color=color,
                    label=f"Run {repetition}", 
                    alpha=0.7)
        any_curve = True

        # Histogram for each repetition (will overlay)
        ax_den.hist(p, bins=40, range=(0, 1), stacked=True, density=True, alpha=0.5, 
                   color=color, edgecolor="white", linewidth=0.3)

    if not any_curve:
        ax_main.text(0.5, 0.5, "Insufficient classes", ha="center", va="center")
        ax_den.text(0.5, 0.5, "Insufficient classes", ha="center", va="center")
        return None

    ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
    ax_main.set_xlabel("Mean predicted probability")
    ax_main.set_ylabel("Fraction of positives" if idx % 2 == 0 else "")
    ax_main.set_xticklabels([])
    # ax_main.set_aspect("equal")

    ax_den.set_xlim(0, 1)
    ax_den.set_xlabel("Predicted probability")
    ax_den.set_ylabel("Density" if idx % 2 == 0 else "")
    # Lighter grid
    for ax in (ax_main, ax_den):
        ax.grid(True, alpha=0.25, linestyle="--")

    # Return average Brier if more than one repetition
    return brier_vals if brier_vals else None


# ---------- Main overview (multiple targets) ----------
def overview(dt_preds: pd.DataFrame,
             targets: Optional[Iterable[str]] = None,
             ncols: int = 3,
             filename: str = "compact_overview_RF.png",
             calib_bins: int = 10,
             calib_strategy: str = "quantile") -> None:
    """
    Plot a compact overview with ROC, PR, Calibration, and Probability Density
    for each target (columns). If targets exceed ncols, wrap to additional rows.
    """
    sns.set_theme(style="white", font_scale=1.5)
    print("Plotting and saving...")

    dt_melted = generate_dt_roc(dt_preds)

    all_targets = list(pd.unique(dt_melted["target"]))

    print(all_targets)
    if targets is None:
        targets = all_targets
    else:
        targets = [t for t in targets if t in all_targets]
        if not targets:
            raise ValueError("None of the requested targets are present in dt_preds.")

    n_t = len(targets)
    ncols = max(1, int(ncols))
    
    # Calculate number of rows needed (each target needs 2 plot rows: ROC + PR)
    nrows_targets = int(np.ceil(n_t / ncols))  # How many target rows we need
    total_plot_rows = nrows_targets * 2  # Each target row has 2 plot types (ROC, PR)
    
    plot_size = 7
    fig_w = ncols * plot_size
    fig_h = nrows_targets * 2.3 * plot_size * 0.65 + 0.8

    # Create height ratios: alternating [1, 1] for each target row
    height_ratios = [1, 1] * nrows_targets
    
    # Create subplots with custom spacing
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        total_plot_rows, ncols,
        height_ratios=height_ratios,
        hspace=0.3,
        wspace=0.25
    )
    
    # Create axes from gridspec
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ncols)] 
                     for i in range(total_plot_rows)])
    
    plt.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.95)

    # Add colored band for alternating target rows (only blue)
    from matplotlib.patches import Rectangle
    
    for target_row in range(nrows_targets):
        # Only color every other row (0, 2, 4, ...) in blue
        if target_row % 2 != 0:
            roc_row = target_row * 2
            pr_row = target_row * 2 + 1
            
            # Get the position of subplots to calculate band position
            ax_top = axes[roc_row, 0]
            ax_bottom = axes[pr_row, 0]
            
            # Get positions in figure coordinates
            bbox_top = ax_top.get_position()
            bbox_bottom = ax_bottom.get_position()
            
            # Extend the band to include titles and labels
            # Add padding above (for titles) and below (for x-labels)
            padding_top = 0.015  # Extra space for titles
            padding_bottom = 0.015  # Extra space for x-labels
            
            # Create a rectangle spanning the entire width and both rows + padding
            rect = Rectangle(
                (0, bbox_bottom.y0 - padding_bottom),  # x, y (bottom-left corner, extended down)
                1.05,  # width (full figure width)
                (bbox_top.y1 - bbox_bottom.y0) + padding_top + padding_bottom,  # height (span both rows + padding)
                transform=fig.transFigure,
                facecolor='#E3F2FD',  # Light blue
                edgecolor='none',
                zorder=-1,  # Behind everything
                alpha=0.5  # Slightly transparent
            )
            fig.patches.append(rect)

    for target_row in range(nrows_targets):
        # Only color every other row (0, 2, 4, ...) in blue
        if target_row % 2 != 0:
            roc_row = target_row * 2
            pr_row = target_row * 2 + 1
        
        # Add background to both ROC and PR rows for this target group
        for col in range(ncols):
            # Background for ROC
            # axes[roc_row, col].set_facecolor('#E3F2FD')
            axes[roc_row, col].patch.set_alpha(0)
            # Background for PR
            # axes[pr_row, col].set_facecolor('#E3F2FD')
            axes[pr_row, col].patch.set_alpha(0)

    handles_accum, labels_accum = [], []

    for i, tar in enumerate(targets):
        print(tar)
        # Calculate which subplot grid position this target goes to
        target_row = i // ncols  # Which row of targets (0, 1, 2...)
        target_col = i % ncols   # Which column (0, 1, 2)
        
        # Calculate the actual row indices in the subplot grid
        roc_row = target_row * 2      # ROC is at even rows
        pr_row = target_row * 2 + 1   # PR is at odd rows
        
        ax_roc = axes[roc_row, target_col]
        ax_pr = axes[pr_row, target_col]
        
        # Format title
        import re
        title = re.sub(' +', ' ', " ".join(tar.split("_")[2:]).capitalize())
        ax_roc.set_title(title, fontweight="bold", fontsize=20, pad=8)

        scores_roc = overview_internal_plot_roc(dt_melted, [ax_roc], 0, tar)
        scores_pr = overview_internal_plot_pr(dt_melted, [ax_pr], 0, tar)

        print(f"AUROC: {np.mean(scores_roc):.3f} ({np.std(scores_roc):.3f})")
        print(f"AUPR: {np.mean(scores_pr):.3f} ({np.std(scores_pr):.3f})")

        if scores_roc:
            ax_roc.text(
                0.97, 0.03, f"AUROC: {np.mean(scores_roc):.3f} ({np.std(scores_roc):.3f})",
                color="k", bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
                fontsize=14, ha="right", va="bottom", transform=ax_roc.transAxes
            )
        if scores_pr:
            ax_pr.text(
                0.97, 0.03, f"AUPR: {np.mean(scores_pr):.3f} ({np.std(scores_pr):.3f})",
                color="k", bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
                fontsize=14, ha="right", va="bottom", transform=ax_pr.transAxes
            )

        # Collect legend from first ROC plot only
        if i == 0:
            h, l = ax_roc.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll != "No Skill" and ll not in labels_accum:
                    labels_accum.append(ll)
                    handles_accum.append(hh)

        # Y-labels: only left-most column of each row
        if target_col != 0:
            ax_roc.set_ylabel("")
            ax_pr.set_ylabel("")

        # Add grid
        ax_roc.grid(True, axis='both', linestyle=":", linewidth=1, alpha=0.6)
        ax_pr.grid(True, axis='both', linestyle=":", linewidth=1, alpha=0.6)
        print()

    # Hide unused subplots if targets don't fill the grid
    total_positions = nrows_targets * ncols
    for i in range(n_t, total_positions):
        target_row = i // ncols
        target_col = i % ncols
        roc_row = target_row * 2
        pr_row = target_row * 2 + 1
        axes[roc_row, target_col].axis('off')
        axes[pr_row, target_col].axis('off')

    # Get legend handles and labels from first plot
    handles_roc, labels_roc = axes[0, 0].get_legend_handles_labels()
    
    # Remove "No Skill" from legend
    if "No Skill" in labels_roc:
        no_skill_idx = labels_roc.index("No Skill")
        handles_roc.pop(no_skill_idx)
        labels_roc.pop(no_skill_idx)
        
    fig.legend(
        handles_roc,
        labels_roc,
        loc="center left",
        bbox_to_anchor=(0.98, 0.5),
        title="Legend",
        ncol=1,
        fontsize=16
    )

    plt.suptitle(f"{filename.split('_')[-1][:-4].capitalize()} subpopulation", fontsize=28)

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {filename}")
 


# def generate_dt_roc(preds_df: pd.DataFrame | pd.Series) -> pd.DataFrame:
#     """
#     Melt a predictions dataframe to long format for plotting.

#     Expected columns in `preds_df`:
#       id_vars:  "target", "true_class", "fold", "model", "indices"
#       plus one or more prediction columns (e.g., "pred" or "pred_RF", "pred_XGB"...)

#     Returns long frame with:
#       ["target", "true_class", "fold", "model", "indices", "Model", "Pred"]
#     """
#     df = pd.DataFrame(preds_df).copy()
#     df = df.drop(columns="repetition")
#     id_vars = ["target", "true_class", "fold", "model", "indices"]
#     for col in id_vars:
#         if col not in df.columns:
#             raise KeyError(f"Missing required column in preds_df: {col}")

#     value_vars = [c for c in df.columns if c not in id_vars]
#     if not value_vars:
#         # common single-model layout is just a "pred" column
#         raise ValueError("No prediction columns found (columns other than id_vars).")

#     melted = pd.melt(
#         df, id_vars=id_vars, value_vars=value_vars,
#         var_name="Model", value_name="Pred"
#     )

#     melted["true_class"] = pd.to_numeric(melted["true_class"], errors="coerce").fillna(0).astype(int)
#     melted["Pred"] = pd.to_numeric(melted["Pred"], errors="coerce")
#     # Make single-model label nice (strip 'pred' if that's the only column)
#     if set(value_vars) == {"pred"}:
#         melted["Model"] = "Model"
#     else:
#         melted["Model"] = melted["Model"].astype(str)

#     return melted


# # ---------- Plot helpers (one target at a time) ----------

# def _short_name(name: str, maxlen: int = 28) -> str:
#     s = str(name)
#     return s if len(s) <= maxlen else "..." + s[-(maxlen-1):]


# def overview_internal_plot_roc(dt_melted: pd.DataFrame,
#                                ax_list: List[plt.Axes],
#                                idx: int,
#                                tar: str) -> List[float]:
#     """Plot ROC curves for a single target on ax_list[idx], one curve per fold."""
#     ax = ax_list[idx]
#     ax.plot([0, 1], [0, 1], ls="--", color="gray", lw=2, label="No Skill")

#     df = dt_melted.loc[dt_melted["target"] == tar].copy()
#     if df.empty:
#         ax.text(0.5, 0.5, "No data", ha="center", va="center")
#         return []

#     scores = []
#     # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for folds
#     colors = sns.color_palette("mako", n_colors=5)
    
#     for (model, fold), g in df.groupby(["Model", "fold"]):
        
#         y = g["true_class"].to_numpy(dtype=int)
#         p = g["Pred"].to_numpy(dtype=float)
#         if y.size == 0 or y.min() == y.max():
#             log(ERROR, f"y size, y min., y max.: {y.size, y.min(), y.max()}")
#             log(INFO, "Due to error above, not calculating auroc")
#             log(INFO, "Skipping iteration...")
#             continue
#         auroc = roc_auc_score(y, p)
        
#         fpr, tpr, _ = roc_curve(y, p)
#         scores.append(auroc)
        
#         # # Use different colors for different folds
#         # color_idx = int(fold) % len(colors)
#         # rep_num = int(df_group['fold'].mean())
#         color = colors[int(fold) % len(colors)]
#         ax.plot(fpr, tpr, lw=2, color=color, label=f"Run {fold}", alpha=0.7)
#     ax.set_xlim(0, 1); ax.set_ylim(0, 1)
#     ax.set_xlabel("False Positive Rate")
#     ax.set_ylabel("True Positive Rate" if idx % 2 == 0 else "")
#     ax.set_xticklabels([])
#     # ax.set_aspect("equal")
#     return scores


# def overview_internal_plot_pr(dt_melted: pd.DataFrame,
#                               ax_list: List[plt.Axes],
#                               idx: int,
#                               tar: str) -> List[float]:
#     """Plot PR curves for a single target on ax_list[idx], one curve per fold."""
#     ax = ax_list[idx]
#     df = dt_melted.loc[dt_melted["target"] == tar].copy()
#     if df.empty:
#         ax.text(0.5, 0.5, "No data", ha="center", va="center")
#         return []

#     y_all = df["true_class"].to_numpy(dtype=int)
#     pos_rate = float((y_all == 1).mean()) if y_all.size else 0.0
#     ax.plot([0, 1], [pos_rate, pos_rate], ls="--", color="gray", lw=2, label="No Skill")

#     scores = []
#     # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for folds
#     colors = sns.color_palette("mako", n_colors=5)
    
#     for (model, fold), g in df.groupby(["Model", "fold"]):
#         y = g["true_class"].to_numpy(dtype=int)
#         p = g["Pred"].to_numpy(dtype=float)
#         if y.size == 0 or y.sum() == 0:
#             continue
#         prec, rec, _ = precision_recall_curve(y, p)
#         scores.append(auc(rec, prec))
        
#         # Use different colors for different folds
#         color = colors[int(fold) % len(colors)]
#         ax.plot(rec, prec, lw=2, color=color, 
#                 label=f"Run {fold}", alpha=0.7)

#     ax.set_xlim(0, 1); ax.set_ylim(0, 1)
#     ax.set_xlabel("Recall")
#     ax.set_ylabel("Precision" if idx % 2 == 0 else "")
#     ax.set_xticklabels([])
#     # ax.set_aspect("equal")
#     return scores


# def overview_internal_plot_calib(dt_melted: pd.DataFrame,
#                                  ax_pairs: List[tuple[plt.Axes, plt.Axes]],
#                                  idx: int,
#                                  tar: str,
#                                  n_bins: int = 10,
#                                  strategy: str = "quantile") -> float | None:
#     """
#     Plot Calibration (reliability) curve + probability density for a single target,
#     one curve per fold.
#     Returns average Brier score (float) or None if not plottable.
#     """
#     ax_main, ax_den = ax_pairs[idx]

#     df = dt_melted.loc[dt_melted["target"] == tar].copy()
#     if df.empty:
#         ax_main.text(0.5, 0.5, "No data", ha="center", va="center")
#         ax_den.text(0.5, 0.5, "No data", ha="center", va="center")
#         return None

#     brier_vals = []
#     any_curve = False
#     # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for folds
#     colors = sns.color_palette("mako", n_colors=5)

    

#     for (model, fold), g in df.groupby(["Model", "fold"]):
#         y = g["true_class"].to_numpy(dtype=int)
#         p = g["Pred"].to_numpy(dtype=float)

#         if y.size == 0:
#             continue

#         # Brier score (lower is better)
#         bs = brier_score_loss(y, p)
#         brier_vals.append(bs)

#         # Calibration curve
#         try:
#             prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
#         except Exception:
#             prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")

#         if not any_curve:  # Draw reference line only once
#             ax_main.plot([0, 1], [0, 1], ls="--", color="gray", lw=2)
            
#         color = colors[int(fold) % len(colors)]
#         ax_main.plot(prob_pred, prob_true, marker="o", lw=2, 
#                     color=color,
#                     label=f"Run {fold}", 
#                     alpha=0.7)
#         any_curve = True

#         # Histogram for each fold (will overlay)
#         ax_den.hist(p, bins=40, range=(0, 1), stacked=True, density=True, alpha=0.5, 
#                    color=color, edgecolor="white", linewidth=0.3)

#     if not any_curve:
#         ax_main.text(0.5, 0.5, "Insufficient classes", ha="center", va="center")
#         ax_den.text(0.5, 0.5, "Insufficient classes", ha="center", va="center")
#         return None

#     ax_main.set_xlim(0, 1); ax_main.set_ylim(0, 1)
#     ax_main.set_xlabel("Mean predicted probability")
#     ax_main.set_ylabel("Fraction of positives" if idx % 2 == 0 else "")
#     ax_main.set_xticklabels([])
#     # ax_main.set_aspect("equal")

#     ax_den.set_xlim(0, 1)
#     ax_den.set_xlabel("Predicted probability")
#     ax_den.set_ylabel("Density" if idx % 2 == 0 else "")
#     # Lighter grid
#     for ax in (ax_main, ax_den):
#         ax.grid(True, alpha=0.25, linestyle="--")

#     # Return average Brier if more than one fold
#     return brier_vals if brier_vals else None


# def plot_debug(dt_melted: pd.DataFrame, tar: str):
#     """Debug plot to identify the diagonal line issue."""
#     df = dt_melted.loc[dt_melted["target"] == tar].copy()
    
#     fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#     ax_roc, ax_pr = axes
    
#     print(f"Target: {tar}")
#     print(f"Total rows: {len(df)}")
#     print(f"Unique (Model, fold) combinations: {df.groupby(['Model', 'fold']).ngroups}")
    
#     for (model, fold), g in df.groupby(["Model", "fold"]):
#         y = g["true_class"].to_numpy(dtype=int)
#         p = g["Pred"].to_numpy(dtype=float)
        
#         print(f"\n--- Model: {model}, Fold: {fold} ---")
#         print(f"  Samples: {len(y)}")
#         print(f"  y unique: {np.unique(y)}")
#         print(f"  p range: [{p.min():.4f}, {p.max():.4f}]")
#         print(f"  p has NaN: {np.any(np.isnan(p))}")
#         print(f"  Duplicate indices in group: {g.index.duplicated().sum()}")
        
#         # ROC
#         fpr, tpr, _ = roc_curve(y, p)
#         print(f"  ROC points: {len(fpr)}")
#         ax_roc.plot(fpr, tpr, marker='.', markersize=3, label=f"{model}-{fold}")
        
#         # PR - WITHOUT reversal
#         prec, rec, _ = precision_recall_curve(y, p)
#         print(f"  PR points: {len(prec)}")
#         ax_pr.plot(rec, prec, marker='.', markersize=3, label=f"{model}-{fold}")
    
#     ax_roc.plot([0, 1], [0, 1], 'k--', alpha=0.5)
#     ax_roc.set_title("ROC")
#     ax_roc.set_xlabel("FPR")
#     ax_roc.set_ylabel("TPR")
#     ax_roc.legend(fontsize=8)
    
#     ax_pr.set_title("PR")
#     ax_pr.set_xlabel("Recall")
#     ax_pr.set_ylabel("Precision")
#     ax_pr.legend(fontsize=8)
    
#     plt.tight_layout()
#     plt.savefig("figures/RF_figures/men/compact_overview_RF_men_DEBUG.png")
# # ---------- Main overview (multiple targets) ----------

# def overview(dt_preds: pd.DataFrame,
#              targets: Optional[Iterable[str]] = None,
#              ncols: int = 2,
#              filename: str = "compact_overview_RF.png",
#              calib_bins: int = 10,
#              calib_strategy: str = "quantile") -> None:
#     """
#     Plot a compact overview with ROC, PR, Calibration, and Probability Density
#     for each target (columns).
#     """
#     sns.set_theme(style="white", font_scale=1.5)
#     print("Plotting and saving...")

#     dt_melted = generate_dt_roc(dt_preds)

#     all_targets = list(pd.unique(dt_melted["target"]))

#     print(all_targets)
#     if targets is None:
#         targets = all_targets
#     else:
#         targets = [t for t in targets if t in all_targets]
#         if not targets:
#             raise ValueError("None of the requested targets are present in dt_preds.")

#     n_t = len(targets)
#     ncols = max(1, int(ncols))
    
#     plot_size = 7
#     fig_w = ncols * plot_size
#     fig_h = 4 * plot_size * 0.65 + 0.8

#     # Create subplots: 4 rows, ncols columns
#     # Height ratios: ROC=1, PR=1, Calibration=1, Density=1/3
#     fig, axes = plt.subplots(
#         4, ncols,
#         figsize=(fig_w, fig_h),
#         sharex='col',        # share x within each column (calib ↔ density)
#         sharey='row',        # share y within each row (so all density plots share y)
#         gridspec_kw={
#             'height_ratios': [1, 1, 1, 1/3],
#             'hspace': 0.1,
#             'wspace': 0.25
#         }
#     )    
#     # Handle single column case
#     if ncols == 1:
#         axes = axes.reshape(-1, 1)
    
#     plt.subplots_adjust(top=0.95, bottom=0.12, left=0.08, right=0.95)

#     # Extract axes for each plot type
#     ax_roc = [axes[0, i] for i in range(min(n_t, ncols))]
#     ax_pr = [axes[1, i] for i in range(min(n_t, ncols))]
#     ax_calib = [(axes[2, i], axes[3, i]) for i in range(min(n_t, ncols))]

#     ax_calib[0][1].invert_yaxis()

#     handles_accum, labels_accum = [], []

#     for i, tar in enumerate(targets):
#         title = _short_name(tar)
#         ax_roc[i].set_title(title, fontweight="bold", fontsize=17, pad=8)

#         scores_roc = overview_internal_plot_roc(dt_melted, ax_roc, i, tar)
#         scores_pr = overview_internal_plot_pr(dt_melted, ax_pr, i, tar)
#         brier = overview_internal_plot_calib(dt_melted, ax_calib, i, tar,
#                                            n_bins=calib_bins, strategy=calib_strategy)

#         # Ensure same x-axis limits for calibration and density
#         ax_calib[i][0].set_xlim(0, 1)
#         ax_calib[i][1].set_xlim(0, 1)

#         print(f"\nAUROC: {np.mean(scores_roc):.2f}")
#         print(f"AUPR: {np.mean(scores_pr):.2f}")

#         if scores_roc:
#             ax_roc[i].text(
#                 0.97, 0.03, f"AUROC: {np.mean(scores_roc):.3f} ({np.std(scores_roc):.3f})",
#                 color="k", bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
#                 fontsize=14, ha="right", va="bottom", transform=ax_roc[i].transAxes
#             )
#         if scores_pr:
#             ax_pr[i].text(
#                 0.97, 0.03, f"AUPR: {np.mean(scores_pr):.3f} ({np.std(scores_pr):.3f})",
#                 color="k", bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
#                 fontsize=14, ha="right", va="bottom", transform=ax_pr[i].transAxes
#             )
#         if brier is not None:
#             ax_calib[i][0].text(
#                 0.97, 0.06, f"Brier: {np.mean(brier):.3f} ({np.std(brier):.3f})",
#                 color="k", bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
#                 fontsize=14, ha="right", va="bottom", transform=ax_calib[i][0].transAxes
#             )

#         # Collect legend from ROC
#         h, l = ax_roc[i].get_legend_handles_labels()
#         for hh, ll in zip(h, l):
#             if ll != "No Skill" and ll not in labels_accum:
#                 labels_accum.append(ll)
#                 handles_accum.append(hh)

#         # Y-labels: only left-most column
#         if i != 0:
#             ax_roc[i].set_ylabel("")
#             ax_pr[i].set_ylabel("")
#             ax_calib[i][0].set_ylabel("")
#             ax_calib[i][1].set_ylabel("")

#     for ax_row in axes:
#         for ax in ax_row:
#             ax.grid(True, axis='both', linestyle=":", linewidth=1, alpha=0.6)  # dotted grid
#     # Get legend handles and labels
#     handles_roc, labels_roc = ax_roc[0].get_legend_handles_labels()
    
#     # Remove "No Skill" from legend
#     if "No Skill" in labels_roc:
#         no_skill_idx = labels_roc.index("No Skill")
#         handles_roc.pop(no_skill_idx)
#         labels_roc.pop(no_skill_idx)
        
#     fig.legend(
#         handles_roc,
#         labels_roc,
#         loc="center left",
#         bbox_to_anchor=(0.98, 0.5),
#         title="Legend",
#         ncol=min(1, len(labels_roc)),
#         fontsize=16
#     )

#     plt.savefig(filename, bbox_inches="tight", dpi=300)
#     plt.close()
#     print(f"Saved: {filename}")

def rename_features(data):
    tmp_mapping = pd.read_csv("src/src_data/Spearhead_UTI_variable_mapping.csv", index_col=0)
    tmp_mapping = tmp_mapping.drop(columns="Notes")

    mapping = tmp_mapping.to_dict()["Clinical Name "]

    data.rename(columns=mapping, inplace=True)

    return data

def rename_title(title):
    replacements = {
        "Amoxicillin clavulansäure": "Amoxicillin Clavulanate",
        "Amoxicillin - Clavulansäure": "Amoxicillin Clavulanate"
    }
    for old, new in replacements.items():
        title = title.replace(old, new)
    return title


def shap_helper(targets, subpop_ids, subpop_name, data_folder, suffix=""):

    for target in targets:
        if is_ukbb:
            antib_dict = {
                "AC": ["Amoxicillin - Clavulansäure"],
                "ACI": ["Amoxicillin - Clavulansäure in", "Co-Amoxicillin iv"],
                "ACO": ["Amoxicillin - Clavulansäure or", "Co-Amoxicillin HWI oral"],
                "ACU": ["Amoxicillin - Clavulansäure or", "Co-Amoxicillin unkomp HWI oral"],
                "AMI": ["Amikacin"],
                "AMP": ["Ampicillin", "Ampicillin / Amoxicillin"],
                "CEZ": ["Cefazolin"],
                "CFE": ["Cefepim"],
                "CIP": ["Ciprofloxacin"],
                "CLI": ["Clindamycin"],
                "CPD": ["Cefpodoxim"],
                "CS": ["Colistin"],
                "CTR": ["Ceftriaxon"],
                "CTZ": ["Ceftazidim"],
                "CXM": ["Cefuroxim"],
                "CXO": ["Cefuroxim-Axetil"],
                "ERT": ["Ertapenem"],
                "ERY": ["Erythromycin"],
                "FDS": ["Fusidinsäure"],
                "FOT": ["Fosfomycin-Trometamol"],
                "GEN": ["Gentamicin"],
                "IMI": ["Imipenem"],
                "LEV": ["Levofloxacin"],
                "LIZ": ["Linezolid"],
                "MER": ["Meropenem"],
                "MUP": ["Mupirocin"],
                "NFT": ["Nitrofurantoin"],
                "OXA": ["Oxacillin"],
                "PM": [
                    "Cefepim"
                ],  # duplicate code in your dict; consider unifying with CFE
                "PT": ["Piperacillin - Tazobactam"],
                "RAM": ["Rifampicin"],
                "SXT": ["Cotrimoxazol"],
                "TE": ["Tetracyclin"],
                "TEI": ["Teicoplanin"],
                "TGC": ["Tigecyclin"],
                "TOB": ["Tobramycin"],
                "VAN": ["Vancomycin"],
            }
            this_target_name = antib_dict[target.split("_")[-1]][0]
        else:
            this_target_name = re.sub(' +', ' ', " ".join(target.split("_")[2:]).capitalize())
        
        # save all data used for shap, so that we can plot without having to rerun all the models every time
        shap_values_combined = np.load(
            f"{data_folder}/for_shap/{model_to_train}/shap_values_combined_{this_target_name}_repetition0{'' if not is_ukbb else '_UKBB'}.npy"
        )
        X_all = pd.read_csv(
            f"{data_folder}/for_shap/{model_to_train}/X_all_{this_target_name}_repetition0{'' if not is_ukbb else '_UKBB'}.csv", index_col=0
        ).reset_index(drop=True)

        X_subpop = X_all.loc[X_all["patient_id_hashed"].isin(subpop_ids)].drop(columns="patient_id_hashed")

        _, (ax_bar, ax_dot) = plt.subplots(
            1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 2]}
        )
        
        print(target, shap_values_combined[X_subpop.index].shape, X_subpop.shape)

        rename_features(X_subpop) # RENAME THE FEATURE NAMES
        
        summary_plot_mod(
            shap_values_combined[X_subpop.index],
            X_subpop,
            curr_axis=ax_dot,
            plot_feature_names=False,
            show=False,
            max_display=20,
            plot_type="dot",
            plot_size=None,
        )
        summary_plot_mod(
            shap_values_combined[X_subpop.index],
            X_subpop,
            curr_axis=ax_bar,
            errorbar_sd=None,
            plot_feature_names=True,
            show=False,
            max_display=20,
            color="grey",
            plot_type="bar",
            plot_size=None,
        )
    
        ax_dot.set_xlabel("SHAP value for subpopulation", fontsize=15)
        ax_bar.set_xlabel("mean(|SHAP value|)", fontsize=15)
        ax_bar.spines[["right", "top", "bottom"]].set_visible(False)
    
        # plt.suptitle(
        #     "title",
        #     fontsize=20,
        # )
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
    
        # # SHAP Summary Plot (Bar Type)
        # shap.summary_plot(mean_shap_values, X_all, plot_type="bar")
        plot_title = rename_title(this_target_name)
        plt.suptitle(plot_title + f" ({' '.join(subpop_name.split('_'))}{' ' + suffix if suffix else suffix})", fontsize=20)
        plt.savefig(
            f"figures/{model_to_train}_figures/{subpop_name}/shap_{plot_title}{'' if not is_ukbb else '_UKBB'}{'_' + suffix if suffix else suffix}.png"
        )
        plt.close()


def validate_male(data, all_preds):
    men_ids = data.loc[data["sex"] == "männlich", "patient_id_hashed"].values  
    men_preds = all_preds.loc[all_preds["indices"].isin(men_ids)]


    for t in TARGETS_CHOSEN:
        temp = men_preds.loc[men_preds["target"] == t]
        # Count positive and total samples
        positive_count = (temp["true_class"] == 1).sum()
        total_count = len(temp)
        
        # Calculate imbalance percentage
        imbalance_percentage = (positive_count / total_count) * 100
        
        print(f"Positive class percentage: {imbalance_percentage:.2f}%")

    # performances
    overview(
        men_preds,
        targets=TARGETS_CHOSEN,
        ncols=3,
        filename=f"figures/{model_to_train}_figures/men/compact_overview_RF_men.png",
        calib_bins=10,
        calib_strategy="quantile"
    )

    # shap
    shap_helper(TARGETS_CHOSEN, men_ids, "men", "Data_sex_stratification")

def validate_female(data, all_preds):
    women_ids = data.loc[data["sex"] == "weiblich", "patient_id_hashed"].values
    women_preds = all_preds.loc[all_preds["indices"].isin(women_ids)]


    for t in TARGETS_CHOSEN:
        temp = women_preds.loc[women_preds["target"] == t]
        # Count positive and total samples
        positive_count = (temp["true_class"] == 1).sum()
        total_count = len(temp)
        
        # Calculate imbalance percentage
        imbalance_percentage = (positive_count / total_count) * 100
        
        print(f"Positive class percentage: {imbalance_percentage:.2f}%")

    # performances
    overview(
        women_preds,
        targets=TARGETS_CHOSEN,
        ncols=3,
        filename=f"figures/{model_to_train}_figures/women/compact_overview_RF_women.png",
        calib_bins=10,
        calib_strategy="quantile"
    )

    # shap
    shap_helper(TARGETS_CHOSEN, women_ids, "women", "Data_sex_stratification")

def validate_pregnant(data, all_preds):
    # ================================ pregnant ==================================
    try:
        preg_ids = data.loc[(data["sex"] == "weiblich") & (data["pregnancy_yn"] == 1), "patient_id_hashed"].values  
        preg_preds = all_preds.loc[all_preds["indices"].isin(preg_ids)]


        for t in TARGETS_CHOSEN:
            temp = preg_preds.loc[preg_preds["target"] == t]
            # Count positive and total samples
            positive_count = (temp["true_class"] == 1).sum()
            total_count = len(temp)
            
            # Calculate imbalance percentage
            imbalance_percentage = (positive_count / total_count) * 100
            
            print(f"Positive class percentage: {imbalance_percentage:.2f}%")

        # performances
        overview(
            preg_preds,
            targets=TARGETS_CHOSEN,
            ncols=3,
            filename=f"figures/{model_to_train}_figures/women/compact_overview_RF_pregnant.png",
            calib_bins=10,
            calib_strategy="quantile"
        )

        # shap
        shap_helper(TARGETS_CHOSEN, preg_ids, "women", "Data_preg_stratification", suffix="pregnant")
    except ValueError as e:
        raise e
        print(f"ERROR: {e}")
        print(f"shape of preg_preds: {preg_preds.shape}")

    # ================================ non pregnant ==================================
    try:
        nonpreg_ids = data.loc[(data["sex"] == "weiblich") & (data["pregnancy_yn"] == 0), "patient_id_hashed"]
        nonpreg_preds = all_preds.loc[all_preds["indices"].isin(nonpreg_ids)]


        for t in TARGETS_CHOSEN:
            temp = nonpreg_preds.loc[nonpreg_preds["target"] == t]
            # Count positive and total samples
            positive_count = (temp["true_class"] == 1).sum()
            total_count = len(temp)
            
            # Calculate imbalance percentage
            imbalance_percentage = (positive_count / total_count) * 100
            
            print(f"Positive class percentage: {imbalance_percentage:.2f}%")
    
        # performances
        overview(
            nonpreg_preds,
            targets=TARGETS_CHOSEN,
            ncols=3,
            filename=f"figures/{model_to_train}_figures/women/compact_overview_RF_nonpregnant.png",
            calib_bins=10,
            calib_strategy="quantile"
        )

        # shap
        shap_helper(TARGETS_CHOSEN, nonpreg_ids, "women", "Data_preg_stratification", suffix="nonpregnant")
    except ValueError:
        print(f"shape of preg_preds: {preg_preds.shape}")

def validate_age_groups(data, all_preds):
    # Age 65-79 And above 80
    print("----------------------- 18 < y < 65 -----------------------")
    below_65_ids = data.loc[(18 <= data["age"]) & (data["age"] < 65), "patient_id_hashed"].values  
    below_65_preds = all_preds.loc[all_preds["indices"].isin(below_65_ids)]

    for t in TARGETS_CHOSEN:
        temp = below_65_preds.loc[below_65_preds["target"] == t]
        # Count positive and total samples
        positive_count = (temp["true_class"] == 1).sum()
        total_count = len(temp)
        
        # Calculate imbalance percentage
        imbalance_percentage = (positive_count / total_count) * 100
        
        print(f"Positive class percentage: {imbalance_percentage:.2f}%")

    
    # performances
    overview(
        below_65_preds,
        targets=TARGETS_CHOSEN,
        ncols=3,
        filename=f"figures/{model_to_train}_figures/age_groups/below_65/compact_overview_RF.png",
        calib_bins=10,
        calib_strategy="quantile"
    )

    # shap
    shap_helper(TARGETS_CHOSEN, below_65_ids, "age_groups/below_65", "Data_age_stratification")

    print("----------------------- 65-79  ----------------------- ")
    _65_79_ids = data.loc[(65 <= data["age"]) & (data["age"] < 80), "patient_id_hashed"].values
    _65_79_preds = all_preds.loc[all_preds["indices"].isin(_65_79_ids)]
    

    for t in TARGETS_CHOSEN:
        temp = _65_79_preds.loc[_65_79_preds["target"] == t]
        # Count positive and total samples
        positive_count = (temp["true_class"] == 1).sum()
        total_count = len(temp)
        
        # Calculate imbalance percentage
        imbalance_percentage = (positive_count / total_count) * 100
        
        print(f"Positive class percentage: {imbalance_percentage:.2f}%")

    
    # performances
    overview(
        _65_79_preds,
        targets=TARGETS_CHOSEN,
        ncols=3,
        filename=f"figures/{model_to_train}_figures/age_groups/65_79/compact_overview_RF.png",
        calib_bins=10,
        calib_strategy="quantile"
    )

    # shap
    shap_helper(TARGETS_CHOSEN, _65_79_ids, "age_groups/65_79", "Data_age_stratification")

    print("----------------------- above 80  -----------------------")
    above_80_ids = data.loc[data["age"] >= 80, "patient_id_hashed"].values
    above_80_preds = all_preds.loc[all_preds["indices"].isin(above_80_ids)]

    
    for t in TARGETS_CHOSEN:
        temp = above_80_preds.loc[above_80_preds["target"] == t]
        # Count positive and total samples
        positive_count = (temp["true_class"] == 1).sum()
        total_count = len(temp)
        
        # Calculate imbalance percentage
        imbalance_percentage = (positive_count / total_count) * 100
        
        print(f"Positive class percentage: {imbalance_percentage:.2f}%")

    # performances
    overview(
        above_80_preds,
        targets=TARGETS_CHOSEN,
        ncols=3,
        filename=f"figures/{model_to_train}_figures/age_groups/above_80/compact_overview_RF.png",
        calib_bins=10,
        calib_strategy="quantile"
    )

    # shap
    shap_helper(TARGETS_CHOSEN, above_80_ids, "age_groups/above_80", "Data_age_stratification")



# def validate_res_and_no_prev_resistance(data, all_preds):
#     """
#     Validate the model on test samples that are non-resistant (true_class == 0).
#     Ignores whether resistance exists in other folds.
#     Returns the subset of preds used for validation.
#     """

#     for target in TARGETS_CHOSEN:
#         try:
#             this_target_name = re.sub(' +', ' ', " ".join(target.split("_")[2:]).capitalize())
#             X_all = pd.read_csv(
#                 f"Data_res_stratification/for_shap/{model_to_train}/X_all_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition0.csv", index_col=0
#             )
    
#             print(all_preds["target"].unique())
    
#             pat_no_res_ids = X_all.loc[(X_all['had_prev_resistance'] == 0) | (X_all['had_prev_resistance'] == -1), "patient_id_hashed"]

#             preds_subset = all_preds.loc[(all_preds["target"] == target) & (X_all["patient_id_hashed"].isin(pat_no_res_ids))]

#             temp = preds_subset.loc[preds_subset["target"] == target]
#             # Count positive and total samples
#             positive_count = (temp["true_class"] == 1).sum()
#             total_count = len(temp)
            
#             # Calculate imbalance percentage
#             imbalance_percentage = (positive_count / total_count) * 100
            
#             print(f"Positive class percentage: {imbalance_percentage:.2f}%")
    
#             overview(
#                 preds_subset,
#                 targets=[target],
#                 ncols=1,
#                 filename=f"figures/{model_to_train}_figures/no_resistance/nonresistant_overview.png",
#                 calib_bins=10,
#                 calib_strategy="quantile"
#             )
        
#             shap_helper([target], pat_no_res_ids, "no_resistance")
#         except ValueError:
#             print(f"shape of preg_preds: {preds_subset.shape}")
#             print("if the shape has 0 rows, it means the filtering for finding the relevant indices did not find any record")


#     for target in TARGETS_CHOSEN:
#         try:
#             this_target_name = re.sub(' +', ' ', " ".join(target.split("_")[2:]).capitalize())
#             X_all = pd.read_csv(
#                 f"Data_res_stratification/for_shap/{model_to_train}/X_all_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition0.csv", index_col=0
#             )
    
#             print(all_preds["target"].unique())
    
#             pat_res_ids = X_all.loc[X_all['had_prev_resistance'] == 1, "patient_id_hashed"]
    
#             preds_subset = all_preds.loc[(all_preds["target"] == target) & (X_all["patient_id_hashed"].isin(pat_res_ids))]

#             temp = preds_subset.loc[preds_subset["target"] == target]
#             # Count positive and total samples
#             positive_count = (temp["true_class"] == 1).sum()
#             total_count = len(temp)
            
#             # Calculate imbalance percentage
#             imbalance_percentage = (positive_count / total_count) * 100
            
#             print(f"Positive class percentage: {imbalance_percentage:.2f}%")
    
#             overview(
#                 preds_subset,
#                 targets=[target],
#                 ncols=1,
#                 filename=f"figures/{model_to_train}_figures/resistance/resistant_overview.png",
#                 calib_bins=10,
#                 calib_strategy="quantile"
#             )
        
#             shap_helper([target], pat_res_ids, "resistance")
#         except ValueError:
#             print(f"shape of preg_preds: {preds_subset.shape}")
#             print("if the shape has 0 rows, it means the filtering for finding the relevant indices did not find any record")


def validate_res_and_no_prev_resistance(data, all_preds):
    """
    Validate the model on test samples that are non-resistant (true_class == 0).
    Ignores whether resistance exists in other folds.
    Returns the subset of preds used for validation.
    """

    # Collect all non-resistant predictions
    all_no_res_preds = []
    all_no_res_pat_ids = []

    print("NO RES")
    for target in TARGETS_CHOSEN:
        try:
            this_target_name = re.sub(' +', ' ', " ".join(target.split("_")[2:]).capitalize())
            X_all = pd.read_csv(
                f"Data_res_stratification/for_shap/{model_to_train}/X_all_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition0.csv", index_col=0
            )
    
            pat_no_res_ids = X_all.loc[(X_all['had_prev_resistance'] == 0) & (X_all['multiple_occurences_prev_resistance'] == 0), "patient_id_hashed"].values
            preds_subset = all_preds.loc[(all_preds["target"] == target) & (all_preds["indices"].isin(pat_no_res_ids))]

            print(len(pat_no_res_ids))
            print(preds_subset.shape)

            temp = preds_subset.loc[preds_subset["target"] == target]

            # Count positive and total samples
            positive_count = (temp["true_class"] == 1.0).sum()
            total_count = len(temp)

            # Calculate imbalance percentage
            imbalance_percentage = (positive_count / total_count) * 100
            
            print(f"Positive class percentage: {imbalance_percentage:.2f}%")
            
            all_no_res_preds.append(preds_subset)
            all_no_res_pat_ids.append(pat_no_res_ids)
        
            shap_helper([target], pat_no_res_ids, "no_resistance", "Data_res_stratification")
        except ValueError:
            print(f"shape of preg_preds: {preds_subset.shape}")
            print("if the shape has 0 rows, it means the filtering for finding the relevant indices did not find any record")


    # Call overview outside the loop for non-resistant
    if all_no_res_preds:
        combined_no_res_preds = pd.concat(all_no_res_preds, ignore_index=True)
        overview(
            combined_no_res_preds,
            targets=TARGETS_CHOSEN,
            ncols=3,
            filename=f"figures/{model_to_train}_figures/no_resistance/nonresistant_overview.png",
            calib_bins=10,
            calib_strategy="quantile"
        )

    # Collect all resistant predictions
    all_res_preds = []
    all_res_pat_ids = []

    
    print()
    print()
    print("RES")
    for target in TARGETS_CHOSEN:
        try:
            this_target_name = re.sub(' +', ' ', " ".join(target.split("_")[2:]).capitalize())
            X_all = pd.read_csv(
                f"Data_res_stratification/for_shap/{model_to_train}/X_all_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition0.csv", index_col=0
            )
    
            pat_res_ids = X_all.loc[X_all['had_prev_resistance'] == 1, "patient_id_hashed"].values
            preds_subset = all_preds.loc[(all_preds["target"] == target) & (all_preds["indices"].isin(pat_res_ids))]

            print(len(pat_res_ids))
            print(preds_subset.shape)

            temp = preds_subset.loc[preds_subset["target"] == target]
            # Count positive and total samples
            positive_count = (temp["true_class"] == 1.0).sum()
            total_count = len(temp)
            
            # Calculate imbalance percentage
            imbalance_percentage = (positive_count / total_count) * 100
            
            print(f"Positive class percentage: {imbalance_percentage:.2f}%")
            
            all_res_preds.append(preds_subset)
            all_res_pat_ids.append(pat_res_ids)
        
            shap_helper([target], pat_res_ids, "resistance", "Data_res_stratification")
        except ValueError:
            print(f"shape of preg_preds: {preds_subset.shape}")
            print("if the shape has 0 rows, it means the filtering for finding the relevant indices did not find any record")

    # Call overview outside the loop for resistant
    if all_res_preds:
        combined_res_preds = pd.concat(all_res_preds, ignore_index=True)
        overview(
            combined_res_preds,
            targets=TARGETS_CHOSEN,
            ncols=3,
            filename=f"figures/{model_to_train}_figures/resistance/resistant_overview.png",
            calib_bins=10,
            calib_strategy="quantile"
        )

def validate_adults_on_children(targets):

    is_ukbb = True

    def to_timedelta_safe(s):
        if pd.isna(s):
            return pd.NaT
        # normalize whitespace and minus signs
        s = str(s).strip()
        s = s.replace("\u2212", "-")        # U+2212 minus → ASCII '-'
        s = re.sub(r"\s+", " ", s)          # collapse spaces
        # let pandas parse; coerce bad rows to NaT instead of raising
        return pd.to_timedelta(s, errors="coerce")


    pid = "patnr"
    caseid = "fallnr"
    shift_col = "date_shifted_to_last_uti"
    
    
    # load the children data, preprocessed
    # load adult model
    # align columns so it has the same colum names expected by the model
    # possibly have to train a "compact" model since we will not have all the same columns

    usb_target_name = {
        "urine_antibiogram_AC": "urine_antibiogram_amoxicillin___clavulansäure",
        "urine_antibiogram_CXM": "urine_antibiogram_cefuroxim",
        "urine_antibiogram_SXT": "urine_antibiogram_cotrimoxazol",
        "urine_antibiogram_FOT": "urine_antibiogram_fosfomycin_trometamol",
        "urine_antibiogram_NFT": "urine_antibiogram_nitrofurantoin",
        # does not exist in UKBB: "urine_antibiogram_norfloxacin",
        "urine_antibiogram_CIP": "urine_antibiogram_ciprofloxacin",
        "urine_antibiogram_CTR": "urine_antibiogram_ceftriaxon",
        "urine_antibiogram_PT": "urine_antibiogram_piperacillin___tazobactam",
    }

    # rf_model = joblib.load(f"Data/models/saved_models_RF/{usb_target_name[target_col]}/RF_fold_0_repetition0.pkl")
    # for i in sorted(rf_model.feature_names_in_):
    #     print(i)
    all_results = {}
    y_dict = {}

    for target_col in targets:

        child_data = pd.read_csv("src/src_data/all_UKBB_data.csv", index_col=0) # NOTE: CANT use this directly, have to preprocess it at least a bit (to create the needed columns like case type one hot encoded)
        prescriptions = pd.read_csv("dataset_UKBB/drug_prescriptions_final.csv")
        prescriptions = prescriptions.drop("date_shifted_to_last_uti", axis=1)
        
        # STEP 0 # preprocess
        ####################################################################
        is_eucast_rules = False
    
        log(INFO, "Running R script for overwriting resistances using AMR R package")
        os.system("Rscript src/test_amr_ukbb.r")
        log(INFO, "Finished running R script")
    
        if os.path.exists("src/src_data/processed_amr_data_binary.csv"):
            is_eucast_rules = True
            child_data = pd.read_csv("src/src_data/processed_amr_data_binary.csv")
        
        # transform date columns to number of days
        # Single columns
        date_cols = child_data.columns[
            child_data.columns.str.contains("date", case=False)
        ].to_list()
    
        for c in date_cols:
            child_data[c] = child_data[c].map(to_timedelta_safe) / pd.Timedelta(days=1)
    
        needs_completion_or_removal = [
            "rare_disease",
            "cancer_immunosuppression",
            "penicillin_allergies",
            "hypertension_disease",
            "diabetus_mellitus",
            "cystitis",
            "dysuria",
            "inflammatory_diseases_of_prostate",
            "pyelonephritis_or_renal_tubulo",
            "persistent_proteinurie",
            "urethritis",
            "urosepsis",
            "chronic_uti",
            "chronic_pyelonephritis",
            "chronic_cystitis_other",
            "chronic_cystitis_interstitial",
            "indwelling_foley_catheter",
            "suprapubic_catheter",
            "ileal_conduit",
            "ureteral_catheterization",
            "lithotripsy_ultrasound",
            "closure_urinary_fistula",
            "operations_kidney",
            "operations_ureter",
            "operations_urinary_bladder",
            "operations_urethra",
            "other_operations_urinary_tract",
            "operations_male_reproductive_organs",
            "operations_female_genital_organs",
        ]
    
        for c in needs_completion_or_removal:
            if child_data[c].eq("Yes").any():
                child_data[c] = child_data[c].fillna("No")
            else:
                child_data = child_data.drop(columns=[c])
                
        ####################################################################
    
        # ------------------------- CREATE VARIABLES -------------------------
        # Step 1: Compute `previous_resistance`
        temp = child_data.copy()
    
        # ------------------------ PREV RESISTANCE WINDOWED ------------------------
    
        WINDOWS = {"1W": 7, "2W": 14, "1M": 30, "6M": 180, "1Y": 365, "ALL": np.inf}
    
        def build_prev_resistance(data: pd.DataFrame, target_col: str) -> pd.DataFrame:
            keys = [pid, caseid]
        
            # index rows that receive features
            idx = data[data[shift_col] == 0][keys].drop_duplicates().copy()
        
            # prior rows (any earlier case for same patient)
            prior = data[data[shift_col] < 0].copy()
            prior["days_to_index"] = prior[shift_col].abs()
        
            out = idx.copy()
    
            # ALL = sum of prior resistant events per patient
            s_all = prior.groupby(pid)[target_col].sum()
            prev_resist_all = out[pid].map(s_all).fillna(0).astype(int)
            out["multiple_occurences_prev_resistance"] = (prev_resist_all > 1).astype(int)
            
            # prior rows for which the event occurred
            prior_true = prior[prior[target_col] == 1]
            
            # for each patient, get days since the most recent true event
            last_true_days = prior_true.groupby(pid)["days_to_index"].min()
            
            # patients that have ANY prior data (regardless of resistance)
            patients_with_prior = prior.groupby(pid).size()
            
            # add days from last resistance (NaN if no prior resistant event)
            out["days_from_last_resistance"] = out[pid].map(last_true_days)
            
            # Create binary column:
            # - 1 if had prior resistance (days_from_last_resistance is not NaN)
            # - 0 if has prior data but no resistance
            # - NaN if no prior data at all
            out["had_prev_resistance"] = pd.NA  # start with all NA
            
            # Set to 1 where there was a prior resistant event
            has_resistance = out[pid].isin(last_true_days.index)
            out.loc[has_resistance, "had_prev_resistance"] = 1
            
            # Set to 0 where patient has prior data but no resistance
            has_prior_data = out[pid].isin(patients_with_prior.index)
            out.loc[has_prior_data & ~has_resistance, "had_prev_resistance"] = 0
            
            # Convert to nullable integer type to preserve NaN
            out["had_prev_resistance"] = out["had_prev_resistance"].astype("Int64")
            # Convert NaN to a string category for stratification purposes
            out["had_prev_resistance"] = out["had_prev_resistance"].fillna(-1).astype(int)
        
            return out
    
        def build_prev_exposure(
            prescriptions: pd.DataFrame, data: pd.DataFrame, keyword: str
        ) -> pd.DataFrame:
    
            keys = [pid, caseid]
    
            # filter prescriptions to this antibiotic
            pres = prescriptions.copy()
            kw = str(keyword).lower()
            if is_ukbb:
                # using "antibiotic_code". creating this column using the dictionnary provided by UKBB.
                mask = (
                    pres.get("antibiotic_code", "")
                    .astype(str)
                    .str.lower()
                    .str.contains(kw, na=False)
                )
            else:
                mask = (
                    pres.get("ATC_name", "")
                    .astype(str)
                    .str.lower()
                    .str.contains(kw, na=False)
                    | pres.get("drug_prescribed", "")
                    .astype(str)
                    .str.lower()
                    .str.contains(kw, na=False)
                    | pres.get(substance_name, "")
                    .astype(str)
                    .str.lower()
                    .str.contains(kw, na=False)
                )
            pres = pres[mask]
    
            # attach case shifts (one row per (patient,case) on the right to avoid explosions)
            case_shifts = data[keys + [shift_col]].drop_duplicates(subset=keys)
            pres = pres.merge(case_shifts, on=keys, how="inner")
    
            # prior only; make coarse days-to-index from case shift
            pres = pres[pres[shift_col] < 0].copy()
            pres["days_to_index"] = pres[shift_col].abs()
    
            # (optional) de-dup: one exposure per (patient, case) is enough for flags
            pres = pres.drop_duplicates(subset=keys)
    
            # index rows that receive features
            idx = data[data[shift_col] == 0][keys].drop_duplicates().copy()
            out = idx.copy()
    
            # ALL = any prior exposure per patient
            s_all_flag = pres.groupby(pid).size().gt(0).astype(int)
            out["prev_exp_ALL"] = out[pid].map(s_all_flag).fillna(0).astype(int)
            print(out["prev_exp_ALL"].unique())
    
            # finite windows (binary flags)
            for lbl, lim in WINDOWS.items():
                if not np.isfinite(lim):
                    continue
                sub = pres[pres["days_to_index"] <= lim]
                s_flag = sub.groupby(pid).size().gt(0).astype(int)
                out[f"prev_exp_{lbl}"] = out[pid].map(s_flag).fillna(0).astype(int)
    
            return out


        # 1) Resistance (per current target antibiotic)
        res_feats = build_prev_resistance(
            temp, target_col
        )  # temp contains all history rows for patients
        child_data = child_data.merge(res_feats, on=[pid, caseid], how="left")
    
        # 2) Exposure (per current target antibiotic)
        split_target = target_col.rsplit("_", 1)
    
        if is_ukbb:
    
            def _norm(s: str) -> str:
                s = s.lower()
                s = (
                    s.replace("ä", "ae").replace("ö", "oe").replace("ü", "ue")
                )  # simple de-umlaut
                s = s.replace("-", " ").replace("/", " ")
                s = re.sub(r"\s+", " ", s).strip()
                return s
    
            # 1) Your base dict (code -> list of canonical names)
            antib_dict = {
                "AC": ["Amoxicillin - Clavulansäure"],
                "ACI": ["Amoxicillin - Clavulansäure in", "Co-Amoxicillin iv"],
                "ACO": ["Amoxicillin - Clavulansäure or", "Co-Amoxicillin HWI oral"],
                "ACU": ["Amoxicillin - Clavulansäure or", "Co-Amoxicillin unkomp HWI oral"],
                "AMI": ["Amikacin"],
                "AMP": ["Ampicillin", "Ampicillin / Amoxicillin"],
                "CEZ": ["Cefazolin"],
                "CFE": ["Cefepim"],
                "CIP": ["Ciprofloxacin"],
                "CLI": ["Clindamycin"],
                "CPD": ["Cefpodoxim"],
                "CS": ["Colistin"],
                "CTR": ["Ceftriaxon"],
                "CTZ": ["Ceftazidim"],
                "CXM": ["Cefuroxim"],
                "CXO": ["Cefuroxim-Axetil"],
                "ERT": ["Ertapenem"],
                "ERY": ["Erythromycin"],
                "FDS": ["Fusidinsäure"],
                "FOT": ["Fosfomycin-Trometamol"],
                "GEN": ["Gentamicin"],
                "IMI": ["Imipenem"],
                "LEV": ["Levofloxacin"],
                "LIZ": ["Linezolid"],
                "MER": ["Meropenem"],
                "MUP": ["Mupirocin"],
                "NFT": ["Nitrofurantoin"],
                "OXA": ["Oxacillin"],
                "PM": [
                    "Cefepim"
                ],  # duplicate code in your dict; consider unifying with CFE
                "PT": ["Piperacillin - Tazobactam"],
                "RAM": ["Rifampicin"],
                "SXT": ["Cotrimoxazol"],
                "TE": ["Tetracyclin"],
                "TEI": ["Teicoplanin"],
                "TGC": ["Tigecyclin"],
                "TOB": ["Tobramycin"],
                "VAN": ["Vancomycin"],
            }
    
            # 2) Alias table to catch your unmapped values and salt forms / components
            alias_to_code = {
                # TMP-SMX (SXT)
                "co trimoxazol": "SXT",
                "co-trimoxazol": "SXT",
                "cotrimoxazol": "SXT",
                "trimethoprim": "SXT",
                "sulfamethoxazol": "SXT",
                # AC (amox/clav)
                "co amoxicillin": "AC",
                "co-amoxicillin": "AC",
                "amoxicillin clavulansaeure": "AC",
                "clavulansaeure": "AC",  # often prescribed as component alongside amoxicillin
                # plain amoxicillin / ampicillin → AMP (adjust if you want separate)
                "amoxicillin": "AMP",
                "ampicillin": "AMP",
                # cephalosporins
                "cefpodoxim": "CPD",
                "ceftriaxon": "CTR",
                "cefepim": "CFE",  # or "PM" if that’s what you use
                # fluoroquinolones
                "ciprofloxacin": "CIP",
                "levofloxacin": "LEV",
                # aminoglycosides
                "amikacin": "AMI",
                "gentamicin": "GEN",
                "tobramycin": "TOB",
                # colistin (salt form)
                "colistimethat natrium": "CS",
                "colistin": "CS",
                # tetracyclines
                "doxycyclin": "TE",  # treat as tetracycline class exposure
                "tetracyclin": "TE",
                "phenoxymethylpenicillin kalium": "PEN",
            }
    
            # 3) Build a normalized keyword → code table from your base dict (fallback matching)
            canon_keywords = []
            for code, names in antib_dict.items():
                for name in names:
                    canon_keywords.append((_norm(name), code))
    
            def map_antibiotic_name(s: str) -> str | None:
                if pd.isna(s):
                    return None
                sn = _norm(s)
                # explicit alias first
                if sn in alias_to_code:
                    return alias_to_code[sn]
                # fallback: substring match against canonical names
                for key_norm, code in canon_keywords:
                    if key_norm in sn or sn in key_norm:
                        return code
                return None
    
            # Apply
            prescriptions["antibiotic_code"] = prescriptions["active_substance"].apply(
                map_antibiotic_name
            )
    
            # Inspect remaining unmapped
            unmapped = (
                prescriptions.loc[
                    prescriptions["antibiotic_code"].isna(), "active_substance"
                ]
                .dropna()
                .unique()
            )
            log(WARNING, f"Still unmapped: {unmapped}")
    
        keyword = (
            split_target[-1].lower() if len(split_target) > 1 else split_target[0].lower()
        )
    
        exp_feats = build_prev_exposure(prescriptions, temp, keyword)
        exp_feats = exp_feats.fillna(0) # making sure there are no NANs
    
        child_data = child_data.merge(exp_feats, on=[pid, caseid], how="left")
        
    
        not_features = ["report_id", "urine_organismsubid", "mo", "gram_binary"]
    
        X, y = (child_data.drop(columns=[col for col in list(child_data.columns) if col not in not_features and "antibiogram" in col]), # drop all columns that are not used as features
                child_data.loc[:, [col for col in list(child_data.columns) if "antibiogram" in col]]
               )
        
        # using preprocessing pipeline
        pipeline = pipeline_func_UKBB(True, False) # use_risk=True, is_premodel=False
        X = pipeline.fit_transform(X)
        
        y = y.loc[:, target_col].dropna()
        # print("target remaining rows:", target.shape[0])
        X = X.loc[y.index]  # Ensure data and target have the same index
        X, y = X.reset_index(drop=True), y.reset_index(drop=True)
    
        # step 2
        # do i just make a dictionary mapping and rename?
        # seems the best.
        # child_col_name: model_col_name
        rename_mapping = {
                         # general
                         "crp": "blood_crp__mg_l_",
                         "case_type_A": "case_type_ambulant",
                         "case_type_S": "case_type_stationär",
                         "case_type_TS": "case_type_teil_stationär",
            
                         # Urine material one hot encoded
                         "urine_material_Urin aus Einmalkatheter": "urine_material_Urin_aus_Einmalkatheter",
                         "urine_material_Urin aus Dauerkatheter": "urine_material_Urin_aus_Dauerkatheter",
                         "urine_material_Urin aus Blasenpunktion": "urine_material_Urin_aus_Blasenpunktion",
                         "urine_material_Urin nicht genauer bezeichnet": "urine_material_Urin_nicht_genauer_bezeichnet",
    
                         # renaming risk factors
                         "hypertension_disease": "risk_hypertension_any",
                         "diabetus_mellitus": "risk_diabetes_any",
                         "cystitis": "risk_cystitis_30d",
                         "pyelonephritis_or_renal_tubulo": "risk_pyelonephritis_or_renal_tubulo_30d",
                         "urosepsis": "risk_urosepsis_30d",
                         "indwelling_foley_catheter": "risk_indwelling_foley_catheter_30d",
                         "suprapubic_catheter": "risk_suprapubic_catheter_30d",
                         "operations_kidney": "risk_operations_kidney_30d",
                         "operations_ureter": "risk_operations_ureter_30d",
                         "operations_urinary_bladder": "risk_operations_urinary_bladder_30d",
                         "operations_urethra": "risk_operations_urethra_30d",
                         "other_operations_urinary_tract": "risk_urinary_tract_surgery_30d",
                         "operations_male_reproductive_organs": "risk_operations_male_reproductive_organs_30d",
                         }
    
        X = X.rename(columns=rename_mapping)
        X.loc[:, "sex_männlich"] = (X["sex"] == 1).astype(int)
    
        # setting last remaining to 0 since they are not present in child data
        X.loc[:, "pregnancy_yn"] = 0
    
        accuracies = []
        f1_scores = []
        aurocs = []
        all_fpr = []
        all_tpr = []
        auprs = []
        all_precision = []
        all_recall = []
        all_shap_values = []
        
        for i in range(10): # repetitions
    
            child_cols = list(X.columns)
            
            rf_model = joblib.load(f"Data_usb_on_ukbb/models/saved_models_RF/{usb_target_name[target_col]}/RF_fold_0_repetition{i}.pkl")
            for k in sorted(rf_model.feature_names_in_):
                print(k)
        
            log(INFO, "Already aligned:")
            aligned = sorted(set(sorted(child_cols)).intersection(set(sorted(rf_model.feature_names_in_))))
            for k in aligned:
                log(INFO, k)
            
            # remove them from the list of columns, so I know which ones I still have to convert to correct names
            child_cols = [x for x in child_cols if x not in aligned]
            # remove all "date" columns
            child_cols = [x for x in child_cols if "date" not in x]
            child_cols = [x for x in child_cols if "antibiogram" not in x]
            child_cols = [x for x in child_cols if x != "report_id" and x != "urine_organismsubid" and x != "patnr" and x != "fallnr" and x != "mo" and x != "gram_binary"]
        
            log(INFO, "")
            log(INFO, "")
            log(INFO, "")
        
        
            log(INFO, "Missing alignment:")
            for col in rf_model.feature_names_in_:
                if col not in aligned:
                    log(INFO, col)
                    
            log(INFO, "")
            log(INFO, "")
            log(INFO, "")
            
            log(INFO, "Remaining child cols:")
            for col in child_cols:
                log(INFO, col)
        
        
            X_test = X.loc[:, aligned]
            X_test = X_test[rf_model.feature_names_in_]
            X_test = X_test.fillna(X_test.mean())
        
            y_pred = rf_model.predict(X_test)
            probas = rf_model.predict_proba(X_test)
        
            # Check if class 1 exists in the classifier
            if 1 in rf_model.classes_:
                idx = list(rf_model.classes_).index(1)
                y_proba = probas[:, idx]
            else:
                # Model was trained without class 1 (e.g., only class 0 was present)
                y_proba = np.zeros(X_test.shape[0])
        
            # Compute Metrics
            acc = accuracy_score(y, y_pred)
            balanced_acc = balanced_accuracy_score(y, y_pred)
            f1 = f1_score(y, y_pred, average="binary")  # "binary"
            fpr, tpr, _ = roc_curve(y, y_proba)
            roc_auc = auc(fpr, tpr)
    
            precision, recall, _ = precision_recall_curve(y, y_proba)
            pr_auc = average_precision_score(y, y_proba)
    
            # Store results
            accuracies.append(balanced_acc)
            f1_scores.append(f1)
            aurocs.append(roc_auc)
            all_fpr.append(fpr)
            all_tpr.append(tpr)
    
            auprs.append(pr_auc)
            all_precision.append(precision)
            all_recall.append(recall)
    
            log(
                PERFORMANCE,
                f"Rep {i+1}: Balanced Acc.={balanced_acc:.4f}, F1={f1:.4f}, AUROC={roc_auc:.4f}",
            )
            # print(colored(PERFORMANCE, "green"), f": \t\tFold {fold+1}: Accuracy={acc:.4f}, F1={f1:.4f}, AUROC={roc_auc:.4f}")
    
            import shap
            explainer = shap.TreeExplainer(rf_model)  # for RF, XGB
            shap_values = explainer.shap_values(X_test)
        
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 2:
                shap_values = shap_values[:, :, 1]
    
            all_shap_values.append(shap_values)
    
        rename_features(X_test) # RENAME THE FEATURE NAMES
    
        averaged_shap_values = np.mean(all_shap_values, axis=0)
        
        _, (ax_bar, ax_dot) = plt.subplots(
            1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 2]}
        )
    
        summary_plot_mod(
            averaged_shap_values,
            X_test,
            curr_axis=ax_dot,
            plot_feature_names=False,
            show=False,
            max_display=20,
            plot_type="dot",
            plot_size=None,
        )
        summary_plot_mod(
            averaged_shap_values,
            X_test,
            curr_axis=ax_bar,
            errorbar_sd=None,
            plot_feature_names=True,
            show=False,
            max_display=20,
            color="grey",
            plot_type="bar",
            plot_size=None,
        )
    
        ax_dot.set_xlabel("SHAP value for subpopulation", fontsize=15)
        ax_bar.set_xlabel("mean(|SHAP value|)", fontsize=15)
        ax_bar.spines[["right", "top", "bottom"]].set_visible(False)
    
        # # SHAP Summary Plot (Bar Type)
        # shap.summary_plot(mean_shap_values, X_all, plot_type="bar")
    
        antib_dict = {
            "AC": ["Amoxicillin - Clavulansäure"],
            "ACI": ["Amoxicillin - Clavulansäure in", "Co-Amoxicillin iv"],
            "ACO": ["Amoxicillin - Clavulansäure or", "Co-Amoxicillin HWI oral"],
            "ACU": ["Amoxicillin - Clavulansäure or", "Co-Amoxicillin unkomp HWI oral"],
            "AMI": ["Amikacin"],
            "AMP": ["Ampicillin", "Ampicillin / Amoxicillin"],
            "CEZ": ["Cefazolin"],
            "CFE": ["Cefepim"],
            "CIP": ["Ciprofloxacin"],
            "CLI": ["Clindamycin"],
            "CPD": ["Cefpodoxim"],
            "CS": ["Colistin"],
            "CTR": ["Ceftriaxon"],
            "CTZ": ["Ceftazidim"],
            "CXM": ["Cefuroxim"],
            "CXO": ["Cefuroxim-Axetil"],
            "ERT": ["Ertapenem"],
            "ERY": ["Erythromycin"],
            "FDS": ["Fusidinsäure"],
            "FOT": ["Fosfomycin-Trometamol"],
            "GEN": ["Gentamicin"],
            "IMI": ["Imipenem"],
            "LEV": ["Levofloxacin"],
            "LIZ": ["Linezolid"],
            "MER": ["Meropenem"],
            "MUP": ["Mupirocin"],
            "NFT": ["Nitrofurantoin"],
            "OXA": ["Oxacillin"],
            "PM": [
                "Cefepim"
            ],  # duplicate code in your dict; consider unifying with CFE
            "PT": ["Piperacillin - Tazobactam"],
            "RAM": ["Rifampicin"],
            "SXT": ["Cotrimoxazol"],
            "TE": ["Tetracyclin"],
            "TEI": ["Teicoplanin"],
            "TGC": ["Tigecyclin"],
            "TOB": ["Tobramycin"],
            "VAN": ["Vancomycin"],
        }
        this_target_name = antib_dict[target_col.split("_")[-1]][0]
    
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        plot_title = rename_title(this_target_name)
        plt.suptitle(plot_title, fontsize=20)
        
        plt.savefig(
            f"figures/{model_to_train}_figures/usb_on_ukbb/shap_{this_target_name}.png"
        )
        plt.close()

        # Get the target name
        this_target_name = antib_dict[target_col.split("_")[-1]][0]
        
        # Store results for combined plot
        all_results[this_target_name] = {
            'all_fpr': all_fpr,
            'all_tpr': all_tpr,
            'all_precision': all_precision,
            'all_recall': all_recall,
            'aurocs': aurocs,
            'auprs': auprs
        }
        y_dict[this_target_name] = y.copy()

    # ============================================================
    # Create combined overview figure
    # ============================================================
    
    def plot_combined_overview(all_results: dict, 
                               y_dict: dict,
                               ncols: int = 3,
                               filename: str = "figures/RF_figures/usb_on_ukbb/combined_overview.png") -> None:
        """Plot ROC and PR curves for all targets in a single figure."""
        
        sns.set_theme(style="white", font_scale=1.5)
        
        targets = list(all_results.keys())
        n_t = len(targets)
        ncols = min(ncols, n_t)
        
        nrows_targets = int(np.ceil(n_t / ncols))
        total_plot_rows = nrows_targets * 2  # ROC + PR
        
        plot_size = 7
        fig_w = ncols * plot_size
        fig_h = nrows_targets * 2 * plot_size * 0.8 + 1
        
        height_ratios = [1, 1] * nrows_targets
        
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs = fig.add_gridspec(
            total_plot_rows, ncols,
            height_ratios=height_ratios,
            hspace=0.35,
            wspace=0.25
        )
        
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(ncols)] 
                         for i in range(total_plot_rows)])
    
    
        for i, tar in enumerate(targets):
            target_row = i // ncols
            target_col = i % ncols
            roc_row = target_row * 2
            pr_row = target_row * 2 + 1
            
            ax_roc = axes[roc_row, target_col]
            ax_pr = axes[pr_row, target_col]
            
            res = all_results[tar]
            y = y_dict[tar]
            
            n_reps = len(res['all_fpr'])
            colors = sns.color_palette("mako", n_colors=n_reps)
            
            # Title
            plot_title = rename_title(tar)
            ax_roc.set_title(plot_title, fontweight="bold", fontsize=16, pad=8)
            
            # ---------- ROC Curve ----------
            ax_roc.plot([0, 1], [0, 1], ls="--", color="gray", lw=2, label="No Skill")
            
            for fold_idx, ((fpr, tpr), c) in enumerate(zip(zip(res['all_fpr'], res['all_tpr']), colors)):
                ax_roc.plot(fpr, tpr, lw=2, color=c, alpha=0.7, zorder=3,
                           label=f"Run {fold_idx}" if i == 0 else None)
            
            ax_roc.set_xlim(0, 1)
            ax_roc.set_ylim(0, 1)
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate" if target_col == 0 else "")
            ax_roc.grid(True, axis='both', linestyle=":", linewidth=1, alpha=0.6)
            
            # Imbalance calculation
            target_class_distribution = y.value_counts()
            if len(target_class_distribution) < 2:
                imbalance_percentage = np.nan
            else:
                imbalance_percentage = (target_class_distribution[1] / target_class_distribution.sum()) * 100
            
            ax_roc.text(
                0.97, 0.03,
                f"AUROC: {np.mean(res['aurocs']):.3f} ({np.std(res['aurocs']):.3f})\nPos: {imbalance_percentage:.1f}%",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
                fontsize=11, ha="right", va="bottom", transform=ax_roc.transAxes
            )
            
            # ---------- PR Curve ----------
            no_skill = len(y[y == 1]) / len(y) if len(y) > 0 else 0
            ax_pr.plot([0, 1], [no_skill, no_skill], ls="--", color="gray", lw=2)
            
            for (precision, recall), c in zip(zip(res['all_precision'], res['all_recall']), colors):
                ax_pr.plot(recall, precision, lw=2, color=c, alpha=0.7, zorder=3)
            
            ax_pr.set_xlim(0, 1)
            ax_pr.set_ylim(0, 1)
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision" if target_col == 0 else "")
            ax_pr.grid(True, axis='both', linestyle=":", linewidth=1, alpha=0.6)
            
            ax_pr.text(
                0.97, 0.03,
                f"AUPR: {np.mean(res['auprs']):.3f} ({np.std(res['auprs']):.3f})",
                color="k",
                bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
                fontsize=11, ha="right", va="bottom", transform=ax_pr.transAxes
            )
    
        # Hide unused subplots
        total_positions = nrows_targets * ncols
        for i in range(n_t, total_positions):
            target_row = i // ncols
            target_col = i % ncols
            axes[target_row * 2, target_col].axis('off')
            axes[target_row * 2 + 1, target_col].axis('off')
    
        # Legend from first plot
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if "No Skill" in labels:
            idx = labels.index("No Skill")
            handles.pop(idx)
            labels.pop(idx)
        
        fig.legend(
            handles, labels,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            title="Legend",
            ncol=1,
            fontsize=12
        )
    
        # plt.suptitle("USB on UKBB - Model Performance Overview", fontsize=22)
        plt.savefig(filename, bbox_inches="tight", dpi=300)
        plt.close()
        print(f"Saved: {filename}")
    
    # Call after the loop ends
    plot_combined_overview(all_results, y_dict, ncols=3)
    

if __name__ == "__main__":
    
    data = pd.read_csv("src/src_data/all_USB_data_risk_consent_dateshift.csv", index_col=0)
    
    preds_sex = pd.concat([pd.read_csv(f"Data_sex_stratification/predictions/dt_preds_RF_repetition{i}.csv", index_col=0) for i in range (10)], ignore_index=True).reset_index(drop=True)
    preds_preg = pd.concat([pd.read_csv(f"Data_preg_stratification/predictions/dt_preds_RF_repetition{i}.csv", index_col=0) for i in range (10)], ignore_index=True).reset_index(drop=True)
    preds_age = pd.concat([pd.read_csv(f"Data_age_stratification/predictions/dt_preds_RF_repetition{i}.csv", index_col=0) for i in range (10)], ignore_index=True).reset_index(drop=True)
    preds_res = pd.concat([pd.read_csv(f"Data_res_stratification/predictions/dt_preds_RF_repetition{i}.csv", index_col=0) for i in range (10)], ignore_index=True).reset_index(drop=True)
    
    # print("\nmale")
    # validate_male(data, preds_sex)

    # print("\nfemale")
    # validate_female(data, preds_sex)

    # print("\npreg")
    # validate_pregnant(data, preds_preg)

    # print("\nage")
    # validate_age_groups(data, preds_age)

    # print("\nres and nores")
    # validate_res_and_no_prev_resistance(data, preds_res)

    
    these_targets = [
        "urine_antibiogram_AC",
        # "urine_antibiogram_CXM",
        "urine_antibiogram_SXT",
        # "urine_antibiogram_FOT",
        # "urine_antibiogram_NFT",
        # # Norfloxacin does not exist in UKBB,
        # "urine_antibiogram_CIP",
        # "urine_antibiogram_CTR",
        # "urine_antibiogram_PT",
    ]
    validate_adults_on_children(targets=these_targets)

