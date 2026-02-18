# -*- coding: utf-8 -*-
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beeswarm import summary_legacy as summary_plot_mod
import matplotlib.gridspec as gridspec
import seaborn as sns
from typing import Iterable, Optional, List
import re

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, auc,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

import matplotlib 

from ut import *

print("Matplotlib version:", matplotlib.__version__)

# -----------------------------
is_ukbb = True
model_to_train = "RF"
# -----------------------------

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
    colors = sns.color_palette("mako", n_colors=10)
    
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
    # ax.set_xticklabels([])
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
    colors = sns.color_palette("mako", n_colors=10)
    
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
    # ax.set_xticklabels([])
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

    # brier_vals = []
    # any_curve = False
    # # colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Color palette for repetitions
    # colors = sns.color_palette("mako", n_colors=10)

    

    # for (model, repetition), g in df.groupby(["Model", "repetition"]):
    #     y = g["true_class"].to_numpy(dtype=int)
    #     p = g["Pred"].to_numpy(dtype=float)

    #     if y.size == 0:
    #         continue

    #     # Brier score (lower is better)
    #     bs = brier_score_loss(y, p)
    #     brier_vals.append(bs)

    #     # Calibration curve
    #     try:
    #         prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy=strategy)
    #     except Exception:
    #         prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy="uniform")

    #     if not any_curve:  # Draw reference line only once
    #         ax_main.plot([0, 1], [0, 1], ls="--", color="gray", lw=2)
            
    #     color = colors[int(repetition) % len(colors)]
    #     ax_main.plot(prob_pred, prob_true, marker="o", lw=2, 
    #                 color=color,
    #                 label=f"Run {repetition}", 
    #                 alpha=0.7)
    #     any_curve = True

    #     # Histogram for each repetition (will overlay)
    #     ax_den.hist(p, bins=40, range=(0, 1), stacked=True, density=True, alpha=0.5, 
    #                color=color, edgecolor="white", linewidth=0.3)

    brier_vals = []
    any_curve = False
    colors = sns.color_palette("mako", n_colors=10)
    
    # Collect histogram data
    all_p = []
    all_colors = []
    
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
    
        if not any_curve:
            ax_main.plot([0, 1], [0, 1], ls="--", color="gray", lw=2)
            
        color = colors[int(repetition) % len(colors)]
        ax_main.plot(prob_pred, prob_true, marker="o", lw=2, 
                    color=color,
                    label=f"Run {repetition}", 
                    alpha=0.7)
        any_curve = True
        
        # Collect for stacked histogram
        all_p.append(p)
        all_colors.append(color)
    
    # Single stacked histogram call AFTER the loop
    ax_den.hist(all_p, bins=40, range=(0, 1), stacked=True, density=True, 
                alpha=0.7, color=all_colors, edgecolor="white", linewidth=0.3)

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
    fig_h = nrows_targets * 2 * plot_size * 0.8 + 1

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

    # # Add colored band for alternating target rows (only blue)
    # from matplotlib.patches import Rectangle
    
    # for target_row in range(nrows_targets):
    #     # Only color every other row (0, 2, 4, ...) in blue
    #     if target_row % 2 != 0:
    #         roc_row = target_row * 2
    #         pr_row = target_row * 2 + 1
            
    #         # Get the position of subplots to calculate band position
    #         ax_top = axes[roc_row, 0]
    #         ax_bottom = axes[pr_row, 0]
            
    #         # Get positions in figure coordinates
    #         bbox_top = ax_top.get_position()
    #         bbox_bottom = ax_bottom.get_position()
            
    #         # Extend the band to include titles and labels
    #         # Add padding above (for titles) and below (for x-labels)
    #         padding_top = 0.015  # Extra space for titles
    #         padding_bottom = 0.015  # Extra space for x-labels
            
    #         # Create a rectangle spanning the entire width and both rows + padding
    #         rect = Rectangle(
    #             (0, bbox_bottom.y0 - padding_bottom),  # x, y (bottom-left corner, extended down)
    #             1.05,  # width (full figure width)
    #             (bbox_top.y1 - bbox_bottom.y0) + padding_top + padding_bottom,  # height (span both rows + padding)
    #             transform=fig.transFigure,
    #             facecolor='#E3F2FD',  # Light blue
    #             edgecolor='none',
    #             zorder=-1,  # Behind everything
    #             alpha=0.5  # Slightly transparent
    #         )
    #         fig.patches.append(rect)

    # for target_row in range(nrows_targets):
    #     # Only color every other row (0, 2, 4, ...) in blue
    #     if target_row % 2 != 0:
    #         roc_row = target_row * 2
    #         pr_row = target_row * 2 + 1
        
        # # Add background to both ROC and PR rows for this target group
        # for col in range(ncols):
        #     # Background for ROC
        #     # axes[roc_row, col].set_facecolor('#E3F2FD')
        #     axes[roc_row, col].patch.set_alpha(0)
        #     # Background for PR
        #     # axes[pr_row, col].set_facecolor('#E3F2FD')
        #     axes[pr_row, col].patch.set_alpha(0)

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
        antib_dict = {
                "AC": ["Amoxicillin clavulansäure"],
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
        title = antib_dict[tar.split("_")[-1]][0]
        plot_title = rename_title(title)
        ax_roc.set_title(plot_title, fontweight="bold", fontsize=20, pad=8)

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

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {filename}")
    
def overview_calib(dt_preds: pd.DataFrame,
             targets: Optional[Iterable[str]] = None,
             ncols: int = 3,
             filename: str = "compact_overview_calib_RF.png",
             calib_bins: int = 10,
             calib_strategy: str = "quantile") -> None:
    """
    Plot a compact overview with Calibration and Probability Density
    for each target (columns). If targets exceed ncols, wrap to additional rows.
    """
    sns.set_theme(style="white", font_scale=1.5)
    print("Plotting and saving calibration plots...")

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
    
    # Calculate number of rows needed (each target needs 2 plot rows: Calib + Density)
    nrows_targets = int(np.ceil(n_t / ncols))
    total_plot_rows = nrows_targets * 2
    
    plot_size = 5
    fig_w = ncols * plot_size * 1.2
    # Account for calibration (square) + density (shorter) per target row
    fig_h = nrows_targets * (plot_size + plot_size * 0.05) + 1.5

    # Height ratios: calibration=1, density=0.3
    height_ratios = [1, 0.3] * nrows_targets
    
    fig = plt.figure(figsize=(fig_w, fig_h))
    
    # Use nested gridspec: outer for target rows, inner for calib+density pairs
    outer_gs = fig.add_gridspec(
        nrows_targets, 1,
        hspace=0.25  # Space BETWEEN target groups (for titles)
    )
    
    axes = []
    for target_row in range(nrows_targets):
        # Inner gridspec for this target row's calib + density
        inner_gs = outer_gs[target_row].subgridspec(
            2, ncols,
            height_ratios=[1, 0.3],
            hspace=0.08,  # Tight spacing between calib and density
            wspace=0.25
        )
        row_axes = [[fig.add_subplot(inner_gs[i, j]) for j in range(ncols)] for i in range(2)]
        axes.append(row_axes)
    
    plt.subplots_adjust(top=0.94, bottom=0.08, left=0.08, right=0.92)

    # Antibiotic name mapping
    antib_dict = {
        "AC": ["Amoxicillin clavulansäure"],
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
        "PM": ["Cefepim"],
        "PT": ["Piperacillin - Tazobactam"],
        "RAM": ["Rifampicin"],
        "SXT": ["Cotrimoxazol"],
        "TE": ["Tetracyclin"],
        "TEI": ["Teicoplanin"],
        "TGC": ["Tigecyclin"],
        "TOB": ["Tobramycin"],
        "VAN": ["Vancomycin"],
    }

    handles_accum, labels_accum = [], []

    for i, tar in enumerate(targets):
        print(tar)
        target_row = i // ncols
        target_col = i % ncols
        
        ax_calib = axes[target_row][0][target_col]
        ax_density = axes[target_row][1][target_col]
        
        # Format title using antib_dict (same as overview function)
        title = antib_dict[tar.split("_")[-1]][0]
        plot_title = rename_title(title)
        ax_calib.set_title(plot_title, fontweight="bold", fontsize=20, pad=8)

        # Call calibration plot function
        brier = overview_internal_plot_calib(
            dt_melted, 
            [(ax_calib, ax_density)], 
            0, 
            tar,
            n_bins=calib_bins, 
            strategy=calib_strategy
        )

        # Flip the density plot y-axis (bars hang down from calibration plot)
        ax_density.invert_yaxis()
        
        # Remove x-axis labels from calibration plot (shared with density below)
        ax_calib.set_xlabel("")
        # ax_calib.set_xticklabels([])
        ax_calib.tick_params(axis='x', length=0)  # Hide x tick marks
        
        # Set x-axis label only on density plot
        ax_density.set_xlabel("Predicted probability")
        
        # Reduce density y-ticks to avoid overlap (keep consistent font size)
        ax_density.yaxis.set_major_locator(plt.MaxNLocator(3))

        if brier is not None:
            ax_calib.text(
                0.97, 0.03, f"Brier: {np.mean(brier):.3f} ({np.std(brier):.3f})",
                color="k", bbox=dict(facecolor="white", edgecolor="k", boxstyle="round", alpha=0.85),
                fontsize=14, ha="right", va="bottom", transform=ax_calib.transAxes
            )
            print(f"Brier: {np.mean(brier):.3f} ({np.std(brier):.3f})")

        if i == 0:
            h, l = ax_calib.get_legend_handles_labels()
            for hh, ll in zip(h, l):
                if ll not in labels_accum:
                    labels_accum.append(ll)
                    handles_accum.append(hh)

        # Y-labels: only left-most column
        if target_col != 0:
            ax_calib.set_ylabel("")
            ax_density.set_ylabel("")
            # ax_calib.set_yticklabels([])
            # ax_density.set_yticklabels([])
        else:
            ax_calib.set_ylabel("Fraction of positives")
            ax_density.set_ylabel("Density")

        ax_calib.grid(True, axis='both', linestyle=":", linewidth=1, alpha=0.6)
        print()

    # Hide unused subplots
    total_positions = nrows_targets * ncols
    for i in range(n_t, total_positions):
        target_row = i // ncols
        target_col = i % ncols
        axes[target_row][0][target_col].axis('off')
        axes[target_row][1][target_col].axis('off')

    if handles_accum:
        fig.legend(
            handles_accum,
            labels_accum,
            loc="center left",
            bbox_to_anchor=(0.98, 0.5),
            title="Legend",
            ncol=1,
            fontsize=16
        )

    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved: {filename}")


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

def shap_helper(targets, data_folder, suffix=""):

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
        
        # # save all data used for shap, so that we can plot without having to rerun all the models every time
        # shap_values_combined = np.load(
        #     f"{data_folder}/for_shap/{model_to_train}/shap_values_combined_{this_target_name}_repetition0{'' if not is_ukbb else '_UKBB'}.npy"
        # )
        # X_all = pd.read_csv(
        #     f"{data_folder}/for_shap/{model_to_train}/X_all_{this_target_name}_repetition0{'' if not is_ukbb else '_UKBB'}.csv", index_col=0
        # ).reset_index(drop=True)

        # Load all 10 repetitions and average
        shap_values_list = []
        X_all_list = []
        
        for rep in range(10):
            shap_values = np.load(
                f"{data_folder}/for_shap/{model_to_train}/shap_values_combined_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition{rep}.npy"
            )
            shap_values_list.append(shap_values)
            
            X_all = pd.read_csv(
                f"{data_folder}/for_shap/{model_to_train}/X_all_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition{rep}.csv", index_col=0
            ).reset_index(drop=True)
            X_all_list.append(X_all)
        
        # Stack all repetitions: shape (n_repetitions, n_samples, n_features)
        shap_values_stacked = np.array(shap_values_list)
        
        # Average SHAP values across repetitions: shape (n_samples, n_features)
        shap_values_combined = np.mean(shap_values_stacked, axis=0)
        
        # Compute global SHAP values per repetition: mean(|SHAP|) per feature
        # Shape: (n_repetitions, n_features)
        global_shap_per_rep = np.array([np.abs(sv).mean(axis=0) for sv in shap_values_list])
        
        # Standard deviation of global SHAP values across repetitions: shape (n_features,)
        errorbar_sd = np.std(global_shap_per_rep, axis=0)
        
        # Use the first X_all
        X_all = X_all_list[0]

        print("SHAP:", shap_values_combined.shape, ", X:", X_all.shape)

        _, (ax_bar, ax_dot) = plt.subplots(
            1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 2]}
        )
        
        rename_features(X_all) # RENAME THE FEATURE NAMES
        
        summary_plot_mod(
            shap_values_combined[X_all.index],
            X_all.drop(columns="patnr"),
            curr_axis=ax_dot,
            plot_feature_names=False,
            show=False,
            max_display=20,
            plot_type="dot",
            plot_size=None,
        )
        summary_plot_mod(
            shap_values_combined[X_all.index],
            X_all.drop(columns="patnr"),
            curr_axis=ax_bar,
            errorbar_sd=errorbar_sd,
            plot_feature_names=True,
            show=False,
            max_display=20,
            color="grey",
            plot_type="bar",
            plot_size=None,
        )
    
        ax_dot.set_xlabel("SHAP value", fontsize=15)
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
        plt.suptitle(plot_title, fontsize=20)
        
        plt.savefig(
            f"figures/{model_to_train}_figures/shap_{this_target_name}{'' if not is_ukbb else '_UKBB'}.png"
        )
        plt.close()


if __name__ == "__main__":
    
    # for rep in range(10):
    #     df = pd.concat([
    #         pd.read_csv(f"Data/predictions/dt_preds_LR_urine_antibiogram_{i}_UKBB_repetition{rep}.csv", index_col=0) 
    #         for i in ["AC", "SXT", "FOT", "CIP", "PT"]
    #     ], ignore_index=True).reset_index(drop=True)
        
    #     df.to_csv(f"Data/predictions/dt_preds_LR_UKBB_repetition{rep}.csv")
        
    targets = [
        "urine_antibiogram_AC",
        # "urine_antibiogram_CXM",
        "urine_antibiogram_SXT",
        # "urine_antibiogram_FOT",
        # "urine_antibiogram_NFT",
        # # Norfloxacin does not exist in UKBB,
        # "urine_antibiogram_CIP",
        # "urine_antibiogram_CTR",
        "urine_antibiogram_PT",
    ]

    dt_preds = pd.concat([pd.read_csv(f"Data/predictions/dt_preds_RF_UKBB_repetition{i}.csv", index_col=0) for i in range (10)], ignore_index=True).reset_index(drop=True)
    
    overview(
        dt_preds,
        targets=targets,
        ncols=3,
        filename="figures/compact_overview_RF_ukbb.png",
        calib_bins=10,           # tweak as you like
        calib_strategy="quantile"  # "uniform" also OK
    )

    overview_calib(
        dt_preds,
        targets=targets,
        ncols=3,
        filename="figures/compact_overview_calib_RF_ukbb.png",
        calib_bins=10,           # tweak as you like
        calib_strategy="quantile"  # "uniform" also OK
    )

    shap_helper(targets, "Data")