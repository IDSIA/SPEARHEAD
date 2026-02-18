# --------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import warnings
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    auc,
    roc_auc_score
)

from scipy.stats import chi2_contingency, fisher_exact, mannwhitneyu, spearmanr

from ut import *


warnings.filterwarnings("ignore")
matplotlib.use("Agg")
#sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------

def calculate_univariate_pvalues(X, targets_data, test_type='auto'):
    """
    Calculate univariate p-values between predictors and targets.
    
    Args:
        X (pd.DataFrame): Predictor variables
        targets_data (pd.DataFrame): Target variables
        test_type (str): 'auto', 'chi2', 'fisher', 'mannwhitney', 'ttest'
    
    Returns:
        pd.DataFrame: P-values matrix (predictors x targets)
    """
    
    p_values = pd.DataFrame(index=X.columns, columns=targets_data.columns)
    
    for predictor in X.columns:
        for target in targets_data.columns:
            
            # Get non-null pairs
            mask = ~(X[predictor].isna() | targets_data[target].isna())
            x_vals = X[predictor][mask]
            y_vals = targets_data[target][mask]
            
            if len(x_vals) < 2:
                p_values.loc[predictor, target] = np.nan
                continue
            
            try:
                # Determine test type automatically
                x_unique = len(x_vals.unique())
                y_unique = len(y_vals.unique())
                
                if test_type == 'auto':
                    # Binary target
                    if y_unique == 2:
                        # Binary predictor
                        if x_unique == 2:
                            # Fisher's exact test for 2x2 contingency
                            crosstab = pd.crosstab(x_vals, y_vals)
                            if crosstab.shape == (2, 2):
                                _, p_val = fisher_exact(crosstab)
                            else:
                                p_val = np.nan
                        # Continuous predictor
                        elif x_unique > 10:
                            # Mann-Whitney U test (non-parametric t-test)
                            groups = [x_vals[y_vals == val] for val in y_vals.unique()]
                            if len(groups) == 2 and len(groups[0]) > 0 and len(groups[1]) > 0:
                                _, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                            else:
                                p_val = np.nan
                        # Categorical predictor
                        else:
                            # Chi-square test
                            crosstab = pd.crosstab(x_vals, y_vals)
                            if crosstab.min().min() >= 5:  # Expected frequency check
                                _, p_val, _, _ = chi2_contingency(crosstab)
                            else:
                                p_val = np.nan
                    
                    # Continuous target
                    else:
                        # Binary predictor
                        if x_unique == 2:
                            groups = [y_vals[x_vals == val] for val in x_vals.unique()]
                            if len(groups) == 2 and len(groups[0]) > 0 and len(groups[1]) > 0:
                                _, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                            else:
                                p_val = np.nan
                        # Continuous predictor
                        else:
                            # Spearman correlation (non-parametric)
                            corr, p_val = spearmanr(x_vals, y_vals)
                
                else:
                    # Manual test type selection
                    if test_type == 'chi2':
                        crosstab = pd.crosstab(x_vals, y_vals)
                        _, p_val, _, _ = chi2_contingency(crosstab)
                    elif test_type == 'fisher':
                        crosstab = pd.crosstab(x_vals, y_vals)
                        _, p_val = fisher_exact(crosstab)
                    elif test_type == 'mannwhitney':
                        groups = [x_vals[y_vals == val] for val in y_vals.unique()]
                        _, p_val = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    elif test_type == 'ttest':
                        groups = [x_vals[y_vals == val] for val in y_vals.unique()]
                        _, p_val = ttest_ind(groups[0], groups[1])
                
                p_values.loc[predictor, target] = p_val
                
            except Exception as e:
                warnings.warn(f"Error calculating p-value for {predictor} vs {target}: {e}")
                p_values.loc[predictor, target] = np.nan
    
    return p_values.astype(float)

def plot_pvalue_heatmap(p_values, title="Univariate P-values: Predictors vs Targets", 
                       figsize=(15, 11), save_path=None, remove_empty=True):
    """
    Create a heatmap of p-values with significance annotations.
    
    Args:
        p_values (pd.DataFrame): P-values matrix
        title (str): Plot title
        figsize (tuple): Figure size
        save_path (str): Path to save the plot
        remove_empty (bool): Whether to remove rows/columns with all NaN or non-significant values
    """
    
    # Remove completely empty rows and columns
    if remove_empty:
        # Remove rows where all values are NaN
        p_values_clean = p_values.dropna(axis=0, how='all')
        
        # Remove columns where all values are NaN
        p_values_clean = p_values_clean.dropna(axis=1, how='all')
        
        # Remove rows where all values are either NaN or > 0.05 (non-significant)
        significant_rows = []
        for idx in p_values_clean.index:
            row_vals = p_values_clean.loc[idx]
            has_significant = (row_vals < 0.05).any()
            if has_significant:
                significant_rows.append(idx)
        
        if significant_rows:
            p_values_clean = p_values_clean.loc[significant_rows]
        
        # Remove columns where all values are either NaN or > 0.05
        significant_cols = []
        for col in p_values_clean.columns:
            col_vals = p_values_clean[col]
            has_significant = (col_vals < 0.05).any()
            if has_significant:
                significant_cols.append(col)
        
        if significant_cols:
            p_values_clean = p_values_clean[significant_cols]
        
        print(f"Original shape: {p_values.shape}")
        print(f"After removing empty/non-significant: {p_values_clean.shape}")
        
        if p_values_clean.empty:
            print("Warning: No significant associations found!")
            return
    else:
        p_values_clean = p_values
    
    # Create significance annotations
    def significance_stars(p_val):
        if pd.isna(p_val):
            return ''
        elif p_val < 0.001:
            return '***'
        elif p_val < 0.01:
            return '**'
        elif p_val < 0.05:
            return '*'
        else:
            return ''
    
    # Create annotation matrix
    annot_matrix = p_values_clean.applymap(significance_stars)
    
    # Create the heatmap
    plt.figure(figsize=figsize)
    
    # Use -log10 transformation for better visualization
    log_p_values = -np.log10(p_values_clean.replace(0, 1e-300))  # Avoid log(0)
    
    ax = sns.heatmap(
        log_p_values,
        annot=annot_matrix,
        fmt='',
        cmap='viridis_r',  # Reverse viridis (yellow = significant)
        cbar_kws={'label': '-log10(p-value)\n(Higher = More Significant)'},
        xticklabels=True,
        yticklabels=True
    )
    
    # Add significance level lines to colorbar
    cbar = ax.collections[0].colorbar
    
    # Calculate threshold positions
    threshold_05 = -np.log10(0.05)   # ≈ 1.30
    threshold_01 = -np.log10(0.01)   # ≈ 2.00
    threshold_001 = -np.log10(0.001) # ≈ 3.00

    lw = 1.5
    cbar.ax.axhline(threshold_05, color='red', linestyle='--', linewidth=lw)
    cbar.ax.axhline(threshold_01, color='black', linestyle='--', linewidth=lw)
    cbar.ax.axhline(threshold_001, color='blue', linestyle='--', linewidth=lw)
    
    # Create custom legend for significance thresholds
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='red', linestyle='--', linewidth=lw, label='p = 0.05 (Significant)'),
        Line2D([0], [0], color='black', linestyle='--', linewidth=lw, label='p = 0.01 (Highly Significant)'),
        Line2D([0], [0], color='blue', linestyle='--', linewidth=lw, label='p = 0.001 (Very Highly Significant)'),
    ]
    
    # Add legend with explanation
    plt.legend(handles=legend_elements, 
              title='Significance Thresholds\n(Values ABOVE lines are significant)',
              bbox_to_anchor=(1.15, 1), 
              loc='upper left')
    
    plt.title(title)
    plt.xlabel('Target Variables (Antibiotics)')
    plt.ylabel('Predictor Variables')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.close()
    
    # Print summary with clear explanation
    significant_05 = (p_values_clean < 0.05).sum().sum()
    significant_01 = (p_values_clean < 0.01).sum().sum()
    significant_001 = (p_values_clean < 0.001).sum().sum()
    total_tests = p_values_clean.notna().sum().sum()
    
    print(f"\nSignificance Summary:")
    print(f"Total tests: {total_tests}")
    print(f"Significant (p < 0.05): {significant_05} ({significant_05/total_tests*100:.1f}%)")
    print(f"Highly significant (p < 0.01): {significant_01} ({significant_01/total_tests*100:.1f}%)")
    print(f"Very highly significant (p < 0.001): {significant_001} ({significant_001/total_tests*100:.1f}%)")
    print(f"\nInterpretation: Values ABOVE the threshold lines in the colorbar are significant.")
    print(f"Higher -log10(p-value) = Lower p-value = More significant association")
    
    return p_values_clean

def create_univariate_heatmap(X, targets_data):
    """
    Main function to create univariate p-value heatmap.
    
    Args:
        X (pd.DataFrame): Predictor variables
        targets_data (pd.DataFrame): Target variables (antibiotic resistance)
    """
    
    # Calculate p-values
    log(INFO, "Calculating univariate p-values...")
    p_values = calculate_univariate_pvalues(X, targets_data)
    
    # Create heatmap
    plot_pvalue_heatmap(
        p_values, 
        title="Univariate Association: Predictors vs Antibiotic Resistance",
        figsize=(30, 18),
        save_path="univariate_pvalues_heatmap.png"
    )
    
    return p_values
# ---------------------------------------------------------------



# ------------------------
# Evaluation Methods
# ------------------------
def __generate_dt_roc(preds_df: pd.DataFrame | pd.Series):
    """Function to melt predictions dataframe and return a groupby dataframe, to later use for plotting boxplots and other.

    Parameters
    ----------
    preds_df : pd.DataFrame
        The predictions dataframe

    Returns
    -------
    pd.DataFrameGroupBy
        A dataframe grouped by ["target", "repetition", "model"]
    """

    melted = pd.melt(
        pd.DataFrame(preds_df),
        id_vars=[
            # "ID",
            "target",
            "true_class",
            "fold",
            "model",
        ],
        var_name="var_name",
        value_name="Predicted_value",
    )

    # Factorize 'true_class' column
    melted["true_class"] = pd.Categorical(
        melted["true_class"], categories=[1, 0], ordered=True
    )

    # Group by columns and compute ROC AUC
    return_frame = melted.groupby(["target", "model"])

    return return_frame


def __prepare_for_boxplot(targets: list[str], dt_roc) -> pd.DataFrame:
    """
    Prepare data for boxplot by calculating ROC and PR curves.

    Parameters and Returns remain the same
    """
    results = []

    for group_name, df_group in dt_roc:
        # Check if we have at least two classes
        unique_classes = df_group["true_class"].nunique()

        if unique_classes < 2:
            # Skip or use default values for single-class cases
            auc_roc_score = float("nan")  # or 0.0 or 1.0 depending on your needs
            auc_pr_score = float("nan")
        else:
            try:
                auc_roc_score = roc_auc_score(
                    df_group["true_class"], df_group["Predicted_value"]
                )

                lr_precision, lr_recall, _ = precision_recall_curve(
                    df_group["true_class"], df_group["Predicted_value"]
                )
                auc_pr_score = auc(lr_recall, lr_precision)
            except ValueError as e:
                print(
                    f"Warning: Could not calculate metrics for group {group_name}: {str(e)}"
                )
                auc_roc_score = float("nan")
                auc_pr_score = float("nan")

        results.append(
            group_name
            + (
                df_group["true_class"].values,
                auc_roc_score,
                auc_pr_score,
            )
        )

    return pd.DataFrame(
        results,
        columns=[
            "target",
            "model",
            "true_class",
            "AUC-ROC",
            "AUC-PR",
        ],
    )


def __plot_boxplot(
    df: pd.DataFrame,
    custom_categories: list | None = None,
    plot_PR=False,
) -> None:
    """
    Plots the boxplot for the given data, adding boxplots for AUC-PR data.

    Parameters
    ----------
    df : pd.DataFrame
        The data transformed from a pd.DataFrameGroupBy object
    custom_categories : list | None, optional
        A list of custom categories to include in the plot (will be the categories that you see in the legend), by default None.
    plot_PR : bool
        plot the PR boxplots as well

    """

    sns.set_theme(style="whitegrid", font_scale=1)

    if custom_categories is None:
        hue_order = df["model"].unique()
        # Generate a color palette with a unique color for each category
        palette = sns.color_palette("tab10", len(hue_order))

    else:
        hue_order = custom_categories
        # Generate a color palette with a unique color for each category
        palette = sns.color_palette("tab10", len(hue_order))

    # # Map the palette to the categories
    custom_palette = dict(zip(sorted(hue_order, reverse=False), palette))



    df_long = pd.melt(
        df,
        id_vars=["model", "target", "true_class"],
        value_vars=(["AUC-ROC", "AUC-PR"] if plot_PR else ["AUC-ROC"]),
        var_name="Metric",
        value_name="Value",
    )

    g = sns.FacetGrid(
        df_long,
        row="Metric",
        margin_titles=True,
        height=4,
        aspect=1.5,
        sharey=True,
    )

    #g.map_dataframe(lambda **kwargs: plt.axhline(y=0.5, linestyle="--", color="grey"))

    g.map_dataframe(
        sns.boxplot,
        x="target",
        y="Value",
        fill=False,
        linewidth=2,
        hue="model",
        hue_order=hue_order,
        palette=custom_palette,
    )
    """
    def add_class_info(data, **kwargs):
        # Calculate class balance for this target
        pos_count = (
            data.loc[data["target"] == "urine_antibiogram_CFE", "true_class"]
            .values[0]
            .tolist()
            .count(True)
        )
        total_CFE = len(
            data.loc[data["target"] == "urine_antibiogram_CFE", "true_class"].values[0].tolist()
        )
        balance = (pos_count / total_CFE) * 100

        # Add text to the plot
        plt.text(
            0.02,
            0.12,
            f"Pos: {pos_count}/{total_CFE}\nBalance: {balance:.1f}%",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            fontsize=12,
        )

        # Calculate class balance for this dataset
        pos_count = (
            data.loc[data["target"] == "urine_antibiogram_CIP", "true_class"]
            .values[0]
            .tolist()
            .count(True)
        )
        total_CIP = len(
            data.loc[data["target"] == "urine_antibiogram_CIP", "true_class"].values[0].tolist()
        )
        balance = (pos_count / total_CIP) * 100

        # Add text to the plot
        plt.text(
            0.50,
            0.12,
            f"Pos: {pos_count}/{total_CIP}\nBalance: {balance:.1f}%",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="center",
            fontsize=12,
        )

        # Calculate class balance for this dataset
        pos_count = (
            data.loc[data["target"] == "urine_antibiogram_IMI", "true_class"]
            .values[0]
            .tolist()
            .count(True)
        )
        total_IMI = len(
            data.loc[data["target"] == "urine_antibiogram_IMI", "true_class"].values[0].tolist()
        )
        balance = (pos_count / total_IMI) * 100

        # Add text to the plot
        plt.text(
            1.0,
            0.12,
            f"Pos: {pos_count}/{total_IMI}\nBalance: {balance:.1f}%",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=12,
        )

    # Map the text addition function to each subplot
    #g.map_dataframe(add_class_info)
    """
    g.set(ylim=(0, 1))
    g.tick_params(axis='x', which='both', bottom=True, top=False)
    g.set_xticklabels(plt.gca().get_xticklabels(), rotation=45, ha="right")
    g.add_legend(title="Model")
    g.set_titles("")  # Removes all titles
    g.tight_layout()
    plt.savefig(f"figures/boxplot.png")
    plt.close()
    # print(f"Saved to: {name}")


def __shap_plots(
    X: pd.DataFrame,
    vals: np.ndarray,
    var_target: str,
    errorbar_data: np.ndarray,
    title: str,
):

    _, (ax_bar, ax_dot) = plt.subplots(
        1, 2, figsize=(20, 15), gridspec_kw={"width_ratios": [1, 2]}
    )
    print("SHAPES: %s, %s" % (vals.shape, X.shape))

    shap.summary_plot(
        vals,
        X,
        curr_axis=ax_dot,
        plot_feature_names=False,
        show=False,
        max_display=10,
        plot_type="dot",
        plot_size=None,
    )
    shap.summary_plot(
        vals,
        X,
        curr_axis=ax_bar,
        errorbar_sd=errorbar_data,
        plot_feature_names=True,
        show=False,
        max_display=10,
        color="grey",
        plot_type="bar",
        plot_size=None,
    )

    plt.suptitle(
        f"{title}",
        fontsize=20,
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    ax_dot.set_xlabel("SHAP value", fontsize=15)
    # ax_dot.set_yticklabels([])
    ax_bar.set_xlabel("mean(|SHAP value|)", fontsize=15)
    ax_bar.spines[["right", "top", "bottom"]].set_visible(False)
    # Save the combined plot
    # plt.show()
    plt.savefig(f"figures/{var_target}_shap.png")
    plt.close()


def plot_results_shap(targets, Xs, mean_shap_values, sd_shap_values):

    for var_target in targets:
        
        stacked_df = pd.concat(Xs[var_target], axis=0, keys=range(10))

        print(f"Shape of stacked_df for {var_target}: {stacked_df.shape}")

        if stacked_df.empty:
            print(f"stacked_df for {var_target} is empty. Skipping...")
            continue

        numerical_cols = stacked_df.columns[
            (stacked_df.dtypes != "bool")
            & (~stacked_df.apply(lambda col: set(col.unique()) <= {0, 1, 0.0, 1.0}))
        ]

        boolean_cols = stacked_df.columns[
            (stacked_df.dtypes == "bool")
            | (stacked_df.apply(lambda col: set(col.unique()) <= {0, 1, 0.0, 1.0}))
        ]

        # print(f"Numerical columns for {var_target}: {numerical_cols}")
        # print(f"Boolean columns for {var_target}: {boolean_cols}")

        if numerical_cols.empty:
            mean_numerical = pd.DataFrame()
            print(f"No numerical columns found for {var_target}. Skipping...")
        else:
            mean_numerical = stacked_df[numerical_cols].groupby(level=1).mean()
            # print(f"Mean numerical for {var_target}: {mean_numerical}")

        if boolean_cols.empty:
            mode_boolean = pd.DataFrame()
            print(f"No boolean columns found for {var_target}. Skipping...")

        else:
            mode_boolean = (
                stacked_df[boolean_cols]
                .groupby(level=1)
                .agg(lambda x: x.mode().iloc[0])
            )
            # print(f"Mode boolean for {var_target}: {mode_boolean}")

        if not mean_numerical.empty and not mode_boolean.empty:
            temp = pd.concat([mean_numerical, mode_boolean], axis=1)
        elif not mean_numerical.empty:
            temp = mean_numerical
        elif not mode_boolean.empty:
            temp = mode_boolean
        else:
            print(
                f"No valid data found for {var_target}. Skipping... Shap plots will not be generated."
            )
            continue

    
        #columns = [
        #    col for col in __get_train_columns(self.X) if col in temp.columns
        #]

        X_enc = temp#[columns]  # reorder columns as they should be

        shap_values_m = mean_shap_values.get(var_target)
        shap_values_sd = sd_shap_values.get(var_target)

        __shap_plots(
            X=X_enc,
            vals=shap_values_m,
            var_target=var_target,
            errorbar_data=shap_values_sd,
            title=var_target,
        )


def plot_results(targets, dt_preds):
    
    # Plot boxplots
    step1 = __generate_dt_roc(dt_preds)
    
    dt_roc = __prepare_for_boxplot(
        targets=targets,
        dt_roc=step1,
    )
    __plot_boxplot(dt_roc, plot_PR=False)
