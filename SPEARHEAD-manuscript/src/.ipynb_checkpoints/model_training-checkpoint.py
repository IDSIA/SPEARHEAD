# --------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------
import math
import re
from pathlib import Path
import joblib

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import StratifiedGroupKFold, train_test_split, cross_validate
from sklearn.metrics import (
    roc_curve, auc, f1_score, balanced_accuracy_score, accuracy_score,
    confusion_matrix, precision_recall_curve, average_precision_score, log_loss,
)
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier
import miceforest as mf

from hyperopt import hp, fmin, tpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss
from hyperopt.exceptions import AllTrialsFailed
from hyperopt.pyll.base import scope

from data_processing import pipeline_func, pipeline_func_UKBB
from ut import *
from beeswarm import summary_legacy as summary_plot_mod

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def _nunique(y):
    return np.unique(y) if not hasattr(y, "nunique") else y.nunique()

def _slice(X, idx):
    return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

def tuning_logreg(X_train, y_train, groups, max_evals=300, seed=42):
    """Return an unfitted Pipeline(StandardScaler -> LogisticRegression) with tuned params."""

    def constant_baseline_pred(y_tr, n_te):
        p_pos = float(np.mean(y_tr)) if len(y_tr) else 0.0
        return np.full(n_te, p_pos, dtype=float)

    # quick global sanity
    if _nunique(y_train) < 2:
        # Return a simple baseline model so outer code can proceed.
        clf = LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            max_iter=300,
            n_jobs=4,
            random_state=seed,
            class_weight="balanced",
        )
        scaler = StandardScaler(
            with_mean=not getattr(getattr(X_train, "sparse", None), "to_coo", None)
        )
        return Pipeline([("scaler", scaler), ("clf", clf)])

    # ----- search space -----
    space = {
        "penalty": hp.choice("penalty", ["l2", "l1", "elasticnet"]),
        "C": hp.loguniform("C", np.log(1e-4), np.log(1e2)),
        "l1_ratio": hp.uniform("l1_ratio", 0.0, 1.0),  # only for elasticnet
        "class_weight": hp.choice("class_weight", [None, "balanced"]),
        "max_iter": hp.quniform("max_iter", 200, 1500, 50),
        "fit_intercept": hp.choice("fit_intercept", [True, False]),
        "tol": hp.loguniform("tol", np.log(1e-6), np.log(1e-3)),
    }

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    def _objective(params):
        penalty = params["penalty"]
        C = float(params["C"])
        max_iter = int(params["max_iter"])
        fit_intercept = bool(params["fit_intercept"])
        class_weight = params["class_weight"]
        tol = float(params["tol"])
        l1_ratio = float(params["l1_ratio"]) if penalty == "elasticnet" else None

        lr = LogisticRegression(
            solver="saga",
            penalty=penalty,
            C=C,
            l1_ratio=l1_ratio,
            max_iter=max_iter,
            tol=tol,
            n_jobs=4,
            random_state=seed,
            class_weight=class_weight,
            fit_intercept=fit_intercept,
        )

        # scaler: if sparse, don't center
        sparse_like = bool(getattr(getattr(X_train, "sparse", None), "to_coo", None))
        scaler = StandardScaler(with_mean=not sparse_like, with_std=True)

        pipe = Pipeline([("scaler", scaler), ("clf", lr)])

        losses = []
        valid_folds = 0

        for tr_idx, te_idx in cv.split(X_train, y_train, groups=groups):
            X_tr, X_te = _slice(X_train, tr_idx), _slice(X_train, te_idx)
            y_tr, y_te = _slice(y_train, tr_idx), _slice(y_train, te_idx)

            # skip unusable folds
            if _nunique(y_tr) < 2 or _nunique(y_te) < 2:
                continue

            try:
                pipe.fit(X_tr, y_tr)
                proba = pipe.predict_proba(X_te)
                classes_ = pipe.named_steps["clf"].classes_
                pos_idx = int(np.where(classes_ == 1)[0][0]) if 1 in classes_ else None
                p = proba[:, pos_idx] if pos_idx is not None else np.zeros(len(X_te))
                loss = log_loss(y_te, p, labels=[0, 1])
                losses.append(loss)
                valid_folds += 1
            except Exception:
                # fold failed—penalize
                losses.append(10.0)

        if valid_folds == 0 or len(losses) == 0 or not np.isfinite(np.mean(losses)):
            return {"loss": float("inf"), "status": STATUS_FAIL}

        return {"loss": float(np.mean(losses)), "status": STATUS_OK}

    trials = Trials()

    try:
        best = fmin(
            fn=_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            rstate=np.random.default_rng(seed),
            trials=trials,
            early_stop_fn=no_progress_loss(30),
        )
    except AllTrialsFailed:
        log(ERROR, "All trials have failed. Skipping this outer fold.")
        return "empty"

    # decode choices
    penalties = ["l2", "l1", "elasticnet"]
    fit_opts = [True, False]
    cw_opts = [None, "balanced"]

    penalty = penalties[best["penalty"]]
    best_params = {
        "penalty": penalty,
        "C": float(best["C"]),
        "l1_ratio": float(best["l1_ratio"]) if penalty == "elasticnet" else None,
        "max_iter": int(best["max_iter"]),
        "fit_intercept": fit_opts[best["fit_intercept"]],
        "class_weight": cw_opts[best["class_weight"]],
        "solver": "saga",
        "n_jobs": 4,
        "tol": float(best["tol"]),
        "random_state": seed,
    }

    # final estimator (unfitted; fit it in your outer flow)
    sparse_like = bool(getattr(getattr(X_train, "sparse", None), "to_coo", None))
    scaler = StandardScaler(with_mean=not sparse_like, with_std=True)
    clf = LogisticRegression(**{k: v for k, v in best_params.items() if v is not None})
    pipe_best = Pipeline([("scaler", scaler), ("clf", clf)])
    return pipe_best


def tuning_xgboost(X_train, y_train, groups, max_evals=300, seed=42):

    # bail out early if only one class overall
    if _nunique(y_train) < 2:
        return XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            min_child_weight=1,
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
        )

    n_splits = 5
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ---- search space ----
    max_d = calculate_max_depth(X_train.shape)
    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 100),
        "max_depth": scope.int(hp.quniform("max_depth", 2, max_d, 1)),
        "learning_rate": hp.loguniform("learning_rate", np.log(1e-3), np.log(1e-1)),
        "subsample": hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "min_child_weight": hp.quniform("min_child_weight", 1, 10, 1),
        # optionally add:
        # "reg_lambda": hp.loguniform("reg_lambda", np.log(1e-3), np.log(10)),
        # "reg_alpha":  hp.loguniform("reg_alpha",  np.log(1e-4), np.log(1)),
        # "scale_pos_weight": hp.loguniform("scale_pos_weight", np.log(0.2), np.log(10)),
    }

    def _objective(params):
        xgb = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]),
            min_child_weight=int(params["min_child_weight"]),
            objective="binary:logistic",
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            eval_metric="logloss",
            # reg_lambda=float(params.get("reg_lambda", 1.0)),
            # reg_alpha=float(params.get("reg_alpha", 0.0)),
            # scale_pos_weight=float(params.get("scale_pos_weight", 1.0)),
        )

        losses = []
        valid = 0

        for tr_idx, te_idx in cv.split(X_train, y_train, groups=groups):
            X_tr, X_te = _slice(X_train, tr_idx), _slice(X_train, te_idx)
            y_tr, y_te = _slice(y_train, tr_idx), _slice(y_train, te_idx)

            if _nunique(y_tr) < 2 or _nunique(y_te) < 2:
                continue

            try:
                xgb.fit(X_tr, y_tr)
                proba = xgb.predict_proba(X_te)
                classes_ = xgb.classes_
                pos_idx = int(np.where(classes_ == 1)[0][0]) if 1 in classes_ else None
                p = proba[:, pos_idx] if pos_idx is not None else np.zeros(len(X_te))
                losses.append(log_loss(y_te, p, labels=[0, 1]))
                valid += 1
            except Exception:
                losses.append(10.0)

        if valid == 0 or len(losses) == 0 or not np.isfinite(np.mean(losses)):
            return {"loss": float("inf"), "status": STATUS_FAIL}

        return {"loss": float(np.mean(losses)), "status": STATUS_OK}

    trials = Trials()

    try:
        best = fmin(
            fn=_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            rstate=np.random.default_rng(seed),
            trials=trials,
            early_stop_fn=no_progress_loss(30),
        )
    except AllTrialsFailed:
        log(ERROR, "All trials have failed. Skipping this outer fold.")
        return "empty"

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(max_d),  # your space used a fixed max_d
        "learning_rate": float(best["learning_rate"]),
        "subsample": float(best["subsample"]),
        "colsample_bytree": float(best["colsample_bytree"]),
        "min_child_weight": int(best["min_child_weight"]),
        # "reg_lambda": float(best.get("reg_lambda", 1.0)),
        # "reg_alpha":  float(best.get("reg_alpha", 0.0)),
        # "scale_pos_weight": float(best.get("scale_pos_weight", 1.0)),
    }

    clf = XGBClassifier(
        **best_params,
        objective="binary:logistic",
        tree_method="hist",
        n_jobs=1,
        random_state=seed,
        eval_metric="logloss",
    )
    return clf


def tuning_rf(X_train, y_train, groups, max_evals=300, seed=42):

    # quick global sanity: if only one class overall, return a baseline RF to keep pipeline alive
    if _nunique(y_train) < 2:
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=4,
            random_state=seed,
        )
        return clf

    n_splits = 5
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # ---- search space ----
    max_d = calculate_max_depth(X_train.shape)
    space = {
        "n_estimators": hp.quniform("n_estimators", 100, 1000, 100),
        "max_depth": scope.int(hp.quniform("max_depth", 2, max_d, 1)),
        "min_samples_split": hp.quniform("min_samples_split", 2, 10, 1),
        "min_samples_leaf": hp.quniform("min_samples_leaf", 1, 10, 1),
    }

    def _objective(params):
        rf = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]),
            min_samples_leaf=int(params["min_samples_leaf"]),
            n_jobs=4,
            random_state=seed,
        )

        losses = []
        valid = 0

        for tr_idx, te_idx in cv.split(X_train, y_train, groups=groups):
            X_tr, X_te = _slice(X_train, tr_idx), _slice(X_train, te_idx)
            y_tr, y_te = _slice(y_train, tr_idx), _slice(y_train, te_idx)

            if _nunique(y_tr) < 2 or _nunique(y_te) < 2:
                continue

            try:
                rf.fit(X_tr, y_tr)
                proba = rf.predict_proba(X_te)
                classes_ = rf.classes_
                pos_idx = int(np.where(classes_ == 1)[0][0]) if 1 in classes_ else None
                p = proba[:, pos_idx] if pos_idx is not None else np.zeros(len(X_te))
                losses.append(log_loss(y_te, p, labels=[0, 1]))
                valid += 1
            except Exception:
                losses.append(10.0)  # penalize failed fold

        if valid == 0 or len(losses) == 0 or not np.isfinite(np.mean(losses)):
            return {"loss": float("inf"), "status": STATUS_FAIL}

        return {"loss": float(np.mean(losses)), "status": STATUS_OK}

    trials = Trials()

    try:
        best = fmin(
            fn=_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=max_evals,
            rstate=np.random.default_rng(seed),
            trials=trials,
            early_stop_fn=no_progress_loss(30),
        )
    except AllTrialsFailed:
        log(ERROR, "All trials have failed. Skipping this outer fold.")
        return "empty"

    best_params = {
        "n_estimators": int(best["n_estimators"]),
        "max_depth": int(max_d),  # your space used a fixed max_d, keep it
        "min_samples_split": int(best["min_samples_split"]),
        "min_samples_leaf": int(best["min_samples_leaf"]),
    }

    clf = RandomForestClassifier(**best_params, n_jobs=4, random_state=seed)
    return clf

def calculate_max_depth(shape):

    n_rows, n_feat = shape

    log_depth = math.log2(n_feat)

    # Adjust depth based on dataset size
    if n_rows < 1000:  # Small dataset
        max_depth_complex = round(1.5 * log_depth)
    else:  # Larger dataset
        max_depth_complex = round(2 * log_depth)

    return max_depth_complex


def append_fold_predictions(
    dt_preds: pd.DataFrame,
    var_target: str,
    ytest: pd.DataFrame,
    yhat: np.ndarray,
    fold: int,
    current_rep: int,
    model: str,
    indices,
) -> pd.DataFrame:

    # df containing all results
    fold_predictions = pd.DataFrame(
        {
            "target": var_target,
            "true_class": ytest.values.ravel(),  # Actual labels or values
            "pred": yhat,  # Predictions from the model
            "fold": fold,
            "repetition": current_rep,
            "model": model,  # Concatenating model type name with suffix
            "indices": indices,
        }
    )

    # Append to existing predictions DataFrame
    return pd.concat([dt_preds, fold_predictions], ignore_index=True)


def _save_predictions(
    y_hat,
    target,
    fold: int,
    is_ukbb=False,
    model_to_train="RF",
    saveFileSuffix: str = "",
):
    """Save model predictions."""
    base_path = f"Data/models/saved_models_{model_to_train}"

    path = f"{base_path}/{target}"
    np.save(
        f"{path}/yhat_fold_{fold}{saveFileSuffix}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.npy",
        y_hat,
    )


def _save_model_data(
    y_test,
    target,
    fold: int,
    is_ukbb=False,
    model_to_train="RF",
    saveFileSuffix: str = "",
):
    """Save model training/test data."""
    base_path = f"Data/models/saved_models_{model_to_train}"

    path = f"{base_path}/{target}"

    y_test.to_csv(
        f"{path}/ytest_fold_{fold}{saveFileSuffix}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.csv"
    )
def _save_model_object(
    clf,
    target,
    fold: int,
    is_ukbb=False,
    model_to_train="RF",
    saveFileSuffix: str = "",
):
    """Save model training/test data."""
    base_path = f"Data/models/saved_models_{model_to_train}"

    path = f"{base_path}/{target}"

    joblib.dump(clf,
        f"{path}/{model_to_train}_fold_{fold}{saveFileSuffix}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.pkl"
    )

def to_timedelta_safe(s):
    if pd.isna(s):
        return pd.NaT
    # normalize whitespace and minus signs
    s = str(s).strip()
    s = s.replace("\u2212", "-")  # U+2212 minus → ASCII '-'
    s = re.sub(r"\s+", " ", s)  # collapse spaces
    # let pandas parse; coerce bad rows to NaT instead of raising
    return pd.to_timedelta(s, errors="coerce")


def main(
    data,
    y,
    drop_nan_perc: float = 40.0,
    use_risk=False,
    is_premodel=False,
    model_to_train: str = "RF",
    tuning_active=True,
    is_ukbb=False,
    rep=None,
):
    """
    Main function. Performs last step of preprocessing before fitting a model to the processed data.
    Preprocessing function is `pipeline_fun()` defined inside src/data_processing.py
    """

    import numpy as np
    import pandas as pd

    global pid
    global caseid
    global shift_col
    global current_rep; current_rep = rep

    pid = "patnr" if is_ukbb else "patient_id_hashed"
    caseid = "fallnr" if is_ukbb else "case_id_hashed"
    shift_col = "date_shifted_to_last_uti" if is_ukbb else "patient_date_shift"

    dt_preds = pd.DataFrame()
    target_col = y.name

    log(TARGET_NAME, target_col)
    # print(colored("\nTARGET NAME", "blue"), f": \t\t{target_col}") # antibiotic name

    # Variables with missing values in more than drop_nan_perc% of the patients were excluded.
    log(
        INFO,
        f"Variables with missing values in more than {drop_nan_perc}% of the patients were excluded.",
    )
    # print(colored(INFO, "green"), f": \t\t\tdropping columns that have more than {100 - drop_nan_perc}% of nans")

    # istantiate the pipeline defined inside src/data_processing.py
    if is_ukbb:
        pipeline = pipeline_func_UKBB(use_risk, is_premodel)
    else:
        pipeline = pipeline_func(use_risk, is_premodel)

    # clean targets and align with feature data
    # print(target.isna().sum())
    target = y.dropna()
    # print("target remaining rows:", target.shape[0])
    data = data.loc[target.index]  # Ensure data and target have the same index
    data, target = data.reset_index(drop=True), target.reset_index(drop=True)

    # plt.figure(figsize=(5, 4))

    # data.age.plot(kind="hist")
    # plt.xlabel("Age")
    # plt.ylabel("Frequency")
    # plt.tight_layout()
    # plt.savefig(f"figures/age_distr{'' if not is_ukbb else '_UKBB'}.png")

    # plt.figure(figsize=(3, 4))
    # data.sex.value_counts().plot(kind="bar")
    # plt.xlabel("Sex")
    # plt.ylabel("Frequency")
    # plt.tight_layout()
    # plt.savefig(f"figures/sex_distr{'' if not is_ukbb else '_UKBB'}.png")

    if not is_ukbb:
        prescriptions = pd.read_csv(
            "data_spearhead/df_SPEARHEAD_antibiotics_20250709.csv"
        )
        substance_name = "substance_name"
    else:
        prescriptions = pd.read_csv("dataset_UKBB/drug_prescriptions_final.csv")
        prescriptions = prescriptions.drop("date_shifted_to_last_uti", axis=1)

        # date_cols = prescriptions.columns[
        #     prescriptions.columns.str.contains("date", case=False)
        # ].to_list()

        # for c in date_cols:
        #     prescriptions[c] = prescriptions[c].map(to_timedelta_safe) / pd.Timedelta(days=1)
        substance_name = "active_substance"

    # plt.figure(figsize=(6, 4))
    # this = prescriptions[substance_name].value_counts().head(30)
    # x_labels = pd.Series(this.index.astype(str))
    # x_labels = [
    #     string if len(string) < 17 else string[:17] + "..." for string in x_labels
    # ]
    # this.index = x_labels
    # this.plot(kind="bar")
    # plt.xlabel("Drug prescribed")
    # plt.ylabel("Frequency")

    # plt.tight_layout()
    # plt.savefig(f"figures/most_common_AB_distr{'' if not is_ukbb else '_UKBB'}.png")
    # plt.close()

    # # ----------------------------------------
    # plt.figure(figsize=(6, 4))
    # this = data["mo"].value_counts().head(30)
    # x_labels = pd.Series(this.index.astype(str))
    # x_labels = [
    #     string if len(string) < 30 else string[:30] + "..." for string in x_labels
    # ]
    # this.index = x_labels
    # this.plot(kind="bar")
    # plt.xlabel("Bacteria")
    # plt.ylabel("Frequency")

    # plt.tight_layout()
    # plt.savefig(f"figures/most_common_bacteria{'' if not is_ukbb else '_UKBB'}.png")
    # plt.close()


    target_class_distribution = target.value_counts()

    if len(target_class_distribution) < 2:
        log(ERROR, "target_class_distribution length is less than 2 (there is only one unique value in target class).")
        return "empty"

    # imbalance_percentage = (
    #     target_class_distribution[1] / target_class_distribution.sum()
    # ) * 100

    # log(
    #     INFO,
    #     f"Imbalance percentage in {target.name}'s positive class: {imbalance_percentage:.3f}",
    # )


    # # ---------------------- imbalance stuff for subpopulation
    # men_ids = data.loc[data["sex"] == "männlich", "patient_id_hashed"].index

    # target_class_distribution = target[men_ids].value_counts()

    # if len(target_class_distribution) < 2:
    #     log(ERROR, "target_class_distribution length is less than 2 (there is only one unique value in target class).")
    #     return "empty"

    # imbalance_percentage = (
    #     target_class_distribution[1] / target_class_distribution.sum()
    # ) * 100

    # log(
    #     INFO,
    #     f"Imbalance percentage SUBPOPULATION MALE in {target.name}'s positive class: {imbalance_percentage:.3f}",
    # )

    # # ----------------------
    # women_ids = data.loc[data["sex"] == "weiblich", "patient_id_hashed"].index

    # target_class_distribution = target[women_ids].value_counts()

    # if len(target_class_distribution) < 2:
    #     log(ERROR, "target_class_distribution length is less than 2 (there is only one unique value in target class).")
    #     return "empty"

    # imbalance_percentage = (
    #     target_class_distribution[1] / target_class_distribution.sum()
    # ) * 100

    # log(
    #     INFO,
    #     f"Imbalance percentage SUBPOPULATION FEMALE in {target.name}'s positive class: {imbalance_percentage:.3f}",
    # )

    # # ----------------------
    # above65_ids = data.loc[data["age"] >= 65, "patient_id_hashed"].index

    # target_class_distribution = target[above65_ids].value_counts()

    # if len(target_class_distribution) < 2:
    #     log(ERROR, "target_class_distribution length is less than 2 (there is only one unique value in target class).")
    #     return "empty"

    # imbalance_percentage = (
    #     target_class_distribution[1] / target_class_distribution.sum()
    # ) * 100

    # log(
    #     INFO,
    #     f"Imbalance percentage SUBPOPULATION above65 in {target.name}'s positive class: {imbalance_percentage:.3f}",
    # )

    # # ----------------------
    # below65_ids = data.loc[(18 <= data["age"]) & (data["age"] < 65), "patient_id_hashed"].index

    # target_class_distribution = target[below65_ids].value_counts()

    # if len(target_class_distribution) < 2:
    #     log(ERROR, "target_class_distribution length is less than 2 (there is only one unique value in target class).")
    #     return "empty"

    # imbalance_percentage = (
    #     target_class_distribution[1] / target_class_distribution.sum()
    # ) * 100

    # log(
    #     INFO,
    #     f"Imbalance percentage SUBPOPULATION below65 in {target.name}'s positive class: {imbalance_percentage:.3f}",
    # )


    # print(colored(INFO, "green"), f": \t\t\tImbalance percentage in {target.name}'s positive class: {imbalance_percentage:.3f}")

    # ------------------------- CREATE VARIABLES -------------------------
    # Step 1: Compute `previous_resistance`
    temp = pd.concat([data, target], axis=1)  # No need for `reset_index(drop=True)`

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

    # def build_prev_exposure(
    #     prescriptions: pd.DataFrame, data: pd.DataFrame, keyword: str
    # ) -> pd.DataFrame:

    #     keys = [pid, caseid]

    #     # filter prescriptions to this antibiotic
    #     pres = prescriptions.copy()
    #     kw = str(keyword).lower()
    #     if is_ukbb:
    #         # using "antibiotic_code". creating this column using the dictionnary provided by UKBB.
    #         mask = (
    #             pres.get("antibiotic_code", "")
    #             .astype(str)
    #             .str.lower()
    #             .str.contains(kw, na=False)
    #         )
    #     else:
    #         mask = (
    #             pres.get("ATC_name", "")
    #             .astype(str)
    #             .str.lower()
    #             .str.contains(kw, na=False)
    #             | pres.get("drug_prescribed", "")
    #             .astype(str)
    #             .str.lower()
    #             .str.contains(kw, na=False)
    #             | pres.get(substance_name, "")
    #             .astype(str)
    #             .str.lower()
    #             .str.contains(kw, na=False)
    #         )
    #     pres = pres[mask]

    #     # attach case shifts (one row per (patient,case) on the right to avoid explosions)
    #     case_shifts = data[keys + [shift_col]].drop_duplicates(subset=keys)
    #     pres = pres.merge(case_shifts, on=keys, how="inner")

    #     # prior only; make coarse days-to-index from case shift
    #     pres = pres[pres[shift_col] < 0].copy()
    #     pres["days_to_index"] = pres[shift_col].abs()

    #     # (optional) de-dup: one exposure per (patient, case) is enough for flags
    #     pres = pres.drop_duplicates(subset=keys)

    #     # index rows that receive features
    #     idx = data[data[shift_col] == 0][keys].drop_duplicates().copy()
    #     out = idx.copy()

    #     # ALL = any prior exposure per patient
    #     s_all_flag = pres.groupby(pid).size().gt(0).astype(int)
    #     out["prev_exp_ALL"] = out[pid].map(s_all_flag).fillna(0).astype(int)
    #     print(out["prev_exp_ALL"].unique())

    #     # finite windows (binary flags)
    #     for lbl, lim in WINDOWS.items():
    #         if not np.isfinite(lim):
    #             continue
    #         sub = pres[pres["days_to_index"] <= lim]
    #         s_flag = sub.groupby(pid).size().gt(0).astype(int)
    #         out[f"prev_exp_{lbl}"] = out[pid].map(s_flag).fillna(0).astype(int)

    #     return out

    def build_prev_exposure(
        prescriptions: pd.DataFrame, data: pd.DataFrame, keyword: str
    ) -> pd.DataFrame:
    
        keys = [pid, caseid]
    
        # filter prescriptions to this antibiotic
        pres = prescriptions.copy()
        kw = str(keyword).lower()
        if is_ukbb:
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
    
        # attach case shifts
        case_shifts = data[keys + [shift_col]].drop_duplicates(subset=keys)
        pres = pres.merge(case_shifts, on=keys, how="inner")
    
        # prior only; coarse days-to-index from case shift
        pres = pres[pres[shift_col] < 0].copy()
        pres["days_to_index"] = pres[shift_col].abs()
    
        # de-dup: one exposure per (patient, case)
        pres = pres.drop_duplicates(subset=keys)
    
        # index rows that receive features
        idx = data[data[shift_col] == 0][keys].drop_duplicates().copy()
        out = idx.copy()
    
        # # --- existing windowed flags ---
        # # ALL = any prior exposure per patient
        # s_all_flag = pres.groupby(pid).size().gt(0).astype(int)
        # out["prev_exp_ALL"] = out[pid].map(s_all_flag).fillna(0).astype(int)
    
        # # finite windows (binary flags)
        # for lbl, lim in WINDOWS.items():
        #     if not np.isfinite(lim):
        #         continue
        #     sub = pres[pres["days_to_index"] <= lim]
        #     s_flag = sub.groupby(pid).size().gt(0).astype(int)
        #     out[f"prev_exp_{lbl}"] = out[pid].map(s_flag).fillna(0).astype(int)
    
        # --- new variables mirroring build_prev_resistance ---
    
        # multiple_occurrences_prev_exposure: >1 prior exposure events
        s_all_count = pres.groupby(pid).size()
        prev_exp_count = out[pid].map(s_all_count).fillna(0).astype(int)
        out["multiple_occurrences_prev_exposure"] = (prev_exp_count > 1).astype(int)
    
        # days_from_last_exposure: days since most recent prior exposure
        last_exp_days = pres.groupby(pid)["days_to_index"].min()
        out["days_from_last_exposure"] = out[pid].map(last_exp_days)
    
        # had_prev_exposure (ternary: 1 / 0 / -1)
        #   1  = had prior exposure
        #   0  = has prior data (any earlier case) but no exposure
        #  -1  = no prior data at all
        prior_all = data[data[shift_col] < 0]
        patients_with_prior = prior_all.groupby(pid).size()
    
        out["had_prev_exposure"] = pd.NA
        has_exposure = out[pid].isin(last_exp_days.index)
        out.loc[has_exposure, "had_prev_exposure"] = 1
        has_prior_data = out[pid].isin(patients_with_prior.index)
        out.loc[has_prior_data & ~has_exposure, "had_prev_exposure"] = 0
        out["had_prev_exposure"] = out["had_prev_exposure"].astype("Int64")
        out["had_prev_exposure"] = out["had_prev_exposure"].fillna(-1).astype(int)
    
        return out

    # 1) Resistance (per current target antibiotic)
    res_feats = build_prev_resistance(
        temp, target_col
    )  # temp contains all history rows for patients
    data = data.merge(res_feats, on=[pid, caseid], how="left")

    # 2) Exposure (per current target antibiotic)
    split_target = target.name.rsplit("_", 1)

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

    data = data.merge(exp_feats, on=[pid, caseid], how="left")

    accuracies, f1_scores, aurocs, auprs = [], [], [], []
    all_fpr, all_tpr, all_precision, all_recall = [], [], [], []
    # Store SHAP values from all folds
    all_shap_values = []
    all_features = []
    for_clinician_comparison = []


    # --------------------------------------------
    if is_ukbb:
        # drop all ID columns but the patient ID for stratification
        data = data.drop(["report_id", "urine_organismsubid"], axis=1)
    else:
        data = data.drop(["urine_sample_id_hashed", "urine_organism_id_hashed"], axis=1)

    # drop all date columns
    date_cols = [col for col in data.columns if "date" in col]
    data = data.drop(date_cols, axis=1)

    n_splits = 5

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # ---------------------------------------------------------------------------------------------------------------------
    log(WARNING, "USING ONLY LAST RECORD FOR EACH PATIENT AS DATASET")

    # Step 2: Take only the last record per patient
    X = data.groupby(pid, as_index=False).tail(1)

    # Step 3: Align target to X
    myTarget = target.loc[X.index]
    
    log(INFO, f"X shape after filtering to last record: {X.shape}")

    target_class_distribution = myTarget.value_counts()
    imbalance_percentage = (
        target_class_distribution[1] / target_class_distribution.sum()
    ) * 100

    log(
        INFO,
        f"Imbalance percentage in {target.name}'s positive class: {imbalance_percentage:.3f}",
    )
    
    SHAPE_LOG_PATH = Path(f"training_shapes{'_UKBB' if is_ukbb else ''}.log")
    
    def log_shape(target_name, X, imbalance):
        with open(SHAPE_LOG_PATH, "a") as f:
            f.write(f"{target_name}: X={X.shape}, Imbalance={imbalance}%\n")

    log_shape(target.name, X, imbalance_percentage)

    # def plot_target_nans(target_series):
    #     """
    #     Return counts for bar plot: positive, negative, and NaN values.
    #     """
    #     clean = target_series.dropna()
    #     positive = (clean == 1).sum()
    #     negative = (clean == 0).sum()
    #     nans = target_series.isna().sum()
        
    #     return {'Positive': positive, 'Negative': negative, 'NaN': nans}
        
    # log(INFO, "Computing data for antibiogram nan count.")
    
    # return plot_target_nans(y)
    
    # save the ids for gram predictions alignment
    all_test_indices = []
    all_valid_cols = []

    # Create age groups for stratification
    # X['age_groups'] = pd.cut(
    #     X['age'],
    #     bins=[18, 65, 80, float('inf')],
    #     labels=['0', '1', '2'], # 0: 18 ≤ age < 65, 1: 65 ≤ age < 80, 2: age ≥ 80
    #     right=False
    # )

    # stratification_col = "age_groups"
    # stratification_labels = X[stratification_col].astype(str) + "_" + myTarget.astype(str)

    # log(WARNING, f"stratification across '{stratification_col}' column")

    # check if there are columns that after splitting become less than 40% populated, and if so remove them completely
    cols_with_single_value = set()  # Track columns with only 1 unique non-null value across folds

    # check if there are columns that after splitting become less than 40% populated, and if so remove them completely
    for fold, (train_idx, test_idx) in enumerate(
        sgkf.split(X, myTarget, groups=X[pid])
    ):
        # for i, (train, test) in enumerate(create_patient_folds(data)):

        xtrain, xtest = X.iloc[train_idx], X.iloc[test_idx]
        ytrain, ytest = myTarget.iloc[train_idx], myTarget.iloc[test_idx]

        # print(f"Fold {i}: Train size: {len(train)}, Test size: {len(test)}")

        # xtrain, ytrain = train.drop(col, axis=1), train[col]
        # xtest, ytest = test.drop(col, axis=1), test[col]

        this_test_ids = xtest.loc[:, [pid, caseid]]

        xtrain, xtest = xtrain.drop([pid, caseid], axis=1), xtest.drop(
            [pid, caseid], axis=1
        )

        pipeline.fit(xtrain)

        xtrain = pipeline.transform(xtrain)
        xtest = pipeline.transform(xtest)

        if xtrain.isna().sum().sum() > 0 and xtest.isna().sum().sum() > 0:
            # Drop columns that are entirely NaN in X_train (miceforest can't impute these)
            all_nan_cols = xtrain.columns[xtrain.isna().all()].tolist()
            if all_nan_cols:
                log(INFO, f"Fold {fold+1}: Dropping {len(all_nan_cols)} columns with all NaN values: {all_nan_cols}")
                xtrain = xtrain.drop(columns=all_nan_cols)
                xtest = xtest.drop(columns=all_nan_cols)

        # if not is_premodel:
        #     print(xtrain["gram_prediction"].isna().mean() * 100)

        valid_cols_train = xtrain.columns[xtrain.isna().mean() * 100 < (drop_nan_perc)]
        valid_cols_test = xtest.columns[xtest.isna().mean() * 100 < (drop_nan_perc)]

        # Add intersection of valid columns in this fold
        all_valid_cols.append(set(valid_cols_train).intersection(valid_cols_test))

    final_valid_cols = set.intersection(*all_valid_cols)

    # Remove columns that have only 1 unique value and missing data (problematic for miceforest)
    if cols_with_single_value:
        log(INFO, f"Excluding {len(cols_with_single_value)} columns with single unique value and missing data: {cols_with_single_value}")
        final_valid_cols = final_valid_cols - cols_with_single_value
    
    log(INFO, f"Final valid columns count: {len(final_valid_cols)}")
    # print(final_valid_cols)

    temps = []
    error_count = 0
    for fold, (train_idx, test_idx) in enumerate(
        sgkf.split(X, myTarget, groups=X[pid])
    ):
        # for i, (train, test) in enumerate(create_patient_folds(data))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = myTarget.iloc[train_idx], myTarget.iloc[test_idx]

        if _nunique(y_train) < 2 or _nunique(y_test) < 2:
            log(ERROR, f"y_train or y_test only contain a single value ({y_train.unique(), y_test.unique()}).")
            error_count +=1
            continue

        if not is_premodel:
            current_test_indices = X.iloc[test_idx][pid].reset_index(drop=True)
            # print(X_test.loc[X_test["sex"]=="männlich", pid].drop_duplicates())

        this_test_ids = X_test.loc[:, [pid, caseid]]
        groups = X_train.loc[:, pid]

        X_train, X_test = X_train.drop([pid, caseid], axis=1), X_test.drop(
            [pid, caseid], axis=1
        )

        pipeline.fit(X_train)

        X_train = pipeline.transform(X_train)
        X_test = pipeline.transform(X_test)
        

        # for col in X_train.columns:
        #     uniques = X_train[col].unique()
        #     print(f"{col}: {list(uniques)}")

        # print()
        # for col in X_test.columns:
        #     uniques = X_test[col].unique()
        #     print(f"{col}: {list(uniques)}")

        X_train, X_test = X_train.astype(float), X_test.astype(float)

        log(
            INFO,
            f"Removing {X_train.shape[1] - len(final_valid_cols)} columns due to insufficient data points during splitting.",
        )

        removed_cols = set(X_train.columns.to_list()) - set(final_valid_cols)

        # log(INFO, X_train.shape)

        X_train = X_train[list(final_valid_cols)]
        X_test = X_test[list(final_valid_cols)]

        def sanitize_df_columns(df):

            df.columns = [re.sub(r"[^äa-zA-Z0-9_]", "_", col) for col in df.columns]
            return df

        X_train = sanitize_df_columns(X_train)
        X_test = sanitize_df_columns(X_test)

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # # dirty feature selection:
        # if not is_premodel:
        #     ids_needed = [col for col in X_train.columns if col.endswith("_hashed")]
        #     general = ["sex_weiblich", "age", "risk_charlson_score", "gram_prediction"]
        #     risk_surgery = [col for col in X_train.columns if any(word in col for word in ['surgery', 'operations'])]

        #     cols_r = [f"prev_resist_{x}" for x in ["2W"]]
        #     # cols_e = [f"prev_exp_{x}"    for x in ["2W","1M","6M","1Y","ALL"]]

        #     # Create feature counting number of surgeries for BOTH datasets
        #     X_train["surgery_count"] = X_train[risk_surgery].sum(axis=1)
        #     X_test["surgery_count"] = X_test[risk_surgery].sum(axis=1)  # Add this line!

        #     manually_selected = ids_needed + general + cols_r + ["surgery_count"] # risk_surgery

        #     X_train_cols = [col for col in manually_selected if col in X_train.columns]
        #     X_test_cols = [col for col in manually_selected if col in X_test.columns]

        #     X_train = X_train.loc[:, X_train_cols]
        #     X_test = X_test.loc[:, X_test_cols]

        #     log(WARNING, "Dirty feature selection Done")
        #     log(INFO, f"shape after feature sel: {X_train.shape} ({X_train.columns.to_list()})")

        X_train, X_test = X_train.astype(float), X_test.astype(float)

        # # # # -----------------------------------------------------------------------------
        # log(DEBUG, "Dropping some risk columns to make the cross models work (adults on children data)")
        # drop_these_temp = ["risk_chronic_cystitis_interstitial_any",
        #                     "risk_ileal_conduit_30d",
        #                     "risk_chronic_pyelonephritis_any",
        #                     "risk_charlson_score",
        #                     "risk_dysuria_30d",
        #                     "risk_bacteria_virus_infections_30d",
        #                     "risk_urostomy_30d",
        #                     "risk_discharge_30_days_unknown",
        #                     "risk_discharge_30_days_still_admitted",
        #                     "risk_penicillin_allergies_any",
        #                     "risk_other_operations_urinary_tract_30d",
        #                     "risk_lithotripsy_ultrasound_30d",
        #                     "risk_chronic_cystitis_other_any",
        #                     "risk_inflammatory_disease_of_prostate_30d",
        #                     "risk_operations_female_genital_organs_30d",
        #                   ]

        # X_train = X_train.drop(columns=drop_these_temp)
        # X_test = X_test.drop(columns=drop_these_temp)
        # # # # -----------------------------------------------------------------------------

        # print("Train test shape:", X_train.shape, X_test.shape)
        # print(X_train.head())
        # print(X_train.isnull().sum().sum())  # Total NaN count
        # print(X_train.isnull().sum())  # NaNs per column

        # Check if imputation is needed for either dataset
        if X_train.isna().sum().sum() > 0 and X_test.isna().sum().sum() > 0:
        
            # Miceforest Imputation
            log(INFO, f"Fold {fold+1}: Imputing data with miceforest")
            log(
                INFO,
                f"Train NaNs: {X_train.isna().sum().sum()}, Test NaNs: {X_test.isna().sum().sum()}",
            )
        
            # === AGGRESSIVE DEBUG & FIX ===
            cols_to_drop = []
            cols_to_fill = {}
            
            for col in X_train.columns.tolist():  # Use tolist() to avoid iteration issues
                col_data = X_train[col]
                null_count = col_data.isna().sum()
                non_null_data = col_data.dropna()
                non_null_count = len(non_null_data)
                n_unique = non_null_data.nunique()
                
                # if null_count > 0:
                #     log(INFO, f"Fold {fold+1}: Column '{col}': nulls={null_count}, non-nulls={non_null_count}, unique={n_unique}")
                
                # Case 1: no valid values at all
                if non_null_count == 0:
                    # log(INFO, f"Fold {fold+1}: DROPPING '{col}' - no valid values")
                    cols_to_drop.append(col)
                # Case 2: has nulls but only 1 unique non-null value
                elif null_count > 0 and n_unique == 1:
                    fill_val = non_null_data.iloc[0]
                    # log(INFO, f"Fold {fold+1}: FILLING '{col}' with constant {fill_val}")
                    cols_to_fill[col] = fill_val
                # Case 3: has nulls but only 0 unique values (all same or empty after dropna edge case)
                elif null_count > 0 and n_unique == 0:
                    # log(INFO, f"Fold {fold+1}: DROPPING '{col}' - 0 unique values")
                    cols_to_drop.append(col)
            
            # fixes
            if cols_to_drop:
                # log(INFO, f"Fold {fold+1}: Dropping {len(cols_to_drop)} columns: {cols_to_drop}")
                X_train = X_train.drop(columns=cols_to_drop)
                X_test = X_test.drop(columns=[c for c in cols_to_drop if c in X_test.columns])
            
            if cols_to_fill:
                # log(INFO, f"Fold {fold+1}: Filling {len(cols_to_fill)} columns with constants")
                for col, val in cols_to_fill.items():
                    X_train[col] = X_train[col].fillna(val)
                    if col in X_test.columns:
                        X_test[col] = X_test[col].fillna(val)
            
            # check before miceforest
            remaining_nans = X_train.isna().sum().sum()
            log(INFO, f"Fold {fold+1}: After fixes - shape={X_train.shape}, remaining NaNs={remaining_nans}")
            
            if remaining_nans > 0:
                # convert to float64
                X_train_clean = X_train.reset_index(drop=True).astype('float64')
                X_test_clean = X_test.reset_index(drop=True).astype('float64')
                
                try:
                    # Create kernel with save_all_iterations_data=True for transform to work
                    kernel = mf.ImputationKernel(
                        X_train_clean.copy(), 
                        num_datasets=1, 
                        random_state=current_rep,
                        save_all_iterations_data=True  # Required for transform()
                    )
                    
                    # Run imputation manually instead of using Pipeline
                    kernel.mice(iterations=5)
                    
                    # Get imputed training data
                    X_train = kernel.complete_data(dataset=0)
                    
                    # Impute test data
                    X_test_imputed = kernel.impute_new_data(X_test_clean.copy())
                    X_test = X_test_imputed.complete_data(dataset=0)
                    
                    last_pipe = None  # Can't use Pipeline with miceforest this way
                    
                    log(INFO, f"Fold {fold+1}: Imputation successful")
                    
                except Exception as e:
                    log(INFO, f"Fold {fold+1}: miceforest failed with: {e}")
                    log(INFO, f"Fold {fold+1}: Falling back to SimpleImputer (median strategy)")
                    
                    from sklearn.impute import SimpleImputer
                    simple_imputer = SimpleImputer(strategy='median')
                    
                    X_train = pd.DataFrame(
                        simple_imputer.fit_transform(X_train_clean),
                        columns=X_train_clean.columns,
                        index=X_train_clean.index
                    )
                    X_test = pd.DataFrame(
                        simple_imputer.transform(X_test_clean),
                        columns=X_test_clean.columns,
                        index=X_test_clean.index
                    )
                    last_pipe = Pipeline([("imputer", simple_imputer)])
            else:
                log(INFO, f"Fold {fold+1}: No imputation needed after pre-filling")
                last_pipe = None
            # === END DEBUG & FIX ===
        
        else:
            log(
                INFO,
                f"Fold {fold+1}: No missing values found, inserting single NaN for imputer compatibility",
            )

            # Insert a single NaN in training data (random position)
            import numpy as np

            np.random.seed(current_rep)  # For reproducibility
            random_row = np.random.randint(0, X_train.shape[0])
            random_col = np.random.randint(0, X_train.shape[1])

            # Store original value for reference
            original_value = X_train.iloc[random_row, random_col]
            log(
                DEBUG,
                f"Inserting NaN at position ({random_row}, {random_col}), original value: {original_value}",
            )

            X_train.iloc[random_row, random_col] = np.nan

            # run imputation
            kernel = mf.ImputationKernel(X_train.copy(), num_datasets=1, random_state=current_rep)
            pipe = Pipeline([("imputer", kernel)])

            X_train = pipe.fit_transform(
                X=X_train.copy(), y=None, imputer__iterations=5
            )
            X_test = pipe.transform(X_test)

            last_pipe = pipe

        # --------------------------------- MODEL TRAINING AND SCORING ---------------------------------
        
        
        # X_test_with_pid = pd.concat([X_test, current_test_indices], axis=1)
        X_test = X_test.dropna()

        # X_test_with_pid = (
        #     X_test
        #     .join(current_test_indices, how="inner")
        # )
        # temp = X_test_with_pid.join(y_test, how="inner")
        

        common_idx = (
            X_test.index
            .intersection(current_test_indices.index)
            .intersection(y_test.index)
        )

        this_test_ids = this_test_ids.reset_index(drop=True)
        this_test_ids = this_test_ids.loc[common_idx]

        current_test_indices = current_test_indices.loc[common_idx]
        y_test = y_test.loc[common_idx]
        
        X_test_with_pid = X_test.join(current_test_indices)
        temp = X_test_with_pid.join(y_test)
        temp = temp.rename(columns={y_train.name: 'target'})
        temps.append(temp)

        if tuning_active:
            if model_to_train == "XGB":
                base_estimator = tuning_xgboost(X_train, y_train, groups, seed=current_rep)

            elif model_to_train == "RF":
                base_estimator = tuning_rf(X_train, y_train, groups, seed=current_rep)

            elif model_to_train == "LR":
                base_estimator = tuning_logreg(X_train, y_train, groups, seed=current_rep)

        if isinstance(base_estimator, str):
            if base_estimator == "empty":
                log(ERROR, "look at error above")
                error_count +=1
                continue

        log(INFO, f"X_train shape before fitting: {X_train.shape}")
        base_estimator.fit(X_train, y_train)

        if (
            hasattr(base_estimator, "named_steps")
            and "clf" in base_estimator.named_steps
        ):
            # case: Pipeline
            clf = base_estimator.named_steps["clf"]
        else:
            clf = base_estimator

        if isinstance(clf, RandomForestClassifier) or isinstance(clf, XGBClassifier):
            explainer = shap.TreeExplainer(clf)  # for RF, XGB
        elif isinstance(clf, LogisticRegression):
            # LR uses LinearExplainer
            explainer = shap.LinearExplainer(
                clf, X_test, feature_perturbation="interventional"
            )
        else:
            explainer = shap.Explainer(clf, X_test)

        shap_values = explainer.shap_values(X_test)

        # print(X_test.isna().sum())
        
        # Predict
        y_pred = clf.predict(X_test)  # soglia in base alla calibrazione
        probas = clf.predict_proba(X_test)

        _save_model_object(clf, target_col, fold, is_ukbb, model_to_train)
        _save_model_data(y_test, target_col, fold, is_ukbb, model_to_train)
        _save_predictions(y_pred, target_col, fold, is_ukbb, model_to_train)
        
        # Check if class 1 exists in the classifier
        if 1 in clf.classes_:
            idx = list(clf.classes_).index(1)
            y_proba = probas[:, idx]
        else:
            # Model was trained without class 1 (e.g., only class 0 was present)
            y_proba = np.zeros(X_test.shape[0])

        # Compute Metrics
        acc = accuracy_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="binary")  # "binary"
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)

        # Store results
        accuracies.append(balanced_acc)
        f1_scores.append(f1)
        aurocs.append(roc_auc)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

        auprs.append(pr_auc)
        all_precision.append(precision)
        all_recall.append(recall)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) > 2:
            shap_values = shap_values[:, :, 1]

        # append to lists for shap plots
        all_shap_values.append(shap_values)

        # all_features.append(
        #     (X_test, current_test_indices)
        # )  # .loc[(X_test["pregnancy_yn"] == True) & (X_test["sex_männlich"] == False)])
        
        all_features.append(X_test_with_pid)

        # concat and append to list for clinician prescription comparison
        X_test = X_test.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        # # this_test_data = pd.concat([this_test_ids, X_test], axis=1)
        # common_idx = this_test_ids.index.intersection(X_test.index)
        
        # this_test_data = pd.concat(
        #     [this_test_ids.loc[common_idx], X_test.loc[common_idx]],
        #     axis=1
        # )

        # print(this_test_data.shape)
        # print(y_proba.shape)

        # this_test_data[f"pred_{target.name}"] = (
        #     y_proba  # y_proba is a np array so no need to reset index
        # )
        # this_test_data[f"true_{target.name}"] = y_test

        # for_clinician_comparison.append(this_test_data)

        log(
            PERFORMANCE,
            f"Fold {fold+1}: Balanced Acc.={balanced_acc:.4f}, F1={f1:.4f}, AUROC={roc_auc:.4f}",
        )
        # print(colored(PERFORMANCE, "green"), f": \t\tFold {fold+1}: Accuracy={acc:.4f}, F1={f1:.4f}, AUROC={roc_auc:.4f}")

        if is_premodel:
            current_test_indices = None

        # append fold prediction for plotting the usual boxplot
        dt_preds = append_fold_predictions(
            dt_preds,
            target.name,
            y_test,
            y_proba,
            fold,
            current_rep,
            model_to_train,
            current_test_indices,
        )

    concat = pd.concat(temps, ignore_index=False)
    log(DEBUG, "Saving data for federated learning to file")
    concat.to_csv("data_for_federated_learning.csv")


    if error_count == n_splits:
        return "empty"

    log(
        PERFORMANCE,
        f"Average internal perf.: Balanced Acc.={np.mean(accuracies):.4f}, F1={np.mean(f1_scores):.4f}, AUROC={np.mean(aurocs):.4f} ({np.std(aurocs):.2f})",
    )
    # print(colored(PERFORMANCE, "green"), f": \t\tAverage internal perf.: Accuracy={np.mean(accuracies):.4f}, F1={np.mean(f1_scores):.4f}, AUROC={np.mean(aurocs):.4f}")

    # Stack SHAP values and features across all folds
    shap_values_combined = np.vstack(
        all_shap_values
    )  # Shape: (total_samples, num_features)

    X_all = pd.concat(all_features, axis=0)

    # print(X_all.columns)

    # clinician_comparison_data = pd.concat(for_clinician_comparison, axis=0).reset_index(
    #     drop=True
    # )

    # Average SHAP values across folds
    mean_shap_values = np.mean(np.abs(shap_values_combined), axis=0).reshape(1, -1)

    _, (ax_bar, ax_dot) = plt.subplots(
        1, 2, figsize=(12, 7), gridspec_kw={"width_ratios": [1, 2]}
    )
    

    # print(shap_values_combined.shape, X_all.shape)
    summary_plot_mod(
        shap_values_combined,
        X_all.drop(columns=pid), # to remove the index used elsewhere for plotting subsets
        curr_axis=ax_dot,
        plot_feature_names=False,
        show=False,
        max_display=20,
        plot_type="dot",
        plot_size=None,
    )
    summary_plot_mod(
        shap_values_combined,
        X_all.drop(columns=pid),
        curr_axis=ax_bar,
        errorbar_sd=None,
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
        this_target_name = antib_dict[target.name.split("_")[-1]][0]
    else:
        # this_target_name = " ".join(target.name.split("_")[2:]).capitalize()
        this_target_name = re.sub(' +', ' ', " ".join(target.name.split("_")[2:]).capitalize())

    # # SHAP Summary Plot (Bar Type)
    # shap.summary_plot(mean_shap_values, X_all, plot_type="bar")
    plt.suptitle(this_target_name, fontsize=20)
    plt.savefig(
        f"figures/{model_to_train}_figures/shap_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.png"
    )
    plt.close()
    
    # save all data used for shap, so that we can plot without having to rerun all the models every time
    np.save(
        f"Data/for_shap/{model_to_train}/shap_values_combined_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.npy",
        shap_values_combined,
    )
    X_all.to_csv(
        f"Data/for_shap/{model_to_train}/X_all_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.csv"
    )


    # Convert to DataFrame for easier plotting
    cv_results = pd.DataFrame(
        {"Accuracy": accuracies, "F1 Score": f1_scores, "AUROC": aurocs}
    )

    plt.figure(figsize=(12, 5))  # 6,5

    # Subplot for ROC Curve
    plt.subplot(1, 2, 1)
    for fpr, tpr in zip(all_fpr, all_tpr):
        plt.plot(fpr, tpr, alpha=0.5)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Across Folds")

    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

    this_ax = plt.gca()
    this_ax.text(
        0.94,
        0.07,
        f"AUROC: {np.mean(aurocs):.3f} ({np.std(aurocs):.2f})\nPos. class: {imbalance_percentage:.1f}%",
        transform=this_ax.transAxes,
        ha="right",
        va="bottom",
        bbox=props,
        fontsize=14,
    )
    this_ax.grid()

    # Subplot for PR Curve
    plt.subplot(1, 2, 2)
    for precision, recall in zip(all_precision, all_recall):
        plt.plot(recall, precision, alpha=0.5)

    # plot the zero skill line
    no_skill = len(myTarget[myTarget == 1]) / len(myTarget)
    plt.plot([0, 1], [no_skill, no_skill], linestyle="--", color="gray")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Across Folds")
    this_ax = plt.gca()
    # Get axis limits
    this_ax.text(
        0.94,
        0.93,
        f"AUPR: {np.mean(auprs):.3f} ({np.std(auprs):.2f})",
        transform=this_ax.transAxes,
        ha="right",
        bbox=props,
        fontsize=14,
    )
    this_ax.grid()

    plt.suptitle(this_target_name, fontsize=20)
    plt.savefig(
        f"figures/{model_to_train}_figures/auroc-pr_curve_{this_target_name}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.png"
    )
    plt.close()

    return dt_preds
