# --------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import re
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, FunctionTransformer

from ut import log, INFO, WARNING, PERFORMANCE, TARGET_NAME, DEBUG, ERROR

matplotlib.use("Agg")


import pandas as pd

def to_timedelta_safe(s):
    if pd.isna(s):
        return pd.NaT
    # normalize whitespace and minus signs
    s = str(s).strip()
    s = s.replace("\u2212", "-")        # U+2212 minus → ASCII '-'
    s = re.sub(r"\s+", " ", s)          # collapse spaces
    # let pandas parse; coerce bad rows to NaT instead of raising
    return pd.to_timedelta(s, errors="coerce")


def load_data(
    all_path, blood_path="", antib_path="", pregnancy_path="", date_shift_path="", risk_factors_path="", use_risk=False, is_ukbb=False
):

    if is_ukbb:
    
        all = pd.read_csv(all_path)
        antib = pd.read_csv(antib_path)

        all.to_csv("all_UKBB_data.csv")
    
        return all, antib

    else:

        risk_factors = pd.read_csv(risk_factors_path, index_col=0)
        risk_factors = risk_factors.drop("alive_30_days", axis=1)
    
        # adding a prefix to distinguish risk factors in shap plots
        merge_cols = [
            "patient_id_hashed",
            "case_id_hashed",
            "urine_sample_id_hashed",
            "urine_organism_id_hashed",
        ]
        prefix = "risk_"
        risk_factors = risk_factors.rename(
            columns={
                col: f"{prefix}{col}"
                for col in risk_factors.columns
                if col not in merge_cols
            }
        )
    
        # dropping columns that only have 1 unique value (bring no information)
        for c in risk_factors.columns:
            uniques = risk_factors[c].unique()
            if len(uniques) < 2:
                risk_factors = risk_factors.drop(c, axis=1)
    
        all = pd.read_csv(all_path)
    
        if use_risk:
            all = all.merge(
                risk_factors,
                on=[
                    "patient_id_hashed",
                    "case_id_hashed",
                    "urine_sample_id_hashed",
                    "urine_organism_id_hashed",
                ],
                how="left",
            )
    
        date_shift = pd.read_csv(date_shift_path)
    
        all = all.merge(
            date_shift,
            on=[
                "patient_id_hashed",
                "case_id_hashed",
                "urine_sample_id_hashed",
                "urine_organism_id_hashed",
            ],
            how="left",
        )

        pregnancy = pd.read_csv(pregnancy_path)

        all = all.merge(
            pregnancy,
            on=[
                "patient_id_hashed",
                "case_id_hashed",
                "urine_sample_id_hashed",
                "urine_organism_id_hashed",
            ],
            how="left",
        )

        # print(all.loc[:, "pregnancy_yn"].value_counts())
    
        blood = pd.read_csv(blood_path)
        antib = pd.read_csv(antib_path)

        if "urine_preg_yn" in all.columns:
            all = all.drop("urine_preg_yn", axis=1)


        return all, blood, antib



# Extract numbers & units
def process_columns(df, columns):
    """
    Some columns are filled with numbers and their respective unit of measure.
    This function extracts the unit of measure from data and appends it to the name of the column.
    This allows for easy transformation to numerical format these columns.
    """
    units = ["Anz./GF", "/µl", "T/l", "mg/l", "g/l", "l/l", "G/l", "%"]
    pattern = rf"^\s*([\d.]+)\s*({'|'.join(map(re.escape, units))})\s*$"

    for col in columns:
        # Skip non-string columns (if needed)
        if not pd.api.types.is_object_dtype(df[col]):
            continue

        # Extract numbers and units
        extracted = df[col].str.extract(pattern)

        # Check if any valid entries match the pattern
        if extracted[0].notna().any():

            # Ensure all non-null entries in the column have the same unit
            unique_units = extracted[1].dropna().unique()
            if len(unique_units) == 1:
                unit = unique_units[0]
                # Replace column with numerical values (keeping NaNs)
                df[col] = pd.to_numeric(extracted[0], errors="coerce")
                # Rename column to include the unit
                df.rename(columns={col: f"{col} ({unit})"}, inplace=True)

    # manually converting urine_mic_erythrocytes for later one-hot encoding
    df["urine_mic_erythrocytes (Anz./GF)"] = df["urine_mic_erythrocytes"].str.replace(
        " Anz./GF", "", regex=False
    )
    df = df.drop("urine_mic_erythrocytes", axis=1)

    return df


def divide_data_from_targets(data, days_tolerance: int = 4, drop_bacteria=False, is_eucast_rules=False, is_ukbb=False):
    """
    From a general dataset containing features and targets, this function divides the features from the targets, returning them in separate pandas DataFrames.
    This function also preprocesses the targets into binary classes. The classes 'resistent' and 'intermediär' become '1', 'sensibel' becomes '0' and 'unbekannt' becomes 'np.nan'.
    """

    if is_ukbb:
        pid = "patnr"
        caseid = "fallnr"
    else:
        pid = "patient_id_hashed"
        caseid = "case_id_hashed"
    
    if not drop_bacteria:
        log(CRITICAL, "USING URINE ORGANISM COLUMN (bacteria names)")


    data_clean = data.copy()

    log(INFO, "before process columns")
    log(INFO, data_clean.loc[:, pid].nunique())

    if is_ukbb:
        data_clean = process_columns_UKBB(data_clean)
    else:
        data_clean = process_columns(data_clean, data_clean.columns)

    log(INFO, data_clean.loc[:, pid].nunique())

    # find all antibiogram columns
    antibiograms_cols = data_clean.columns[
        data_clean.columns.str.contains("antibiogram")
    ].to_list()
    target_pathogen_cols = data_clean.columns[
        data_clean.columns.str.contains("target_")
    ].to_list()  # this are the urine organisms names converted to binary, that we are trying to predict. 2-7-2025

    # only in USB data
    if not is_ukbb:
        # date columns to drop since they are useless:
        # `uti_date` is always 0
        data_clean = data_clean.drop("uti_date", axis=1)

    if drop_bacteria:
        if is_ukbb:
            data_clean = data_clean.drop(["urine_organism", "urine_organism_count"], axis=1)
        else:
            data_clean = data_clean.drop(["urine_organism", "count_of_bacteria"], axis=1)

    if is_ukbb:
        # transform date columns to number of days
        # Single columns
        date_cols = data_clean.columns[
            data_clean.columns.str.contains("date", case=False)
        ].to_list()

        for c in date_cols:
            data_clean[c] = data_clean[c].map(to_timedelta_safe) / pd.Timedelta(days=1)

    # contains also negative date columns: `urine_dipstick_date`, `urine_mic_date`, `urine_flowcyt_date`, `blood_date`
    # contains only positive: `urine_preg_sample_date`, `urine_preg_test_date`
    # use only values within the tolerance for dates. (Antibiogram tests are carried out in +- 4 days from what is the uti_date)
    date_cols = data_clean.columns[
        data_clean.columns.str.contains("date", case=False)
    ].to_list()
    mask = (data_clean[date_cols] <= days_tolerance) | data_clean[date_cols].isna()

    log(INFO, "before days tolerance")
    log(INFO, data_clean.loc[:, pid].nunique())
    
    data_clean = data_clean.loc[mask.all(axis=1)]

    log(INFO, data_clean.loc[:, pid].nunique())
    
    if not is_eucast_rules:
        # converting to a boolean target column
        target_mapping = {
            "sensibel": 0,
            "resistent": 1,
            "intermediär": 0,
            "unbekannt": np.nan,
        }

        for column in antibiograms_cols:
            data_clean[column] = data_clean[column].map(target_mapping)

    else:
        log(
            WARNING,
            "Not transforming target mapping since using EUCAST rules, overwritten with R script and already converted targets to binary",
        )
        rand_antib = antibiograms_cols[np.random.randint(0,len(antibiograms_cols))]
        log(
            INFO,
            f"Sanity check: '{rand_antib}' has values {data_clean.loc[:, rand_antib].unique()}",
        )


    # remove all targets from data to create X
    # Combine antibiograms_cols and target_pathogen_cols to drop from X and include in targets
    if is_eucast_rules:
        target_cols_to_use = (
            [col for col in antibiograms_cols if col in data_clean]
            + [col for col in target_pathogen_cols if col in data_clean]
            + ["gram_binary"] # do not use gram positive or negative as predictor.
        )

    else:
        target_cols_to_use = (
            [col for col in antibiograms_cols if col in data_clean]
            + [col for col in target_pathogen_cols if col in data_clean]
        )

    X = data_clean.drop(columns=target_cols_to_use, axis=1)
    targets = data_clean[target_cols_to_use]

    log(INFO, "X inside data processing")
    log(INFO, X.loc[:, pid].nunique())

    return X, targets

# --- helpers ---
NEGATIVE_TOKENS = {
    "negativ", "neg", "neg.", "-", "negative"
}
MISSING_TOKENS = {
    "nb", "nbb", "nbr", "nm", "n.b.", "kein befund",
    "zu wenig probenmaterial", "falsch.mat.", "falsch. mat.",
    "blutig", "mlv", "Verw.", "verwechsl.", "verwechsl", "s.text", "s. text",
    "nicht beurteilbar", "unbeurteilbar", "k.a.", "ka", "unbekannt", ""
}

PLUS_MAP = {
    "+": "+", "1+": "+",
    "++": "++", "2+": "++",
    "+++": "+++", "3+": "+++",
    "++++": "++++", "4+": "++++"
}

def _norm_str(x: object) -> str:
    """Lowercase, strip, collapse spaces; keep '+' and digits."""
    s = str(x).lower().strip()
    s = s.replace(",", ".")
    s = re.sub(r"\s+", "", s)  # '1 +' -> '1+'
    return s

# 1) FIRST-STAGE CLEANER (dipstick/mic semi-quantitative fields like +, ++, +++)
def clean_value(val):
    if pd.isna(val):
        return "missing"
    s = _norm_str(val)

    if s in NEGATIVE_TOKENS:
        return "negative"
    if s in MISSING_TOKENS:
        return "missing"

    # map semi-quantitative positives
    if s in PLUS_MAP:
        return PLUS_MAP[s]

    # sometimes labs write weird punctuations like 'neg/' or 'neg-'
    if s.startswith("neg"):
        return "negative"

    # fallback: unknown -> treat as missing (safer than inventing a level)
    return "missing"

# 2) SECOND-STAGE CLEANER (ranges, QC flags, etc.)
#    - keeps the microscopy erythrocytes category scheme if present
#    - maps 'massenhaft' to '>40'
def clean_value_2(val):
    if pd.isna(val):
        return np.nan  # keep as NaN for truly numeric columns
    s = _norm_str(val)

    # explicit QC / unevaluable -> missing
    if s in MISSING_TOKENS:
        return np.nan

    # handle "massenhaft" / "massenh..."
    if "massenh" in s:
        return ">40"

    # keep existing microscopy categories as-is
    MIC_ERTH_CATS = {"0-5", "2-6", "5-10", "10-15", "15-20", "20-30", "30-40", ">40"}
    if s in MIC_ERTH_CATS:
        return s

    # ranges written with different dash characters (e.g., '5–10')
    s_dash = s.replace("–", "-").replace("—", "-")

    # If it looks like a numeric, return numeric to let later steps handle it
    try:
        return float(s_dash)
    except ValueError:
        pass

    # if it's something like 'neg', keep as 'nan'
    if s in NEGATIVE_TOKENS or s.startswith("neg"):
        return np.nan

    # else: unknown string -> missing (avoid inventing categories)
    return np.nan

# 3) SPECIFIC GRAVITY CLEANER (urine_*_spez_gewicht)
#    Goal: normalize to ~1.000–1.060. Common data-entry patterns:
#      - 1.025  -> keep
#      - 1025   -> divide by 1000
#      - 1,025  -> 1.025
#      - >1025  -> 1.025 (drop sign, normalize)

def clean_gewicht(val):
    if pd.isna(val):
        return np.nan

    s = _norm_str(val)
    s = s.replace(">", "").replace("<", "")  # drop inequality for descriptive table
    # keep only digits and single dot
    s = re.sub(r"[^0-9.]", "", s)

    if s == "" or s == ".":
        return np.nan

    try:
        x = float(s)
    except ValueError:
        return np.nan

    # Heuristics to normalize:
    if 1.0 <= x <= 1.1:
        return x  # already specific gravity
    if 1000 <= x <= 1100:
        return x / 1000.0
    if 10 < x <= 110:      # e.g., '102' -> 1.02
        return x / 100.0
    if 1.1 < x <= 2.0:     # out-of-range but plausible scaling error like 1.20
        return x           # keep (or clamp if you prefer)
    # everything else likely junk
    return np.nan

def pipeline_func(use_risk, is_premodel=False):
    """
    This is the main cleaning pipeline, which encodes all defined features using ordinal encoding, one hot encoding or simply transforms features to number format.
    Here the pipeline is defined, but only returned and never run. It is called inside the `main()` function inside src/main.py file.
    """
    # all the columns at the end of the pipeline to be selected
    c = [
        "age",
        "count_of_bacteria",
        "pregnancy_yn",
        "urine_dipstick_bilirubin",
        "urine_dipstick_glucose",
        "urine_dipstick_ketone",
        "urine_dipstick_leucocytes",
        "urine_dipstick_nitrites",
        "urine_dipstick_ph",
        "urine_dipstick_spez_gewicht",
        "urine_mic_bilirubin",
        "urine_mic_glucose",
        "urine_mic_hemoglobin",
        "urine_mic_ketokorper",
        "urine_mic_leucocytes",
        "urine_mic_nitrites",
        "urine_mic_ph",
        "urine_mic_spez_gewicht",
        # these below are columns that get renamed by `process_columns()`.
        # they do not exist in `data` yet.
        "urine_dipstick_protein (g/l)",
        "urine_flowcyt_bacteria (/µl)",
        "urine_flowcyt_erythrocytes (/µl)",
        "urine_flowcyt_erythrocytesprogf (Anz./GF)",
        "urine_flowcyt_leucocytes (/µl)",
        "urine_flowcyt_leucocytesprogf (Anz./GF)",
        "urine_flowcyt_squamousepithelia (/µl)",
        "blood_crp (mg/l)",
        "blood_erythrocytes (T/l)",
        "blood_hematocrite (l/l)",
        "blood_hemoglobin (g/l)",
        "blood_leucocytes (G/l)",
        "blood_neutrophiles (%)",
        "blood_neutrophiles_abs (G/l)",
        "blood_thrombocytes (G/l)",
        "urine_mic_erythrocytes (Anz./GF)",
    ]

    # parameter to check if the model training is for the generation of the gram prediction or the actual resistance models
    if not is_premodel:
        c.append("gram_prediction")

    
    # columns to ordinal encode
    first_stage = [
        "urine_dipstick_bilirubin",  # this gives errors when encoding, sometimes it does not get encoded correctly. dunno why
        "urine_dipstick_glucose",
        "urine_dipstick_ketone",
        "urine_dipstick_leucocytes",
        "urine_dipstick_nitrites",
        "urine_mic_bilirubin",
        "urine_mic_glucose",
        "urine_mic_hemoglobin",
        "urine_mic_ketokorper",
        "urine_mic_leucocytes",  # this bs
        "urine_mic_nitrites",
    ]

    # columns to clean after ordinal encoding
    # clean the columns to ordinal encode before, so that it goes at stages.
    # 1. Clean those,
    # 2. ordinal encode,
    # 3. then clean the rest
    second_stage = [
        "urine_mic_erythrocytes (Anz./GF)",
        "urine_mic_spez_gewicht",
        "urine_dipstick_ph",
        "urine_dipstick_spez_gewicht",
        "urine_dipstick_protein (g/l)",
        "urine_mic_ph",
    ]

    def to_dataframe(X, feature_names):
        """Convert NumPy array back to a Pandas DataFrame."""
        return pd.DataFrame(X, columns=feature_names)


    def convert_to_number(val):
        """Convert string numbers to actual numbers without converting non-numeric values to NaN."""
        if isinstance(val, (int, float)):  # Already a number
            return val
        if isinstance(val, str):
            val = val.replace(",", ".")
            val = val.strip()
            if val.isdigit():
                return int(val)
            try:
                return float(val)
            except ValueError:
                return val
        if isinstance(val, object):
            return str(val)
        return val

    # Custom transformers
    class ColumnCleaner(BaseEstimator, TransformerMixin):
        def __init__(self, columns, clean_func):
            self.columns = columns
            self.clean_func = clean_func

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            for col in self.columns:
                if col in X.columns:
                    X[col] = X[col].map(self.clean_func)
            return X

    # Ordinal encoding
    ordinal_categories = [["negative", "+", "++", "+++", "++++"]] * len(first_stage)
    ordinal_encoder = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )

    # Second ordinal encoder
    ordinal_categories_2 = [
        ["0-5", "2-6", "5-10", "10-15", "15-20", "20-30", "30-40", ">40"]
    ]
    ordinal_encoder_2 = OrdinalEncoder(
        categories=ordinal_categories_2,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )

    ordinal_preprocessor = ColumnTransformer(
        [("ordinal_encoder", ordinal_encoder, first_stage)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    ordinal_preprocessor_2 = ColumnTransformer(
        [("ordinal_encoder", ordinal_encoder_2, ["urine_mic_erythrocytes (Anz./GF)"])],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    if use_risk:
        one_hot_encoding_cols = [
            "sex",
            "case_type",
            "urine_material",
            "risk_discharge_30_days",
            # "gram",
            # "risk_alive_30_days",
        ]
    else:
        one_hot_encoding_cols = [
            "sex",
            "case_type",
            "urine_material",
            # "gram",
        ]

    # Define known categories for each categorical column
    known_categories = {
        "sex": ["weiblich", "männlich"],
        "case_type": ["stationär", "teil-stationär", "ambulant"],
        "urine_material": [
            "Urin",
            "Urin aus Dauerkatheter",
            "Urin aus Einmalkatheter",
            "Mittelstrahlurin",
            "Urin aus Blasenpunktion",
            "Urin nicht genauer bezeichnet",
            "Urinsäckchen",
        ],
        # risk factors
        "risk_discharge_30_days": ["discharged", "unknown", "still admitted"],
        # "risk_alive_30_days": ['alive or unknown', 'dead', 'unknown']
        # "gram": ["Gram-negative", "Gram-positive"], # cannot use this as predictor
    }

    # convert dictionary to a list (matching the order of one_hot_encoding_cols)
    categories_list = [known_categories[col] for col in one_hot_encoding_cols]

    onehot_encoder = OneHotEncoder(
        drop="first",
        categories=categories_list,
        handle_unknown="ignore",
        sparse_output=False,
    )

    onehot_preprocessor = ColumnTransformer(
        [("onehot_encoder", onehot_encoder, one_hot_encoding_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    replace_missing = FunctionTransformer(
        lambda X: X.replace("missing", np.nan), validate=False
    )
    replace_yes_no = FunctionTransformer(
        lambda X: X.replace({"yes": 1, "no": 0}), validate=False
    )

    # Define pipeline
    pipeline = Pipeline(
        [
            ("cleaner_first", ColumnCleaner(first_stage, clean_value)),
            ("ordinal-encoder", ordinal_preprocessor),  # Outputs a NumPy array
            (
                "to_dataframe",
                FunctionTransformer(
                    lambda X: to_dataframe(
                        X, ordinal_preprocessor.get_feature_names_out()
                    )
                ),
            ),
            ("cleaner_second", ColumnCleaner(second_stage, clean_value_2)),
            (
                "cleaner_gewicht",
                ColumnCleaner(
                    ["urine_dipstick_spez_gewicht", "urine_mic_spez_gewicht"],
                    clean_gewicht,
                ),
            ),
            (
                "convert2number",
                ColumnCleaner(set(c + first_stage + second_stage), convert_to_number),
            ),
            ("one-hot-encoder", onehot_preprocessor),  # Outputs a NumPy array
            (
                "to_dataframe_2",
                FunctionTransformer(
                    lambda X: to_dataframe(
                        X, onehot_preprocessor.get_feature_names_out()
                    )
                ),
            ),
            ("ordinal-encoder_2", ordinal_preprocessor_2),  # Outputs a NumPy array
            (
                "to_dataframe_final",
                FunctionTransformer(
                    lambda X: to_dataframe(
                        X, ordinal_preprocessor_2.get_feature_names_out()
                    )
                ),
            ),
            (
                "replace_yes_no_riskfactors",
                replace_yes_no,
            ),  # no need to comment it since if it does not find yes or no values it does nothing
            ("replace_missing", replace_missing),
        ]
    )

    return pipeline




# ====================== UKBB-specific related stuff ===========================

def process_columns_UKBB(df):

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
        if df[c].eq("Yes").any():
            df[c] = df[c].fillna("No")
        else:
            df = df.drop(columns=[c])

    return df


def pipeline_func_UKBB(use_risk, is_premodel=False):
    """
    This is the main cleaning pipeline, which encodes all defined features using ordinal encoding, one hot encoding or simply transforms features to number format.
    Here the pipeline is defined, but only returned and never run. It is called inside the `main()` function inside src/main.py file.
    """
    c = [
        "age",
        "count_of_bacteria",
        "urine_dipstick_bilirubin",
        "urine_dipstick_glucose",
        "urine_dipstick_keton",
        "urine_dipstick_nitrites",
        "urine_dipstick_ph",
        "urine_dipstick_spez_gewicht",
        "urine_mic_bilirubin",
        "urine_mic_glucose",
        "urine_mic_hemoglobin",
        "urine_mic_ketokorper",
        "urine_mic_leucocytes",
        "urine_mic_nitrites",
        "urine_mic_ph",
        "urine_mic_spez_gewicht",
        "urine_dipstick_protein",
        "urine_flowcyt_bacteria",
        "urine_flowcyt_erythrocytes",
        "urine_flowcyt_erythrocytesprogf",
        "urine_flowcyt_leucocytes",
        "urine_flowcyt_leucocytesprogf",
        "urine_flowcyt_plattenepithelien",
        "crp",
        "pct",
        "blood_erythrocytes",
        "blood_hematocrite",
        "blood_hemoglobin",
        "blood_leucocytes",
        "blood_neutrophiles",
        "blood_neutrophiles_abs",
        "blood_thrombocytes",

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

    # parameter to check if the model training is for the generation of the gram prediction or the actual resistance models
    if not is_premodel:
        c.append("gram_prediction")

    clean_general = [
        "urine_dipstick_urobilinogen",
        "urine_dipstick_leucocytes",
        "urine_dipstick_erythrocytes",
        "urine_mic_protein",
        "urine_flowcyt_erythrocytes",
        "urine_flowcyt_erythrocytesprogf",
        "urine_flowcyt_bacteria",
        "urine_flowcyt_leucocytes",
        "urine_flowcyt_leucocytesprogf",
        "urine_flowcyt_plattenepithelien",
        "blood_erythrocytes",
        "blood_hematocrite",
        "blood_hemoglobin",
        "blood_leucocytes",
        "blood_neutrophiles",
        "blood_neutrophiles_abs",
        "blood_thrombocytes",
    ]
    
    # columns to ordinal encode
    first_stage = [
        "urine_dipstick_bilirubin",
        "urine_dipstick_glucose",
        "urine_dipstick_keton",
        "urine_dipstick_nitrites",
        "urine_mic_bilirubin",
        "urine_mic_glucose",
        "urine_mic_hemoglobin",
        "urine_mic_ketokorper",
        "urine_mic_leucocytes",
        "urine_mic_nitrites",
    ]

    # columns to clean after ordinal encoding
    # clean the columns to ordinal encode before, so that it goes at stages.
    # 1. Clean those,
    # 2. ordinal encode,
    # 3. then clean the rest
    second_stage = [
        "urine_mic_spez_gewicht",
        "urine_dipstick_ph",
        "urine_dipstick_spez_gewicht",
        "urine_dipstick_protein",
        "urine_mic_ph",
    ]

    def to_dataframe(X, feature_names):
        """Convert NumPy array back to a Pandas DataFrame."""
        return pd.DataFrame(X, columns=feature_names)


    def convert_to_number(val):
        """Convert string numbers to actual numbers without converting non-numeric values to NaN."""
        if isinstance(val, (int, float)):  # Already a number
            return val
        if isinstance(val, str):
            val = val.replace(">", "").replace("<", "")  # drop inequality for descriptive table
            val = val.replace(",", ".")
            val = val.strip()
            if val.isdigit():
                return int(val)
            try:
                return float(val)
            except ValueError:
                return val
        if isinstance(val, object):
            return str(val)
        return val

    # Custom transformers
    class ColumnCleaner(BaseEstimator, TransformerMixin):
        def __init__(self, columns, clean_func):
            self.columns = columns
            self.clean_func = clean_func

        def fit(self, X, y=None):
            return self

        def transform(self, X):
            X = X.copy()
            for col in self.columns:
                if col in X.columns:
                    X[col] = X[col].map(self.clean_func)
            return X

    # Ordinal encoding
    ordinal_categories = [["negative", "+", "++", "+++", "++++"]] * len(first_stage)
    ordinal_encoder = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )

    # Second ordinal encoder
    ordinal_categories_2 = [
        ["0-5", "2-6", "5-10", "10-15", "15-20", "20-30", "30-40", ">40"]
    ]
    ordinal_encoder_2 = OrdinalEncoder(
        categories=ordinal_categories_2,
        handle_unknown="use_encoded_value",
        unknown_value=np.nan,
    )

    ordinal_preprocessor = ColumnTransformer(
        [("ordinal_encoder", ordinal_encoder, first_stage)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    one_hot_encoding_cols = [
        "case_type",
        "urine_material",
        # "gram",
    ]

    # Mapping values → canonical form
    value_mapping = {
        "Urin, Mittelstrahl": "Mittelstrahlurin",
        "Urin, Mittelstrahl, 1": "Mittelstrahlurin",
        "Urin aus Einmalkatheter": "Urin aus Einmalkatheter",
        "Urin aus Einmalkatheter, aus Einmalkatheter": "Urin aus Einmalkatheter",
        "Urin, aus Einmalkatheter": "Urin aus Einmalkatheter",
        "Urin aus Dauerkatheter": "Urin aus Dauerkatheter",
        "Urin aus Dauerkatheter, aus Dauerkatheter": "Urin aus Dauerkatheter",
        "Urin, aus Dauerkatheter": "Urin aus Dauerkatheter",
        "Urin aus Blasenpunktion": "Urin aus Blasenpunktion",
        "Urin nicht genauer bezeichnet": "Urin nicht genauer bezeichnet",
    }


    known_categories = {
        "case_type": ["S", "A", "TS"],
        "urine_material": ['Urin', 
                           'Urinsäckchen', 
                           'Urin nicht genauer bezeichnet', 
                           'Urin aus Einmalkatheter',
                           'Mittelstrahlurin', 
                           'Urin aus Dauerkatheter', 
                           'Urin in Urotube', 
                           'Urin aus Blasenpunktion',
                           'Urin, Urostoma'
                          ],
    }


    # convert dictionary to a list (matching the order of one_hot_encoding_cols)
    categories_list = [known_categories[col] for col in one_hot_encoding_cols]

    onehot_encoder = OneHotEncoder(
        drop="first",
        categories=categories_list,
        handle_unknown="ignore",
        sparse_output=False,
    )

    onehot_preprocessor = ColumnTransformer(
        [("onehot_encoder", onehot_encoder, one_hot_encoding_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )

    replace_missing = FunctionTransformer(
        lambda X: X.replace("missing", np.nan), validate=False
    )
    replace_yes_no = FunctionTransformer(
        lambda X: X.replace({"Yes": 1, "No": 0}), validate=False
    )
    replace_before_onehot = FunctionTransformer(
        lambda X: X.replace(value_mapping), validate=False
    )

    # Define pipeline
    pipeline = Pipeline(
        [
            ("clean_general", ColumnCleaner(clean_general, clean_value_2)),
            ("cleaner_first", ColumnCleaner(first_stage, clean_value)),
            ("ordinal-encoder", ordinal_preprocessor),  # Outputs a NumPy array
            (
                "to_dataframe",
                FunctionTransformer(
                    lambda X: to_dataframe(
                        X, ordinal_preprocessor.get_feature_names_out()
                    )
                ),
            ),
            ("cleaner_second", ColumnCleaner(second_stage, clean_value_2)),
            (
                "cleaner_gewicht",
                ColumnCleaner(
                    ["urine_dipstick_spez_gewicht", "urine_mic_spez_gewicht"],
                    clean_gewicht,
                ),
            ),
            (
                "convert2number",
                ColumnCleaner(set(c + first_stage + second_stage), convert_to_number),
            ),
            ("replace_before_onehot", replace_before_onehot),
            ("one-hot-encoder", onehot_preprocessor),  # Outputs a NumPy array
            (
                "to_dataframe_2",
                FunctionTransformer(
                    lambda X: to_dataframe(
                        X, onehot_preprocessor.get_feature_names_out()
                    )
                ),
            ),
            (
                "replace_yes_no_riskfactors",
                replace_yes_no,
            ),  # no need to comment it since if it does not find yes or no values it does nothing
            ("replace_missing", replace_missing),
        ]
    )

    return pipeline


# =================== ^^^ UKBB-specific related stuff ^^^ ========================

