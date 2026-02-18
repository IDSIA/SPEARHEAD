"""
USAGE: `python src/main.py use_risk=True only_consenting=False load_data_only=True model="RF" is_ukbb=False`

Main script for bacterial resistance prediction analysis.

This script processes medical data to predict antibiotic resistance patterns
and bacterial species identification using machine learning models.
"""

# --------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import  roc_auc_score, roc_curve, precision_recall_curve, auc
from matplotlib.ticker import FormatStrFormatter

import warnings
import time
from multiprocessing import Process
import sys
import os
import re
from typing import Literal
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from data_processing import load_data, divide_data_from_targets
from model_training import main as main_train
from visualization import plot_results, create_univariate_heatmap
from ut import *

# Configuration
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
matplotlib.use("Agg")

# --------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------

REPETITIONS = 10

TARGETS = [
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

TARGETS_UKBB = [
    "urine_antibiogram_AC",
    "urine_antibiogram_CXM",
    "urine_antibiogram_SXT",  # cotrimoxazol, Trimethoprim/Sulfamethoxazo (TMP/SMX)
    "urine_antibiogram_FOT",
    # # "urine_antibiogram_NFT",
    # # "urine_antibiogram_norfloxacin",
    "urine_antibiogram_CIP",
    "urine_antibiogram_CTR",
    "urine_antibiogram_PT",
]


DATA_PATHS = {
    "main_data": "data_spearhead/df_SPEARHEAD_all_20250709.csv",
    "blood_culture": "data_spearhead/df_SPEARHEAD_bloodculture_20250709.csv",
    "antibiotics": "data_spearhead/df_SPEARHEAD_antibiotics_20250709.csv",
    "consent": "data_spearhead/df_SPEARHEAD_general_consent_status_20250709.csv",
    "pregnancy": "data_spearhead/df_SPEARHEAD_pregnancy_info_20250709.csv",
    "date_shift": "data_spearhead/df_SPEARHEAD_patient_secrets_20250709.csv",
    "risk_factors": "data_spearhead/df_SPEARHEAD_riskfactors_20250709.csv",
}

DATA_PATHS_UKBB = {
    "main_data": "data_spearhead/UKBB_data/dataset_20250505/uti_dataset_ukbb.csv",
    "antibiotics": "data_spearhead/UKBB_data/dataset_20250505/drug_prescriptions_final.csv",
    "consent": "data_spearhead/UKBB_data/dataset_with_risk_factors/ukbb_patient_consent_20250410.csv",
}


# --------------------------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------------------------

def load_and_filter_data(only_consenting=None, 
                                     use_risk=None, 
                                     load_data_only=True, 
                                     is_ukbb: bool=False,
                                     model_to_train="RF"):
    """
    DEPRECATION: GENERATING THE GRAM BINARY FEATURE, NOT USING IT ANYWHERE. IT DOES NOT EVEN RETURN IT
    Load data and generate a gram prediction model and return predictions as features for resistance models.
    
    This function runs the complete pipeline for gram prediction, then returns
    the trained model and predictions to be used as features in resistance prediction.
    
    Returns:
        tuple: (gram_predictions, processed_data)
            - gram_predictions: Predictions from the gram model
            - processed_data: Processed dataset ready for resistance prediction
    """

    if is_ukbb:
        # Load and process data (same as main pipeline)
        all_data, antibiotics_data = load_data(
            all_path=DATA_PATHS_UKBB["main_data"],
            antib_path=DATA_PATHS_UKBB["antibiotics"],
            is_ukbb=is_ukbb,
        )
    else:
        # Load and process data (same as main pipeline)
        all_data, blood_data, antibiotics_data = load_data(
            all_path=DATA_PATHS["main_data"],
            blood_path=DATA_PATHS["blood_culture"],
            antib_path=DATA_PATHS["antibiotics"],
            pregnancy_path=DATA_PATHS["pregnancy"],
            date_shift_path=DATA_PATHS["date_shift"],
            risk_factors_path=DATA_PATHS["risk_factors"],
            use_risk=use_risk,  # use_risk parameter
        )

    log(INFO, "before consensus and AMR")
    log(INFO, all_data.loc[:, pid].nunique())
        
    # SAVING DATA FOR FUTURE USE
    if is_ukbb:
        all_data.to_csv("src/src_data/all_UKBB_data.csv")
    else:
        all_data.to_csv("src/src_data/all_USB_data_risk_dateshift.csv")
        
    # Apply EUCAST rules if available
    is_eucast_rules = False

    log(INFO, "Running R script for overwriting resistances using AMR R package")
    if is_ukbb:
        os.system("Rscript src/test_amr_ukbb.r")
    else:
        os.system("Rscript src/test_amr.r")
    log(INFO, "Finished running R script")

    if os.path.exists("src/src_data/processed_amr_data_binary.csv"):
        is_eucast_rules = True
        all_data = pd.read_csv("src/src_data/processed_amr_data_binary.csv")

    if not is_ukbb:
        # Filter by consent based on only_consenting parameter
        general_cons = pd.read_csv(DATA_PATHS["consent"])
        
        if only_consenting:
            # Only include patients who explicitly accepted
            mapping = {
                "Accepted": True,
                "No information": False,
                "Refused": False,
            }
            log(WARNING, "Using only CONSENTING patients")
        else:
            # Include patients who accepted or have no information
            mapping = {
                "Accepted": True,
                "No information": True,
                "Refused": False,
            }
            log(WARNING, "Using ALL patients (Consenting and not known)")
        
        general_cons["consent_status"] = general_cons["consent_status"].map(mapping)
        consented_ids = general_cons.loc[
            general_cons["consent_status"] == True, pid
        ]
        all_data = all_data.loc[all_data[pid].isin(consented_ids)]
    else:
        # Filter by consent based on only_consenting parameter
        general_cons = pd.read_csv(DATA_PATHS_UKBB["consent"])
        
        if only_consenting:
            # Only include patients who explicitly accepted
            mapping = {
                "Ja": True,
                "Keine Entscheidung": False,
                "Nein": False,
            }
            log(WARNING, "Using only CONSENTING patients")
        else:
            # Include patients who accepted or have no information
            mapping = {
                "Ja": True,
                "Keine Entscheidung": True,
                "Nein": False,
            }
            log(WARNING, "Using ALL patients (Consenting and not known)")
        
        general_cons["general_consent"] = general_cons["general_consent"].map(mapping)
        consented_ids = general_cons.loc[
            general_cons["general_consent"] == True, pid
        ]

        log(INFO, "data pid unique: " + str(all_data[pid].nunique()))
        log(INFO, "consensus pid unique: " + str(general_cons[pid].nunique()))
        
        all_data = all_data.loc[all_data[pid].isin(consented_ids)]

    log(INFO, all_data.loc[:, pid].nunique())
        
    # Sort by date
    if is_ukbb:
        all_data = all_data.sort_values(by="date_shifted_to_last_uti").reset_index(drop=True)
    else:
        all_data = all_data.sort_values(by="patient_date_shift").reset_index(drop=True)

    all_data = all_data.drop("mo", axis=1)    

    return all_data


def parse_command_line_args(argv_list):
    """
    Parses command line arguments in 'key=value' format.

    Args:
        argv_list (list): List of command line arguments

    Returns:
        dict: Dictionary of parsed arguments
    """
    args_dict = {}
    for arg in argv_list:
        if "=" in arg:
            key, value_str = arg.split("=", 1)
            # Convert boolean strings to actual booleans
            if value_str.lower() == "true":
                value = True
            elif value_str.lower() == "false":
                value = False
            else:
                value = value_str
            args_dict[key] = value
        else:
            print(f"Warning: Argument '{arg}' ignored (no '=' found).")
    return args_dict


def main(use_bacteria_names=None, load_data_only=True):
    """
    Main function to execute the bacterial resistance prediction pipeline.

    Args:
        use_bacteria_names (bool, optional): Whether to use bacteria names in prediction
    """
    use_bacteria = use_bacteria_names or False

    # Parse command line arguments
    cli_args = sys.argv[1:]
    config = parse_command_line_args(cli_args)
    log(INFO, f"Parsed config: {config}")

    # Extract configuration parameters
    only_consenting = config.get("only_consenting", False)
    use_risk = config.get("use_risk", False)
    load_data_only = config.get("load_data_only", False)
    is_ukbb = config.get("is_ukbb", False)
    model_to_train = config.get("model", "RF")

    log(INFO, "-------------------------------------------------")
    log(INFO, f"*\t\tTraining {model_to_train} model\t\t*")
    log(INFO, "-------------------------------------------------")

    global pid
    global caseid

    if is_ukbb:
        pid = "patnr"
        caseid = "fallnr"
    else:
        pid = "patient_id_hashed"
        caseid = "case_id_hashed"

    if not is_ukbb:
        # Log configuration
        if use_risk:
            log(WARNING, "Using Risk Factors\n")
        else:
            log(WARNING, "Not Using Risk Factors\n")

    if load_data_only:
        log(WARNING, "Loading data only. Not generating gram status prediction feature.")

    # STEP 1: get processed data
    all_data = load_and_filter_data(only_consenting, use_risk, load_data_only=load_data_only, is_ukbb=is_ukbb, model_to_train=model_to_train)
    
    log(INFO, f"Number of patients after gram predictor: {all_data[pid].nunique()}")

    targets = TARGETS_UKBB if is_ukbb else TARGETS

    # STEP 2: Process targets for resistance prediction (excluding gram_binary)
    resistance_targets = [t for t in targets if t != "gram_binary"]

    is_eucast_rules = os.path.exists("src/src_data/processed_amr_data_binary.csv")
    log(CRITICAL, f"{is_eucast_rules=}")
    log(INFO, "sleeping 1 second...")
    time.sleep(1) # to make sure the log is see-able

    # for col in all_data.columns:
    #     uniques = all_data[col].unique()
    #     print(f"{col}: {list(uniques)}")

    X, targets_data = divide_data_from_targets(
        all_data, days_tolerance=4, drop_bacteria=not use_bacteria, is_eucast_rules=is_eucast_rules, is_ukbb=is_ukbb
    )

    # all_data = all_data.reset_index(drop=True)
    log(INFO, "all data after divide data from targets")
    log(INFO, all_data.loc[:, pid].nunique())

    log(INFO, "X after divide data from targets")
    log(INFO, X.loc[:, pid].nunique())
    
    log(INFO, "Loaded data")


    # Use original data filtered to same rows for Table 1
    # for col in all_data.columns:
    #     print(col)

    table1_data = all_data.loc[X.index]  # If you preserve index
    table1_patients = table1_data.drop_duplicates(subset=[pid], keep='first')

    log(INFO, "before dropping duplicate patients")
    log(INFO, table1_data.loc[:, pid].nunique())

    log(INFO, table1_patients.loc[:, pid].nunique())
    
    # log(INFO, f"Shape of {table1_data[pid].nunique()=}")
    # log(INFO, f"Shape of {table1_data.shape=}")

    if not is_ukbb:
        # categorical variables (yes/no, ordered +++ scales, etc.)
        categorical_vars = ["risk_indwelling_foley_catheter_30d", 
                            "risk_urinary_tract_surgery_30d",
                            "sex", 
                            "case_type", 
                            "pregnancy_yn",
        ]
        
        # continuous variables (summarize with mean/SD or median/IQR)
        continuous_vars = [
            "age", 
            "risk_charlson_score", 
            "blood_crp", 
            "blood_leucocytes", 
            "blood_neutrophiles_abs"
        ]
    else:
        categorical_vars = ["sex", 
                            "case_type", 
                            "rare_disease", 
                            "indwelling_foley_catheter", 
                            "other_operations_urinary_tract"]
        continuous_vars = ["age", 
                           "crp", 
                           "blood_leucocytes", 
                           "blood_neutrophiles_abs"]

    for col in continuous_vars:
        if col in table1_data.columns and table1_data[col].dtype == 'object':
            log(INFO, f"Cleaning {col}...")
            table1_data[col] = pd.to_numeric(
                table1_data[col].astype(str).str.extract(r'([\d.\-]+)')[0],
                errors='coerce'
            )
            log(INFO, table1_data[col].min())
            print(table1_data[col].max())
    # log(INFO, f"Age: {table1_data['age'].mean():.1f} ± {table1_data['age'].std():.1f}")
    # log(INFO, f"Male: {(table1_data['sex'] == 'männlich').sum()} ({(table1_data['sex'] == 'männlich').mean()*100:.1f}%)")
    # Deduplicate to one row per patient (keep first occurrence)
    table1_patients = table1_data.drop_duplicates(subset=[pid], keep='first')
    
    log(INFO, "=" * 60)
    log(INFO, f"TABLE 1: Cohort Characteristics (N = {len(table1_patients)} unique patients)")
    log(INFO, f"(Total samples: {len(table1_data)})")
    log(INFO, "=" * 60)
    
    # Clean dirty numeric columns first
    for col in continuous_vars:
        if col in table1_patients.columns and table1_patients[col].dtype == 'object':
            table1_patients[col] = pd.to_numeric(
                table1_patients[col].astype(str).str.extract(r'([\d.\-]+)')[0],
                errors='coerce'
            )
    
    # Continuous variables
    log(INFO, "\n--- Continuous Variables ---")
    for var in continuous_vars:
        if var in table1_patients.columns:
            n_missing = table1_patients[var].isna().sum()
            n_valid = table1_patients[var].notna().sum()
            
            if var == "age":
                # Age: median (IQR)
                median_val = table1_patients[var].median()
                q1 = table1_patients[var].quantile(0.25)
                q3 = table1_patients[var].quantile(0.75)
                log(INFO, f"{var}: {median_val:.2f} ({q1:.2f} - {q3:.2f}) [median (IQR)] (n={n_valid}, missing={n_missing})")
            else:
                # Other continuous: mean ± SD
                mean_val = table1_patients[var].mean()
                std_val = table1_patients[var].std()
                log(INFO, f"{var}: {mean_val:.2f} ± {std_val:.2f} [mean ± SD] (n={n_valid}, missing={n_missing})")
        else:
            log(WARNING, f"{var}: column not found")
    
    # Categorical variables: N (%)
    log(INFO, "\n--- Categorical Variables (N, %) ---")
    for var in categorical_vars:
        if var in table1_patients.columns:
            log(INFO, f"\n{var}:")
            value_counts = table1_patients[var].value_counts(dropna=False)
            total = len(table1_patients)
            for value, count in value_counts.items():
                pct = count / total * 100
                label = value if pd.notna(value) else "Missing"
                log(INFO, f"  {label}: {count} ({pct:.1f}%)")
        else:
            log(WARNING, f"{var}: column not found")
    
    log(INFO, "=" * 60)

    # log(INFO, "Saving Univariate heatmap - predictors vs. targets (including gram predictions)")
    # p_values_matrix = create_univariate_heatmap(X, targets_data)

    # STEP 4: Train resistance models with gram predictions as feature
    data4comparison = {}
    all_dt_preds = []
    summary_dfs = []

    from difflib import SequenceMatcher
    import re
    
    def find_similar_targets(columns, threshold=0.80, prefix_to_remove='urine_antibiogram_', min_common_length=10):
        """
        Find groups of similar column names, ignoring prefix and special characters.
        Only merge if they share a common base (min_common_length).
        """
        groups = {}
        used = set()
        
        def clean_name(col):
            cleaned = col.replace(prefix_to_remove, '')
            cleaned = re.sub(r'[^a-z0-9]', '', cleaned.lower())
            return cleaned
        
        col_mapping = {col: clean_name(col) for col in columns}
        
        for col in columns:
            if col in used:
                continue
            
            col_stripped = col_mapping[col]
            similar = [col]
            
            for other_col in columns:
                if other_col != col and other_col not in used:
                    other_stripped = col_mapping[other_col]
                    
                    # Find longest common substring
                    matcher = SequenceMatcher(None, col_stripped, other_stripped)
                    longest_match = max([block.size for block in matcher.get_matching_blocks()], default=0)
                    
                    ratio = SequenceMatcher(None, col_stripped, other_stripped).ratio()
                    
                    # Merge only if ratio > threshold AND they share a long common base
                    if ratio > threshold and longest_match >= min_common_length:
                        similar.append(other_col)
            
            if len(similar) > 1:
                groups[col] = similar
                used.update(similar)
        
        return groups

    
    # Find and merge similar targets with OR logic
    similar_groups = find_similar_targets(targets_data.columns)

    if is_ukbb:
        for this_target in resistance_targets:
            if this_target == "urine_antibiogram_AC":
                variants = ["urine_antibiogram_AC", "urine_antibiogram_ACI", "urine_antibiogram_ACO", "urine_antibiogram_ACU"]
                log(INFO, f"Merging with OR: {variants}")
                # OR logic: 1 if any column has 1, else 0 (ignoring NaNs)
                targets_data[this_target] = targets_data[variants].max(axis=1)
                targets_data = targets_data.drop(columns=[v for v in variants if v != this_target])
    else:
        for main_target, variants in similar_groups.items():
            log(INFO, f"Merging with OR: {variants}")
            # OR logic: 1 if any column has 1, else 0 (ignoring NaNs)
            targets_data[main_target] = targets_data[variants].max(axis=1)
            targets_data = targets_data.drop(columns=[v for v in variants if v != main_target])

    # targets_data = targets_data.drop(columns="gram_binary")
    
    # Plot distribution of merged targets
    stacked_nan_bar_plot = {}
    for col in targets_data.columns:
        positive = (targets_data[col] == 1).sum()
        negative = (targets_data[col] == 0).sum()
        nan_count = targets_data[col].isna().sum()
        stacked_nan_bar_plot[col] = {'Positive': positive, 'Negative': negative, 'NaN': nan_count}

    index_cols = list(stacked_nan_bar_plot.keys())

    # Plot distribution of merged targets
    stacked_nan_bar_plot_filtered = {}
    for col in TARGETS:
        positive = (targets_data[col] == 1).sum()
        negative = (targets_data[col] == 0).sum()
        nan_count = targets_data[col].isna().sum()
        stacked_nan_bar_plot_filtered[col] = {'Positive': positive, 'Negative': negative, 'NaN': nan_count}

    index_cols_filtered = list(stacked_nan_bar_plot_filtered.keys())
    
    def extract_target_name(col):
        parts = col.split("_")[2:]
        if parts and any(p for p in parts):  # Check if there's actual content
            return re.sub(' +', ' ', " ".join(parts).capitalize())
        else:
            return col  # Fallback to full column name
            
    target_names = [extract_target_name(col) for col in index_cols]
    target_names_filtered = [extract_target_name(col) for col in index_cols_filtered]

    antibiotic_name_mapping = {
        # Standard antibiotics
        'Amikacin': 'Amikacin',
        'Ampicillin': 'Ampicillin',
        'Amoxicillin': 'Amoxicillin',
        'Azithromycin': 'Azithromycin',
        'Aztreonam': 'Aztreonam',
        'Cefazolin': 'Cefazolin',
        'Cefepim': 'Cefepime',
        'Cefiderocol': 'Cefiderocol',
        'Cefotaxim': 'Cefotaxime',
        'Cefotetan': 'Cefotetan',
        'Cefpodoxim': 'Cefpodoxime',
        'Ceftazidim': 'Ceftazidime',
        'Ceftriaxon': 'Ceftriaxone',
        'Cefuroxim': 'Cefuroxime',
        'Cefuroxim axetil': 'Cefuroxime axetil',
        'Ciprofloxacin': 'Ciprofloxacin',
        'Clindamycin': 'Clindamycin',
        'Colistin ': 'Colistin',
        'Daptomycin': 'Daptomycin',
        'Ertapenem': 'Ertapenem',
        'Erythromycin': 'Erythromycin',
        'Fosfomycin': 'Fosfomycin',
        'Gentamicin': 'Gentamicin',
        'Imipenem': 'Imipenem',
        'Levofloxacin': 'Levofloxacin',
        'Linezolid': 'Linezolid',
        'Meropenem': 'Meropenem',
        'Minocyclin': 'Minocycline',
        'Moxifloxacin': 'Moxifloxacin',
        'Mupirocin': 'Mupirocin',
        'Nitrofurantoin': 'Nitrofurantoin',
        'Norfloxacin': 'Norfloxacin',
        'Oxacillin': 'Oxacillin',
        'Penicillin': 'Penicillin',
        'Rifampicin': 'Rifampicin',
        'Tedizolid': 'Tedizolid',
        'Teicoplanin': 'Teicoplanin',
        'Tetracyclin': 'Tetracycline',
        'Tigecyclin': 'Tigecycline',
        'Tobramycin': 'Tobramycin',
        'Vancomycin': 'Vancomycin',
        
        # Combinations (German → English)
        'Amoxicillin clavulansäure': 'Amoxicillin Clavulanate',
        'Cefepim . clavulansäure': 'Cefepime Clavulanate',
        'Cefotetan . cloxacillin': 'Cefotetan-cloxacillin',
        'Ceftazidim avibactam': 'Ceftazidime-avibactam',
        'Ceftolozan tazobactam': 'Ceftolozane-tazobactam',
        'Fosfomycin trometamol': 'Fosfomycin trometamol',
        'Piperacillin tazobactam': 'Piperacillin-tazobactam',
        'Cotrimoxazol': 'Trimethoprim-sulfamethoxazole',
        'Fusidinsäure': 'Fusidic acid',
        
        # Antifungals
        '5 fluorocytosin': '5-Fluorocytosine (Flucytosine)',
        'Amphotericin b': 'Amphotericin B',
        'Anidulafungin': 'Anidulafungin',
        'Caspofungin': 'Caspofungin',
        'Fluconazol': 'Fluconazole',
        'Isavuconazol': 'Isavuconazole',
        'Itraconazol': 'Itraconazole',
        'Micafungin': 'Micafungin',
        'Posaconazol': 'Posaconazole',
        'Voriconazol': 'Voriconazole',
        
        # Specific indications (German → English)
        'Ceftriaxon bei meningitis': 'Ceftriaxone (meningitis)',
        'Meropenem bei meningitis': 'Meropenem (meningitis)',
        'Penicillin bei endokarditis': 'Penicillin (endocarditis)',
        'Penicillin bei meningitis': 'Penicillin (meningitis)',
        'Penicillin bei pneumonie': 'Penicillin (pneumonia)',
        'Penicillin bei anderen infekten': 'Penicillin (other infections)',
        'Ciprofloxacin ohne meningitis': 'Ciprofloxacin (non-meningitis)',
        'Amoxicillin intravenös. ohne meningitis': 'Amoxicillin IV (non-meningitis)',
        'Amoxicillin oral. unkomplizierter hwi': 'Amoxicillin oral (uncomplicated UTI)',
        'Amoxicillin clavulansäure oral. bei unkompliziertem hwi': 'Amoxicillin Clavulanate oral (uncomplicated UTI)',
        'Levofloxacin bei unkompliziertem hwi': 'Levofloxacin (uncomplicated UTI)',
        
        # Resistance testing combinations
        'Gentamicin high level': 'Gentamicin high-level',
        'Imipenem . edta': 'Imipenem + EDTA (MBL screen)',
        'Penem': 'Carbapenem (screen)',
        'Penem . ampc inhibitor': 'Carbapenem + AmpC inhibitor',
        'Penem . mbl inhibitor': 'Carbapenem + MBL inhibitor',
        'Temocillin . mbl inhibitor': 'Temocillin + MBL inhibitor',
        'Cefpodoxim 10 . ampc inhibitor': 'Cefpodoxime + AmpC inhibitor',
        'Cefpodoxim 10 . esbl inhibitor . ampc inhibitor': 'Cefpodoxime + ESBL/AmpC inhibitor',
        
        # Synergy tests
        'Aztreonam .alleine synergietest.': 'Aztreonam (synergy test)',
        'Ceftazidim avibactam .alleine synergietest.': 'Ceftazidime-avibactam (synergy test)',
        
        # Technical/screening tests (may want to exclude from analysis)
        'Cazcv caz30': 'CAZ/CV vs CAZ30 (ESBL screen)',
        'Cpmcv cpm30': 'CPM/CV vs CPM30 (ESBL screen)',
        'Ctxcv ctx30': 'CTX/CV vs CTX30 (ESBL screen)',
        'Quotient ip.ipi': 'IP/IPI ratio (carbapenemase screen)',
        'Quotient imi.imd': 'IMI/IMD ratio (carbapenemase screen)',
        'Quotient atm gekreuzt.alleine': 'ATM ratio (synergy screen)',
        'Fici wert': 'FICI value (synergy index)',
    }

    def translate_antibiotic_name(name):
        """Translate German antibiotic name to English."""
        return antibiotic_name_mapping.get(name, name)
    
    # Apply to your target names
    target_names = [translate_antibiotic_name(name) for name in target_names]
    target_names_filtered = [translate_antibiotic_name(name) for name in target_names_filtered]

    df_results = pd.DataFrame(list(stacked_nan_bar_plot.values()), index=target_names)
    
    # Normalize each row to 100%
    df_normalized = df_results.div(df_results.sum(axis=1), axis=0) * 100
    
    # Sort by NaN percentage (ascending - less NaNs first)
    df_normalized = df_normalized.sort_values('NaN', ascending=True)
    
    # Plot stacked bar
    df_normalized.plot(kind='bar', stacked=True, figsize=(13, 9),
                       color=['tab:blue', 'tab:orange', '#d3d3d3'],
                       width=0.8)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Targets')
    
    plt.gca().set_xticklabels(labels=plt.gca().get_xticklabels(), rotation=90)
    plt.title('Target Distribution After Merging (Normalized)')
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.savefig(f"figures/targets_merged_distribution{'_ukbb' if is_ukbb else ''}.png")
        
    plt.close()


    # Filter df_normalized to only include targets in TARGETS
    df_results_filtered = pd.DataFrame(list(stacked_nan_bar_plot_filtered.values()), index=target_names_filtered)
    
    # Normalize each row to 100%
    df_normalized_filtered = df_results_filtered.div(df_results_filtered.sum(axis=1), axis=0) * 100
    
    # Sort by NaN percentage (ascending - less NaNs first)
    df_normalized_filtered = df_normalized_filtered.sort_values('NaN', ascending=True)

    print(df_normalized_filtered)
    

    # Plot stacked bar
    df_normalized_filtered.plot(kind='bar', stacked=True, figsize=(8, 6),
                     color=['tab:blue', 'tab:orange', '#d3d3d3'],
                     width=0.8)
    plt.ylabel('Percentage (%)')
    plt.xlabel('Targets')
    plt.xticks(rotation=90)
    plt.title('Target Distribution After Merging (Normalized)')
    plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f"figures/targets_chosen_merged_distribution{'_ukbb' if is_ukbb else ''}.png")
        
    plt.close()

    if use_bacteria_names:
        # THERE IS ALREADY A R PLOT THAT MERGES USING EUCAST RULES! compare wich is better
        pathogen_mapping = {
            # Escherichia coli (all variants)
            'Escherichia coli': 'Escherichia coli',
            'Escherichia coli, zweierlei Morphotypen': 'Escherichia coli',
            'Escherichia coli, dreierlei Morphotypen': 'Escherichia coli',
            'Escherichia coli (ESBL)': 'Escherichia coli (ESBL)',
            'Escherichia coli (Carbapenemase produzierend)': 'Escherichia coli (Carbapenemase)',
            'Escherichia coli (Colistin resistent)': 'Escherichia coli (Colistin-resistant)',
            'Escherichia hermannii': 'Escherichia coli',
            'Escherichia marmotae': 'Escherichia coli',
            
            # Klebsiella species
            'Klebsiella pneumoniae': 'Klebsiella pneumoniae',
            'Klebsiella pneumoniae (ESBL)': 'Klebsiella pneumoniae (ESBL)',
            'Klebsiella pneumoniae (Carbapenemase produzierend)': 'Klebsiella pneumoniae (Carbapenemase)',
            'Klebsiella pneumoniae (Colistin resistent)': 'Klebsiella pneumoniae (Colistin-resistant)',
            'Klebsiella variicola (K. pneumoniae-Gruppe)': 'Klebsiella pneumoniae',
            'Klebsiella variicola (K. pneumoniae-Gruppe) (ESBL)': 'Klebsiella pneumoniae (ESBL)',
            'Klebsiella oxytoca': 'Klebsiella oxytoca',
            'Klebsiella aerogenes (früher Enterobacter)': 'Klebsiella aerogenes',
            'Klebsiella species': 'Klebsiella species',
            
            # Enterococcus species
            'Enterococcus faecalis': 'Enterococcus faecalis',
            'Enterococcus faecium': 'Enterococcus faecium',
            'Enterococcus faecium, Vancomycin-resistent (VRE)': 'Enterococcus faecium (VRE)',
            'Enterococcus avium': 'Enterococcus species (other)',
            'Enterococcus gallinarum': 'Enterococcus species (other)',
            'Enterococcus species': 'Enterococcus species (other)',
            'Enterococcus raffinosus': 'Enterococcus species (other)',
            'Enterococcus hirae': 'Enterococcus species (other)',
            'Enterococcus casseliflavus': 'Enterococcus species (other)',
            'Enterococcus thailandicus': 'Enterococcus species (other)',
            
            # Proteus species
            'Proteus mirabilis': 'Proteus mirabilis',
            'Proteus mirabilis (ESBL)': 'Proteus mirabilis (ESBL)',
            'Proteus mirabilis / penneri': 'Proteus mirabilis',
            'Proteus vulgaris': 'Proteus species (other)',
            'Proteus vulgaris-Gruppe': 'Proteus species (other)',
            'Proteus penneri': 'Proteus species (other)',
            
            # Pseudomonas species
            'Pseudomonas aeruginosa': 'Pseudomonas aeruginosa',
            'Pseudomonas aeruginosa, mukös': 'Pseudomonas aeruginosa',
            'Pseudomonas putida': 'Pseudomonas species (other)',
            'Pseudomonas putida-Gruppe': 'Pseudomonas species (other)',
            'Pseudomonas putida (Carbapenemase produzierend)': 'Pseudomonas species (Carbapenemase)',
            'Pseudomonas oryzihabitans (P. putida-Gruppe)': 'Pseudomonas species (other)',
            'Pseudomonas stutzeri': 'Pseudomonas species (other)',
            'Pseudomonas fluorescens-Komplex': 'Pseudomonas species (other)',
            'Pseudomonas species': 'Pseudomonas species (other)',
            
            # Staphylococcus species
            'Staphylococcus aureus': 'Staphylococcus aureus',
            'Staphylococcus aureus, Oxacillin sensibel': 'Staphylococcus aureus',
            'Staphylococcus aureus, Oxacillin-/Methicillin-resistent (MRSA)': 'Staphylococcus aureus (MRSA)',
            'Staphylococcus argenteus (S. aureus-Komplex)': 'Staphylococcus aureus',
            'Staphylococcus epidermidis': 'Coagulase-negative Staphylococci',
            'Staphylococcus haemolyticus': 'Coagulase-negative Staphylococci',
            'Staphylococcus hominis': 'Coagulase-negative Staphylococci',
            'Staphylococcus capitis': 'Coagulase-negative Staphylococci',
            'Staphylococcus caprae': 'Coagulase-negative Staphylococci',
            'Staphylococcus lugdunensis': 'Coagulase-negative Staphylococci',
            'Staphylococcus saprophyticus': 'Staphylococcus saprophyticus',
            'Staphylococcus borealis': 'Coagulase-negative Staphylococci',
            
            # Enterobacter species
            'Enterobacter cloacae': 'Enterobacter cloacae',
            'Enterobacter cloacae-Gruppe': 'Enterobacter cloacae',
            'Enterobacter cloacae-Gruppe (ESBL)': 'Enterobacter cloacae (ESBL)',
            'Enterobacter bugandensis (E. cloacae-Gruppe)': 'Enterobacter cloacae',
            
            # Citrobacter species
            'Citrobacter freundii': 'Citrobacter species',
            'Citrobacter freundii-Gruppe': 'Citrobacter species',
            'Citrobacter freundii-Gruppe (Carbapenemase produzierend)': 'Citrobacter species (Carbapenemase)',
            'Citrobacter koseri': 'Citrobacter species',
            'Citrobacter farmeri': 'Citrobacter species',
            'Citrobacter amalonaticus': 'Citrobacter species',
            
            # Serratia species
            'Serratia marcescens': 'Serratia species',
            'Serratia rubidaea': 'Serratia species',
            'Serratia liquefaciens-Gruppe': 'Serratia species',
            
            # Acinetobacter species
            'Acinetobacter baumannii': 'Acinetobacter baumannii',
            'Acinetobacter baumannii-Gruppe': 'Acinetobacter baumannii',
            'Acinetobacter pittii (A. baumannii-Gruppe)': 'Acinetobacter baumannii',
            'Acinetobacter johnsonii': 'Acinetobacter species (other)',
            'Acinetobacter ursingii': 'Acinetobacter species (other)',
            'Acinetobacter species': 'Acinetobacter species (other)',
            'Acinetobacter species nicht A. baumannii-Gruppe': 'Acinetobacter species (other)',
            
            # Providencia species
            'Providencia stuartii': 'Providencia species',
            'Providencia stuartii (ESBL)': 'Providencia species (ESBL)',
            'Providencia stuartii (Carbapenemase produzierend)': 'Providencia species (Carbapenemase)',
            'Providencia rettgeri': 'Providencia species',
            
            # Morganella
            'Morganella morganii': 'Morganella morganii',
            
            # Salmonella species
            'Salmonella Serovar Enteritidis': 'Salmonella species',
            'Enteritische Salmonella Serogruppe E': 'Salmonella species',
            'Enteritische Salmonella Serogruppe B': 'Salmonella species',
            'Enteritische Salmonella Serogruppe C': 'Salmonella species',
            'Salmonella species': 'Salmonella species',
            
            # Streptococcus species
            'Streptococcus gallolyticus (S. bovis-Gruppe)': 'Streptococcus species',
            'Streptococcus anginosus (S. anginosus / milleri-Gruppe)': 'Streptococcus species',
            'Streptococcus pneumoniae (Pneumokokken)': 'Streptococcus pneumoniae',
            'Streptococcus agalactiae (Serogruppe B)': 'Streptococcus agalactiae',
            'Streptococcus mitis-Gruppe': 'Streptococcus species',
            
            # Candida species
            'Candida albicans': 'Candida species',
            'Candida glabrata (syn. Nakaseomyces glabratus)': 'Candida species',
            'Candida tropicalis': 'Candida species',
            
            # Corynebacterium species
            'Corynebacterium jeikeium': 'Corynebacterium species',
            'Corynebacterium aurimucosum ': 'Corynebacterium species',
            'Corynebacterium glucuronolyticum': 'Corynebacterium species',
            'Corynebacterium amycolatum': 'Corynebacterium species',
            
            # Aerococcus species
            'Aerococcus sanguinicola': 'Aerococcus species',
            'Aerococcus urinae': 'Aerococcus species',
            
            # Other rare/miscellaneous
            'Stenotrophomonas maltophilia': 'Stenotrophomonas maltophilia',
            'Hafnia alvei': 'Other Enterobacterales',
            'Achromobacter xylosoxidans': 'Other non-fermenters',
            'Pasteurella multocida': 'Other pathogens',
            'Sphingomonas paucimobilis (Nonfermenter)': 'Other non-fermenters',
            'Kluyvera species': 'Other Enterobacterales',
            'Alcaligenes faecalis': 'Other non-fermenters',
            'Kerstersia gyiorum (früher Alcaligenes)': 'Other non-fermenters',
            'Burkholderia cepacia-Komplex': 'Other non-fermenters',
            'Bacillus species nicht B. cereus-Gruppe': 'Other pathogens',
            'Aeromonas hydrophila': 'Other pathogens',
            'Haemophilus influenzae': 'Haemophilus influenzae',
            'Grampositive Flora': 'Mixed/unspecified flora',
        }
        
        # Apply the mapping
        X["urine_organism_grouped"] = X["urine_organism"].map(pathogen_mapping)
        
        # Plot the grouped distribution
        fig, ax = plt.subplots(figsize=(14, 7))
        
        organism_counts = X["urine_organism_grouped"].value_counts().sort_values(ascending=False)
        
        organism_counts.plot(kind='bar', color='tab:blue', width=0.8, ax=ax)
        plt.ylabel('Count')
        plt.xlabel('Pathogen')
        plt.xticks(rotation=45, ha='right')
        plt.title('Pathogen Distribution (Grouped)')
        plt.tight_layout()
        
        if is_ukbb:
            plt.savefig("figures/pathogens_distribution_grouped_ukbb.png")
        else:
            plt.savefig("figures/pathogens_distribution_grouped.png")
        
        log(DEBUG, "aborting due to use of pathogen names, using them only for plotting")
        exit(0)


    def __run_rep(current_rep):

        stacked_nan_bar_plot = {}
        for this_target in targets:
            if "antibiogram" in this_target:
                
                y = targets_data[this_target]
        
                dir_path = os.path.join(
                        "Data", "models", f"saved_models_{model_to_train}", this_target
                    )
                dir_path2 = os.path.join(
                        "Data", "predictions"
                    )
                dir_path3 = os.path.join(
                        "figures", f"{model_to_train}_figures"
                    )
                dir_path4 = os.path.join(
                        "Data", "for_shap", f"{model_to_train}"
                    )
                os.makedirs(dir_path, exist_ok=True)
                os.makedirs(dir_path2, exist_ok=True)
                os.makedirs(dir_path3, exist_ok=True)
                os.makedirs(dir_path4, exist_ok=True)
        
                # model_pred, dt_preds = main_train(X, y, drop_nan_perc=40, use_risk=use_risk, is_premodel=False)
        
                # print(X['gram_prediction'])
                # pd.set_option('display.max_rows', None)
        
                # print(pd.Series(X.columns))
                # print(X.loc[:, "pregnancy_yn"].value_counts())

                res = main_train(X.reset_index(drop=True), y.reset_index(drop=True), drop_nan_perc=40, use_risk=use_risk, is_premodel=False, model_to_train=model_to_train, tuning_active=True, is_ukbb=is_ukbb, rep=current_rep)
    
                # # temp just to plot stacked bar plot nans for all targets
                # if isinstance(res, dict):
                #     stacked_nan_bar_plot[this_target] = res

                if isinstance(res, str):
                    if res == "empty":
                        log(ERROR, "look at error above.")
                        continue
        
                # Store results
                # data4comparison[this_target] = model_pred
                all_dt_preds.append(res)
                res.to_csv(f"Data/predictions/dt_preds_{model_to_train}_{this_target}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.csv")


        # index_cols = list(stacked_nan_bar_plot.keys())
        # target_names = [re.sub(' +', ' ', " ".join(target.split("_")[2:]).capitalize()) for target in index_cols]
        
        # df_results = pd.DataFrame(list(stacked_nan_bar_plot.values()), index=target_names)
        
        # # Normalize each row to 100%
        # df_normalized = df_results.div(df_results.sum(axis=1), axis=0) * 100
        
        # # Sort by NaN percentage (ascending - less NaNs first)
        # df_normalized = df_normalized.sort_values('NaN', ascending=True)
        
        # # Plot stacked bar
        # df_normalized.plot(kind='bar', stacked=True, figsize=(13, 9),
        #                    color=['tab:blue', 'tab:orange', '#d3d3d3'],
        #                    width=0.8)
        # plt.ylabel('Percentage (%)')
        # plt.xlabel('Targets')
        
        # plt.gca().set_xticklabels(labels=plt.gca().get_xticklabels(), rotation=90)
        # plt.title('Target Distribution (Normalized)')
        # plt.legend(title='Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        # plt.tight_layout()
        # plt.savefig("nans_colored.png")

        # adasdadadas
    
        to_save = pd.concat(all_dt_preds, ignore_index=True).reset_index(drop=True)
        to_save.to_csv(f"Data/predictions/dt_preds_{model_to_train}{'' if not is_ukbb else '_UKBB'}_repetition{current_rep}.csv")
    
        SHAPE_LOG_PATH = Path(f"training_shapes{'_UKBB' if is_ukbb else ''}.log")
        import datetime
        now = datetime.datetime.now()
        with open(SHAPE_LOG_PATH, "a") as f:
            f.write(f"{now} - EOF\n\n")

    def run_multiproc():
        start = time.time()
        
        log(INFO, f"Main process PID: {os.getpid()}")
        master_processes = []
        
        # Start ALL processes before waiting
        for i in range(REPETITIONS):
            log(INFO, f"[...] Starting master process for round {i}, PID: {os.getpid()}")
            p = Process(target=__run_rep, args=(i,))
            p.start()
            master_processes.append(p)
        
        # Wait for ALL processes to complete
        for p in master_processes:
            p.join()
        
        end = time.time()
        log(INFO, f"\nTotal execution time: {end - start} seconds")

    # --------------------------------------------------------------------------------
    # Main Loop call
    # --------------------------------------------------------------------------------


    
    run_multiproc()

    # for i in range(REPETITIONS):
    #     __run_rep(i)


# --------------------------------------------------------------------------------
# Entry Point
# --------------------------------------------------------------------------------
if __name__ == "__main__":

    # if os.path.exists("dt_preds_urine_antibiogram_amoxicillin___clavulansäure.csv"):
    #     dt_preds = pd.read_csv("dt_preds_urine_antibiogram_amoxicillin___clavulansäure.csv", index_col=0)
    #     # dt_preds must include columns:
    #     # "target", "true_class", "fold", "model", "indices", and your prediction columns (e.g., "pred_RF", "pred_XGB", ...)
    #     overview(dt_preds, targets=None, ncols=2, filename="compact_overview_RF.png")

    main()
