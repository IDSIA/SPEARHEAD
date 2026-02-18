import seaborn as sns
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import shap

from shiny import reactive
from shiny.express import input, render, ui

from shared import app_dir

# Load feature name mapping for display
_var_map_df = pd.read_csv(f"{app_dir}/www/Spearhead_UTI_variable_mapping.csv", index_col=0)
FEATURE_LABELS = _var_map_df["Clinical Name "].to_dict()

# Define available antibiotic models
ANTIBIOTIC_MODELS = {
    "Amoxicillin Clavulanate": "models/RF_fold_0_repetition0_amoxicillin.pkl",
    "Cefuroxim": "models/RF_fold_0_repetition0_cefuroxim.pkl",
    "Cotrimoxazol": "models/RF_fold_0_repetition0_cotrimoxazol.pkl",
    "Fosfomicyn Trometamol": "models/RF_fold_0_repetition0_fosfomicyn.pkl",
    "Nitrofurantoin": "models/RF_fold_0_repetition0_nitrofurantoin.pkl",
    "Norfloxacin": "models/RF_fold_0_repetition0_norfloxacin.pkl",
    "Ciprofloxacin": "models/RF_fold_0_repetition0_ciprofloxacin.pkl",
    "Ceftriaxon": "models/RF_fold_0_repetition0_ceftriaxon.pkl",
    "Piperacillin Tazobactam": "models/RF_fold_0_repetition0_piperacillin.pkl",

}

# authoritative feature order (fallback used if model.feature_names_in_ absent)
FEATURES = [
    "had_prev_resistance", "prev_exp_6M", "risk_chronic_cystitis_other_any",
    "urine_material_Urin_aus_Blasenpunktion", "risk_discharge_30_days_unknown",
    "blood_crp__mg_l_", "risk_operations_female_genital_organs_30d",
    "urine_mic_bilirubin", "risk_urinary_tract_surgery_30d", "sex_männlich",
    "urine_mic_glucose", "risk_other_operations_urinary_tract_30d",
    "urine_material_Urin_nicht_genauer_bezeichnet", "risk_bacteria_virus_infections_30d",
    "risk_operations_ureter_30d", "case_type_teil_stationär", "risk_urosepsis_30d",
    "risk_operations_kidney_30d", "risk_penicillin_allergies_any",
    "urine_mic_nitrites", "age", "prev_exp_1M", "pregnancy_yn", "prev_exp_1W",
    "prev_exp_2W", "risk_operations_urinary_bladder_30d", "risk_ileal_conduit_30d",
    "risk_pyelonephritis_or_renal_tubulo_30d", "risk_operations_urethra_30d",
    "urine_material_Urin_aus_Dauerkatheter", "risk_chronic_pyelonephritis_any",
    "risk_indwelling_foley_catheter_30d", "urine_material_Urinsäckchen",
    "urine_mic_ph", "prev_exp_ALL", "risk_lithotripsy_ultrasound_30d",
    "urine_mic_spez_gewicht", "urine_material_Urin_aus_Einmalkatheter",
    "urine_mic_hemoglobin", "risk_inflammatory_disease_of_prostate_30d",
    "risk_dysuria_30d", "urine_material_Mittelstrahlurin",
    "risk_discharge_30_days_still_admitted", "risk_chronic_cystitis_interstitial_any",
    "risk_operations_male_reproductive_organs_30d", "urine_mic_ketokorper",
    "urine_mic_leucocytes", "risk_diabetes_any", "multiple_occurences_prev_resistance",
    "risk_suprapubic_catheter_30d", "risk_charlson_score", "risk_hypertension_any",
    "risk_urostomy_30d", "case_type_ambulant", "risk_cystitis_30d", "prev_exp_1Y"
]


ui.page_opts(
    title=ui.tags.span(
        ui.img(src="logo.jpg", height="100px",
               style="vertical-align: middle; margin-right: 10px;"),
        ui.tags.span("Spearhead Antibiotic Resistance Calculator",
                     style="font-size: 1.5em; font-weight: bold; flex-grow: 1; text-align: center;"),
        style="display: flex; align-items: center; width: 100%;"
    ),
    fillable=False
)

# Short description at top
ui.markdown("""
**Welcome to the Spearhead Antibiotic Resistance Calculator.** 
Enter patient and laboratory data in the sidebar to predict resistance probabilities across multiple antibiotics.
""")

# UI
# ui.page_opts(title="Resistance prediction", fillable=True)

with ui.sidebar(title="Prediction inputs"):
    ui.h4("Patient & lab inputs")

    # numeric / continuous
    ui.input_numeric("age", FEATURE_LABELS.get("age", "Age"), value=65, min=0, max=120, step=1)
    ui.input_numeric("blood_crp__mg_l_", FEATURE_LABELS.get("blood_crp__mg_l_", "CRP (mg/L)"),
                     value=5.0, min=0.0, max=500.0, step=0.1)
    ui.input_numeric("urine_mic_ph", FEATURE_LABELS.get("urine_mic_ph", "Urine pH"), value=6.0,
                     min=0.0, max=14.0, step=0.1)
    ui.input_numeric("urine_mic_spez_gewicht", FEATURE_LABELS.get("urine_mic_spez_gewicht", "Urine specific gravity"),
                     value=1.015, min=1.000, max=1.050, step=0.001)
    ui.input_numeric("urine_mic_leucocytes", FEATURE_LABELS.get("urine_mic_leucocytes", "Urine leucocytes"),
                     value=0.0, min=0.0, max=1e6, step=1.0)
    ui.input_numeric("urine_mic_nitrites", FEATURE_LABELS.get("urine_mic_nitrites", "Urine nitrites"),
                     value=0, min=0, max=1, step=1)
    ui.input_numeric("urine_mic_glucose", FEATURE_LABELS.get("urine_mic_glucose", "Urine glucose"),
                     value=0.0, min=0.0, max=1e6, step=1.0)
    ui.input_numeric("urine_mic_hemoglobin", FEATURE_LABELS.get("urine_mic_hemoglobin", "Urine hemoglobin"),
                     value=0.0, min=0.0, max=1e6, step=1.0)
    ui.input_numeric("urine_mic_bilirubin", FEATURE_LABELS.get("urine_mic_bilirubin", "Urine bilirubin"),
                     value=0.0, min=0.0, max=1e6, step=1.0)
    ui.input_numeric("urine_mic_ketokorper", FEATURE_LABELS.get("urine_mic_ketokorper", "Urine ketone bodies"),
                     value=0.0, min=0.0, max=1e6, step=1.0)
    ui.input_numeric("risk_charlson_score", FEATURE_LABELS.get("risk_charlson_score", "Charlson score"),
                     value=0.0, min=0.0, max=50.0, step=1.0)

    # binary / flags
    binary_flags = [
        "had_prev_resistance", "prev_exp_6M", "risk_chronic_cystitis_other_any",
        "urine_material_Urin_aus_Blasenpunktion", "risk_discharge_30_days_unknown",
        "risk_operations_female_genital_organs_30d", "risk_urinary_tract_surgery_30d",
        "sex_männlich", "risk_other_operations_urinary_tract_30d",
        "urine_material_Urin_nicht_genauer_bezeichnet", "risk_bacteria_virus_infections_30d",
        "risk_operations_ureter_30d", "case_type_teil_stationär", "risk_urosepsis_30d",
        "risk_operations_kidney_30d", "risk_penicillin_allergies_any", "prev_exp_1M",
        "pregnancy_yn", "prev_exp_1W", "prev_exp_2W", "risk_operations_urinary_bladder_30d",
        "risk_ileal_conduit_30d", "risk_pyelonephritis_or_renal_tubulo_30d",
        "risk_operations_urethra_30d", "urine_material_Urin_aus_Dauerkatheter",
        "risk_chronic_pyelonephritis_any", "risk_indwelling_foley_catheter_30d",
        "urine_material_Urinsäckchen", "prev_exp_ALL", "risk_lithotripsy_ultrasound_30d",
        "urine_material_Urin_aus_Einmalkatheter", "risk_inflammatory_disease_of_prostate_30d",
        "risk_dysuria_30d", "urine_material_Mittelstrahlurin",
        "risk_discharge_30_days_still_admitted", "risk_chronic_cystitis_interstitial_any",
        "risk_operations_male_reproductive_organs_30d", "risk_diabetes_any",
        "multiple_occurences_prev_resistance", "risk_suprapubic_catheter_30d",
        "risk_hypertension_any", "risk_urostomy_30d", "case_type_ambulant",
        "risk_cystitis_30d", "prev_exp_1Y"
    ]

    for fid in binary_flags:
        # Use clinical name from mapping, fallback to formatted id
        lbl = FEATURE_LABELS.get(fid, fid.replace("_", " ").replace("  ", " ")).strip()
        ui.input_checkbox(fid, lbl, value=False)

    ui.hr()


# NEW: Multi-Antibiotic Comparison Card
with ui.card(full_screen=True):
    ui.card_header("Compare All Antibiotics")
    
    @render.plot
    def antibiotic_comparison_plot():
        import matplotlib.pyplot as plt
        
        X = _input_row()
        results = []
        
        for ab_name, model_path in ANTIBIOTIC_MODELS.items():
            try:
                model = joblib.load(f"{app_dir}/{model_path}")
                features = list(getattr(model, "feature_names_in_", FEATURES))
                
                # Rebuild input row for this model's features
                row = {}
                for f in features:
                    try:
                        val = getattr(input, f)()
                    except Exception:
                        val = None
                    if isinstance(val, bool):
                        val = int(val)
                    if val is None:
                        val = np.nan
                    row[f] = val
                X_model = pd.DataFrame([row], columns=features)
                
                probs = model.predict_proba(X_model)[0]
                # Get probability of resistance (class 1)
                p_res = probs[1] if len(probs) > 1 else 1 - probs[0]
                results.append({'antibiotic': ab_name, 'p_resistance': p_res})
            except Exception as e:
                results.append({'antibiotic': ab_name, 'p_resistance': np.nan})
        
        res_df = pd.DataFrame(results).sort_values('p_resistance', ascending=True)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        bars = ax.barh(res_df['antibiotic'], res_df['p_resistance'] * 100, color="grey")
        ax.axvline(x=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
        ax.set_xlim(0, 100)
        ax.set_xlabel('Probability of Resistance (%)')
        ax.set_title('Resistance Risk Across All Antibiotics')
        
        # Add value labels
        for bar, val in zip(bars, res_df['p_resistance']):
            ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                    f'{val*100:.1f}%', va='center', fontsize=9)
        
        # plt.tight_layout()
        return fig


# NEW: Feature Importance Card
with ui.card(full_screen=True):
    ui.card_header("Feature Importance")

    @reactive.calc
    def _loaded_model():
        """Reactively load the model based on selected antibiotic."""
        model_file = ANTIBIOTIC_MODELS[input.antibiotic()]
        return joblib.load(f"{app_dir}/{model_file}")

    @reactive.calc
    def _input_row():
        model = _loaded_model()
        # get features in model order if available
        features = list(getattr(model, "feature_names_in_", FEATURES))
        row = {}
        for f in features:
            try:
                val = getattr(input, f)()
            except Exception:
                # input not present -> NaN
                val = None
            # convert checkboxes (bool) -> int
            if isinstance(val, bool):
                val = int(val)
            # None -> np.nan so pipeline can handle missing
            if val is None:
                val = np.nan
            row[f] = val
        return pd.DataFrame([row], columns=features)
    
    # Antibiotic selection
    ui.input_select(
        "antibiotic",
        "Select Antibiotic",
        choices=list(ANTIBIOTIC_MODELS.keys()),
        selected=list(ANTIBIOTIC_MODELS.keys())[0]
    )
    
    @render.plot
    def feature_importance_plot():
        model = _loaded_model()
        X = _input_row()
        features = list(getattr(model, "feature_names_in_", FEATURES))
        
        # Get the classifier from pipeline if needed
        if hasattr(model, 'named_steps'):
            clf = model.named_steps.get('classifier') or model.named_steps.get('clf') or list(model.named_steps.values())[-1]
        else:
            clf = model
        
        # Calculate SHAP values
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(X)
        
        # Extract values - shap_values.values has shape (n_samples, n_features) or (n_samples, n_features, n_classes)
        vals = shap_values.values
        if vals.ndim == 3:
            # Multi-class: take class 1 (resistant)
            shap_vals = vals[0, :, 1]
        else:
            shap_vals = vals[0, :]
        
        # Create DataFrame and get top 15 by absolute value
        imp_df = pd.DataFrame({
            'feature': features,
            'importance': np.abs(shap_vals)
        }).sort_values('importance', ascending=True).tail(15)
        
        # Rename features using mapping, keep original if not found
        imp_df['feature'] = imp_df['feature'].map(FEATURE_LABELS).fillna(imp_df['feature'])
        
        # Assign colors based on SHAP sign per feature
        # We need the original signed SHAP values in the same order as imp_df
        imp_df_signed = pd.DataFrame({
            'feature': features,
            'importance': np.abs(shap_vals),
            'sign': np.sign(shap_vals)
        }).sort_values('importance', ascending=True).tail(15)

        imp_df_signed['feature'] = imp_df_signed['feature'].map(FEATURE_LABELS).fillna(imp_df_signed['feature'])

        colors = ['tab:red' if s >= 0 else 'tab:blue' for s in imp_df_signed['sign']]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(imp_df_signed['feature'], imp_df_signed['importance'], color=colors)
        ax.set_xlabel('|SHAP value|')
        ax.set_title(f'Top 15 Features for {input.antibiotic()} Resistance Prediction')

        # Manual legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color='tab:red', label='Toward resistance'),
            Patch(color='tab:blue', label='Toward susceptibility')
        ])

        # plt.tight_layout()
        return fig


# Longer description at bottom
ui.markdown("""
---

### About This Tool

This calculator uses machine learning models trained on clinical data to predict the probability of antibiotic resistance for urinary tract infections. 

**How to use:**
- Enter patient demographics and laboratory values in the sidebar
- View resistance predictions across all antibiotics in the comparison chart
- Select a specific antibiotic to see feature importance for that model

**Disclaimer:** This tool is intended for research and educational purposes only. Clinical decisions should always be made in consultation with qualified healthcare professionals.

For more information about the Spearhead project, please contact [placeholder@example.com](mailto:placeholder@example.com).
""")
