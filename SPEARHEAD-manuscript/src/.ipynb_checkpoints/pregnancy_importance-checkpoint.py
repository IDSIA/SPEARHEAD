import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X = pd.read_csv(
    "Data/for_shap/RF/X_all_Amoxicillin clavulansäure_repetition1.csv",
    index_col=0
)
shap_vals = np.load(
    "Data/for_shap/RF/shap_values_combined_Amoxicillin clavulansäure_repetition1.npy"
)

X = X.drop(columns="patient_id_hashed")
# sanity check
assert shap_vals.shape[0] == X.shape[0]
assert shap_vals.shape[1] == X.shape[1]

preg_idx = X.columns.get_loc("pregnancy_yn")

mask_preg = X["pregnancy_yn"] == 1
mask_nonpreg = X["pregnancy_yn"] == 0

mean_abs_shap = {
    "Nonpregnant": np.abs(shap_vals[mask_nonpreg, preg_idx]).mean(),
    "Pregnant": np.abs(shap_vals[mask_preg, preg_idx]).mean(),
}

df_plot = pd.DataFrame.from_dict(
    mean_abs_shap, orient="index", columns=["mean_abs_shap"]
)

plt.figure(figsize=(6, 4))
plt.bar(df_plot.index, df_plot["mean_abs_shap"])
plt.ylabel("Mean(|SHAP value|)")
plt.title("Use of Pregnancy Feature by Test Population")
plt.tight_layout()
plt.savefig("src/Fig_use_of_pregnancy_feature.png")


plt.figure(figsize=(8, 6))

plt.hist(
    shap_vals[mask_nonpreg, preg_idx],
    bins=40,
    alpha=0.6,
    label=f"Nonpregnant (n={len(shap_vals[mask_nonpreg, preg_idx])})",
    density=True,
)

plt.hist(
    shap_vals[mask_preg, preg_idx],
    bins=40,
    alpha=0.6,
    label=f"Pregnant (n={len(shap_vals[mask_preg, preg_idx])})",
    density=True,
)

plt.axvline(0, linestyle="--", linewidth=1)
plt.xlabel("SHAP value (pregnancy)")
plt.ylabel("Density")
plt.legend()
plt.gca().set_yscale('log')
plt.title("Distribution of SHAP Values for Pregnancy Feature")
plt.tight_layout()
plt.savefig("src/Fig_distribution_SHAP_pregnancy.png")


# ---------------------------------------------------------------


# import numpy as np
# import pandas as pd
# import shap 
# import matplotlib.pyplot as plt

# X = pd.read_csv(
#     "Data/for_shap/RF/X_all_Amoxicillin clavulansäure_repetition1.csv",
#     index_col=0
# )
# shap_vals = np.load(
#     "Data/for_shap/RF/shap_values_combined_Amoxicillin clavulansäure_repetition1.npy"
# )
# X = X.drop(columns="patient_id_hashed")
# # sanity check
# assert shap_vals.shape[0] == X.shape[0]
# assert shap_vals.shape[1] == X.shape[1]

# mask_preg = X["pregnancy_yn"] == 1
# mask_nonpreg = X["pregnancy_yn"] == 0

# # ---- exclude pregnancy feature by position ----
# preg_idx = X.columns.get_loc("pregnancy_yn")
# col_mask = np.arange(X.shape[1]) != preg_idx

# features = X.columns[col_mask]

# mean_preg = np.abs(shap_vals[mask_preg][:, col_mask]).mean(axis=0)
# mean_nonpreg = np.abs(shap_vals[mask_nonpreg][:, col_mask]).mean(axis=0)

# df = pd.DataFrame({
#     "feature": features,
#     "pregnant": mean_preg,
#     "nonpregnant": mean_nonpreg
# })


# df["max_importance"] = df[["pregnant", "nonpregnant"]].max(axis=1)
# df = df.sort_values(
#     ["pregnant", "nonpregnant"], ascending=False
# )


# fig, ax = plt.subplots(figsize=(8, 10))

# ax.barh(df["feature"], -df["nonpregnant"], label="Nonpregnant")
# ax.barh(df["feature"],  df["pregnant"], label="Pregnant")

# ax.axvline(0, linewidth=1)
# ax.set_xlabel("Mean(|SHAP|) within subgroup")
# ax.set_title("Conditional Feature Importance from Pooled Model")

# ax.legend()
# plt.tight_layout()
# plt.savefig("src/Conditional Feature Importance from Pooled Model.png")


# df["delta"] = df["pregnant"] - df["nonpregnant"]
# df_delta = df.sort_values("delta")

# fig, ax = plt.subplots(figsize=(8, 10))
# ax.barh(df_delta["feature"], df_delta["delta"])
# ax.axvline(0, linewidth=1)

# ax.set_xlabel("Δ Mean(|SHAP|) (Pregnant − Nonpregnant)")
# ax.set_title("Change in Predictor Importance Conditional on Pregnancy")

# plt.tight_layout()
# plt.savefig("src/Change in Predictor Importance Conditional on Pregnancy.png")


# feature = "had_prev_resistance"
# idx = X.columns.get_loc(feature)

# plt.figure(figsize=(6, 4))

# plt.scatter(
#     X.loc[mask_nonpreg, feature],
#     shap_vals[mask_nonpreg, idx],
#     alpha=0.25,
#     label="Nonpregnant"
# )

# plt.scatter(
#     X.loc[mask_preg, feature],
#     shap_vals[mask_preg, idx],
#     alpha=0.25,
#     label="Pregnant"
# )

# plt.xlabel(feature)
# plt.ylabel("SHAP value")
# plt.legend()
# plt.title(f"Effect Modification by Pregnancy: {feature}")
# plt.tight_layout()
# plt.savefig(f"src/Effect Modification by Pregnancy: {feature}.png")
