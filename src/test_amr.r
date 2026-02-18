# install.packages("AMR", repos='https://stat.ethz.ch/CRAN/')
# update.packages(ask = FALSE, repos='https://stat.ethz.ch/CRAN/')
library(dplyr)
library(AMR)
library(ggplot2)

# Read your data
df <- read.csv("src/src_data/all_USB_data_risk_consent_dateshift.csv")

df <- df[,-1] # remove first column which is the index (range of numbers)

# Select all columns with "antibiogram" in their name
antibiograms_cols <- grep("antibiogram", colnames(df), value = TRUE)

# Step 1: Set cleaning regex BEFORE using as.mo()
options(AMR_cleaning_regex = mo_cleaning_regex())

# Step 2: If needed, reset session MO memory
mo_reset_session()

# Step 3: Rerun the classification with regex cleaning active
df$mo <- as.mo(df$urine_organism)

# Step 4: Check failures again
mo_failures()

# Check unique microorganisms (optional)
unique(df$mo)

df <- df %>%
  mutate(gram_binary = case_when(
    mo_is_gram_positive(mo) ~ 1,
    mo_is_gram_negative(mo) ~ 0,
    TRUE ~ NA  # for unknown or unclassified
  ))

df %>%
  group_by(gram_binary) %>%
  summarise(n = n())

# mapping susceptibility results to S, R, or NA
target_mapping <- c(
  "sensibel" = "S",
  "resistent" = "R",
  "intermediär" = "R",
  "unbekannt" = NA
)


data_clean <- df %>%
  mutate(across(all_of(antibiograms_cols),
                ~ recode(.x, !!!target_mapping)))

# Convert eligible columns to 'sir' class and save
data_clean <- data_clean %>%
  mutate(across(where(is_sir_eligible), as.sir))

# Apply EUCAST expert rules, overwriting original values
data_clean <- eucast_rules(data_clean, overwrite = TRUE)

# Define a function to convert SIR values to binary resistance
sir_to_binary <- function(x) {
  case_when(
    x == "R" ~ 1,
    x %in% c("S", "I", "NI", "SDD") ~ 0,
    TRUE ~ NA  # keep NA for unknown/missing values
  )
}

# Apply this conversion to your antibiotic columns
data_clean_binary <- data_clean %>%
  mutate(across(all_of(antibiograms_cols), sir_to_binary))

# First: get the top microorganisms from the SIR data (not binary)
top_mo <- data_clean %>%
  count(mo, sort = TRUE) %>%
  slice_head(n = 20) %>%
  mutate(name = mo_fullname(mo))

# Now make the bar plot
ggplot(top_mo, aes(x = reorder(name, n), y = n)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_text(aes(label = n), hjust = -0.2, size = 3) +
  coord_flip() +
  labs(
    title = "Top 20 Most Common Microorganisms",
    x = "Microorganism",
    y = "Count"
  ) +
  theme_minimal() +
  expand_limits(y = max(top_mo$n) * 1.1)

ggsave("figures/top_microorganisms.png", width = 8, height = 8, dpi = 300)

# Save to csv for use in modelling
write.csv(data_clean_binary, "src/src_data/processed_amr_data_binary.csv", row.names = FALSE)

# write.csv(data_clean, "src/src_data/processed_amr_data_sir.csv", row.names = FALSE)

# Print summary of processed data
cat("Data processing complete!\n")
cat("Original data shape:", dim(df), "\n")
cat("Processed data shape:", dim(data_clean_binary), "\n")
cat("unique patients:", length(unique(data_clean_binary$patient_id_hashed)), "\n")
cat("Number of antibiogram columns:", length(antibiograms_cols), "\n")