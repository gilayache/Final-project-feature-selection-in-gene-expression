import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO)

# Open a file with relative paths
file_dir = os.path.dirname(os.path.realpath(__file__))
back_two_levels = os.path.join(file_dir, "..", "..")
sampleinfo_SCANB_t_path = os.path.join(back_two_levels, "data", "raw/sampleinfo_SCANB_t.csv")
SCANB_path = os.path.join(back_two_levels, "data", "raw/SCANB.csv")
merged_dataset_path = os.path.join(back_two_levels, "data", "processed/merged_dataset.csv")

# read the sampleinfo_SCANB_t
info_SCANB = pd.read_csv(sampleinfo_SCANB_t_path).drop('Unnamed: 0', axis=1)
# read the SCANB
SCANB = pd.read_csv(SCANB_path).T.reset_index()
# Get the first row and assign it as the new header
SCANB.columns = SCANB.iloc[0]
# Remove the first row and reset_index
SCANB = SCANB.iloc[1:]
# Rename the 'samplename'
SCANB = SCANB.rename(columns={'Unnamed: 0': 'samplename'})
merged_dataset = SCANB.merge(info_SCANB, left_on='samplename', right_on='samplename')

# check the number of columns is correct
assert info_SCANB.shape[1] + SCANB.shape[1] == merged_dataset.shape[1] + 1, logging.info(
        "number of column is not correct"
    )
# Create the LumA column for classification
merged_dataset['LumA_target'] = merged_dataset['PAM50'].apply(lambda x: 1 if x == 'LumA' else 0)

merged_dataset.to_csv(merged_dataset_path,index=False)