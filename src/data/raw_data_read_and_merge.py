import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

sampleinfo_SCANB_t_path = "/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/data/raw/sampleinfo_SCANB_t.csv"
SCANB_path = "/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/data/raw/SCANB.csv"
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

# check the number of column is correct
assert info_SCANB.shape[1] + SCANB.shape[1] == merged_dataset.shape[1] + 1, logging.info(
        "number of column is not correct"
    )

merged_dataset.to_csv("/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/data/processed/merged_dataset.py")
