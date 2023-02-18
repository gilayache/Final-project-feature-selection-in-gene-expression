import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

# Load the dataset
data = pd.read_csv('/Users/gilayache/PycharmProjects/Final-project-feature-selection-in-gene-expression/data/processed/merged_dataset.csv')



