import pandas as pd
import yaml


def load_data(path):
    """
    load the data
    """
    data = pd.read_csv(path)

    return data

def load_params(path):
    """
    load the params
    """
    with open(path, 'r') as f:
        params = yaml.safe_load(f)



    return params
