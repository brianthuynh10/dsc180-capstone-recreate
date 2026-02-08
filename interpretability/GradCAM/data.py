import pandas as pd 
import numpy as np 
import regex as re
import h5py as h5
import os


def grab_labled_bnpp_reports(path='~/teams/b1/bnpp-reports-clean.csv'):
    """
    Load cleaned BNP reports CSV and return DataFrame
    """
    return pd.read_csv(path, usecols=['Phonetic','ReportClean', 'LLM_Output'])

def connect_data(phonetic_name, base="~/teams/b1/"):
    """
    Given a phonetic name, return the HDF5 file name that contains it
    """
    hdf5_names = [f'bnpp_frontalonly_1024_{i}' for i in range(1,11)]
    hdf5_names.append('bnpp_frontalonly_1024_0_1')

    base = os.path.expanduser(base)

    for name in hdf5_names:
        path = os.path.join(base, f"{name}.hdf5")

        if not os.path.exists(path):
            continue

        with h5py.File(path, "r") as f:
            for key in f.keys():
                if phonetic_name.lower() in key.lower():
                    return name  # <-- this is the file it lives in

    return None

def main(): 
    df = grab_labled_bnpp_reports()
    df['hdf5_file'] = df['Phonetic'].apply(connect_data)

    # if no file found, drop
    df = df.dropna(subset=['hdf5_file'])

    return df