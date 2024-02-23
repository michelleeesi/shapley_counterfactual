#!/usr/bin/python3
#SBATCH --array=1-660
import os

from functions import *
import multiprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json

X_train_res = pd.read_csv("reservation_train.csv")
X_test_res = pd.read_csv("reservation_test.csv")
unique_categories = X_train_res["arrival_month"].unique()

with open("parameters_natural1.json", "r") as json_file:
    # Load the JSON content
    parameter_dict = json.load(json_file)

job_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
job_id = str(job_id)
compareNatural(X_train_res, X_test_res, *parameter_dict[job_id])
