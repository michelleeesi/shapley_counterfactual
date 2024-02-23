#!/usr/bin/python3
#SBATCH --array=1-250
import os

from functions import *
import multiprocessing
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import json

X_train_cancer = pd.read_csv("breast_cancer_train_copy.csv")
X_test_cancer = pd.read_csv("breast_cancer_test_copy.csv")
scaler = MinMaxScaler()

columns_to_scale_c = X_train_cancer.columns.drop("target")
X_train_cancer[columns_to_scale_c] = scaler.fit_transform(X_train_cancer[columns_to_scale_c])
X_test_cancer[columns_to_scale_c] = scaler.fit_transform(X_test_cancer[columns_to_scale_c])

with open("parameters_zipfian_logreg.json", "r") as json_file:
    parameter_dict = json.load(json_file)

job_id = int(os.getenv('SLURM_ARRAY_TASK_ID'))
job_id = str(job_id)
compareZipfian(X_train_cancer, X_test_cancer, *parameter_dict[job_id])
