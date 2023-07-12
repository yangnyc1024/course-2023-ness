import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
import pandas as pd
from sklearn.model_selection import train_test_split


import sys
sys.path.append('./') 
import split_cp_regression
from split_cp_regression import conformal_prediction_intervals, cqr_prediction_intervals, naive_prediction_intervals
from split_cp_regression import RFQR
from split_cp_regression import evaluate_predictions

from tqdm import tqdm


# import data
dataset_full = pd.read_csv('./data/blogData_train.csv', header=None)
_, dataset = train_test_split(dataset_full, test_size=1000, random_state=2023)

# import the algorithm we choose
black_box_qr = RFQR()
# black_box = GradientBoostingRegressor(n_estimators=100, min_samples_split=1, random_state=2023)
black_box = RandomForestRegressor(n_estimators=100,random_state=2023)


## input alpha
alpha = 0.1

def run_experiment(dataset, black_box, black_box_qr, random_state=2023):
    # Divide data
    X_data, X_test, Y_data, Y_test = train_test_split(dataset.iloc[:,0:280].values, dataset.iloc[:,-1].values, test_size=0.2, random_state=random_state)

    
    # Run and evaluate naive
    lower_naive, upper_naive = naive_prediction_intervals(X_data, Y_data, X_test, black_box, alpha)
    metrics_naive = evaluate_predictions(lower_naive, upper_naive, X_test, Y_test, verbose=False)
    
    # Run and evaluate conformal
    lower_conformal, upper_conformal = conformal_prediction_intervals(X_data, Y_data, X_test, black_box, alpha, random_state=random_state)
    metrics_conformal = evaluate_predictions(lower_conformal, upper_conformal, X_test, Y_test, verbose=False)
    
    # Run and evaluate CQR
    lower_cqr, upper_cqr = cqr_prediction_intervals(X_data, Y_data, X_test, black_box_qr, alpha, random_state=random_state)
    metrics_cqr = evaluate_predictions(lower_cqr, upper_cqr, X_test, Y_test, verbose=False)
       
    # Return results
    results_exp = pd.DataFrame({"Method":["Naive", "Conformal", "CQR"], 
                                "Coverage":[metrics_naive[0], metrics_conformal[0], metrics_cqr[0]],
                                "Length":[metrics_naive[1], metrics_conformal[1], metrics_cqr[1]],
                  })
    
    return results_exp

# Run many experiments
results = pd.DataFrame()


for experiment in tqdm(range(20)):
    
    # Random state for this experiment
    random_state = 2023 + experiment
    
    # Run the experiment
    result_exp = run_experiment(dataset, black_box, black_box_qr, random_state=random_state)
    
    # Store results
    results = pd.concat([results,result_exp])

print(results)