import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier


import sys
sys.path.append('./') 
import cp_kernel
from cp_kernel import ProbabilityCal

import split_cp_classification
from split_cp_classification import split_cp_classification, oracle, evaluate_predictions


import mfpi
from mfpi import data_gen_models as data
from mfpi.deep_classification import Net as ClassNet




# Pre-defined model
p = 10                                                 # Number of features
K = 10                                                 # Number of possible labels
data_model = data.Model_Class2(K,p)                    # Data generating model

# Sample data
n = 2000                                               # Number of data samples
X_data = data_model.sample_X(n)                        # Generate the data features
Y_data = data_model.sample_Y(X_data)                   # Generate the data labels conditional on the features

# Sample test data
n_test = 1000                                          # Number of test samples
X_test = data_model.sample_X(n_test)                   # Generate independent test data
Y_test = data_model.sample_Y(X_test)



black_box = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
black_box.fit(X_data, Y_data)
pi_hat = black_box.predict_proba(X_test)
Y_hat = np.argmax(pi_hat,1)


# Compute true class probabilities for every sample
pi = data_model.compute_prob(X_test) # 对于test集合里面的每一个的概率

# Nominal coverage: 1-alpha 
alpha = 0.1

# Oracle prediction sets
S_oracle = oracle(pi, alpha)     ##这个是返回set    


# Desired coverage level (1-alpha)
alpha = 0.1

# Apply split conformal classification
S = split_cp_classification(X_data, Y_data, X_test, black_box, alpha)

# Evaluate prediction sets
metrics = evaluate_predictions(S, X_test, Y_test)



# print(S)
# print(metrics)


## 确认data是general的形式？