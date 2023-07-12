from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from scipy.stats import norm
import statsmodels.api as sm
from scipy.stats.mstats import mquantiles
from quantile_forest import RandomForestQuantileRegressor


def naive_prediction_intervals(X, Y, X_test, black_box, alpha):
    """
    Compute naive prediction bands based on the distribution of
      residuals within the training data set
      
    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level 
    """
    
    # Output placeholder
    lower = None
    upper = None
    
    # Fit the black box model on the training data
    black_box.fit(X, Y)
    
    # Compute residuals on the training data
    residuals_calib = np.abs(Y - black_box.predict(X))
    
    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y)
    level_adjusted = 1.0-alpha
    Q_hat = mquantiles(residuals_calib, prob=level_adjusted)[0]
    
    # Construct prediction bands
    Y_hat = black_box.predict(X_test)
    lower = Y_hat - Q_hat
    upper = Y_hat + Q_hat
    
    return lower, upper 

def conformal_prediction_intervals(X, Y, X_test, black_box, alpha, random_state=2023):
    """
    Compute conformal prediction bands
    
    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : sklearn model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level 
    """
    
    # Output placeholder
    lower = None
    upper = None
    
    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=2023)
    
    # Fit the black box model on the training data
    """TODO: write your code here (1 line)"""
    black_box.fit(X_train, Y_train)

    
    # Compute residuals on the calibration data
    """TODO: write your code here (1 line)"""
    residuals_calib = np.abs(Y_calib - black_box.predict(X_calib))

    
    # Compute suitable empirical quantile of absolute residuals
    """TODO: write your code here (3 lines)"""
    n_calib = len(Y_calib)
    level_adjusted = 1.0-alpha
    Q_hat = mquantiles(residuals_calib, prob=level_adjusted)[0]
    
    # Construct prediction bands
    """TODO: write your code here (3 lines)"""
    Y_hat = black_box.predict(X_test)
    lower = Y_hat - Q_hat
    upper = Y_hat + Q_hat
    
    return lower, upper   


def cqr_prediction_intervals(X, Y, X_test, black_box, alpha, random_state=2023):
    """
    Compute split-conformal quantile regression prediction bands.
    Uses quantile random forests as a black box 
    
    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    black_box : quantile regression model object with 'fit' and 'predict' methods
    alpha     : 1 - target coverage level 
    """
    
    # Output placeholder
    lower = None
    upper = None
    
    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=2023)
    
    # Fit the quantile regression model
    black_box.fit(X_train, Y_train)

    # Estimate conditional quantiles for calibration set
    lower_qr, upper_qr = black_box.predict(X_calib)

    
    # Compute conformity scores on the calibration data
    residuals_calib = np.maximum(Y_calib - upper_qr, lower_qr - Y_calib)

    
    # Compute suitable empirical quantile of absolute residuals
    n_calib = len(Y_calib)
    level_adjusted = (1.0-alpha)  #* (1+ 1/ len(Y))
    Q_hat = mquantiles(residuals_calib, prob=level_adjusted)[0]
    
    # Construct prediction bands
    lower_test, upper_test = black_box.predict(X_test)
    lower = lower_test - Q_hat
    upper = upper_test + Q_hat
    
    return lower, upper 


def evaluate_predictions(lower, upper, X, Y, verbose=True):
    """
    Evaluate performance metrics for a set of regression predictions
    Computes:
    - marginal coverage
    - average size of sets
    
    Input
    lower     : n x 1 vector of prediction lower bounds
    upper     : n x 1 vector of prediction upper upper
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    """
    
    # Number of samples
    n = len(Y)
    
    # Evaluate the empirical coverage
    covered = (Y>=lower) * (Y <= upper)

    # Compute marginal coverage
    marginal_coverage = np.mean(covered)
    
    # Compute average size of prediction sets
    size = np.mean(upper-lower)
    
    # Compute average size of prediction sets that contain the true label
    idx_cover = np.where(covered)[0]
    size_cover = np.mean(upper[idx_cover]-lower[idx_cover])
    
    # Print summary
    if verbose:
        print('Marginal coverage       : {:2.3%}'.format(marginal_coverage))
        print('Average length          : {:2.3f}'.format(size))
        
    return marginal_coverage, size



class RFQR:
    def __init__(self, alpha=0.1, n_estimators=100, min_samples_split=10):
        self.alpha = alpha
        self.qr = RandomForestQuantileRegressor(min_samples_split=min_samples_split, min_samples_leaf=5,
                                                n_estimators=n_estimators)

    def fit(self, X, Y):
        self.qr.fit(X, Y)
        return self

    def predict(self, X):
        pred = self.qr.predict(X, quantiles=[self.alpha/2, 1-self.alpha/2])
        lower = pred[:,0]
        upper = pred[:,1]
        # Correct any possible quantile crossings
        pred = np.concatenate([lower.reshape(len(lower),1), upper.reshape(len(upper),1)],1)
        pred = np.sort(pred,1)
        return lower, upper