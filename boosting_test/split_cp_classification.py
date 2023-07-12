import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats.mstats import mquantiles
from sklearn.neural_network import MLPClassifier
import sys
sys.path.append('./') 
import cp_kernel
from cp_kernel import ProbabilityCal




def split_cp_classification(X, Y, X_test, black_box, alpha):
    """
    Compute split-conformal classification prediction sets.
    Uses neural network as a black box 
    
    Input
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    X_test    : n x p test data matrix of explanatory variables
    alpha     : 1 - target coverage level 
    """
    
    # Output placeholder
    S = None
    
    # Split the data into training and calibration sets
    X_train, X_calib, Y_train, Y_calib = train_test_split(X, Y, test_size=0.5, random_state=2024)
    
    # Fit a quantile regression model
    # black_box = MLPClassifier(learning_rate_init = 0.01, max_iter = 1000, hidden_layer_sizes = 64, 
    #                           random_state = 2023)
    black_box.fit(X_train, Y_train)
    
    # Estimate class probabilities for calibration points (store them in a variable called 'pi_hat')
    pi_hat = black_box.predict_proba(X_calib) #这个是你对于每一个y值的概率，相当于现在你预测概率


    # Define prediction rule with plugin probability estimates
    prediction_rule = cp_kernel.ProbabilityCal(pi_hat) ##这个就是根据你的pi我来建立score函数？
    
    # Generate independent uniform random variables for the calibration data (store them in a variable called 'epsilon')
    n_calib = len(Y_calib)
    epsilon = np.random.uniform(low = 0.0, high = 1.0, size = n_calib)

    
    
    # Compute conformity scores
    scores = prediction_rule.calibrate_scores(Y_calib, epsilon = epsilon)   
            #你看所以这里用prediction_rule来找对应的score，也就是概率？所以这个返回的是distribution？
    
    # Compute suitable empirical quantile of absolute residuals
    level_adjusted = (1.0-alpha)  * (1 + 1/n_calib)
    Q_hat = mquantiles(scores, prob=level_adjusted)[0]

    # Construct prediction sets for test data
    pi_hat_test = black_box.predict_proba(X_test)
    S = oracle(pi_hat_test, 1 - Q_hat)
    
    return S



def oracle(pi, alpha, randomize=True):
    '''
    find the prediction set
    Input:
        pi:     the true probability ? 
        alpha:      the coverage ?  
    '''
    prediction_rule = cp_kernel.ProbabilityCal(pi)  # 这个是score函数？
        ## 但是这个第一步是什么？？？？   
    S = prediction_rule.predict_sets(alpha, randomize=randomize) # 这个是把set找到
    return S


def evaluate_predictions(S, X, Y, verbose=True):
    """
    Evaluate performance metrics for a set of classification predictions
    Computes:
    - marginal coverage
    - unbiased estimate of worst-slab coverage
    - average size of sets
    - average size of sets that contain the true Y
    
    Input
    S         : n - long list of prediction sets (each set is a discrete array)
    X         : n x p data matrix of explanatory variables
    Y         : n x 1 vector of response variables
    """
    
    # Estimate worst-slab coverage
    wsc_coverage = cp_kernel.wsc_unbiased(X, Y, S)
    
    # Number of samples
    n = len(Y)
    
    # Output placeholder
    marginal_coverage = None
    size = None
    size_cover = None
    
    # Compute marginal coverage
    marginal_coverage = np.mean([Y[i] in S[i] for i in range(n)])
    
    # Compute average size of prediction sets
    size = np.mean([len(S[i]) for i in range(n)])
    
    # Compute average size of prediction sets that contain the true label
    idx_cover = np.where([Y[i] in S[i] for i in range(n)])[0]
    size_cover = np.mean([len(S[i]) for i in idx_cover])
    
    # Print summary
    if verbose:
        print('Marginal coverage       : {:2.3%}'.format(marginal_coverage))
        print('WS conditional coverage : {:2.3%}'.format(wsc_coverage))
        print('Average size            : {:2.3f}'.format(size))
        print('Average size | cover    : {:2.3f}'.format(size_cover))
        
    return marginal_coverage, wsc_coverage, size, size_cover