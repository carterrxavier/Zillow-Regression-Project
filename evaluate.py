import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression, RFE



def get_risiduals(df ,act, pred):
    df['risiduals'] = act - pred
    df['baseline_risiduals'] = act - act.mean()
    return df

def plot_residuals(act, pred, res, baseline):
    plt.figure(figsize=(16,9))
    plt.subplot(221)
    plt.title('Residuals')
    res.hist()
    plt.subplot(222)
    plt.title('Baseline Residuals')
    baseline.hist()
    
    
    ax = plt.subplot(223)
    ax.scatter(act, pred)
    ax.set(xlabel='actual', ylabel='prediction')
    ax.plot(act, act,  ls=":", color='black')
    
    ax = plt.subplot(224)
    ax.scatter(act, res)
    ax.set(xlabel='actual', ylabel='residual')
    ax.hlines(0, *ax.get_xlim(), ls=":",color='black')
    
    plt.show()
    
def regression_errors(y, yhat):
    sse = ((y-yhat) ** 2).sum()
    mse = sse / y.shape[0]
    rmse = math.sqrt(mse)
    ess = ((yhat - y.mean())**2).sum()
    tss = ((y - y.mean())**2).sum()
    r_2 = (ess/tss)
    
    return sse, mse, rmse, ess, tss, r_2

def baseline_errors(y, measure="Mean"):
    if measure == "Mean":
        sse_baseline = ((y-y.mean()) ** 2).sum()
        mse_baseline = sse_baseline / y.shape[0]
        rmse_baseline = math.sqrt(mse_baseline)
    else:
        sse_baseline = ((y-y.median()) ** 2).sum()
        mse_baseline = sse_baseline / y.shape[0]
        rmse_baseline = math.sqrt(mse_baseline)
        
    
    return sse_baseline, mse_baseline, rmse_baseline

def better_than_baseline(y, yhat, measure="Mean"):
    if measure == "Mean":
        return regression_errors(y,yhat)[2] < baseline_errors(y, measure = "Mean")[2]
    else:
        return regression_errors(y,yhat)[2] < baseline_errors(y, measure = "Median")[2]
    

def select_kbest(X,y, top=3):
    f_selector = SelectKBest(f_regression, k =top)
    f_selector.fit(X,y)
    result = f_selector.get_support()
    f_feature = X.loc[:,result].columns.tolist()
    return f_feature

def select_rfe(X,y, n_features_to_select = 3,model_type = LinearRegression()):
    lm = model_type
    rfe = RFE(lm, n_features_to_select)
    X_rfe = rfe.fit_transform(X,y)
    mask = rfe.support_
    rfe_feautures = X.loc[:,mask].columns.tolist()
    return rfe_feautures
    
    
    

    



