import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.feature_selection import SelectKBest, f_regression, RFE
from sklearn.preprocessing import PolynomialFeatures
import scipy.stats as stats


def get_risiduals(df ,act, pred):
    df['risiduals'] = act - pred
    df['baseline_risiduals'] = act - act.mean()
    return df

def plot_residuals(act, pred, res, baseline):
    plt.figure(figsize=(20,12))
    plt.subplot(221)
    plt.title('Residuals')
    res.hist(bins=20,color='#dbd86e')
    plt.grid(b=None)
    plt.subplot(222)
    plt.title('Baseline Residuals')
    baseline.hist(bins=20)
    plt.grid(b=None)
    
   
    
    ax = plt.subplot(223)
    ax.scatter(act, pred, color='#5e338a')
    ax.set(xlabel='actual', ylabel='prediction')
    ax.plot(act, act,  ls=":", color='black')
    
    ax = plt.subplot(224)
    ax.scatter(act, res, color ='#5e338a')
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
    
    
    
def get_t_test(t_var, df, target, alpha):
    '''
        This method will produce a 2 tailed t test  equate the p value to the alpha to determine whether the null hypothesis can be rejected.
    '''
    for i in t_var:
        t, p = stats.ttest_ind(df[i],df[target], equal_var=False)
        print('Null Hypothesis: {} is not correlated to value '.format(i))
        print('Alternative hypothesis:  {} is correlated to value '.format(i))
        if p < alpha:
            print('p value {} is less than alpha {} , we reject our null hypothesis'.format(p,alpha))
        else:
            print('p value {} is not less than alpha {} , we  fail to reject our null hypothesis'.format(p,alpha))
        print('-------------------------------------')
        
def get_pearsons(con_var, target, alpha, df):
     for i in con_var:
        t, p = stats.pearsonr(df[i],df[target])
        print('Null Hypothesis: {} is not correlated to value '.format(i))
        print('Alternative hypothesis:  {} is correlated to value '.format(i))
        if p < alpha:
            print('p value {} is less than alpha {} , we reject our null hypothesis'.format(p,alpha))
        else:
            print('p value {} is not less than alpha {} , we  fail to reject our null hypothesis'.format(p,alpha))
        print('-------------------------------------')
        

def get_model_results(X_train, y_train, X, y, target, model='linear', alpha = 0, power = 0, graph = False, degree=2):
    results = y.copy()
    
    if model == "linear":
        lm = LinearRegression(normalize=True)
        lm.fit(X_train, y_train)  
        results['pred'] = lm.predict(X)
        
    elif model == 'lasso':
        lasso = LassoLars(alpha=alpha)
        lasso.fit(X_train, y_train)
        results['pred'] = lasso.predict(X)
        
    elif model == 'glm':
        glm = TweedieRegressor(power=power, alpha=alpha)
        glm.fit(X_train, y_train)
        results['pred'] = glm.predict(X)
    elif model == 'poly':
        pf = PolynomialFeatures(degree=degree)
        
        X_train_degree2 = pf.fit_transform(X_train)
        X_degree_2 = pf.transform(X)
        
        
        lm = LinearRegression(normalize=True)
        lm.fit(X_train_degree2, y_train)  
        results['pred'] = lm.predict(X_degree_2)
        
        
        
    
    results = get_risiduals(results, y[target],results.pred)
    rmse =    regression_errors(y[target],results.pred)[2]
    r_2  =    regression_errors(y[target],results.pred)[5]
    btb = better_than_baseline(y[target],results.pred)
    
    print('RMSE Score: {}'.format(rmse)) 
    print('r2 Score: {}'.format(r_2))
    print('Better than Basline: {}'.format(btb))
    
    if graph == True:
        plot_residuals(results[target], results.pred, results.risiduals, results.baseline_risiduals)
        
    return rmse

    
    
    

