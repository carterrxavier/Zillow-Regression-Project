import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import itertools

def get_distribution(df):
    for i in df.columns:
        plt.figure(figsize=(9,5))
        sns.histplot(data = df, x=i)
    plt.show()
    
def graph_distribution(df, target, x):
        g = sns.FacetGrid(df, col='County', hue='County',height=5)
        plt.figure(figsize=(9,5))
        g.map(sns.kdeplot, x)
        plt.xlim(0,5)
        plt.show()
      
    
    
    
def get_heatmap(df, target):
    '''
    This method will return a heatmap of all variables and there relation to churn
    '''
    plt.figure(figsize=(15,12))
    heatmap = sns.heatmap(df.corr()[[target]].sort_values(by=target, ascending=False), annot=True)
    heatmap.set_title('Feautures  Correlating with {}'.format(target))
    
    return heatmap


    
def plot_variable_pairs(df, cont_vars = 2):
    combos = itertools.combinations(df,cont_vars)
    for i in combos:
        plt.figure(figsize=(8,3))
        sns.regplot(data=df, x=i[0], y =i[1],line_kws={"color":"red"})
        plt.show()
        
def month_to_year(df):
    df["tenure_in_years"] = df.apply(lambda df: math.floor(df.tenure / 12), axis=1)
    #df["tenure_in_months"] = df.apply(lambda df: df.tenure % 12, axis = 1)
    return df


def plot_cat_and_cont(cat_var, con_var, df):
    for i in cat_var:
        for j in con_var:
            plt.figure(figsize=(20,12))
            plt.subplot(131)
            sns.swarmplot(x=i, y=j, data=df)
            plt.subplot(132)
            sns.boxplot(x=i, y=j, data=df)
            plt.subplot(133)
            sns.barplot(x=i, y=j, data=df)
            plt.show()
        
        