import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import itertools

def get_distribution(df):
    df.hist(grid = False, bins = 10)
    plt.show()

def get_heatmap(df):
    sns.heatmap(df.corr(),annot=True, mask= np.triu(df.corr()))
    plt.show()
    
def plot_variable_pairs(df, cont_vars = 2):
    combos = itertools.combinations(df,cont_vars)
    for i in combos:
        plt.figure(figsize=(10,5))
        sns.regplot(data=df, x=i[0], y =i[1],line_kws={"color":"red"})
        plt.show()
        
def month_to_year(df):
    df["tenure_in_years"] = df.apply(lambda df: math.floor(df.tenure / 12), axis=1)
    #df["tenure_in_months"] = df.apply(lambda df: df.tenure % 12, axis = 1)
    return df


def plot_cat_and_cont(cat_var, con_var, df):
    for i in cat_var:
        for j in con_var:
            plt.figure(figsize=(10,5))
            plt.subplot(131)
            sns.swarmplot(x=i, y=j, data=df)
            plt.subplot(132)
            sns.boxplot(x=i, y=j, data=df)
            plt.subplot(133)
            sns.barplot(x=i, y=j, data=df)
            plt.show()
        
        