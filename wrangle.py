import numpy as np
import pandas as pd
from env import host, user ,password
from sklearn.model_selection import train_test_split
import os


def get_connection(db, user = user, host = host, password = password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'
    
    
def get_telco_tenure():
    '''
    This function gets the tenure information from the telco data set for customers with 2 year contracts
    '''
    file_name = 'telco_tenure.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
        select customer_id, monthly_charges, tenure, total_charges
        from customers
        where contract_type_id = 3
        '''
    df = pd.read_sql(query, get_connection('telco_churn'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df

def clean_telco_tenure(df):
    '''
    cleans telco tenure data
    
    '''
    #fill total charges with monthly charges
    df['total_charges'].fillna(df['monthly_charges'], inplace = True)
    
    #convert total_charges object type into a float
    df['total_charges'] = pd.to_numeric(df['total_charges'],errors='coerce')
    
    #change tenure from zero to one
    df.loc[df['tenure'] == 0, 'tenure'] = 1
    
    return df
    

def get_zillow_data():
    '''
    This function gets the zillow data needed to predict single unit properities.
    '''
    file_name = 'zillow.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
      select properties_2017.parcelid, bedroomcnt,bathroomcnt,calculatedfinishedsquarefeet,taxvaluedollarcnt,taxamount,yearbuilt,regionidzip, fips, transactiondate
        from properties_2017
        join predictions_2017 on properties_2017.parcelid = predictions_2017.parcelid
        where transactiondate between '2017-05-01' and '2017-08-31' and (propertylandusetypeid ='261' or propertylandusetypeid = '279' or propertylandusetypeid = '262' or propertylandusetypeid = '264')'''
    df = pd.read_sql(query, get_connection('zillow'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df

def summerize_df(df):
    print('-----shape------')
    print('{} rows and {} columns'.format(df.shape[0], df.shape[1]))
    print('---info---')
    print(df.info())
    print(df.describe())
    print('--nulls--')
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    print(df.isna().sum())


def clean_zillow_data(zillow_df):
    '''
    this function cleans data for zillow data 
    '''
    zillow_df = zillow_df.dropna(axis=0, subset=['bedroomcnt'])
    zillow_df = zillow_df.dropna(axis=0, subset=['bathroomcnt'])
    zillow_df = zillow_df.dropna(axis=0, subset=['regionidzip'])
    zillow_df['regionidzip'] = zillow_df['regionidzip'].astype(str)
    zillow_df = zillow_df.dropna(how='all')
    zillow_df = zillow_df.dropna(axis=0, subset=['calculatedfinishedsquarefeet'])
    zillow_df['taxvaluedollarcnt'].fillna(zillow_df['taxvaluedollarcnt'].mean(), inplace = True)
    mode = zillow_df[(zillow_df['yearbuilt'] > 1947) & (zillow_df['yearbuilt'] <= 1957)].yearbuilt.mode()
    zillow_df['yearbuilt'].fillna(value=mode[0], inplace = True)
    
    zillow_df['transactiondate'] = pd.to_datetime(zillow_df['transactiondate'],dayfirst=True)
    zillow_df['transactiondate'] = zillow_df['transactiondate'].dt.month
    zillow_df = zillow_df.rename(columns={'transactiondate':'transactionmonth'})
    




    
    return zillow_df

def zillow_engineering(zillow_df):
    '''
    this function creates the new columns specified by client, includes, state and county columns, as well as tax rate
    '''
    zillow_df['fips'] = zillow_df['fips'].astype(str)
    zillow_df.loc[zillow_df['fips'].str[0] == '6','State'] = 'California'
    zillow_df.loc[zillow_df['fips'].str.contains('111'),'County'] = 'Ventura'
    zillow_df.loc[zillow_df['fips'].str.contains('037'),'County'] = 'Los Angeles'
    zillow_df.loc[zillow_df['fips'].str.contains('059'),'County'] = 'Orange'
    zillow_df['fips'] = zillow_df['fips'].astype(float)
    
    zillow_df['taxrate'] = round(zillow_df['taxamount']/zillow_df['taxvaluedollarcnt'] * 100 ,2)
    
    return zillow_df

def handle_outliers(df , col, lquan, upquan):
    q1 = df[col].quantile(lquan)
    q3 = df[col].quantile(upquan)
    iqr = q3-q1 #Interquartile range
    lower_bound  = q1-1.5*iqr
    upper_bound = q3+1.5*iqr
    if lower_bound < 0:
        lower_bound = 0
    if upper_bound > df[col].max():
        upper_bound = df[col].max()
    df_out = df.loc[(df[col] > lower_bound) & (df[col] < upper_bound)]
    return df_out

def get_dummies(df, col_names):
    for i in col_names:
        dummy = pd.get_dummies(df[i], drop_first=True)
        df = pd.concat([df,dummy] , axis=1)
    df = df.drop(columns = col_names)
    return df
    
    
def split_for_model(df):
    '''
    This function take in the telco_churn data acquired,
    performs a split into 3 dataframes. one for train, one for validating and one for testing 
    Returns train, validate, and test dfs.
    '''
    train_validate, test = train_test_split(df, test_size=.2, 
                                        random_state=333)
    train, validate = train_test_split(train_validate, test_size=.3, 
                                   random_state=333)
    
    print('train{},validate{},test{}'.format(train.shape, validate.shape, test.shape))
    return train, validate, test


def get_mall_customers():
    '''
    This function gets the tenure information from the telco data set for customers with 2 year contracts
    '''
    file_name = 'mall_customers.csv'
    if os.path.isfile(file_name):
        return pd.read_csv(file_name)
    
    else:
        query =  '''
        select *
        from customers
        '''
    df = pd.read_sql(query, get_connection('mall_customers'))  
    
    #replace white space with nulls
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    df.to_csv(file_name, index = False)
    return df