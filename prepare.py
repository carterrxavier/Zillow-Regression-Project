import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler


def scale_telco_data(train, validate, test, scale_type = None):
    '''
    returns scaled data of specified type into data frame
    '''
    X_train = train
    X_validate = validate
    X_test = test
    
    min_max_scaler = MinMaxScaler()
    robust_scaler = RobustScaler()
    standard_scaler = StandardScaler()
    
    min_max_scaler.fit(X_train)
    robust_scaler.fit(X_train)
    standard_scaler.fit(X_train)
    
    mmX_train_scaled = min_max_scaler.transform(X_train)
    rX_train_scaled = robust_scaler.transform(X_train)
    sX_train_scaled = standard_scaler.transform(X_train)
    
    mmX_validate_scaled = min_max_scaler.transform(X_validate)
    rX_validate_scaled = robust_scaler.transform(X_validate)
    sX_validate_scaled = standard_scaler.transform(X_validate)
    
    mmX_test_scaled = min_max_scaler.transform(X_test)
    rX_test_scaled = robust_scaler.transform(X_test)
    sX_test_scaled = standard_scaler.transform(X_test)
    
    
    mmX_train_scaled = pd.DataFrame(mmX_train_scaled, columns=X_train.columns)
    mmX_validate_scaled = pd.DataFrame(mmX_validate_scaled, columns=X_validate.columns)
    mmX_test_scaled = pd.DataFrame(mmX_test_scaled, columns=X_test.columns)

    rX_train_scaled = pd.DataFrame(rX_train_scaled, columns=X_train.columns)
    rX_validate_scaled = pd.DataFrame(rX_validate_scaled, columns=X_validate.columns)
    rX_test_scaled = pd.DataFrame(rX_test_scaled, columns=X_test.columns)


    sX_train_scaled = pd.DataFrame(sX_train_scaled, columns=X_train.columns)
    sX_validate_scaled = pd.DataFrame(sX_validate_scaled, columns=X_validate.columns)
    sX_test_scaled = pd.DataFrame(sX_test_scaled, columns=X_test.columns)
    
    
    if scale_type == 'MinMax':
        return mmX_train_scaled, mmX_validate_scaled, mmX_test_scaled
    elif scale_type == 'Robust':
        return rX_train_scaled, rX_validate_scaled, rX_test_scaled
    elif scale_type == 'Standard':
        return sX_train_scaled, sX_validate_scaled, sX_test_scaled
    else:
        return train, validate, test
        