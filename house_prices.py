import numpy as np
import pandas as pd


from sklearn.ensemble import RandomForestRegressor
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from xgboost import XGBRegressor
import catboost
import lightgbm

def random_forest_rmse(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rms = mean_squared_error(y_test, y_pred, squared=False)
    return rms

def xgboost_rmse(model, X_train, X_val, y_train, y_val, esr):
    model.fit(X_train, y_train, early_stopping_rounds=esr,
             eval_set=[(X_valid, y_valid)],
             verbose=False)
    model.predict(X_test)

#Importing train and test datasets
training_data_path = '/Users/raviachan/Documents/Programming/Kaggle/House_Prices/train.csv'
train_df_full = pd.read_csv(training_data_path)
test_data_path ='/Users/raviachan/Documents/Programming/Kaggle/House_Prices/test.csv'
test_df_full = pd.read_csv(test_data_path)

train_df = train_df_full.copy()
test_df = test_df_full.copy()
combine = [train_df, test_df]

for dataset in combine:
    #Drop columns with scarce data
    dataset.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature'], axis=1, inplace=True)

    #Drop preliminary of very weak correlations
    dataset.drop(['MSSubClass', 'OverallCond', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath',
                            'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'PoolArea',
                             'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)

    #Drop unhelpful categorical features
    dataset.drop(['Exterior2nd', 'MSZoning', 'LandSlope', 'LotConfig', 'SaleCondition', 'SaleType', 'PavedDrive', 'GarageCond', 'Functional',
                'Electrical', 'Heating', 'BsmtFinType2', 'ExterCond', 'Exterior1st', 'RoofMatl', 'RoofStyle',
                'Condition2', 'Condition1', 'Utilities', 'LandContour', 'LotShape', 'Street', 'MasVnrType' ], axis=1, inplace=True)

    #Manual ordinal encoding
    dataset['GarageQual'] = dataset['GarageQual'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).fillna(0).astype(int)
    dataset['GarageFinish'] = dataset['GarageFinish'].map({'Unf':1, 'RFn':2, 'Fin':3}).fillna(0).astype(int)
    dataset['KitchenQual'] = dataset['KitchenQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)
    dataset['CentralAir'] = dataset['CentralAir'].map({'N':0, 'Y':1}).astype(int)
    dataset['HeatingQC'] = dataset['HeatingQC'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype(int)
    dataset['BsmtExposure'] = dataset['BsmtExposure'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)
    dataset['BsmtQual'] = dataset['BsmtQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)
    dataset['ExterQual'] = dataset['ExterQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)

    #Fill missing values
    dataset.fillna({'BsmtFinType1': 'no_bas', 'BsmtCond': 'no_bas', 'GarageType': 'no_gar'}, inplace=True)

#Train test split before further wrangling
X = train_df.drop(['Id', 'SalePrice'], axis=1)
X_test = test_df.drop(['Id'], axis=1)
y = train_df['SalePrice']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.25, random_state=1)

#One hot encoding
features_one_hot_encoding = [col for col in X_train.columns if X_train[col].dtype == 'object']

OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

OH_X_train = pd.DataFrame(OH_encoder.fit_transform(X_train[features_one_hot_encoding]))
OH_X_val = pd.DataFrame(OH_encoder.transform(X_val[features_one_hot_encoding]))
OH_X_test = pd.DataFrame(OH_encoder.transform(X_test[features_one_hot_encoding]))

OH_X_train.index = X_train.index
OH_X_val.index = X_val.index
OH_X_test.index = X_test.index

num_X_train = X_train.drop(features_one_hot_encoding, axis=1)
num_X_val = X_val.drop(features_one_hot_encoding, axis=1)
num_X_test = X_test.drop(features_one_hot_encoding, axis=1)

X_train = pd.concat([num_X_train, OH_X_train], axis=1)
X_val = pd.concat([num_X_val, OH_X_val], axis=1)
X_test = pd.concat([num_X_test, OH_X_test], axis=1)

#Impute values to columns using RandomForestRegressor
imp = IterativeImputer(estimator=RandomForestRegressor(), random_state=1, max_iter=50)
X_train_imp = pd.DataFrame(imp.fit_transform(X_train))
X_val_imp = pd.DataFrame(imp.transform(X_val))
X_test_imp = pd.DataFrame(imp.transform(X_test))

X_train_imp.columns = X_train.columns
X_val_imp.columns = X_val.columns
X_test_imp.columns = X_test.columns

X_train = X_train_imp
X_val = X_val_imp
X_test = X_test_imp

#Modelling
model_1 = RandomForestRegressor(n_estimators=100, random_state=1)
model_2 = RandomForestRegressor(n_estimators=200, random_state=1)
model_3 = XGBRegressor(n_estimators=1000, learning_rate=0.05)

rf_parameters = {'n_estimators': [10, 20, 50, 100, 200, 500, 1000],
                'max_depth': [2, 5, 10, 15, 20]}



print(random_forest_rmse(model_1, X_train, X_val, y_train, y_val))
print(random_forest_rmse(model_2, X_train, X_val, y_train, y_val))
print(xgboost_rmse(model_3, X_train, X_val, y_train, y_val, esr=5))
