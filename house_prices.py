import numpy as np
import pandas as pd

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
    dataset.drop(['MSSubClass', 'OverallCond', 'Id', 'BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath',
                            'BedroomAbvGr', 'KitchenAbvGr', 'EnclosedPorch', '3SsnPorch', 'PoolArea',
                             'MiscVal', 'MoSold', 'YrSold'], axis=1, inplace=True)

    #Drop unhelpful categorical features
    dataset.drop(['Exterior2nd', 'MSZoning', 'LandSlope', 'LotConfig', 'SaleCondition', 'SaleType', 'PavedDrive', 'GarageCond', 'Functional',
                'Electrical', 'Heating', 'BsmtFinType2', 'ExterCond', 'Exterior1st', 'RoofMatl', 'RoofStyle',
                'Condition2', 'Condition1', 'Utilities', 'LandContour', 'LotShape', 'Street', 'MasVnrType' ], axis=1, inplace=True)

    #Manual ordinal encoding
    dataset['GarageQual'] = dataset['GarageQual'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5})
    dataset['GarageQual'] = dataset['GarageQual'].fillna(0).astype(int)

    dataset['GarageFinish'] = dataset['GarageFinish'].map({'Unf':1, 'RFn':2, 'Fin':3})
    dataset['GarageFinish'] = dataset['GarageFinish'].fillna(0).astype(int)

    dataset['KitchenQual'] = dataset['KitchenQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)
    dataset['CentralAir'] = dataset['CentralAir'].map({'N':0, 'Y':1}).astype(int)
    dataset['HeatingQC'] = dataset['HeatingQC'].map({'Po':1, 'Fa':2, 'TA':3, 'Gd':4, 'Ex':5}).astype(int)

    dataset['BsmtExposure'] = dataset['BsmtExposure'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)
    dataset['BsmtQual'] = dataset['BsmtQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)

    dataset['ExterQual'] = dataset['ExterQual'].map({'Fa':1, 'TA':2, 'Gd':3, 'Ex':4}).fillna(0).astype(int)

    #Fill missing values
    dataset['GarageType'].fillna('no_gar', inplace=True)


object_features = ['Neighborhood', 'BldgType', 'HouseStyle', 'Foundation', 'BsmtFinType1', 'GarageType']

for feature in object_features:
    table = train_df[[feature, 'SalePrice']].groupby([feature], as_index=False).mean().sort_values(by='SalePrice', ascending=False)
    print(table)
    print(train_df[feature].value_counts())
    print('')

print(test_df.info())
print(train_df.info())
