#importing the libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing the data
train_set=pd.read_csv('train.csv')
test_set=pd.read_csv('test.csv')
Y_test=pd.read_csv('sample_submission.csv')
Y_test=Y_test['SalePrice']
Y_train=train_set['SalePrice']
del train_set['SalePrice']

train_set.info()
test_set.info()

#missing values
mis_val=['BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtFinType2','FireplaceQu','GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'] 
for i in mis_val:
  train_set[i]=train_set[i].replace(np.nan, 'NA')
  test_set[i]=test_set[i].replace(np.nan, 'NA')
  
replace_map=[]  
replace_map.append({'IR1':2, 'IR2':1, 'IR3':0, 'Reg':3})
replace_map.append({'NoSewr':2, 'NoSeWa':1, 'EL0':0, 'AllPub':3})
replace_map.append({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0})
replace_map.append({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0})
replace_map.append({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
replace_map.append({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
replace_map.append({'Gd':4, 'Av':3, 'Mn':2, 'No':1, 'NA':0})
replace_map.append({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0})
replace_map.append({'GLQ':6, 'ALQ':5, 'BLQ':4, 'Rec':3, 'LwQ':2, 'Unf':1, 'NA':0})
replace_map.append({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0})
replace_map.append({'Ex':4, 'Gd':3, 'TA':2, 'Fa':1, 'Po':0})
replace_map.append({'Typ':7, 'Min1':6, 'Min2':5, 'Mod':4, 'Maj1':3, 'Maj2':2, 'Sev':1, 'Sal':0})
replace_map.append({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
replace_map.append({'Fin':3, 'RFn':2, 'Unf':1, 'NA':0})
replace_map.append({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})
replace_map.append({'Ex':5, 'Gd':4, 'TA':3, 'Fa':2, 'Po':1, 'NA':0})

cat_data_ORD=['LotShape','Utilities','ExterQual','ExterCond','BsmtQual','BsmtCond',
              'BsmtExposure','BsmtFinType1','BsmtFinType2','HeatingQC','KitchenQual',
              'Functional','FireplaceQu','GarageFinish','GarageQual','GarageCond']

j=0
for i in cat_data_ORD:
  train_set[i].replace(replace_map[j], inplace=True)
  test_set[i].replace(replace_map[j], inplace=True)
  j+=1
   
  
del_col=['Alley','PoolQC', 'Fence', 'MiscFeature', 'GarageYrBlt']
for i in del_col:
  del train_set[i] 
  del test_set[i]

train_set['LotFrontage']=train_set['LotFrontage'].replace(np.nan, train_set['LotFrontage'].mean())
test_set['LotFrontage']=test_set['LotFrontage'].replace(np.nan, test_set['LotFrontage'].mean())

#searching for the rows with NaN values and dropping them
df_null_train = train_set.isnull().unstack()
t = df_null_train[df_null_train]
null_rows=[]
for i in range(len(t)):
  null_rows.append(t.index[i][1])
train_set=train_set.drop(null_rows)
Y_train=Y_train.drop(null_rows)
Y=Y_train.append(Y_test)
Y=pd.DataFrame(Y)

#replace by mode
mode_test=['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'KitchenQual',
           'Functional', 'SaleType']
for i in mode_test:
  test_set[i]=test_set[i].replace(np.nan, test_set[i].mode()[0])
 
#replace by mean  
mean_test=['MasVnrArea', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
           'BsmtHalfBath', 'GarageCars', 'GarageArea', 'BsmtFinSF2'] 
for i in mean_test:
  test_set[i]=test_set[i].replace(np.nan, test_set[i].mean())

train_set.info()
test_set.info()

dataset_X=train_set.append(test_set, sort=False)

#Encoding categorical data
cat_data_NOM=['MSSubClass','MSZoning','LandContour','LotConfig','LandSlope',
              'Neighborhood','Condition1','Condition2','BldgType','HouseStyle',
              'RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType',
              'Foundation','Heating','Electrical','GarageType',
              'PavedDrive','SaleType','SaleCondition']

from sklearn.preprocessing import LabelEncoder
label_encoder_X=LabelEncoder()

for i in ['Street', 'CentralAir']:
  dataset_X[i]=label_encoder_X.fit_transform(dataset_X[i])

#dummy variables
for i in cat_data_NOM:
    dataset_X=pd.concat((dataset_X, pd.get_dummies(dataset_X[i], drop_first=True)), axis=1)
    dataset_X=dataset_X.drop([i], axis=1)

#dropping the duplicates
final_dataset_X=dataset_X.loc[:, ~dataset_X.columns.duplicated()]
dataset=pd.concat((final_dataset_X, Y), axis=1)

#correlation between X and Y, dropping the nonsignificant variables
correlations_data_Y = dataset.corr()['SalePrice'].sort_values()
correlations_data_Y=correlations_data_Y[(correlations_data_Y<0.05) & (correlations_data_Y>-0.05)]
for i in correlations_data_Y.index:
  del dataset[i]
del dataset['SalePrice']

#multicollinearity
final_dataset_X=dataset
correlations_data_X=final_dataset_X.corr()
CorField = []
for i in correlations_data_X:
    for j in correlations_data_X.index[correlations_data_X[i] > 0.9]:
        if i != j and j not in CorField and i not in CorField:
            CorField.append(j)
            print ("%s-->%s: r^2=%f" % (i,j, correlations_data_X[i][correlations_data_X.index==j].values[0]))
for i in CorField:
  del final_dataset_X[i]

#splitting into the train and test set
X_train=final_dataset_X[0:len(train_set)]
X_test=final_dataset_X[len(train_set):]

#objects to integers
for i in X_train:
  if X_train[i].dtype=='O':
    X_train[i]=X_train[i].astype(int)
    X_test[i]=X_test[i].astype(int)

#Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler() 

X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

sc_Y=StandardScaler()
Y_train=sc_Y.fit_transform(pd.DataFrame(Y_train))
Y_test=sc_Y.transform(pd.DataFrame(Y_test))

from xgboost import XGBRegressor
regressor=XGBRegressor()

#tune hyperparameters
from sklearn.model_selection import GridSearchCV
parameters={'n_estimators' : [100, 500, 900, 1100, 1500],
            'learning_rate' : [0.05,0.1,0.2,0.3],
            'booster' : ['gbtree','gblinear']}

grid_search = GridSearchCV(
    estimator=regressor,
    param_grid=parameters,
    scoring = 'neg_mean_squared_error',
    n_jobs = -1,
    cv = 5,
    verbose=True)

grid_search.fit(X_train, Y_train)
grid_search.best_estimator_
grid_search.best_score_

#fitting the regressor to the train set
regressor=XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.1, max_delta_step=0,
             max_depth=3, min_child_weight=1, missing=None, n_estimators=1100,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)
regressor.fit(X_train, Y_train)

#cross validation
from sklearn.model_selection import cross_val_score
accuracies=cross_val_score(estimator=regressor, X=X_train, y=Y_train, cv=10)
accuracies.mean() 
#0.892

#predicting the results
y_pred=pd.DataFrame(regressor.predict(X_test))
y_pred=pd.DataFrame(sc_Y.inverse_transform(y_pred))
#RMSE=0.13061

#saving the results
sub_df=pd.read_csv('sample_submission.csv')
datasets=pd.concat([sub_df['Id'],y_pred],axis=1)
datasets.columns=['Id','SalePrice']
datasets.to_csv('sixth_submission.csv',index=False)
