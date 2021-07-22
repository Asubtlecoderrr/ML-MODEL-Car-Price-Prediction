import numpy as np
import pandas as pd
import matplotlib as plt
import sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import RandomizedSearchCV,GridSearchCV
df = pd.read_csv("car data.csv")
df.drop('Car_Name',axis=1,inplace=True)
df['current_year']=2021
df['no_year']=df['current_year']-df['Year']
df.drop(['current_year','Year'],axis=1,inplace=True)
df=pd.get_dummies(df,drop_first=True,)
corr=df.corr()
X=df.iloc[:,1:]
Y=df.iloc[:,0]
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=5)
from sklearn.linear_model import LinearRegression

mymodel = LinearRegression()

mymodel.fit(x_train,y_train)

y_pred = mymodel.predict(x_test)

pred_df = pd.DataFrame(y_pred,y_test)

print("MAE ",metrics.mean_absolute_error(y_test,y_pred))
print("MSE ",metrics.mean_squared_error(y_test,y_pred))
print("R2 ",metrics.r2_score(y_test,y_pred))

from sklearn.ensemble import RandomForestRegressor

regressor = RandomForestRegressor()

regressor.fit(x_train,y_train)

y_pred2=regressor.predict(x_test)

pred_df2 = pd.DataFrame(y_pred2,y_test)



print("RMSE ",metrics.mean_squared_log_error(y_test,y_pred2))
print("MAE ",metrics.mean_absolute_error(y_test,y_pred2))
print("MSE ",metrics.mean_squared_error(y_test,y_pred2))
print("R2 ",metrics.r2_score(y_test,y_pred2))

from sklearn.tree import DecisionTreeRegressor

tree=DecisionTreeRegressor()

tree.fit(x_train,y_train)

y_pred3=tree.predict(x_test)

pred_df3 = pd.DataFrame(y_pred3,y_test)

pred_df3

print("RMSE ",metrics.mean_squared_log_error(y_test,y_pred3))
print("MAE ",metrics.mean_absolute_error(y_test,y_pred3))
print("MSE ",metrics.mean_squared_error(y_test,y_pred3))
print("R2 ",metrics.r2_score(y_test,y_pred3))

from sklearn.ensemble import ExtraTreesRegressor

model= ExtraTreesRegressor()

model.fit(X,Y)

print(model.feature_importances_)

feat_imp = pd.Series(model.feature_importances_,index=X.columns)

n_estimators =[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30,num=6)]
min_samples_split=[2,5,10,15,100]
min_samples_leaf=[1,2,5,10]

random_grid ={'n_estimators':n_estimators,
'max_features':max_features,
'max_depth':max_depth,
'min_samples_split':min_samples_split,
'min_samples_leaf':min_samples_leaf}
rf=RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf,param_distributions=random_grid,scoring='neg_mean_squared_error',n_iter=10,cv=5,verbose=2,random_state=5)
rf_random.fit(x_train,y_train)
rf_random.best_params_
rf_random.best_score_

y_pred = rf_random.predict(x_test)
print("RMSE ",metrics.mean_squared_log_error(y_test,y_pred))
print("MAE ",metrics.mean_absolute_error(y_test,y_pred))
print("MSE ",metrics.mean_squared_error(y_test,y_pred))
print("R2 ",metrics.r2_score(y_test,y_pred))

import pickle
file=open('car_price_model.pkl','wb')

pickle.dump(rf_random,file)

