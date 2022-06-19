import pandas as pd
import numpy as np

cols = X_train.select_dtypes("object").columns
for col in cols:
    print("\n=====", col, "=====")
    print("[train]")
    print(X_train[col].value_counts())
    print("[test]")
    print(X_test[col].value_counts())

X_train = pd.get_dummies(X_train,columns = cols)
X_test = pd.get_dummies(X_test,columns = cols)

y_train['charges'] = np.log1p(y_train['charges'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train['bmi'] = scaler.fit_transform(X_train[['bmi']])
X_test['bmi'] = scaler.transform(X_test[['bmi']])

X_train['age'] = X_train['age'].apply(lambda x: x//10)
X_test['age'] = X_test['age'].apply(lambda x: x//10)

from sklearn.model_selection import train_test_split
target = y_train['charges']
X_train = X_train.drop('id',axis = 1)
X_tr,X_val,y_tr,y_val = train_test_split(X_train,target,test_size = 0.15,random_state = 2022)

from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

model1 = RandomForestRegressor()
model1.fit(X_tr,y_tr)
pred1 = model1.predict(X_val)

model2 = XGBRegressor()
model2.fit(X_tr,y_tr)
pred2 = model2.predict(X_val)

from sklearn.metrics import mean_squared_error
def rmse(y,y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

print(rmse(y_val,pred1))
print(rmse(y_val,pred2))

model1.fit(X_train,y_train['charges'])
pred = model1.predict(X_test.drop('id',axis = 1))

pred = np.exp(pred)
output = pd.DataFrame({'id':y_test['id'],'charges':pred})
output.head()