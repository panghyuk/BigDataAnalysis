import pandas as pd

pd.set_option("display.max_columns",None)
display(X_train.head())

print(X_train.isnull().sum().sort_values(ascending=False)[:20])
print(X_train.info())

X_train = X_train.select_dtypes(exclude = 'object')
X_test = X_test.select_dtypes(exclude = 'object')
target = y_train['SalePrice']

X_train.dropna(axis = 1,inplace = True)
X_test.dropna(axis = 1,inplace = True)

from sklearn.preprocessing import MinMaxScaler
model = MinMaxScaler()
X_train = model.fit_transform(X_train)
X_test = model.transform(X_test)

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train,target,test_size = 0.2, random_state = 2022)

from sklearn.metrics import mean_squared_error
def rmse(y, y_pred):
    return np.sqrt(mean_squared_error(y,y_pred))

from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

model1 = RandomForestRegressor()
model1.fit(X_tr,y_tr)
pred1 = model1.predict(X_val)
print(rmse(y_val,pred1))

model2 = SVR()
model2.fit(X_tr,y_tr)
pred2 = model2.predict(X_val)
print(rmse(y_val,pred2))

model3 = XGBRegressor()
model3.fit(X_tr,y_tr)
pred3 = model3.predict(X_val)
print(rmse(y_val,pred3))

y = y_train["SalePrice"]

# 최종모델
model1 = RandomForestRegressor()
model1.fit(X_train,y)
pred1 = model1.predict(X_test)

model3 = XGBRegressor()
model3.fit(X_train,y)
pred3 = model3.predict(X_test)

output1 = pd.DataFrame({'ID':y_test['Id'],'income':pred1})
output2 = pd.DataFrame({'ID':y_test['Id'],'income':pred3})

print(rmse(y_test['SalePrice'],pred1))
print(rmse(y_test['SalePrice'],pred3))