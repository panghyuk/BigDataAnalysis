import pandas as pd
#데이터 로드
X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/X_test.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/studentscore/y_test.csv")

# print(X_train.isnull().sum()) # 결츨치 X
# print(X_test.isnull().sum()) # 결측치 X

X_train = X_train.drop('StudentID',axis = 1)
s_id = X_test['StudentID']
X_test = X_test.drop('StudentID',axis = 1)
y = y_train['G3']
y_test = y_test['G3']

X_train = X_train.select_dtypes(exclude = 'object')
X_test = X_test.select_dtypes(exclude = 'object')

from sklearn.preprocessing import MinMaxScaler,RobustScaler
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge,Lasso
from xgboost import XGBRegressor

X_tr,X_val,y_tr,y_val = train_test_split(X_train,y,test_size = 0.15, random_state = 2022)

model1 = RandomForestRegressor()
model1.fit(X_tr,y_tr)
pred = model1.predict(X_val)
print("RF")
print(mean_squared_error(y_val,pred))
print(r2_score(y_val,pred))

model2 = Ridge()
model2.fit(X_tr,y_tr)
pred2 = model2.predict(X_val)
print("Ridge")
print(mean_squared_error(y_val,pred2))
print(r2_score(y_val,pred2))

model3 = Lasso()
model3.fit(X_tr,y_tr)
pred3 = model3.predict(X_val)
print("Lasso")
print(mean_squared_error(y_val,pred3))
print(r2_score(y_val,pred3))

model4 = XGBRegressor()
model4.fit(X_tr,y_tr)
pred4 = model4.predict(X_val)
print("XGB")
print(mean_squared_error(y_val,pred4))
print(r2_score(y_val,pred4))

pred_final = model1.predict(X_test)
print("final")
print(mean_squared_error(y_test,pred_final))
print(r2_score(y_test,pred_final))