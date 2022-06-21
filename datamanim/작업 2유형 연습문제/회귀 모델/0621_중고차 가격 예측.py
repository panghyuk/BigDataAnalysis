import pandas as pd
#데이터 로드
X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/y_train.csv")
X_test= pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/X_test.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/carsprice/y_test.csv")

print(X_train.info())
# print(X_test.info())

X_train = X_train.drop('carID',axis = 1)
X_test = X_test.drop('carID',axis = 1)
y = y_train['price']
y_test = y_test['price']
# print(y)

num = [col for col in X_train.columns if X_train[col].dtypes != object]
cat = [col for col in X_train.columns if X_train[col].dtypes == object]
# print(num)
# print(cat)

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error,r2_score

scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])
# print(X_train[num].describe())

for col in cat:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# print(X_train)
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y, test_size = 0.2, random_state = 2022)

model = RandomForestRegressor()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
print(mean_squared_error(y_val,pred))
print(r2_score(y_val,pred))

model.fit(X_val,y_val)
pred_final = model.predict(X_test)
print(mean_squared_error(y_test,pred_final))
print(r2_score(y_test,pred_final))