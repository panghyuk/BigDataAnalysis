import pandas as pd
#데이터 로드
X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/X_test.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/churnk/y_test.csv")

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,LabelEncoder

# 결측치 확인
# print(X_train.isnull().sum())
# print(X_test.isnull().sum())
print(X_train.info())

num = [col for col in X_train.columns if X_train[col].dtypes != 'object']
num = num[1:]
cat = [col for col in X_train.columns if X_train[col].dtypes == 'object']
scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])
for col in cat:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.fit_transform(X_test[col])

X_train = X_train.drop('CustomerId',axis = 1)
X_test = X_test.drop('CustomerId',axis = 1)
y = y_train['Exited']
y_test = y_test['Exited']

X_tr,X_val,y_tr,y_val = train_test_split(X_train,y)
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,pred))
print(roc_auc_score(y_val,pred_proba))

pred_final = model.predict(X_test)
pred_proba_final = model.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred_final))
print(roc_auc_score(y_test,pred_proba_final))