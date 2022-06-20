import pandas as pd
#데이터 로드
X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/X_test.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/shipping/y_test.csv")
# print(X_train.info()) # 결측치 X
# print(X_test.info()) # 결측치 X

num = [col for col in X_train.columns if X_train[col].dtypes != 'object']
num = num[1:]
cat = [col for col in X_train.columns if X_train[col].dtypes == 'object']
# print(num)
# print(cat)

X_train = X_train.drop('ID',axis = 1)
X_test = X_test.drop('ID',axis = 1)
y = y_train['Reached.on.Time_Y.N']

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[cat] = X_train[cat].apply(le.fit_transform)
X_test[cat] = X_test[cat].apply(le.fit_transform)

scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y,test_size = 0.2, random_state = 2022)

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

model1 = RandomForestClassifier()
model1.fit(X_tr,y_tr)
pred = model1.predict(X_val)
pred_proba = model1.predict_proba(X_val)[:,1]
print("RF")
print(roc_auc_score(y_val,pred))
print(roc_auc_score(y_val,pred_proba))
# print(pred_proba)

model2 = XGBClassifier()
model2.fit(X_tr,y_tr)
pred2 = model2.predict(X_val)
pred_proba2 = model2.predict_proba(X_val)[:,1]
print("XGB")
print(roc_auc_score(y_val,pred2))
print(roc_auc_score(y_val,pred_proba2))

pred_final = model1.predict(X_test)
pred_proba_final = model1.predict_proba(X_test)[:,1]
y_final = y_test['Reached.on.Time_Y.N']
print("final")
print(roc_auc_score(y_final,pred_final))
print(roc_auc_score(y_final,pred_proba_final))