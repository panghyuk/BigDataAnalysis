import pandas as pd
import numpy as np
X_train = pd.read_csv("data/X_train.csv", encoding = 'euc-kr')
X_test = pd.read_csv("data/X_test.csv", encoding = 'euc-kr')
y_train = pd.read_csv("data/y_train.csv", encoding = 'euc-kr')

pd.set_option('display.max_columns',None)
y = y_train['gender']
cust_id = X_test['cust_id']

print(X_train.info()) # 환불금액 결측치
# print(X_test.info())

# 결측치 0으로 채우기
X_train['환불금액'] = X_train['환불금액'].fillna(0)

# print(X_train.describe())
# print(X_train['환불금액'].skew())
# X_train['환불금액'] = np.log1p(X_train['환불금액'])
# print(X_train.describe())
# print(X_train['환불금액'].skew())

num = X_train.select_dtypes(exclude = 'object').columns
cat = X_train.select_dtypes(include = 'object').columns

# for col in cat:
#     print('*' * 20)
#     print(col)
#     print(X_train[col].value_counts())
#     print(X_test[col].value_counts())

train_cat = pd.get_dummies(X_train[cat])
test_cat = pd.get_dummies(X_test[cat])

train_cat, test_cat = train_cat.align(test_cat, join = 'inner', axis = 1)

X_train = X_train.drop(cat,axis = 1)
X_test = X_test.drop(cat,axis = 1)
# print(X_train)

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print(X_train.shape)
print(X_test.shape)

X_train = pd.concat([pd.DataFrame(X_train),train_cat],axis = 1)
X_test = pd.concat([pd.DataFrame(X_test),test_cat],axis = 1)
# print(X_train.shape)
# print(X_test.shape)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

X_tr,X_val,y_tr,y_val = train_test_split(X_train,y, stratify = y, test_size = 0.2, random_state = 2022)
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,pred))

pred_final = model.predict_proba(X_test)[:,1]
output = pd.DataFrame({'cust_id':cust_id, 'gender':pred_final})
output.to_csv('20220624.csv',index = False)