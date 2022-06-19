import pandas as pd

# print(X_train.head())
print(X_train.info()) # null 값 확인
# print(X_test.info()) # null 값 확인

# 포도당 이상치 삭제
del_idx = X_train[(X_train['Glucose'] == 0)].index
# print(del_idx)

X_train = X_train.drop(index = del_idx, axis = 0)
y_train = y_train.drop(index = del_idx, axis = 0)

# 포도당을 제외한 이상치, 평균값으로 대체
cols = ['BloodPressure','SkinThickness','Insulin','BMI']
cols_mean = X_train[cols].mean()
# print(cols_mean)
X_train[cols].replace(0,cols_mean)

# 스케일링
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_cols = X_train.columns
# print(X_cols[1:])
X_train[X_cols] = scaler.fit_transform(X_train[X_cols])
X_test[X_cols] = scaler.transform(X_test[X_cols])

X_train = X_train.drop('id',axis = 1)
X_test = X_test.drop('id',axis = 1)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train,y_train['Outcome'])
pred = model.predict(X_test)
output = pd.DataFrame({'idx': X_test.index, 'Outcome': pred})
output.head()

print(model.score(X_train,y_train['Outcome']))
print(model.score(X_test,y_test['Outcome']))