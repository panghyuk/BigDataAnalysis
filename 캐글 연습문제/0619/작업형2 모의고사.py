import pandas as pd
X_test = pd.read_csv("../input/hr-data/X_test.csv")
X_train = pd.read_csv("../input/hr-data/X_train.csv")
y_train = pd.read_csv("../input/hr-data/y_train.csv")

pd.set_option('display.max_columns',None)
print(X_train.head())

print(X_train.info())

# 결측치 확인
print(X_train.isnull().sum())
print()
print(X_test.isnull().sum())

# 결측치 대체
X_train = X_train.fillna("null")
X_test = X_test.fillna('null')

# 이상치 제거 1
q1 = X_train['training_hours'].quantile(0.25)
q3 = X_train['training_hours'].quantile(0.75)
iqr = q3-q1
outdata1 = X_train[X_train['training_hours'] < q1 - 1.5 * iqr].index
outdata2 = X_train[X_train['training_hours'] > q3 + 1.5 * iqr].index

X_train = X_train.drop(index = outdata1, axis = 0)
X_train = X_train.drop(index = outdata2, axis = 0)
y_train = y_train.drop(index = outdata1, axis = 0)
y_train = y_train.drop(index = outdata2, axis = 0)

# 이상치 제거 2
q1 = X_train['city_development_index'].quantile(0.25)
q3 = X_train['city_development_index'].quantile(0.75)
iqr = q3-q1
outdata1 = X_train[X_train['city_development_index'] < q1 - 1.5 * iqr].index
outdata2 = X_train[X_train['city_development_index'] > q3 + 1.5 * iqr].index

X_train = X_train.drop(index = outdata1, axis = 0)
X_train = X_train.drop(index = outdata2, axis = 0)
y_train = y_train.drop(index = outdata1, axis = 0)
y_train = y_train.drop(index = outdata2, axis = 0)

cat = [col for col in X_train.columns if X_train[col].dtypes == 'object']
print(cat)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[cat] = X_train[cat].apply(le.fit_transform)
X_test[cat] = X_test[cat].apply(le.fit_transform)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
nums = ['city_development_index','training_hours']
X_train[nums] = scaler.fit_transform(X_train[nums])
X_test[nums] = scaler.transform(X_test[nums])

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y_train['target'],test_size = 0.2,random_state = 2022)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,pred))

pred = model.predict_proba(X_test)[:,1]

output = pd.DataFrame({'enrollee_id':X_test['enrollee_id'],'target':pred})
output.to_csv("20220619.csv",index = False)
