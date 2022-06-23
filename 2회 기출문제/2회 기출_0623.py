### 작업 1유형
''' Q1 '''
# import pandas as pd
# df = pd.read_csv('data/boston_housing.csv')
# # print(df.head())
# df = df.sort_values('crim',ascending = False)
# # print(df.head(10))
# top10 = df.iloc[9,0]
# df.iloc[:10,0] = top10
# # print(df.head(10))
# ans = df[df['age'] >= 80]['crim'].mean()
# print(ans)

''' Q2 '''
# import pandas as pd
# df = pd.read_csv("data/california_housing.csv")
# # print(df.shape)
# new_df = df[:16512].copy()
# # print(new_df.shape)
# before_std = new_df['total_bedrooms'].std()
# new_df['total_bedrooms'].fillna(new_df['total_bedrooms'].median(), inplace = True)
# after_std = new_df['total_bedrooms'].std()
# ans = abs(after_std - before_std)
# print(ans)

''' Q3 '''
# import pandas as pd
# df = pd.read_csv("data/insurance.csv")
# charge_mean = df['charges'].mean()
# charge_std = df['charges'].std()
# top = charge_mean + 1.5 * charge_std
# bottom = charge_mean - 1.5 * charge_std
# new_df = df[(df['charges'] > top) | (df['charges'] < bottom)]
# ans = new_df['charges'].sum()
# print(ans)

### 작업 2유형
import pandas as pd
X_train = pd.read_csv("data/X_train.csv")
X_test = pd.read_csv("data/X_test.csv")
y_train = pd.read_csv("data/y_train.csv")

# 결측치 X
# print(X_train.isnull().sum())
# print(X_test.isnull().sum())

# id 삭제 & y 타겟값
# X_train = X_train.drop('ID',axis = 1)
# id = X_test.pop("ID")
# y = y_train['Reached.on.Time_Y.N']

train = pd.merge(X_train,y_train, on = 'ID')
yes = train.loc[train['Reached.on.Time_Y.N'] == 1, ['ID']].count()[0]
no = train.loc[train['Reached.on.Time_Y.N'] == 0, ['ID']].count()[0]

diff = yes - no
train_one = train.loc[train['Reached.on.Time_Y.N'] == 1]
train_one_os = train_one.sample(n = diff,random_state = 2022)
train_os = pd.concat([train,train_one_os])

X_train = train_os.iloc[:,1:11]
y = train_os.iloc[:,11]
X_test = X_test.iloc[:,1:]

print(X_train.info())
num = [col for col in X_train.columns if X_train[col].dtypes != object]
cat = [col for col in X_train.columns if X_train[col].dtypes == object]

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])

for col in cat:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train, y, stratify = y, test_size = 0.2, random_state = 2022)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,pred))
print(roc_auc_score(y_val,pred_proba))

# 최종 test 데이터 셋 예측
pred_final = model.predict(X_test)
#
# output = pd.DataFrame({'ID':id, 'Reached.on.Time_Y.N':pred_final})
# output.to_csv("data/20220623.csv",index = False)