import pandas as pd
X_test = pd.read_csv("data/X_test.csv",encoding = 'euc-kr')
X_train = pd.read_csv("data/X_train.csv",encoding = 'euc-kr')
y_train = pd.read_csv("data/y_train.csv",encoding = 'euc-kr')

# 사용자 코딩
cust_id = X_test['cust_id'].copy()
pd.set_option("display.max_columns",None)
X_train = pd.merge(X_train,y_train,on = 'cust_id')

X_train['환불금액'] = X_train['환불금액'].fillna(0)
X_test['환불금액'] = X_test['환불금액'].fillna(0)
X_train.loc[X_train['총구매액']<0,'총구매액'] = 0
X_train.loc[X_train['최대구매액']<0,'총구매액'] = 0
X_test.loc[X_test['총구매액']<0,'총구매액'] = 0
X_test.loc[X_test['최대구매액']<0,'총구매액'] = 0

# gender 열 데이터 불균형 확인
g_zero = X_train.loc[X_train['gender'] == 0, ['cust_id']].count()[0]
g_one = X_train.loc[X_train['gender'] == 1, ['cust_id']].count()[0]
# print(g_zero,g_one)

# 데이터 불균형 차이
diff = g_zero - g_one
# train 셋에서 gender가 1인 대상만 데이터 셋으로 저장
train_one = X_train.loc[X_train['gender'] == 1]
# 데이터 불균형 차이만큼 비복원 랜덤 샘플링해서 별도 데이터 셋 구성
train_one_os = train_one.sample(n = diff, random_state = 2022)
# 랜덤 샘플링한 대상과 기존 train 데이터셋 행을 합침
train_os = pd.concat([X_train,train_one_os])

X_train = train_os.iloc[:,1:10]
y_train = train_os.iloc[:,10]
X_test = X_test.iloc[:,1:]

# print(X_train.head())
# print(y_train.head())
num = [col for col in X_train.columns if X_train[col].dtypes != object]
cat = [col for col in X_train.columns if X_train[col].dtypes == object]
# print(num)
# print(cat)

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])

for col in cat:
	le = LabelEncoder()
	X_train[col] = le.fit_transform(X_train[col])
	X_test[col] = le.transform(X_test[col])

X_tr,X_val,y_tr,y_val = train_test_split(X_train, y_train, stratify = y_train, test_size = 0.2, random_state = 2022)

model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:,1]
# print(roc_auc_score(y_val,pred))
print(roc_auc_score(y_val,pred_proba))

pred_final = model.predict_proba(X_test)[:,1]
output = pd.DataFrame({"cust_id":cust_id,'gender':pred_final})
# output.to_csv("20220622.csv",index = False)
# print(output.head())