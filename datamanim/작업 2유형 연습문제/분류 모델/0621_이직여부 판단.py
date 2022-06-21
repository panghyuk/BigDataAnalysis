import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

#데이터 로드
X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/X_test.csv")
y_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/HRdata/y_test.csv")

print(X_train.isnull().sum())
print()
print(X_test.isnull().sum())

num = [col for col in X_train.columns if X_train[col].dtypes != object]
num = num[1:]
cat = [col for col in X_train.columns if X_train[col].dtypes == object]
cat = cat[1:]

print(X_train['gender'].unique())
print(X_train['enrolled_university'].unique())
print(X_train['education_level'].unique())
print(X_train['major_discipline'].unique())
print(X_train['experience'].unique())
print(X_train['company_size'].unique())
print(X_train['company_type'].unique())
print(X_train['last_new_job'].unique())

X_train.gender = X_train.gender.fillna('Other')
X_train.enrolled_university = X_train.enrolled_university.fillna('no_enrollment')
X_train.major_discipline = X_train.major_discipline.fillna('No Major')
X_train.experience = X_train.experience.fillna('0')
X_train.company_size = X_train.company_size.fillna('<10')
X_train.company_type = X_train.company_type.fillna('Other')
X_train.last_new_job = X_train.last_new_job.fillna('never')
X_train.education_level = X_train.education_level.fillna('Primary School')

X_test.gender = X_test.gender.fillna('Other')
X_test.enrolled_university = X_test.enrolled_university.fillna('no_enrollment')
X_test.major_discipline = X_test.major_discipline.fillna('No Major')
X_test.experience = X_test.experience.fillna('0')
X_test.company_size = X_test.company_size.fillna('<10')
X_test.company_type = X_test.company_type.fillna('Other')
X_test.last_new_job = X_test.last_new_job.fillna('never')
X_test.education_level = X_test.education_level.fillna('Primary School')

y = y_train['target']
y_test = y_test['target']

X_train = X_train.drop(['enrollee_id','city'],axis = 1)
X_test = X_test.drop(['enrollee_id','city'],axis = 1)
print(X_train.info())

for col in cat:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])

X_tr,X_val,y_tr,y_val = train_test_split(X_train, y, stratify = y,test_size = 0.2, random_state = 2022)
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:,1]
# print(roc_auc_score(y_val,pred))
# print(roc_auc_score(y_val,pred_proba))

model.fit(X_val,y_val)
pred_final = model.predict(X_test)
pred_proba_final = model.predict_proba(X_test)[:,1]
print(roc_auc_score(y_test,pred_final))
print(roc_auc_score(y_test,pred_proba_final))