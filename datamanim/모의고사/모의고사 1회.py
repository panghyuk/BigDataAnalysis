### 작업 1유형
import pandas as pd
pd.set_option("display.max_columns",None)
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
# print(df.info())
# print(df.head())

''' Q1 '''
# print(df.head())
ans = ((df['age']//10) * 10).value_counts().index[0]
# print(ans)

''' Q2 '''
ans = ((df['age']//10) * 10).value_counts().values[0]
# print(ans)

''' Q3 '''
ans = df[(df['age'] >= 25) & (df['age'] < 29)]
ans = ans[ans['housing'] == 'yes']
# print(len(ans))

'''' Q4 '''
cat = []
for col in df.select_dtypes(exclude = 'int'):
    target = df[col]
    cat.append([col,target.nunique()])

cat.sort(key = lambda x:x[1], reverse = True)
ans = cat[0][0]
# print(ans)

''' Q5 '''
new_df = df[df['balance'] > df['balance'].mean()]
new_df = new_df.sort_values(['ID'],ascending = False)
new_df = new_df[:100]
ans = new_df['balance'].mean()
# print(new_df)
# print(ans)

''' Q6 '''
new_df = df[['day','month']].value_counts()
ans = new_df.index[0]
# print(ans)

''' Q7 '''
from scipy.stats import shapiro
ans = shapiro(df[df['job'] == 'unknown']['age'])[1]
# print(ans)

''' Q8 '''
new_df = df[['age','balance']].corr()
ans = new_df.iloc[0,1]
# print(ans)

''' Q9 '''
y = pd.crosstab(df['y'],df['education'])
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(y)
# print(y)
# print(p)

''' Q10 '''
new_df = df.groupby(['job','marital']).size().reset_index()
# print(type(new_df))
pivot_df = new_df.pivot_table(index = 'job', columns = 'marital')[0]
pivot_df = pivot_df.fillna(0)
pivot_df['ratio'] = pivot_df['divorced'] / pivot_df['married']
# print(pivot_df)
ans = pivot_df.sort_values('ratio',ascending = False)['ratio']
# print(ans[0])

### 작업 2유형
import pandas as pd
train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
test= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/test.csv')
submission= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/submission.csv')

print(train)
# print(test)
# print(submission)
def noyes(x):
    if x == 'no':
        x = 0
    else:
        x = 1
    return x

y = train.pop('y').apply(noyes)
# print(y)
# print(train.info())
# print(train.isnull().sum()) # 결측치 없음
num = [col for col in train.columns if train[col].dtypes != 'object']
num = num[1:]
cat = [col for col in train.columns if train[col].dtypes == 'object']
print(num)
print(cat)
# print(train[num].describe())

# 이상치 제거 X

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[num] = scaler.fit_transform(train[num])
test[num] = scaler.transform(test[num])

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train[cat] = train[cat].apply(le.fit_transform)
test[cat] = test[cat].apply(le.fit_transform)

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(train, y, test_size = 0.2,random_state = 2022)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

model1 = RandomForestClassifier()
model1.fit(X_tr,y_tr)
pred = model1.predict_proba(X_val)
print(roc_auc_score(y_val,pred[:,1]))

model3 = XGBClassifier()
model3.fit(X_tr,y_tr)
pred3 = model3.predict_proba(X_val)
print(roc_auc_score(y_val,pred3[:,1]))

# pred_rf = model1.predict_proba(test)
# pred_xgb = model3.predict_proba(test)


''' DM 풀이 '''
train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/train.csv')
test= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/test.csv')
submission= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bank/submission.csv')
from sklearn.model_selection import train_test_split

x = train.drop(columns =['ID','y'])
xd = pd.get_dummies(x)
y = train['y']

x_train,x_test,y_train,y_test = train_test_split(xd,y,stratify =y ,random_state=1)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(x_train,y_train)
pred = rf.predict_proba(x_test)

from sklearn.metrics import roc_auc_score,classification_report

print('test roc score : ',roc_auc_score(y_test,pred[:,1]))

test_pred = rf.predict_proba(pd.get_dummies(test.drop(columns=['ID'])))
submission['predict'] = test_pred[:,1]

print('submission file')
submission.to_csv('00000000000000.csv',index=False)