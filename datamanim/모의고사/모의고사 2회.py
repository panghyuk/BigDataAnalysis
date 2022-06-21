### 작업 1유형
import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv')

pd.set_option("display.max_columns",None)
# print(df.head())
# print(df.info())

''' Q1 '''
# df['age'] = df['age'].str.replace('*',"",regex = True).astype('int')
# ans = df[df['gender'] == "Male"]['age'].mean()
# print(ans)

''' Q2 '''
# df['bmi'] = df['bmi'].fillna(df['bmi'].median())
# ans = round(df['bmi'].mean(),3)
# print(ans)

''' Q3 '''
# df['bmi'] = df['bmi'].fillna(method = 'ffill')
# ans = round(df['bmi'].mean(),3)
# print(ans)

''' Q4 *** '''
# df['age'] = df['age'].str.replace("*","",regex = True).astype('int')
# age_mean = df[df['bmi'].notnull()].groupby((df['age']//10) * 10)['bmi'].mean()
# age_dict = {x:y for x,y in age_mean.items()}
#
# idx = df.loc[df['bmi'].isnull(), ['age','bmi']].index
# df.loc[df['bmi'].isnull(),'bmi'] = (df[df['bmi'].isnull()]['age']//10 * 10).map(lambda x : age_dict[x])
# ans = df['bmi'].mean()
# print(ans)

''' Q5 * '''
# df.loc[df['avg_glucose_level'] >= 200,'avg_glucose_level'] = 199
# ans = round(df[df['stroke'] == 1]['avg_glucose_level'].mean(),3)
# print(ans)

''' 다른 데이터 '''
df = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/pok/Pokemon.csv")
# print(df.head())
# print(df.info())

''' Q6 '''
# new_df = df.sort_values(by = 'Attack',ascending = False)
# top_df = new_df[:400]
# bottom_df = new_df[400:800]
# top_leg = len(top_df[top_df['Legendary'] == True])
# bottom_leg = len(bottom_df[bottom_df['Legendary'] == True])
# ans = top_leg - bottom_leg
# print(ans)

''' Q7 '''
# new_df = df.groupby("Type 1")['Total'].mean()
# new_df = new_df.sort_values(ascending = False)
# ans = new_df.index[2]
# print(ans)

''' Q8 '''
# new_df = df.dropna(axis = 0)
# new_df = new_df[:int(len(new_df) * 0.6)]
# ans = new_df['Defense'].quantile(0.25)
# print(ans)

''' Q9 '''
# target = df[df['Attack'] > df[df['Type 1'] == "Fire"]['Attack'].mean()]
# ans = target[target['Type 1'] == 'Water'].shape[0]
# print(ans)

''' Q10 *** '''
# ans = abs(df.groupby(['Generation'])[['Speed','Defense']].mean().T.diff().T).sort_values('Defense',ascending = False).index[0]
# print(ans)


### 작업 2유형
import pandas as pd
train = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/train.csv')
test = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/stroke_/test.csv')

# print(train.shape,test.shape)
X_train = train.iloc[:,1:11].copy()
y_train = train['stroke']
X_test = test.iloc[:,1:].copy()
# print(X_train.head())
# print(y_train.head())
# print(X_train.info())
# print(X_test.info())
# print(X_train['age'].unique())
# print(X_test['age'].unique())

X_train['age'] = X_train['age'].str.replace('*',"",regex = True).astype('int64')
X_train['bmi'] = X_train['bmi'].fillna(X_train['bmi'].mean())
X_test['bmi'] = X_test['bmi'].fillna(X_test['bmi'].mean())

print(X_train.info())
# print(X_test.info())
num = ['age','hypertension','heart_disease','avg_glucose_level','bmi']
cat = ['gender','ever_married','work_type','Residence_type','smoking_status']

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])


for col in cat:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

# print(X_train.info())
# print(X_test.info())

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size = 0.15, random_state = 2022)

print(X_tr.shape,X_val.shape,y_tr.shape,y_val.shape)

model1 = RandomForestClassifier()
model1.fit(X_tr,y_tr)
pred1 = model1.predict(X_val)
pred_proba1 = model1.predict_proba(X_val)[:,1]
print("RF")
print(roc_auc_score(y_val,pred1))
print(roc_auc_score(y_val,pred_proba1))

model2 = XGBClassifier()
model2.fit(X_tr,y_tr)
pred2 = model2.predict(X_val)
pred_proba2 = model2.predict_proba(X_val)[:,1]
print("XGB")
print(roc_auc_score(y_val,pred2))
print(roc_auc_score(y_val,pred_proba2))

pred_final = model2.predict(X_test)
pred_proba_final = model2.predict_proba(X_test)[:,1]
output = pd.DataFrame({'id':test['id'],'stroke':pred_proba_final})
print(output.head())
# output.to_csv("20220621.csv",index = False)
