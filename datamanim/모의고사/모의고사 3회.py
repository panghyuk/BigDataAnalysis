### 작업 1유형
import pandas as pd
pd.set_option("display.max_columns",None)
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/spotify/spotify.csv')
# print(df.head())

''' Q1 '''
df = df.dropna()
df['rank'] = list(range(1,101)) * 10
# ans = df[df['rank'] == 1]['bpm'].mean()
# print(ans)

''' Q2 '''
# ans = df[df['top year'] == 2015]['artist'].value_counts().index[0]
# print(ans)

''' Q3 '''
# ans = df[df['rank'].isin(range(1,11))]['top genre'].value_counts().index[2]
# print(ans)

''' Q4 '''
# ans = df['title'].str.split('feat. ').str[1].dropna().str[:-1].str.strip().value_counts().index[0]
# print(ans)

''' Q5 '''
# ans = df[df['year released'] != df['top year']]['top year'].value_counts().index[0]
# print(ans)

''' Q6 '''
# ans = df[df['artist'].str.lower().str.contains('q')]['artist'].nunique()
# print(ans)

''' Q7 '''
# df_50 = df[df['rank']<= 50]
# df_100 = df[(df['rank']>50) & (df['rank'] <= 100)]
# df_50_mean = df_50['dur'].mean()
# df_100_mean = df_100['dur'].mean()
# ans = df_50_mean - df_100_mean
# print(ans)

''' Q8 '''

''' Q9 '''
# m = df.groupby(['top year'])['nrgy'].mean().sort_values()
# ans = m.values[-1] - m.values[0]
# print(ans)

''' Q10 '''
# ans = df[['artist','artist type']].value_counts().reset_index()['artist'].value_counts().index[0]
# print(ans)


### 작업 2유형
import pandas as pd
#데이터 로드
X_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/x_train.csv")
y_train = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/y_train.csv")
X_test = pd.read_csv("https://raw.githubusercontent.com/Datamanim/datarepo/main/muscle/x_test.csv")

# 결측값 확인
# print(X_train.info())
# print(X_test.info())
id = X_test['ID']
X_train = X_train.drop('ID',axis = 1)
X_test = X_test.drop('ID',axis = 1)
y = y_train['pose']

# print(X_train.describe())

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_tr,X_val,y_tr,y_val = train_test_split(X_train,y, stratify = y, test_size = 0.2, random_state = 2022)

model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
pred_proba = model.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,pred))
print(roc_auc_score(y_val,pred_proba))

pred_final = model.predict_proba(X_test)[:,1]
output = pd.DataFrame({"id":id,'pose':pred_final})
print(output)
# output.to_csv("20220622.csv",index = False)