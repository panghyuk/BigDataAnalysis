### 작업 1유형
''' Q1 '''
# import pandas as pd
# df = pd.read_csv("data/boston_housing.csv")
# # print(df.info())
# df.dropna(inplace = True)
# # print(df.info())
# new_df = df[:int(df.shape[0] * 0.7)].copy()
# ans = new_df['tax'].quantile(0.25)
# print(ans)

''' Q2 '''
# import pandas as pd
# df = pd.read_csv("data/tour.csv",encoding = "euc-kr",index_col = 0)
# mean_2000 = df.loc[2000,:].mean()
# ans = sum(df.loc[2000,:] > mean_2000)
# print(ans)

# ''' Q3 '''
# import pandas as pd
# df = pd.read_csv("data/titanic.csv")
# ans = (df.isnull().sum()/len(df)).sort_values(ascending = False).index[0]
# print(ans)

## 작업 2유형
import pandas as pd
train = pd.read_csv("data/TravelInsurancePrediction_train.csv")
test = pd.read_csv('data/TravelInsurancePrediction_test.csv')

# 결측치 X
print(train.info())
# print(test.info())
id = test['ID']

print(train['TravelInsurance'].value_counts())
zero = train['TravelInsurance'].value_counts().values[0]
one = train['TravelInsurance'].value_counts().values[1]
diff = zero - one

train_one = train.loc[train['TravelInsurance'] == 1]
train_one_os = train_one.sample(n = diff, random_state = 2022)
train_os = pd.concat([train,train_one_os])

X_train = train_os.iloc[:,1:9]
y = train_os.iloc[:,9]
X_test = test.iloc[:,1:]

num = [col for col in X_train.columns if X_train[col].dtypes != object]
cat = [col for col in X_train.columns if X_train[col].dtypes == object]
# print(num,cat)

from sklearn.preprocessing import MinMaxScaler,LabelEncoder
scaler = MinMaxScaler()
X_train[num] = scaler.fit_transform(X_train[num])
X_test[num] = scaler.transform(X_test[num])

for col in cat:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col])
    X_test[col] = le.transform(X_test[col])

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y,stratify = y, test_size = 0.2, random_state = 2022)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict_proba(X_val)[:,1]
print(roc_auc_score(y_val,pred))

pred_final = model.predict_proba(X_test)[:,1]
output = pd.DataFrame({'id':id,'TravelInsurance':pred_final})
output.to_csv("data/20220623.csv",index = False)