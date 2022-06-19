# 피처 구분
num_features = [col for col in X_train.columns if X_train[col].dtypes != 'object']
cat_features = [col for col in X_train.columns if X_train[col].dtypes == 'object']
num_features.remove('id')

# 결측치 처리
def data_fillna(df):
    df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
    df['occupation'] = df['occupation'].fillna('null')
    df['native.country'] = df['native.country'].fillna(df['native.country'].mode()[0])
    return df

X_train = data_fillna(X_train)
X_test = data_fillna(X_test)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[cat_features] = X_train[cat_features].apply(le.fit_transform)
X_test[cat_features] = X_test[cat_features].apply(le.fit_transform)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

y = (y_train['income'] != '<=50K').astype(int)

from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y,test_size = 0.2,random_state = 2022)

X_tr = X_tr.drop('id',axis = 1)
X_val = X_val.drop('id',axis = 1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier()
model.fit(X_tr,y_tr)
pred = model.predict(X_val)
print(accuracy_score(y_val,pred))

X_id = X_test.pop('id')
pred = model.predict(X_test)
output = pd.DataFrame({"id": X_id, 'income':pred})
output.to_csv("20220619.csv",index = False)
output.head()

# 수험자는 확인 불가
y_test = (y_test['income'] != '<=50K').astype(int)
from sklearn.metrics import accuracy_score
print('accuracy score:', (accuracy_score(y_test, pred))) # 0.85