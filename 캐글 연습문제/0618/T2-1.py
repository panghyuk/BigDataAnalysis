import pandas as pd

# print(X_train.head())
print(X_train.info())

y = y_train['Survived']

features = ["Pclass","Sex","SibSp","Parch"]
X = pd.get_dummies(X_train[features])
test = pd.get_dummies(X_test[features])
# print(X)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators = 200, max_depth = 7, random_state = 42)
model.fit(X,y)
pred = model.predict(test)
# print(model.score(X,y))

output = pd.DataFrame({'PassengerID':X_test['PassengerId'],'Survived':pred})
print(output.head())

print(model.score(test,y_test['Survived']))