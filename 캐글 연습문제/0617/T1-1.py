# 라이브러리 및 데이터 불러오기
import pandas as pd
df = pd.read_csv("../input/titanic/train.csv")
print(df.info())

# 간단한 탐색적 데이터 분석 (EDA)
print(df.isnull().sum())

# IQR 구하기
print(df.describe())
fare_1q = df.describe()['Fare']['25%']
fare_3q = df.describe()['Fare']['75%']
fare_iqr = fare_3q - fare_1q
top = fare_3q + 1.5 * fare_iqr
bottom = fare_1q - 1.5 * fare_iqr
print(top,bottom)

# 이상치 데이터 구하기
new_df = df[(df['Fare'] > top) | (df['Fare'] < bottom)]
print(new_df)

# 이상치 데이터에서 여성 수 구하기, 출력하기 print()
ans = len(new_df[df['Sex'] == 'female'])
print(ans)