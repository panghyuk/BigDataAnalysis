# 라이브러리 및 데이터 불러오기
import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
print(df.head())

# EDA - 결측값 확인(비율 확인)
print(df.isnull().sum() / df.shape[0])

# 80%이상 결측치 컬럼, 삭제
df.drop(['f3'],axis = 1, inplace = True)
print(df.info())

# 80%미만 결측치 컬럼, city별 중앙값으로 대체
print(df['city'].unique())

s = df[df['city'] == '서울']['f1'].median()
b = df[df['city'] == '부산']['f1'].median()
k = df[df['city'] == '경기']['f1'].median()
d = df[df['city'] == '대구']['f1'].median()

df['f1'] = df['f1'].fillna(df['city'].map({'서울':s, '경기':k, '부산':b, '대구':d}))

# f1 평균값 결과 출력
print(df['f1'].mean())