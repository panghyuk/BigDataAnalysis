# yeo-johnson/box-cox

# 라이브러리 및 데이터 불러오기
import pandas as pd
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(df.head(5))

# 조건에 맞는 데이터
df = df[df['age'] >= 20]
print(df.head())


# 최빈값으로 'f1' 컬럼 결측치 대체
print(df['f1'].mode())
df['f1'] = df['f1'].fillna(df['f1'].mode()[0])
df.isnull().sum()

# 'f1'데이터 여-존슨 yeo-johnson 값 구하기
from sklearn.preprocessing import power_transform
df['yeo-johnson'] = power_transform(df[['f1']], standardize = False)
print(df['yeo-johnson'].head())

# 'f1'데이터 박스-콕스 box-cox 값 구하기
df['box-cox'] = power_transform(df[['f1']], method = 'box-cox',standardize = False)
print(df['box-cox'].head())

# 두 값의 차이를 절대값으로 구한다음 모두 더해 소수점 둘째 자리까지 출력(반올림)
import numpy as np
ans = round(sum(np.abs(df['yeo-johnson'] - df['box-cox'])),2)
print(ans)