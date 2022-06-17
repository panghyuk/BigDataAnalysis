import pandas as pd
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(df.head())

# f1컬럼 결측치 제거
print(df.info())
df.dropna(subset = ['f1'],inplace = True)
print(df)
print(df.info())

# 그룹 합계 계산
df2 = df.groupby(['city','f2']).sum()
print(df2)

# 조건에 맞는 값 출력
print(df2.iloc[0]['f1'])