import pandas as pd
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(df.head())

# ESFJ 값을 가진 데이터 확인
new_df = df[df['f4'] == 'ESFJ'].copy()
print(new_df)

# 값 변경하기
new_df['f4'] = new_df['f4'].replace('ESFJ','ISFJ')
print(new_df)

# 2개의 조건에 맞는 값중 age컬럼의 최대값
ans = df[(df['city'] == '경기') & (df['f4'] == 'ISFJ')]['age'].max()
print(ans)