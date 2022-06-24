# 라이브러리 및 데이터 불러오기
import pandas as pd
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
# ESFJ -> ISFJ 교체
df['f4'] = df['f4'].replace('ESFJ','ISFJ')
# 해당 조건에 맞는 최대값 찾기
ans = df[(df['city'] == '경기') & (df['f4'] == "ISFJ")]['age'].max()
print(ans)