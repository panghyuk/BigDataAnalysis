# 라이브러리 및 데이터 불러오기
import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
print(df.head())

# 조건에 맞는 데이터 (ENFJ, INFP)
df_enfj = df[(df['f4'] == 'ENFJ')]
df_infp = df[df['f4'] == 'INFP']

# 조건에 맞는 f1의 표준편차 (ENFJ, INFP)
enfj_std = df_enfj['f1'].std()
infp_std = df_infp['f1'].std()

# 두 표준편차 차이 절대값 출력
ans = abs(enfj_std - infp_std)
print(ans)