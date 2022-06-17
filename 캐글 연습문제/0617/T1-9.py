# 라이브러리 및 데이터 불러오기
import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
print(df.head())

# 표준화
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df[['f5']])
df['f5'] = scaler.transform(df[['f5']])

# 중앙값 출력
ans = df['f5'].median()
print(ans)