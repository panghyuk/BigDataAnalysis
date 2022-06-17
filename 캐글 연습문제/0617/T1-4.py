# 라이브러리 및 데이터 불러오기
import pandas as pd
import numpy as np
df = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")
print(df['SalePrice'].head())

# 'SalePrice'컬럼 왜도와 첨도계산 
sk = df['SalePrice'].skew()
ku = df['SalePrice'].kurt()
print(sk,ku)

# 'SalePrice'컬럼 로그변환
df['SalePrice'] = np.log1p(df['SalePrice'])

# 'SalePrice'컬럼 왜도와 첨도계산 
sk2 = df['SalePrice'].skew()
ku2 = df['SalePrice'].kurtosis()
print(sk2,ku2)

# 모두 더한 다음 출력
ans = sk + ku + sk2 + ku2
print(round(ans,2))