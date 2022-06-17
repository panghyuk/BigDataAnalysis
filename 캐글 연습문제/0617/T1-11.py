import pandas as pd
df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
print(df.head())

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df['f5'] = scaler.fit_transform(df[['f5']])
print(df.head())

low = df['f5'].quantile(0.05)
high = df['f5'].quantile(0.95)
ans = low + high
print(ans)