import pandas as pd

df = pd.read_csv("../input/bigdatacertificationkr/basic2.csv")
# print(df)

df['previous'] = df['PV'].shift(1)
df['previous'] = df['previous'].fillna(method = 'bfill')
# print(df.head())

new_df = df[(df['Events'] == 1) & (df['Sales'] <= 1000000)]
new_df

ans = new_df['previous'].sum()
print(ans)