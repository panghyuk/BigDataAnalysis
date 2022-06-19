import pandas as pd

df = pd.read_csv("../input/bigdatacertificationkr/basic2.csv", parse_dates=['Date'], index_col=0)
print(df)

df_w = df.resample('W').sum()
w_max = df_w['Sales'].max()
w_min = df_w['Sales'].min()
ans = w_max - w_min
print(ans)