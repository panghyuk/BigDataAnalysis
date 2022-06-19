import pandas as pd

b1 = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
b3 = pd.read_csv("../input/bigdatacertificationkr/basic3.csv")

df = pd.merge(left = b1, right = b3, how = "left", on = 'f4')
df = df.dropna(subset = ['r2'])
print(df)

new_df = df[:20]
ans = new_df['f2'].sum()
print(ans)