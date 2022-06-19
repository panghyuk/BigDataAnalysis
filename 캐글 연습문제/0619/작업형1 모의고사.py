# 1번
import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
print(df.head())
df1 = df[:50].copy()
df2 = df[50:].copy()
df1['f1'] = df1['f1'].fillna(df1['f1'].median())
df2['f1'] = df2['f1'].fillna(df2['f1'].max())
df1_std = df1['f1'].std()
df2_std = df2['f1'].std()
ans = df1_std + df2_std
print(round(ans,1))

# 2번
import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
df = df.sort_values(['f4','f5'],ascending = [False, True]).reset_index(drop = True)
print(df)
f5_min = df['f5'][:10].min()
df['f5'][:10] = f5_min
print(df)
ans = df['f5'].mean()
print(round(ans,2))

# 3번
import pandas as pd
df = pd.read_csv("../input/bigdatacertificationkr/basic1.csv")
print(df)
print(df['age'].describe())
q1 = df['age'].quantile(0.25)
q3 = df['age'].quantile(0.75)
iqr = q3 -q1
new_df = df[(df['age'] > q3 + 1.5*iqr) | (df['age'] < q1 - 1.5*iqr)]
print(len(new_df))
mean = df['age'].mean()
std = df['age'].std()
top = mean + 1.5 * std
bottom = mean - 1.5 * std
new_df2 = df[(df['age'] > top) | (df['age'] < bottom)]
print(len(new_df2))