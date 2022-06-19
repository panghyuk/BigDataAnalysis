import pandas as pd

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')

num = df.sort_values('f1',ascending = False).iloc[9,3]
# print(num)
df['f1'] = df['f1'].fillna(num)

res1 = df['f1'].median()
# print(res1)
df = df.drop_duplicates(subset = ['age'])
res2 = df['f1'].median()
# print(res2)

ans = res1 - res2
print(abs(ans))