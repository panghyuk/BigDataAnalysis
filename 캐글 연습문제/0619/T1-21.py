import pandas as pd

df = pd.read_csv('../input/bigdatacertificationkr/basic1.csv')
# print(df)
df = df[df['age'] > 0]
df = df[df['age'] == round(df['age'])]
# print(df)

df['range'] = pd.qcut(df['age'], q = 3, labels = ['group1','group2','group3'])
df['range'].value_counts()

range1 = df[df['range'] == 'group1']['age'].median()
range2 = df[df['range'] == 'group2']['age'].median()
range3 = df[df['range'] == 'group3']['age'].median()

ans = range1 + range2 + range3
print(ans)