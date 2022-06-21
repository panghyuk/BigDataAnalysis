import pandas as pd
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/bicycle/seoul_bi.csv')

''' Q1 '''
# new_df = df['대여일자'].value_counts().sort_index().to_frame()
# print(new_df)
# ans = new_df[new_df['대여일자'] == new_df['대여일자'].max()].index[0]
# print(ans)

''' Q2 '''
# df['대여일자'] = pd.to_datetime(df['대여일자'])
# df['day_name'] = df['대여일자'].dt.day_name()
# ans = df['day_name'].value_counts().to_frame()
# print(ans)

''' Q3 '''
