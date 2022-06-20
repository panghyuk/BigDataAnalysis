import pandas as pd
pd.set_option('display.max_columns',None)
df =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/happy2/happiness.csv',encoding='utf-8')
print(df.head())

''' Q1 '''
ans = df[df['행복랭킹'] == 10]['점수'].mean()
# print(ans)

''' Q2 '''
ans = df[df['행복랭킹'] <= 50][['년도','점수']].groupby('년도').mean()
# print(ans)

''' Q3 '''
ans = df[df['년도'] == 2018][['점수','부패에 대한인식']].corr().iloc[0,1]
# print(ans)

''' Q4 '''
ans = len(df[['행복랭킹','나라명']]) - len(df[['행복랭킹','나라명']].drop_duplicates())
# print(ans)

''' Q5 상관관계 '''

''' Q6 '''
ans = df.groupby('년도').tail(5).groupby('년도').mean()[['점수']]
print(ans)

''' Q7 '''
new_df = df[df['년도'] == 2019]
df_up = new_df[new_df['상대GDP'] >= new_df['상대GDP'].mean()]['점수'].mean()
df_down = new_df[new_df['상대GDP'] < new_df['상대GDP'].mean()]['점수'].mean()
ans = df_up - df_down
# print(ans)

''' Q8 '''
ans = df.sort_values(by = ['년도','부패에 대한인식'],ascending = False).groupby('년도').head(20).groupby('년도').mean()[['부패에 대한인식']]
# print(ans)

''' Q9 '''
ans = len(set(df[(df['행복랭킹'] <= 50) & (df['년도'] == 2018)]['나라명']) - set(df[(df['행복랭킹'] <= 50) & (df['년도'] == 2019)]['나라명']))
# print(ans)

''' Q10 행복 점수 차이'''

