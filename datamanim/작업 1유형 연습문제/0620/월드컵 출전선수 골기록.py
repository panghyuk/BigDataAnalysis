import pandas as pd
pd.set_option("display.max_columns",None)
df = pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')
print(df.head())
print(df.info())

''' Q1 '''
ans = df.groupby('Country').sum().sort_values(by = 'Goals',ascending = False).head(5)
# print(ans)

''' Q2 '''
ans = df.groupby('Country').size().sort_values(ascending = False).head(5)
# print(ans)

''' Q3 '''
df['yearlist'] = df['Years'].str.split('-')
def checkfour(x):
    for value in x:
        if len(value) != 4:
            return False
    return True
df['check'] = df['yearlist'].apply(checkfour)
ans = len(df[df['check'] == False])
# print(ans)

''' Q4 '''
df2 = df[df['check'] == True].reset_index(drop = True)
# print(df2.shape[0])
print(df2.head())

''' Q5 '''
df2['LenCup'] = df2['yearlist'].str.len()
ans = df2['LenCup'].value_counts()[4]
# print(ans)

''' Q6 '''
ans = len(df2[(df2['LenCup'] == 2) & (df2['Country'] == 'Yugoslavia')])
# print(ans)

''' Q7 '''
ans = len(df2[df2['Years'].str.contains('2002')])
# print(ans)

''' Q8 '''
ans = len(df2[df2['Player'].str.lower().str.contains('carlos')])
# print(ans)

''' Q9 '''
ans = df2[df2['LenCup'] == 1].sort_values(by = 'Goals',ascending = False)['Player'].values[0]
# print(ans)

''' Q10 '''
ans = df2[df2['LenCup'] == 1]['Country'].value_counts().index[0]
print(ans)