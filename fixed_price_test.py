import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from: https://www2.electricityinfo.co.nz/historic_data/

d2020 = pd.read_csv('Old_data/2020_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
d2021 = pd.read_csv('Old_data/2021_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
d2022 = pd.read_csv('Old_data/2022_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
d2023 = pd.read_csv('Old_data/2023_ROS1101.csv', names = ['node','date','TP','price','date2','data'])
df = pd.concat([d2020, d2021,d2022,d2023])

td_arr = []
for i in range(len(df)):
    td_arr.append(pd.Timedelta(df.iloc[i]['TP']*0.5-0.5,'hours'))

df['dt'] = pd.to_datetime(df['date'], dayfirst=True) + pd.to_timedelta(td_arr)
df = df.drop(['data','date2','date','TP','node'],axis=1)

df = df.reset_index()
df = df.rename({'dt':'date'},axis=1)
df = df.drop(['index'],axis=1)
df.set_index('date', inplace=True)

def price_curve(x):
    N = 400
    pct_save = []
    price_arr = np.linspace(0,N,N-1)
    for price in price_arr:
        df['bool'] = df['price'] < price
        df_daily = df.resample('D').sum()

        # calculate percentage of days where duration is greater than x hours
        pct_above_3hrs = (1-((len(df_daily[df_daily['bool']<x*2])) / len(df_daily)))
        pct_save.append(pct_above_3hrs)
    pct_save = np.array(pct_save)
    return price_arr, pct_save

price_arr, pct_1 = price_curve(1)
price_arr, pct_2 = price_curve(2)
price_arr, pct_3 = price_curve(3)
price_arr, pct_4 = price_curve(4)
price_arr, pct_5 = price_curve(5)
price_arr, pct_6 = price_curve(6)

plt.plot(pct_1, price_arr)
plt.plot(pct_2, price_arr)
plt.plot(pct_3, price_arr)
plt.plot(pct_4, price_arr)
plt.plot(pct_5, price_arr)
plt.plot(pct_6, price_arr)
plt.ylabel('Price / $/MWh')
plt.xlabel('% of days with >xhrs below price')
plt.plot([0,1],[183.5,183.5],'k--')
plt.legend(['1','2','3','4','5','6','Mercury'])


# time based limits

grouped = df.groupby(pd.Grouper(key='date', freq='D')) # group by calendar day

# define a function to find the average of the lowest 6 prices in each group
def avg_of_lowest(x):
    lowest = x.nsmallest(6, 'price')
    return lowest['price'].mean()

def max_of_lowest(x):
    lowest = x.nsmallest(6, 'price')
    return lowest['price'].max()

result_av = grouped.apply(avg_of_lowest) # apply the function to each group
result_max = grouped.apply(max_of_lowest) # apply the function to each group

res_max_smooth = savgol_filter(result_max.to_numpy(),120,3)
plt.plot(result_max)
plt.plot(result_max.index, res_max_smooth)
plt.ylabel('Price / $/MWh')

# Lowest 6 prices in 24 hours

boool = []
for i in range(len(df)):
    if i < 48:
        boool.append(True)
    else:
        price_current = df.iloc[i]
        df_day = df.iloc[i-48:i]
        N_less = (price_current < df_day).sum().to_numpy()
        # print(N_less)
        if N_less < 7:
            boool.append(True)
        else:
            boool.append(False)
df['bool'] = boool

df_daily_sum = df.resample('D').sum()
df2 = df.copy()
df2['TWP'] = df2['bool']*df2['price']

df_daily_av = df2.resample('D').mean()
df_daily = df_daily_sum.copy()
df_daily['price'] = df_daily_av['TWP']

plt.plot(df_daily['price'])
plt.ylabel('$/MWh')
plt.plot([18200,19500],[183.5,183.5],'k--')
plt.legend(['simple_algo','Mercury'])

print(len(df_daily[df_daily['bool']<6]))



