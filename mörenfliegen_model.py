#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 15:20:17 2024

@author: philipprutimann
"""


import pandas as pd
import numpy as np
from datetime import datetime
from matplotlib import pyplot



df = pd.read_excel('Möhrenfliegendaten2006_2023.xlsx')
df_new=df[["datum","Mittelwert","Median"]].copy()
df_new["week"]=[x.date().isocalendar().week for x in list(df_new.datum)]
df_new["year"]=[x.date().isocalendar().year for x in list(df_new.datum)]
df_new["key"]=[str(str(x) + str(y)) for x,y in zip(list(df_new["year"]),list(df_new["week"]))]
df_new["key_1lag"]=[str(int(x)-1) for x in list(df_new["key"])]
df_new["key_2lag"]=[str(int(x)-2) for x in list(df_new["key"])]
df_new["class"]=[int(x>0) for x in list(df_new.Mittelwert)]
df_new.head()



df_new.groupby('class').count()


# df_wetter = pd.read_excel('Temperatur_Niederschlag_Muri_Days.ods')
# df_wetter["datum1"]=pd.to_datetime(df_wetter.Datum)
# df_wetter["datetime"]=[x.date() for x in list(df_wetter.datum1)]
# df_wetter["week"]=[x.date().isocalendar().week for x in list(df_wetter.datum1)]
# df_wetter["year"]=[x.date().isocalendar().year for x in list(df_wetter.datum1)]
# #df_wetter["temp"]=pd.to_numeric(df_wetter['MURI - Temperatur Durchschnitt +2 m (Â°C)'], errors='ignore')
# df_wetter["temp"]=df_wetter['MURI - Temperatur Durchschnitt +2 m (Â°C)']
# #df_wetter["rain"]=pd.to_numeric(df_wetter['MURI - Niederschlag (mm oder Liter/m2)'], errors='ignore')
# df_wetter["rain"]=df_wetter['MURI - Niederschlag (mm oder Liter/m2)']
# df_wetter["key"]=[str(x) + str(y) for x,y in zip(list(df_wetter["year"]),list(df_wetter["week"]))]
# df_wetter=df_wetter[['key','temp','rain']]
# #df_wetter=df_wetter[['key','MURI - Niederschlag (mm oder Liter/m2)']]
# df_wetter.head()



df_wohlen = pd.read_excel('Wetterdaten_Wohlen.xlsx')
df_wohlen["datum1"]=pd.to_datetime(df_wohlen.Datum)
df_wohlen["datetime"]=[x.date() for x in list(df_wohlen.datum1)]
df_wohlen["week"]=[x.date().isocalendar().week for x in list(df_wohlen.datum1)]
df_wohlen["year"]=[x.date().isocalendar().year for x in list(df_wohlen.datum1)]
df_wohlen["tempmax_wohlen"]=df_wohlen['Tmax (°C)']
df_wohlen["tempmin_wohlen"]=df_wohlen['Tmin (°C)']
df_wohlen["tempavg_wohlen"]=df_wohlen['Mittelwert Temp']
df_wohlen["rainsum_wohlen"]=df_wohlen['NStag (mm)']
df_wohlen["windavg_wohlen"]=df_wohlen['Wmax (km/h)']
df_wohlen["key"]=[str(x) + str(y) for x,y in zip(list(df_wohlen["year"]),list(df_wohlen["week"]))]
df_wohlen.head()


# df_group_wetter=df_wetter.groupby("key").mean()
# df_group_wetter= df_group_wetter.reset_index()
# df_group_wetter["key_lag_1week"]=[str(int(x)-1) for x in list(df_group_wetter["key"])]
# df_group_wetter.head()


df_boden_in = pd.read_excel('Temp_Boden.xlsx')
df_boden=df_boden_in.copy()
df_boden["soiltemp_sum"]=df_boden['GradSummeKW']
df_boden["soiltemp_avg"]=df_boden['GRAENICHEN - Temperatur Durchschnitt -10 cm (°C)']
df_boden["datum1"]=pd.to_datetime(df_boden.Datum)
df_boden["datetime"]=[x.date() for x in list(df_boden.datum1)]
df_boden["week"]=[x.date().isocalendar().week for x in list(df_boden.datum1)]
df_boden["year"]=[x.date().isocalendar().year for x in list(df_boden.datum1)]
df_boden["key"]=[str(x) + str(y) for x,y in zip(list(df_boden["year"]),list(df_boden["week"]))]
df_boden.head()


df_lag=df_new[['key','Mittelwert']].copy()
df_total=pd.merge(df_new, df_lag, how="left",  left_on='key_1lag', right_on='key')
df_total=pd.merge(df_total, df_wohlen, how="left", left_on='key_x', right_on='key')
df_total=pd.merge(df_total, df_boden, how="left", left_on='key_x', right_on='key',suffixes=('', '_zz'))
df_total["trend"] = range(1,467)
df_total['Y']=np.sqrt(df_total.Median)
df_total.head()


df_total.describe()


pyplot.plot('datum', 'Mittelwert_x', data=df_total)


pyplot.plot('datum', 'Y', data=df_total)


fig, ax = pyplot.subplots()
for i in range(2006,2023,5):
    df_zw=df_total[['week_x','Mittelwert_x']][df_total.year_x==i]
    ax.plot('week_x', 'Mittelwert_x', data= df_zw,label=i)
ax.legend()
pyplot.show()


df_total.shape


df_total_train = df_total[1:401]
df_total_test = df_total[401:466]
Xtrain = np.array(df_total_train[['tempmax_wohlen','tempmin_wohlen','tempavg_wohlen','rainsum_wohlen','windavg_wohlen','trend','week_x','soiltemp_sum','soiltemp_avg']])
Ytrain = np.array(df_total_train['Y'])
Xtest = np.array(df_total_test[['tempmax_wohlen','tempmin_wohlen','tempavg_wohlen','rainsum_wohlen','windavg_wohlen','trend','week_x','soiltemp_sum','soiltemp_avg']])
Ytest = np.array(df_total_test['Y'])

from sklearn.ensemble import RandomForestRegressor
clf = RandomForestRegressor(n_estimators=1000)
clf = clf.fit(Xtrain, Ytrain)


from sklearn.metrics import mean_squared_error, explained_variance_score
Ypredict=clf.predict(Xtest)
df_zz=pd.DataFrame(Ypredict)
df_total_predict=df_total_test.copy()
df_total_predict = pd.concat([df_total_predict, df_zz], ignore_index=True, axis=0)
print(f'Standartfehler: {(mean_squared_error(Ytest**2,Ypredict**2))}')
print(f'Erklährte Variabilität: {explained_variance_score(Ytest**2, Ypredict**2, force_finite=False)}')


importances = clf.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)


feature_names=df_total_train[['tempmax_wohlen','tempmin_wohlen','tempavg_wohlen','rainsum_wohlen','windavg_wohlen','trend','week_x','soiltemp_sum','soiltemp_avg']].columns
forest_importances = pd.Series(importances, index=feature_names)

fig, ax = pyplot.subplots()
#forest_importances.plot.bar(yerr=std, ax=ax)
forest_importances.plot.bar(ax=ax)
ax.set_title("Feature importances Möhrenfliegen Model")
ax.set_ylabel("Variable Importance")
fig.tight_layout()



import pickle

# save the iris classification model as a pickle file
model_pkl_file = "pest_möhrenfliege_model.pkl"  

with open(f'/Users/philipprutimann/Documents/Open Farming Hackdays/pest_monitoring/{model_pkl_file}', 'wb') as file:  
    pickle.dump(clf, file)


from sklearn.ensemble import GradientBoostingRegressor
est = GradientBoostingRegressor().fit(Xtrain, Ytrain)
Ypredict_boost=est.predict(Xtest)
est.score(Xtest, Ytest)
print(f'Standartfehler: {mean_squared_error(Ytest,Ypredict_boost)}')
print(f'Erklährte Variabilität: {explained_variance_score(Ytest, Ypredict_boost, force_finite=False)}')


from sklearn.ensemble import RandomForestClassifier
df_total_train = df_total[1:401]
df_total_test = df_total[401:466]
Xtrain = np.array(df_total_train[['tempmax_wohlen','tempmin_wohlen','tempavg_wohlen','rainsum_wohlen','windavg_wohlen','trend','week_x','soiltemp_sum','soiltemp_avg']])
Ytrain = np.array(df_total_train['class'])
Xtest = np.array(df_total_test[['tempmax_wohlen','tempmin_wohlen','tempavg_wohlen','rainsum_wohlen','windavg_wohlen','trend','week_x','soiltemp_sum','soiltemp_avg']])
Ytest = np.array(df_total_test['class'])
clf = RandomForestClassifier(n_estimators=1000)
clf = clf.fit(Xtrain, Ytrain)


from sklearn.metrics import accuracy_score, balanced_accuracy_score
Ypredict=clf.predict(Xtest)
print(f'Standartfehler: {accuracy_score(Ytest,Ypredict)}')
print(f'Erklährte Variabilität: {balanced_accuracy_score(Ytest, Ypredict)}')


