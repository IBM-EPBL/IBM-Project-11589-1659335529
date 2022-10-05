#!/usr/bin/env python
# coding: utf-8

# # #Importing reqired packages
# 

# In[293]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings


# # #Dataset collection 

# In[295]:


data=pd.read_csv('water_dataX.csv',encoding='ISO-8859-1',low_memory=False)


# In[296]:


data.head()


# In[297]:


data.describe()


# # # Data preprocessing

# In[224]:


data.info


# In[225]:


data.shape


# # Handling Missing Values

# In[226]:


data.isnull().any()


# In[227]:


data.isnull().sum()


# In[228]:


data.dtypes


# In[229]:


data.head()


# In[230]:


data['Temp']=pd.to_numeric(data[ 'Temp'],errors='coerce')
data['D.O. (mg/l)']=pd.to_numeric(data['D.O. (mg/l)'],errors='coerce')
data['PH']=pd.to_numeric(data['PH'], errors='coerce') 
data['B.O.D. (mg/l)']=pd.to_numeric(data['B.O.D. (mg/l)'],errors='coerce')
data['CONDUCTIVITY (µmhos/cm)']=pd.to_numeric(data['CONDUCTIVITY (µmhos/cm)'], errors='coerce')
data['NITRATENAN N+ NITRITENANN (mg/l)']=pd.to_numeric(data['NITRATENAN N+ NITRITENANN (mg/l)'],errors='coerce')
data['TOTAL COLIFORM (MPN/100ml)Mean']=pd.to_numeric(data['TOTAL COLIFORM (MPN/100ml)Mean'],errors='coerce')
data.dtypes


# In[231]:


data.isnull().sum()


# In[232]:


start = 1
end = 1779
station = data.iloc[start:end, 0]
location = data.iloc[start:end ,1]
state = data.iloc[start:end, 2]
do = data.iloc[start:end, 4].astype(np.float64)

value=0

ph = data.iloc[ start:end, 5]  
co = data.iloc [start:end, 6].astype(np.float64)   
  
year = data.iloc[start:end, 11]
tc = data.iloc[2:end, 10].astype(np.float64)


bod = data.iloc[start:end, 7].astype(np.float64)
na = data.iloc[start:end, 8].astype(np.float64)
na.dtype


# In[233]:


data.head()


# In[234]:


data['Temp'].fillna(data['Temp'].mean(), inplace=True) 
data['D.O. (mg/l)'].fillna(data['D.O. (mg/l)'].mean(), inplace=True)

data['PH'].fillna(data['PH'].mean(),inplace=True)

data['CONDUCTIVITY (µmhos/cm)'].fillna(data['CONDUCTIVITY (µmhos/cm)'].mean(), inplace=True)

data['B.O.D. (mg/l)'].fillna(data['B.O.D. (mg/l)'].mean(),inplace=True)

data['NITRATENAN N+ NITRITENANN (mg/l)'].fillna(data['NITRATENAN N+ NITRITENANN (mg/l)'].mean(), inplace=True)
data['TOTAL COLIFORM (MPN/100ml)Mean'].fillna(data['TOTAL COLIFORM (MPN/100ml)Mean'].mean(), inplace=True)


# In[235]:


data.drop(["FECAL COLIFORM (MPN/100ml)"],axis=1,inplace=True)


# In[236]:


data=data.rename(columns={'D.O. (mg/l)':'do'})
data=data.rename(columns={'CONDUCTIVITY (µmhos/cm)':'co'})
data=data.rename(columns={'B.O.D. (mg/l)':'bod'})
data=data.rename(columns={'NITRATENAN N+ NITRITENANN (mg/1)':'na'})
data=data.rename(columns={'TOTAL COLIFORM (MPN/100ml) Mean': 'tc'})
data=data.rename(columns={'STATION CODE': 'station'})
data=data.rename(columns={'LOCATIONS': 'location'})
data=data.rename(columns={'STATE': 'state'})
data=data.rename(columns={'PH':'ph'})


# In[237]:


data = pd.concat([station,location,state,do,ph,co,bod,na,tc,year], axis=1)

data.columns = ['station','location','state','do','ph','co','bod','na','tc','year']


# In[238]:


data.head()


# # Calculating Water Quality Index(WQI)

# In[239]:


#calulation of Ph
data['npH']=data.ph.apply(lambda x: (100 if (8.5>=x>=7)  
                                 else(80 if  (8.6>=x>=8.5) or (6.9>=x>=6.8) 
                                      else(60 if (8.8>=x>=8.6) or (6.8>=x>=6.7) 
                                          else(40 if (9>=x>=8.8) or (6.7>=x>=6.5)
                                              else 0)))))


# In[240]:


#calculation of dissolved oxygen
data['ndo']=data.do.apply(lambda x:(100 if (x>=6)  
                                 else(80 if  (6>=x>=5.1) 
                                      else(60 if (5>=x>=4.1)
                                          else(40 if (4>=x>=3) 
                                              else 0)))))


# In[241]:


#calculation of total coliform
data['nco']=data.tc.apply(lambda x:(100 if (5>=x>=0)  
                                 else(80 if  (50>=x>=5) 
                                      else(60 if (500>=x>=50)
                                          else(40 if (10000>=x>=500) 
                                              else 0)))))


# In[242]:


#calc of B.D.O
data['nbdo']=data.bod.apply(lambda x:(100 if (3>=x>=0)  
                                 else(80 if  (6>=x>=3) 
                                      else(60 if (80>=x>=6)
                                          else(40 if (125>=x>=80) 
                                              else 0)))))


# In[243]:


#calculation of electrical conductivity
data['nec']=data.co.apply(lambda x:(100 if (75>=x>=0)  
                                 else(80 if  (150>=x>=75) 
                                      else(60 if (225>=x>=150)
                                          else(40 if (300>=x>=225) 
                                              else 0)))))


# In[244]:


#Calulation of nitrate
data['nna']=data.na.apply(lambda x:(100 if (20>=x>=0)  
                                 else(80 if  (50>=x>=20) 
                                      else(60 if (100>=x>=50)
                                          else(40 if (200>=x>=100) 
                                              else 0)))))


# In[245]:


data.head()


# In[246]:


data.dtypes


# In[247]:


# Calculate  water quality index WQI
data['wph']=data.npH * 0.165
data['wdo']=data.ndo * 0.281
data['wbdo']=data.nbdo * 0.234
data['wec']=data.nec* 0.009
data['wna']=data.nna * 0.028
data['wco']=data.nco * 0.281


# In[248]:


data['wqi']=data.wph+data.wdo+data.wbdo+data.wec+data.wna+data.wco 


# In[249]:


data.head()


# In[250]:


#Calculating overall wqi for each year
average=data.groupby('year')['wqi'].mean()


# In[251]:


average.head()


# # Data Visualization

# In[252]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (20.0, 10.0)


# In[253]:


from mpl_toolkits.mplot3d import Axes3D


# In[254]:


year = data['year'].values
AQI = data['wqi'].values
data['wqi'] = pd.to_numeric(data['wqi'], errors = 'coerce')
data['wqi'] = pd.to_numeric(data['wqi'], errors = 'coerce')


# In[255]:


fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(year, AQI, color = 'red')
plt.show()


# In[256]:


data = data[np.isfinite(data['wqi'])]
data.head()


# In[257]:


cols = ['year']
y = data['wqi']
x = data[cols]

plt.scatter(x, y)
plt.show()


# In[258]:


#Splitting the data into dependent and independent variables
x = data.iloc[:,0:7].values


# In[259]:


x.shape


# In[260]:


y = data.iloc[:,7:].values


# In[261]:


y.shape


# # Splitting data into Train & Test

# In[270]:


from sklearn import neighbors, datasets
data = data.reset_index(level = 0, inplace = False)


# In[271]:


from sklearn import linear_model


# In[272]:


cols = ['year']


# In[273]:


y = data['wqi']
x = data[cols]


# In[274]:


reg = linear_model.LinearRegression()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 4)


# In[275]:


reg.fit(x_train, y_train)


# In[276]:


a = reg.predict(x_test)
a


# In[277]:


y_test


# In[278]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size = 0.2,random_state = 10)


# In[279]:


X_train


# In[280]:


X_test


# In[281]:


y_train


# In[282]:


y_test


# In[285]:


y_pred = reg.predict(X_test)


# # Model Evaluation

# In[286]:


from sklearn import metrics
print ('MAE:',metrics.mean_absolute_error(y_test,y_pred))


# In[287]:


print(('MSE:',metrics.mean_squared_error(y_test,y_pred)))


# In[288]:


print('RMSE:',np.sqrt(metrics.mean_squared_error(y_test,y_pred)))


# In[289]:


metrics.r2_score(y_test,y_pred)


# In[290]:


import matplotlib.pyplot as plt
data=data.set_index('year')
data.plot(figsize=(15,6))
plt.show()


# In[292]:


import pickle
pickle.dump(reg,open('wqi.pkl','wb'))
model = pickle.load(open('wqi.pkl','rb'))


# In[ ]:




