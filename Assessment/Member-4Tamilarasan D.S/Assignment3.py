#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv("abalone.csv")


# In[4]:


sns.displot(df.Sex)


# In[5]:


df.plot.line()


# In[6]:


sns.lmplot("Diameter","Length",df,hue="Length", fit_reg=False);


# In[7]:


sns.lmplot("Diameter","Length",df,hue="Length", fit_reg=False);


# In[8]:


data = pd.read_csv("abalone.csv")
pd.isnull(data["Sex"])


# In[9]:


df["Rings"] = np.where(df["Rings"] >10, np.median,df["Rings"])
df["Rings"]


# In[10]:


pd.get_dummies(df, columns=["Sex", "Length"], prefix=["Length", "Sex"]).head()


# In[11]:


X = df.iloc[:, :-2].values
print(X)


# In[12]:


Y = df.iloc[:, -1].values
print(Y)


# In[14]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df[["Length"]] = scaler.fit_transform(df[["Length"]]) 
print(df)


# In[17]:


from sklearn.model_selection import train_test_split
train_size=0.8
X = df.drop(columns = ['Sex']).copy()
y = df['Sex']
X_train, X_rem, y_train, y_rem = train_test_split(X,y, train_size=0.8)
test_size = 0.5
X_valid, X_test, y_valid, y_test = train_test_split(X_rem,y_rem, test_size=0.5)
print(X_train.shape), print(y_train.shape)
print(X_valid.shape), print(y_valid.shape)
print(X_test.shape), print(y_test.shape)


# In[18]:


test_size = 0.33
seed = 7
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)


# In[19]:


X_train


# In[20]:


y_train


# In[21]:


X_test


# In[22]:


y_test


# In[29]:


from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import mean_squared_error 
X_train =[5, -1, 2, 10]
y_test = [3.5, -0.9, 2, 9.9]
print ('R Squared =',r2_score(X_train, y_test))
print ('MAE =',mean_absolute_error(X_train, y_test))
print ('MSE =',mean_squared_error(X_train, y_test))


# In[ ]:




