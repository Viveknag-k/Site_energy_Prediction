#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import plotly.express as px
from plotly.offline import init_notebook_mode
init_notebook_mode(connected=True) 


# In[2]:


train=pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\train_dataset.csv")
xtest=pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\x_test.csv")
ytest=pd.read_csv("C:\\Users\\Vivek Nag Kanuri\\Downloads\\y_test.csv")

train


# In[3]:


train.shape


# In[4]:


train.isnull().sum()


# In[5]:


train.info()


# In[6]:


train.dropna(inplace=True)


# In[7]:


train.isnull().sum()


# In[8]:


for i in train:
    if train[i].dtype==object:
        print(i)
        print(train[i].unique())
        print(train[i].nunique())


# In[9]:


train.describe().T


# In[10]:


Skew = train.skew()
print(Skew)


# In[11]:


px.pie(train,train['building_class'],train['site_eui'],template='ggplot2',color='building_class',hole=0.5)


# In[12]:


px.histogram(train,train['year_built'],train['site_eui'],template='ggplot2',)


# In[13]:


train_data=train.copy()


# In[14]:


train_data.head()


# In[15]:


train_data.drop(['Year_Factor','State_Factor','facility_type'],axis=1,inplace=True)


# In[16]:


train_data.head()


# In[17]:


xtest
ytest
xtest.dropna(inplace=True)
ytest.dropna(inplace=True)


# In[18]:


print(xtest.size/ytest.size)


# In[19]:


xtest.head()


# In[20]:


xtest.head()


# In[21]:


from sklearn.preprocessing import LabelEncoder
l=LabelEncoder()
for i in train_data:
    if train_data[i].dtypes==object:
        train_data[i]=l.fit_transform(train_data[i])

        


# In[22]:


train_data.head()


# In[23]:


from sklearn.model_selection import train_test_split
x=train_data.drop('site_eui',axis=1)
y=train_data['site_eui']
xtrain, xtest, ytrain, ytest= train_test_split(x,y,test_size=0.3,random_state=42)


# In[24]:


from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
lr = LinearRegression()
rf = RFE(lr, n_features_to_select=10,verbose=2)
rf.fit(xtrain,ytrain)
names=xtrain.columns.tolist()
b = rf.ranking_

a = pd.DataFrame(sorted(list(map(lambda x,y : (x,y),b,names))),columns=['rank','features'])
rfe_selected=a['features'][a['rank']==1]

rfe_selected
# In[25]:


xtrain=xtrain[rfe_selected]


# In[26]:


xtrain.head()


# In[27]:


xtest=xtest[rfe_selected]


# In[28]:


xtest.head()


# In[29]:


from sklearn.linear_model import LinearRegression
l = LinearRegression()
l.fit(xtrain,ytrain)
ypred=l.predict(xtest)

from sklearn import metrics
print('Mean Absolute Error (MAE):', round(metrics.mean_absolute_error(ytest, ypred),3))  
print('Mean Squared Error (MSE):', round(metrics.mean_squared_error(ytest, ypred),3))  
print('Root Mean Squared Error (RMSE):', round(np.sqrt(metrics.mean_squared_error(ytest, ypred)),3))
print('R2_score:', round(metrics.r2_score(ytest, ypred),6))
print('Root Mean Squared Log Error (RMSLE):', round(np.log(np.sqrt(metrics.mean_squared_error(ytest, ypred))),3))


# In[30]:


Results = pd.DataFrame({'site_eui_act':ytest, 'site_eui_pred':ypred})

# Merge two Dataframes on index of both the dataframes

ResultsFinal = train_data.merge(Results, left_index=True, right_index=True)
ResultsFinal.sample(10)


# In[31]:


px.scatter(Results,'site_eui_pred','site_eui_act',trendline='ols',trendline_color_override='blue',template='plotly_dark',title='Predicted Vs Actual Sales')


# In[ ]:





# In[32]:


import pickle 
pickle_out = open("l.pkl","wb")
pickle.dump(l,pickle_out)
pickle_out.close()


# In[ ]:




