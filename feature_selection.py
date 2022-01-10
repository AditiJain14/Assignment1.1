#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy
import sklearn
import seaborn as sns


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


train_Data=pd.read_csv('/Users/aditijain/Documents/GitHub/COL341/Assignment_1/data/train_large.csv',index_col=[0])


# In[4]:


train_Data.head()


# In[5]:


train_Data.shape


# In[8]:


Y=train_Data["Total Costs"]
del train_Data["Total Costs"]
train_Data.head()


# In[7]:


for col in train_Data.columns:
    print(col)


# In[6]:


for col in train_Data.columns:
    print(train_Data[col].unique().size)


# In[8]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[7]:


sns.heatmap(train_Data.corr())


# In[7]:


for col in train_Data.columns:
    print(np.corrcoef(train_Data[col], train_Data["Total Costs"])[0,1])


# In[7]:


train_Data.head()


# In[8]:


train_Data["Length of Stay"]


# correlated sets: Operating Certification Num: Facility ID
# APR DRG Code: CCS Diagnosis Code
# APR MDC Code: CCS Diagnosis Code
# APR MDC Description : APR DRG Code
# APR MDC Description : APR MDC Code
# APR Severity of Illness Description : APR Medical Surgical Description

# In[9]:


del train_Data["Operating Certificate Number"]
del train_Data["Facility Id"]
del train_Data["Zip Code - 3 digits"]
del train_Data["Gender"]
del train_Data["Race"]
del train_Data["Ethnicity"]
del train_Data["CCS Diagnosis Code"]
del train_Data["CCS Diagnosis Description"]
del train_Data["CCS Procedure Code"]
del train_Data["CCS Procedure Description"]
del train_Data["APR DRG Code"]
del train_Data["APR Severity of Illness Code"]
del train_Data["APR DRG Description"]
del train_Data["APR MDC Code"]
del train_Data["APR Risk of Mortality"]
del train_Data["Birth Weight"]
del train_Data["Payment Typology 3"]
del train_Data["Emergency Department Indicator"]


# In[10]:


train_Data.columns


# In[11]:


X=train_Data.to_numpy()


# In[12]:


X.shape


# In[13]:


length_s=X[:,train_Data.columns.get_loc('Length of Stay')].reshape(X.shape[0],1)


# In[14]:


colmn=[]
for i in train_Data.columns:
    if i!="Length of Stay":
        colmn+=[i]
colmn


# In[42]:


#group of high correlation: Length of Stay, APR Severity of Illness Code 
#APR Severity of Illness Description
#APR Risk of Mortality 
#APR Medical Surgical Description


# In[15]:


train_Data.shape


# In[16]:


length_s


# In[17]:


one_hot_encoded_data = pd.get_dummies(train_Data, columns =colmn)


# In[18]:


one_hot_encoded_data


# In[19]:


one_hot_encoded_data.drop(columns=['Length of Stay'], axis=1, inplace=True)


# In[20]:


column_extend_names=[]


# In[21]:


one_hot_encoded_data=length_s*one_hot_encoded_data
one_hot_encoded_data.to_numpy()


# In[22]:


for col in one_hot_encoded_data.columns:
    column_extend_names+=[col]
column_extend_names+=["Length of Stay"]


# In[23]:


column_extend_names


# In[25]:


one_hot_encoded_data.shape


# In[26]:


X_t=np.c_[one_hot_encoded_data,length_s]
X_t=np.c_[X_t,np.ones(X_t.shape[0])]


# In[27]:


X_t.shape


# In[28]:


X_t[245]


# In[29]:


len(column_extend_names)


# In[30]:


X_extnd=X_t


# In[31]:


X_extnd.shape


# In[32]:


from sklearn import linear_model
reg = linear_model.LassoLars(alpha=0.001)


# In[33]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X_extnd, Y, test_size=0.1, random_state=23,shuffle=True)


# In[34]:


reg.fit(X_train,Y_train)


# In[37]:


validity=reg.score(X_train,Y_train)
print(validity)


# In[38]:


validity=reg.score(X_test,Y_test)
print(validity)


# In[35]:


validity=reg.score(X_train,Y_train)
print(validity) # with manish's 


# In[36]:


validity=reg.score(X_test,Y_test)
print(validity)


# In[63]:


validity=reg.score(X_train,Y_train)
print(validity) # with manish-payment 3


# In[64]:


validity=reg.score(X_test,Y_test)
print(validity)


# In[37]:


print(reg.coef_,reg.coef_.size)


# In[34]:


len(column_extend_names)


# In[35]:


j=0
k=[]
for i in reg.coef_:
    if(abs(i)>0):
        k+=[column_extend_names[j]]
    j=j+1


# In[36]:


len(k)


# In[37]:


k


# In[38]:


c=[]
for j in column_extend_names:
    if(j in k):
        continue
    else:
        c+=[j]


# In[39]:


c


# In[40]:


len(c)


# In[ ]:


#cross validation


# In[51]:


lambd=np.array([0.000000001,0.000001,0.00001,0.0001,0.001,0.01,0.1])


# In[ ]:


def cross_validation(lambd,X,Y,k):
    mean=np.zeros((lambd.size))
    l=0
    for j in lambd:
        for i in range(0,k):
            loss=0
            X_train, X_test, Y_train, Y_test = train_test_split(X_extnd, Y, test_size=0.1, random_state=23,shuffle=True)
            reg = linear_model.LassoLars(alpha=j)
            reg.fit(X_train,Y_train)
            validity=reg.score(X_test,Y_test)
            loss+=validity
        mean[l]=loss/k
        print(mean[l])
        l=l+1
    return mean


# In[ ]:


minimum=min(mean)
i=0
while(mean[i]!=minimum):
    i=i+1
lamb=lambd[i]

