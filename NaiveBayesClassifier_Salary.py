#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
train=pd.read_csv('SalaryData_Train.csv')
train.head()


# In[2]:


train.shape


# In[3]:


train.isnull().any(axis=1)


# In[4]:


train.isnull().any().sum()


# In[5]:


test=pd.read_csv('SalaryData_Test.csv')
test.head()


# In[6]:


test.shape


# In[9]:


test.isnull().any(axis=1)


# In[10]:


test.isnull().any().sum()


# # Using Gaussian Naive Bayes

# In[11]:


from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


# In[12]:


string_columns=["workclass","education","maritalstatus","occupation","relationship","race","sex","native"]

from sklearn import preprocessing
number = preprocessing.LabelEncoder()
for i in string_columns:
    train[i] = number.fit_transform(train[i])
    test[i] = number.fit_transform(test[i])


# In[13]:


colnames = train.columns


# In[14]:


colnames


# In[18]:


len(colnamefrom sklearn.naive_bayes import MultinomialNB


# In[19]:


trainX = train[colnames[0:13]]
trainY = train[colnames[13]]
testX  = test[colnames[0:13]]
testY  = test[colnames[13]]


# In[21]:


sgnb = GaussianNB()
spred_gnb = sgnb.fit(trainX,trainY).predict(testX)


# In[22]:


confusion_matrix(testY,spred_gnb)


# In[30]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(sgnb,testX,testY)


# In[23]:


print ("Accuracy=",(10759+1209)/(10759+601+2491+1209)) 


# In[ ]:





# # Using Multinomial Naive Bayes Classifier

# In[26]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
smnb = MultinomialNB()


# In[27]:


spred_mnb = smnb.fit(trainX,trainY).predict(testX)


# In[28]:


confusion_matrix(testY,spred_mnb)


# In[31]:


from sklearn.metrics import plot_confusion_matrix
plot_confusion_matrix(smnb,testX,testY)


# In[29]:


print ("Accuracy=",(10891+780)/(10891+469+2920+780)) 


# In[ ]:




