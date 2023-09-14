#!/usr/bin/env python
# coding: utf-8

# # credit
# 
# Use the "Run" button to execute the code.

# In[1]:


print('Hello World')


# In[2]:


#importing libary 


# In[3]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[4]:


from sklearn.linear_model import LogisticRegression 


# In[5]:


from sklearn.metrics import accuracy_score


# In[6]:


#loading data to pandas dataframe


# In[7]:


credit_card_data=pd.read_csv('creditcard.csv')


# In[8]:


credit_card_data.head()


# In[9]:


credit_card_data.tail()


# In[10]:


#dataset information 


# In[11]:


credit_card_data.info()


# In[12]:


# checking no of missing values in each coloumn 


# In[13]:


credit_card_data.isnull().sum()


# In[14]:


# distribution of legit tranascation and fradulent transactions


# In[15]:


credit_card_data['Class'].value_counts()


# In[16]:


#this dataset is highly unbalanced 
#0->represents normal tranascation 
#1->represents fradulent tranasaction 


# In[17]:


#seprating data for analysis 


# In[18]:


legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[19]:


print(legit.shape)
print(fraud.shape)


# In[20]:


#statistical measures of data 


# In[21]:


legit.Amount.describe()


# In[22]:


fraud.Amount.describe()


# In[23]:


#compare values of both of tranasactions 


# In[24]:


credit_card_data.groupby('Class').mean()


# In[25]:


# we will take sample of data from orginial data containing similar distributions of normal tranasactions and fradulent tranasactions


# In[26]:


legit_sample = legit.sample(n=492)


# In[27]:


# concate two dataframes 


# In[28]:


new_dataset = pd.concat([legit_sample,fraud],axis=0)


# In[29]:


new_dataset.head()


# In[30]:


new_dataset.tail()


# In[31]:


new_dataset['Class'].value_counts()


# In[32]:


new_dataset.groupby('Class').mean()


# In[33]:


#spliting data into features and targets 


# In[34]:


X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']


# In[35]:


print(X)


# In[36]:


print(Y)


# In[37]:


#split data into training data and testing data 


# In[38]:


X_train , X_test , Y_train , Y_test = train_test_split(X, Y ,test_size=0.2, stratify=Y ,random_state= 2)


# In[39]:


print(X.shape,X_train.shape,X_test.shape)


# In[40]:


# model training - logistic regression 


# In[41]:


model = LogisticRegression()


# In[42]:


#train logistic regression model with training data 


# In[43]:


model.fit(X_train , Y_train)


# In[44]:


#evaluate model on accuracy score


# In[51]:


X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[52]:


print('Accuracy on training data:' , training_data_accuracy)


# In[55]:


X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction , Y_test)


# In[56]:


print('accuracy score on test data:' ,test_data_accuracy)


# In[ ]:




