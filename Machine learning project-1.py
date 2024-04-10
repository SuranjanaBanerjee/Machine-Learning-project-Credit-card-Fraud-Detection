#!/usr/bin/env python
# coding: utf-8

# In[4]:


pip install scikit-learn


# In[6]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[5]:


import os


# In[7]:


os.getcwd()


# In[8]:


os.chdir('C:\\Users\\suran\\Downloads')


# In[9]:


credit_card_data=pd.read_csv("archive (6)\creditcard.csv")


# In[10]:


credit_card_data


# In[11]:


credit_card_data.head()


# In[12]:


credit_card_data.tail()


# In[13]:


credit_card_data.info()


# ### Checking the number of missing values in each column
# 

# In[14]:


credit_card_data.isnull().sum()


# ### Distribution of legit and fraudulent transactions.
# 

# In[15]:


credit_card_data["Class"].value_counts()


# # This dataset is highly unbalanced
#  
#  
#  

# In[24]:


# 0= Normal Transaction
# 1= Fraudulent Transaction


# ### Seperating the data for Analysis
# 

# In[16]:


legit=credit_card_data[credit_card_data.Class==0]
fraud=credit_card_data[credit_card_data.Class==1]


# In[17]:


print(legit.shape)
print(fraud.shape)


# ### Statistical measured of the data

# In[18]:


legit.Amount.describe()


# In[19]:


fraud.Amount.describe()


# ### Compare the values for both transaction

# In[20]:


credit_card_data.groupby("Class").mean()


# ### Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
# 

# In[21]:


# Number of Fraudulent Transaction---> 492
legit_sample=legit.sample(n=492)


# ### Concatening Two DataFrames
# 

# In[22]:


new_data=pd.concat([legit_sample,fraud],axis=0)


# In[23]:


new_data.head()


# In[24]:


new_data.tail()


# In[25]:


new_data["Class"].value_counts()


# In[26]:


new_data.groupby("Class").mean()


# ### Splitting the data into Features and Targets

# In[27]:


x=new_data.drop(columns="Class",axis=1)
y=new_data['Class']


# In[28]:


print(x)


# In[29]:


print(y)


# ### Split the data into Training data and Testing data

# In[30]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)


# In[31]:


print(x.shape,x_train.shape,x_test.shape)


# ### Model Training

# In[33]:


model=LogisticRegression()


# In[34]:


model.fit(x_train,y_train)


# # Model Evaluation
# 
# ### Accuracy score on training data

# In[38]:


x_train_prediction=model.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)
print("Accuracy on training data:  ",training_data_accuracy)


# ### Accuracy on test data

# In[40]:


x_test_prediction=model.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)
print("Accuracy on testing data:  ",test_data_accuracy)


# In[ ]:




