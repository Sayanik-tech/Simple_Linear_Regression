#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[6]:


dataset = pd.read_csv('Salary_Data.csv')
print(dataset)


# In[7]:


dataset.info()


# In[9]:


x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
print(x)


# In[10]:


print(y)


# In[11]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 1)


# In[ ]:


# Training the simple Linear Regression Model on Training Set


# In[13]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)


# In[ ]:


# Training the simple Linear Regression Model on Test Set


# In[ ]:


y_pred = regressor.predict(x_test)


# In[ ]:


## visualizing the train set results


# In[14]:


plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Train Data)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show


# In[ ]:


## visualizing the test set results


# In[15]:


plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary vs Experience (Test Data)')
plt.xlabel('years of Experience')
plt.ylabel('Salary')
plt.show


# In[16]:


print(regressor.coef_)
print(regressor.intercept_)


# In[ ]:


## Salary = 9332.94473799×YearsExperience+25609.89799835482


# In[17]:


## Prediction of salary of an employee with experience of 15 years
y_pred = regressor.predict([[15]])
print(y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




