#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm


# In[4]:


dataset = pd.read_csv("C:\\Users\\hp\\Desktop\\Advertising.csv")


# In[5]:


dataset


# In[6]:


# Rename the column 'Unnamed: 0' to 'Index'
dataset.rename(columns={'Unnamed: 0': 'Index'}, inplace=True)


# In[7]:


dataset


# In[8]:


#Preparing model
x = dataset.drop('Sales', axis=1)
y = dataset[["Sales"]]


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=46)


# In[16]:


# Linear Regression Model
lm = sm.OLS.from_formula("Sales ~ TV + Radio + Newspaper", data=dataset).fit()


# In[17]:


# Print the summary of the regression model
print(lm.summary())


# In[19]:


# Print the coefficients of the model

print(lm.params, "\n")


# In[20]:


# Evaluate the model

results = []
names = []
models = [('LinearRegression', LinearRegression())]


# In[21]:


# Loop through each model, fit it to the data, and calculate the RMSE

for name, model in models:
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    result = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append(result)
    names.append(name)
    msg = "%s: %f" % (name, result)
    print(msg)


# In[23]:


# Make predictions on new data

new_data = pd.DataFrame({'TV': [110], 'Radio': [60], 'Newspaper': [20]})
predicted_sales = lm.predict(new_data)
print("Predicted Sales:", predicted_sales)


# In[ ]:




