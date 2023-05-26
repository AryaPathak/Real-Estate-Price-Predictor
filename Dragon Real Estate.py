#!/usr/bin/env python
# coding: utf-8

# ## Dragon RealEstate Price Prediction

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


#For Plotting Histogram
# import matplotlib.pyplot as plt
# housing.hist(bins=50, figsize=(20, 15))


# ## Train Test Splitting

# In[8]:


#For Learing

# import numpy as np
# def split_train_test(data, test_ratio):
#     np.random.seed(42)
#     shuffled = np.random.permutation(len(data))
#     print(shuffled)
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]
# train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[9]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set:{len(test_set)}\n")


# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[11]:


strat_train_set


# # Looking for Corelations

# In[12]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV", "RM", "ZN", "LSTAT"]
#scatter_matrix(housing[attributes], figsize = (12,8))


# In[13]:


#housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.9)


# In[14]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[15]:


housing["TAXRM"]


# In[16]:


# corr_matrix = housing.corr()
# corr_matrix['MEDV'].sort_values(ascending=False)


# # Scikit-Learn Design

# # Creating Pipeline
# 

# # Feature Scaling

# In[17]:


housing = strat_train_set.copy()


# In[18]:


housing["TAXRM"] = housing['TAX']/housing['RM']


# In[19]:


#housing  = strat_train_set.drop("MEDV", axis=1)
housing_lebels = strat_train_set["MEDV"].copy()


# In[20]:


housing


# In[ ]:





# In[21]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scalar', StandardScaler())
])


# In[22]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[23]:


housing_num_tr.shape


# # Selecting a desired model for Dragon Real Estate

# In[87]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

#model = LinearRegression()
model = DecisionTreeRegressor()
#model = RandomForestRegressor()

model.fit(housing_num_tr, housing_lebels)


# In[75]:


some_data = housing.iloc[:5]


# In[76]:


some_lebels = housing_lebels.iloc[:5]


# In[77]:


prepared_data = my_pipeline.transform(some_data)


# In[78]:


model.predict(prepared_data)


# In[79]:


list(some_lebels)


# # Evaluating the Model

# In[80]:


from sklearn.metrics import mean_squared_error
import numpy as np
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_lebels, housing_predictions)
rmse = np.sqrt(mse)


# In[81]:


rmse


# # Using Better evaluation technique - Cross Validation

# In[82]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_lebels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)


# In[83]:


rmse_scores


# In[84]:


def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())


# In[85]:


print_scores(rmse_scores)


# In[ ]:





# In[ ]:




