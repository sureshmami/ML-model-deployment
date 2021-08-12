#
#Importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Loading the dataset
data = pd.read_csv("insurance.csv")
data


# In[274]:


#To know number of rows and columns
data.shape


# In[275]:


#Checking for any values
data.isnull().sum()


# In[276]:


#checking for unique values from each features
data.nunique()


# In[16]:


#to understand on IQR, mean and STD of continuos variables
data.describe().T


# In[30]:


#
fig,axes = plt.subplots(ncols=3, figsize = (15,6), squeeze = True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)
data.plot(kind='scatter', x='age', y='expenses', ax=axes[0])
data.plot(kind='scatter', x='children', y='expenses', ax=axes[1])
data.plot(kind='scatter', x='bmi', y='expenses', ax=axes[2])


# In[279]:


#To understand the histograms of 
fig,axes = plt.subplots(ncols= 2,nrows= 2 ,figsize = (15,6), squeeze = True)
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

data.plot(kind='hist', y='age', ax=axes[0][0], color = 'blue')
data.plot(kind='hist', y='bmi', ax=axes[0][1], color = 'orange', bins = 54)
data.plot(kind='hist', y='children', ax=axes[1][0], color = 'red', bins = 6)
data.plot(kind='hist', y='expenses', ax=axes[1][1], color = 'green', bins = 80)


# In[73]:


#Sex vs expenses
#Smokers are paying high expenses then non-smokers, more smokers are male
plt.figure(figsize=(15,6))
sns.barplot(x='sex',y='expenses',data=data, hue='smoker')
plt.title('Sex Vs Expenses')


# In[46]:


#Age Vs Expenses
#the more the age , the insurance expenses is high
plt.figure(figsize=(15,6))
sns.barplot(x='age',y='expenses',data=data)
plt.title('Age vs Expenses')


# In[66]:


#Children Vs Expenses
#the number of children is having changes in insurance expenses
plt.figure(figsize=(15,6))
sns.barplot(x='children',y='expenses',data=data)
plt.title('Children vs Expenses')


# In[71]:


# region vs charges
# Insurance charges are higher at southeast & northeast comparatively
plt.figure(figsize = (12, 8))
sns.barplot(x = 'region', y = 'expenses', data = data)
plt.title('region vs charges')


# In[123]:


#Correlation to check for multicollearnity 
corr = data.corr()
sns.heatmap(corr,annot=True,fmt='.2f', linewidths=2)


# #### Data Preprocessing

# In[80]:


# removing unnecassary columns from the dataset
data = data.drop('region', axis = 1)
print(data.shape)
data.columns


# In[86]:


#applying label encoding for sex and smoker feature
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
#Transform all females to 0 and males to 1
data['sex'] = le.fit_transform(data['sex'])
#Transform all nonsmokers to 0 and smokers to 1
data['smoker'] = le.fit_transform(data['smoker'])


# In[85]:


#Printing the count of 'sex' and 'smoker' post encoding
print(data['sex'].value_counts())
print(data['smoker'].value_counts())


# In[99]:


#Splitting the target and predictor variables

x = data.iloc[:,:5]
y = data.iloc[:,5]

print(x.shape)
print(y.shape)


# In[217]:


#splitting the data to training and test sets
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y, test_size= 0.2, random_state=30)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[280]:


#Importing StandardScaler from sklearn

from sklearn.preprocessing import StandardScaler

# creating a standard scaler
sc = StandardScaler()

# feeding independents sets into the standard scaler
X_train = sc.fit_transform(X_train)
X_test = pd.DataFrame(X_test)
X_test = sc.fit_transform(X_test)


# In[281]:


#Importing Linear Regression from Sklearn 

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
model0 = LinearRegression()
model0.fit(X_train,y_train)
y_pred = model0.predict(X_test)

MSE = np.mean((y_test - y_pred)**2,axis = None)
print('MSE:',MSE)

RMSE = np.sqrt(MSE)
print('RMSE:',RMSE)

r2 = r2_score(y_test,y_pred)
print('r2 score :',r2)


# In[222]:


#Regression Line between Predicted and Actual values with LR
fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between Predicted and Actual values with LR')
sns.regplot(x=y_pred,y=y_test,ax=ax1)


# In[252]:


##Importing SVR from Sklearn 

from sklearn.svm import SVR

# creating the model
model1 = SVR()

# feeding the training data to the model
model1.fit(X_train, y_train)

# predicting the test set results
y_pred = model1.predict(X_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)

# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

# Calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2 score :", r2)


# In[225]:


##Regression Line between Predicted and Actual values using SVR 
fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between Predicted and Actual values with SVR')
sns.regplot(x=y_pred,y=y_test,ax=ax1)


# In[253]:


#Importing RandomForestRegressor from sklearn

from sklearn.ensemble import RandomForestRegressor

# creating the model
model2 = RandomForestRegressor(n_estimators = 40, max_depth = 4, n_jobs = -1)

# feeding the training data to the model
model2.fit(X_train, y_train)

# predicting the test set results
y_pred = model2.predict(X_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)

# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

# Calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2 score :", r2)


# In[254]:


##Regression Line between Predicted and Actual values using RFR 
fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between Predicted and Actual values with RFR')
sns.regplot(x=y_pred,y=y_test,ax=ax1)


# In[255]:


print(y_pred)


# In[260]:


print(y_test)


# In[228]:


#Importing DecisionTreeRegressor from sklearn
from sklearn.tree import DecisionTreeRegressor

# creating the model
model3 = DecisionTreeRegressor()

# feeding the training data to the model
model3.fit(X_train, y_train)

# predicting the test set results
y_pred = model3.predict(X_test)

# calculating the mean squared error
mse = np.mean((y_test - y_pred)**2, axis = None)
print("MSE :", mse)

# Calculating the root mean squared error
rmse = np.sqrt(mse)
print("RMSE :", rmse)

# Calculating the r2 score
r2 = r2_score(y_test, y_pred)
print("r2 score :", r2)


# In[285]:


##Regression Line between Predicted and Actual values using DTR
fig , (ax1) = plt.subplots(1,1,figsize=(10,4))

ax1.set_title('Scatter plot between Predicted and Actual values with DTR')
sns.regplot(x=y_pred,y=y_test,ax=ax1)


# In[284]:


#importing pickle library to create a pickle file
import pickle
# Saving model to disk
filename = 'finalized_model.pkl'
pickle.dump(model2, open(filename,'wb'))
model2 = pickle.load(open('finalized_model.pkl','rb'))
print(model2.predict([[18,0,23,0,0]]))

