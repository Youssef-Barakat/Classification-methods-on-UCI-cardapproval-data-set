#!/usr/bin/env python
# coding: utf-8

# # Importing modules

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.ensemble import RandomForestClassifier as RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold 
from sklearn.metrics import accuracy_score
from time import perf_counter
from scipy.stats import zscore
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import category_encoders as ce
from sklearn.ensemble import VotingClassifier
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading the data

# In[7]:


df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.data",
                names = ["Male","Age","Debt","Married","BankCustomer","EducationLevel",
                         "Ethnicity","YearsEmployed","PriorDefault","Employed","CreditScore",
                         "DriversLicense","Citizen","ZipCode","Income","Approved"]
                ,na_values=['?'])


# In[8]:


df.head()


# ## Data information

# In[9]:


#Knowing how many non unique value in each feature
for column in list(df.columns.values):
    print(f"Number of non unique values in column {column} = {df[column].nunique()}")  


# ## Removing unneeded columns

# In[10]:


df = df.loc[:, df.columns != 'ZipCode']


# ## Working on the dependent variable

# In[11]:


df["Approved"].replace("+", 1,inplace = True)
df["Approved"].replace("-", 0,inplace = True)
print(df["Approved"].value_counts())


# In[12]:


df.describe()


# In[13]:


df.info()


# ## infromation about data

# In[14]:


for col in df.select_dtypes(include=np.number).columns:
    print(f"Variance of {col} is {df[col].var()}")


# In[15]:


for col in df.select_dtypes(include=np.number).columns:
    print(f"Skewness of {col} is {df[col].skew()}")


# ## Checking for outliers

# In[16]:


#Filtering the outliers using Zscore method
def filter_zscore(df, column_name):
    print(f"====== {column_name} ======")
    print(f"Mean of values is: {df[column_name].mean()}")
    print(f"Stanndard Deviation of values is: {df[column_name].std()}\n")
    print(f"length of data before filtering is {len(df)}")        
    filtered = df[(np.abs(zscore(df[column_name], nan_policy='omit')) < 3)]
    print(f"length of data after filtering is {len(filtered)}")
    return filtered.index.tolist()

index_set = set(df.index.tolist())
for col in df.select_dtypes(include=np.number).columns:
    index_set = set(filter_zscore(df, col)).intersection(index_set)

df = df.loc[set(index_set)]
df


# ## Some plots

# In[17]:


df.boxplot(column='Age')


# In[18]:


df.boxplot(column='Debt')


# In[19]:


df.boxplot(column='YearsEmployed')


# In[20]:


counts_Approved = df["Approved"].value_counts()
fig = plt.figure()
ax = fig.add_axes([0, 0, 1, .9]) 

ax.barh(list(counts_Approved.index), list(counts_Approved), color='r',
        alpha=0.7, height=0.4)

ax.set_yticks(list(counts_Approved.index))
fig.suptitle("Approved (1 = Accepted, 0 = Rejected)")


# ## Handling missing values 

# In[21]:


#Replacing the nan values with mean value in case of numerical
imputer = SimpleImputer(strategy="mean")
df.loc[:, df.select_dtypes(include=np.number).columns] = imputer.fit_transform(df.loc[:, df.select_dtypes(include=np.number).columns]) 
df.isna().sum() 

#Replacing the nan values with mode value in case of categorical

for column in df.select_dtypes(include=object).columns:
    df[column] = df[column].fillna(value=df[column].mode()[0])


# In[ ]:


print(df.isna().sum(), "\n\n")


# ## Categorical encoding

# In[22]:


#Applying label encoding
enc = ce.OrdinalEncoder(cols=df.select_dtypes(include=object).columns.tolist())
df = enc.fit_transform(df)
df


# In[ ]:


#Checking for nan values
print(df.isna().sum(), "\n\n")
print(df['Male'].value_counts())


# ## Feature Scaling

# In[23]:


#Applying min max scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.loc[:, df.select_dtypes(include=np.number).columns] = scaler.fit_transform(df.loc[:, df.select_dtypes(include=np.number).columns]) 


# In[ ]:


print(df.isna().sum(), "\n\n")
print(df['Male'].value_counts())


# ## Determining the dependent variable

# In[24]:


labels = df.pop("Approved")


# ## Applying k-fold

# In[25]:


k = 5
kf = KFold(n_splits=k, random_state=None)
X_Train , X_Test ,Y_Train , Y_Test = train_test_split(df,labels,test_size=.25,random_state=1)


# # Trying classification models

# ## Logistic Regression

# In[26]:


Logistic = LogisticRegression()
#Applying cross validation across the training data
total_acc = []
for train_index , val_index in kf.split(X_Train):
    X_train , X_val = X_Train.iloc[train_index,:],X_Train.iloc[val_index,:]
    Y_train , Y_val = Y_Train.iloc[train_index] , Y_Train.iloc[val_index]
    Logistic.fit(X_train, Y_train)
    Y_pred = Logistic.predict(X_val)
    acc = accuracy_score(Y_pred , Y_val)
    total_acc.append(acc)    
avg_acc = sum(total_acc)/5

print(avg_acc)
print(Logistic.score(X_Test , Y_Test))


# ## KNN

# In[27]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_Train, Y_Train)
print(knn.score(X_Test , Y_Test))


# ## Support vector classifier

# In[28]:


model = SVC(kernel='rbf', degree=3, C=1E6)
model.fit(X_Train, Y_Train)
print(model.score(X_Test , Y_Test))


# ## Decision trees

# In[29]:


hp = {
    'max_depth': 5,
    'min_samples_split': 10
}
DTC = DecisionTreeClassifier(**hp)
DTC.fit(X_Train,Y_Train)
print(DTC.score(X_Test , Y_Test))


# ## Random forests

# In[30]:


RFC = RandomForestClassifier(random_state=0)
total_acc = []
for train_index , val_index in kf.split(X_Train):
    X_train , X_val = X_Train.iloc[train_index,:],X_Train.iloc[val_index,:]
    Y_train , Y_val = Y_Train.iloc[train_index] , Y_Train.iloc[val_index]
    RFC.fit(X_train, Y_train)
    Y_pred = RFC.predict(X_val)
    acc = accuracy_score(Y_pred , Y_val)
    total_acc.append(acc)    
avg_acc = sum(total_acc)/5
RFC.fit(X_Train,Y_Train)
print(avg_acc)
print(RFC.score(X_Test , Y_Test))


# ## Knowing feature importances

# In[31]:


plt.figure(figsize=(15,10))
pd.Series(RFC.feature_importances_ , index = X_Train.columns).plot(kind="bar",color=['green'])


# ## Removing useless columns

# In[16]:


df = df.drop(['Male','Citizen','Married','BankCustomer','DriversLicense'],axis=1)


# In[32]:


X_Train , X_Test ,Y_Train , Y_Test = train_test_split(df,labels,test_size=.25,random_state=1)


# ## Grid searching to get the best params for random forests

# In[ ]:


params = dict()
params['n_estimators'] = list(range(1,15))
params['max_features'] = [0.6 , 0.7 , 0.8 , 0.9]
params['max_depth'] = list(range(1,10))
params['min_samples_split'] = list(range(10,30))
start = perf_counter()

gs = GridSearchCV(RandomForestClassifier(),param_grid = params , n_jobs = -1)
gs.fit(X_Train,Y_Train)
end = perf_counter()
print("Time = " , end-start)
print("Best score = ", gs.best_score_)
print("Best Hyperparameters = " , gs.best_params_)


# In[33]:


RFC2 = RandomForestClassifier(n_estimators=8, max_features=0.7, max_depth=9, min_samples_split=14)

total_acc = []
for train_index , val_index in kf.split(X_Train):
    X_train , X_val = X_Train.iloc[train_index,:],X_Train.iloc[val_index,:]
    Y_train , Y_val = Y_Train.iloc[train_index] , Y_Train.iloc[val_index]
    RFC2.fit(X_train, Y_train)
    Y_pred = RFC2.predict(X_val)
    acc = accuracy_score(Y_pred , Y_val)
    total_acc.append(acc)    
avg_acc = sum(total_acc)/5
print(avg_acc)
print(RFC2.score(X_Test , Y_Test))


# In[34]:


Logistic2 = LogisticRegression()

total_acc = []
for train_index , val_index in kf.split(X_Train):
    X_train , X_val = X_Train.iloc[train_index,:],X_Train.iloc[val_index,:]
    Y_train , Y_val = Y_Train.iloc[train_index] , Y_Train.iloc[val_index]
    Logistic2.fit(X_train, Y_train)
    Y_pred = Logistic2.predict(X_val)
    acc = accuracy_score(Y_pred , Y_val)
    total_acc.append(acc)    
avg_acc = sum(total_acc)/5

print(avg_acc)
print(Logistic2.score(X_Test , Y_Test))


# ## Grid searching to get the best params for KNN

# In[ ]:


params = dict()
params['n_neighbors'] = list(range(1,15))
params['metric'] = ['euclidean','manhattan','minkowski']
params['weights'] = ['unifrom','distance']

start = perf_counter()

gs = GridSearchCV(KNeighborsClassifier(),param_grid = params , n_jobs = -1)
gs.fit(X_Train,Y_Train)
end = perf_counter()
print("Time = " , end-start)
print("Best score = ", gs.best_score_)
print("Best Hyperparameters = " , gs.best_params_)


# In[35]:


knn2 = KNeighborsClassifier(n_neighbors=13,metric='manhattan',weights='distance')

total_acc = []
for train_index , val_index in kf.split(X_Train):
    X_train , X_val = X_Train.iloc[train_index,:],X_Train.iloc[val_index,:]
    Y_train , Y_val = Y_Train.iloc[train_index] , Y_Train.iloc[val_index]
    knn2.fit(X_train, Y_train)
    Y_pred = knn2.predict(X_val)
    acc = accuracy_score(Y_pred , Y_val)
    total_acc.append(acc)   
    
avg_acc = sum(total_acc)/5
print(avg_acc)
print(knn2.score(X_Test , Y_Test))


# ## Applying ensembling voting classifier between the best models

# In[37]:


eclf = VotingClassifier(estimators=[('rf', RFC2), ('log', Logistic2),
                                    ('knn', knn2)],
                        voting='soft', weights=[1, 1, 1])
eclf.fit(X_Train, Y_Train)
print(eclf.score(X_Test , Y_Test))


# In[ ]:




