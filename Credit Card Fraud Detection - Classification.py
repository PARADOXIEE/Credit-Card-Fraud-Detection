#!/usr/bin/env python
# coding: utf-8

# ## Dataset Information
# 
# The dataset contains transactions made by credit cards in September 2013 by European cardholders.
# This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.
# 
# It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.
# 
# Given the class imbalance ratio, we recommend measuring the accuracy using the Area Under the Precision-Recall Curve (AUPRC). Confusion matrix accuracy is not meaningful for unbalanced classification.

# ## Import modules

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# pandas - used to perform data manipulation and analysis.
# 
# numpy - used to perform a wide variety of mathematical operations on arrays.
# 
# matplotlib - used for data visualization.
# 
# seaborn - built on top of matplotlib with similar functionalities.
# 
# warnings - to manipulate warnings details filterwarnings('ignore') is to ignore the warnings thrown by the modules (gives clean results).
# 
# %matplotlib - to enable the inline plotting.

# ## Loading the dataset

# In[2]:


df = pd.read_csv('creditcard.csv')
df.head()


# Time attributes are in the terms of timestamp.
# 
# The class attribute shows 0 indicates non fraudulent transactions and 1 indicates fraudulent transactions.
# 
# The amount is in EUROs.

# In[3]:


# statistical info
df.describe()


# Due to the sizeable difference in the range of column (min) & (max), we need to run a standard scalar transformation later.

# In[4]:


# datatype info
df.info()


# There are no NULL values present in the dataset.
# 
# If any NULL values are present, we have to fill all the NULL values before proceeding to model training.

# ## Preprocessing the dataset

# In[5]:


# check for null values
df.isnull().sum()


# There are no NULL values present in the dataset.
# 
# If any NULL values are present, we have to fill all the NULL values before proceeding to model training.

# ## Exploratory Data Analysis

# In[6]:


sns.countplot(df['Class'])


# The number of fraudulent classes is low.
# 
# Hence, we need to balance the data for reasonable results.

# # To display all the 28 PCA columns, we need to run a loop¶

# In[10]:


df_temp = df.drop(columns=['Time', 'Amount', 'Class'], axis=1)

# create dist plots
fig, ax = plt.subplots(ncols=4, nrows=7, figsize=(20, 50))
index = 0
ax = ax.flatten()

for col in df_temp.columns:
    sns.distplot(df_temp[col], ax=ax[index])
    index += 1
plt.tight_layout(pad=0.5, w_pad=0.5, h_pad=5)


# Most columns show a bell curve (Normal Distribution), which does not require processing.
# 
# If we have to deal with the basic algorithmic models (like Logistic Regression), we can also pass these data into Standard Scalar.

# # Let us explore the column "Time".

# In[11]:


sns.distplot(df['Time'])


# # To display the column "Amount"

# In[12]:


sns.distplot(df['Amount'])


# Due to the irregularities in range, we need to pass these data into Standard Scalar.
# Before that, we can have a look at Correlation Matrix.

# # Correlation Matrix Analysis
# 
# 

# The correlation matrix is insignificant because of the lack of meaningful information. All the columns containing random pieces of information is dynamically reduced using PCA transformation.

# In[14]:


corr = df.corr()
plt.figure(figsize=(30,40))
sns.heatmap(corr, annot=True, cmap='coolwarm')


# We can observe a correlation with respect to Amount.
# 
# Usually, we had to drop a few columns in the correlation matrix. However, in this specific project, we can skip it.

# ## Input Split

# In[15]:


X = df.drop(columns=['Class'], axis=1)
y = df['Class']


# Store the input attributes in variable X and output attribute in variable y

# ## Standard Scaling

# In[16]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_scaler = sc.fit_transform(X)


# All Input attributes are in the X and y contains the output Class.
# 
# After running the code, we can see an array with a scaled value ranging from 0-1.
# 
# To understand the process, please go through the formula of Standard Scalar

# # Model Training and Testing

# In[18]:


x_scaler[-1]


# We have to use stratify to uniformly distribute class variables (Because the class is not balanced).

# ## Model Training

# In[23]:


# train test split
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.25, random_state=42, stratify=y)


# We have to use stratify to uniformly distribute class variables (Because the class is not balanced).

# # Logistic Regression:

# In[25]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# training
model.fit(x_train, y_train)
# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# Here, we can observe accuracy as 100% (Because of the Standard Scaling). However, the majority of the accuracy is based on Non-Fraudulent samples.
# 
# F1-Score is a combination of Precision and Recall.
# 
# Since the F1 score is around 72%, we have to consider a better Model for training.

# # Random Forest

# In[26]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
# training
model.fit(x_train, y_train)
# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# After running the code, we have to wait longer than usual due to the larger number of Data-Set values.
# 
# Now the F1-Score has improved.
# 
# Due to unbalanced training, we are observing a low score.
# 
# Let us try one boosting model.

# # XGBoost

# In[37]:


from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)
# training
model.fit(x_train, y_train)
# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# We can observe an F1-Score of 86%, which is a good result.
# 
# However, let us try to balance this data and see if the results improve in terms of F1-Score and Macro Average

# ## Class Imbalancement

# In[28]:


sns.countplot(y_train)


# In[29]:


# hint - use combination of over sampling and under sampling
# balance the class with equal distribution
from imblearn.over_sampling import SMOTE
over_sample = SMOTE()
x_smote, y_smote = over_sample.fit_resample(x_train, y_train)


# We can use Random Under_Sampling to reduce the data and Random Over_Sampling for increasing the data.
# 
# The use of these balancing methods will result in good values. Now the sample is equally distributed, the model will give weightage for both of these classes.

# In[30]:


sns.countplot(y_smote)


# # Logistic Regression

# In[33]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
# training
model.fit(x_smote, y_smote)
# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# Due to a large number of samples, the Logistic Regression does not show good results.
# 
# Let us try with Random Forest.

# # Random Forest

# In[34]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_jobs=-1)
# training
model.fit(x_smote, y_smote)
# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# There is no significant change in F1-Score after Imbalancement by Random Forest
# 
# It may be due to the complex algorithm of Random Forest.

# # XGBoost

# In[35]:


from xgboost import XGBClassifier
model = XGBClassifier(n_jobs=-1)
# training
model.fit(x_smote, y_smote)
# testing
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))
print("F1 Score:",f1_score(y_test, y_pred))


# The F1-Score as well as Macro average decline after the Class Imbalancement for XGBoost.
# 
# 

# # Final Thoughts
# We can finalize this project with the model which shows Higher results. 
# 
# Without balancing the classes, we are observing good results with XGB Classifier.
# 
# Out of all the three models we have used the XGBoost gives the best result.
# 
# You can also use Log Transformation for other attributes like Time and Amount.
