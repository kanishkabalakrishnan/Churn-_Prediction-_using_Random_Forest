#!/usr/bin/env python
# coding: utf-8

# # Random Forest

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('CDR-Call-Details.csv')#load the dataset


# In[4]:


df


# In[5]:


df.head()


# In[6]:


df.info()


# ### Data Cleaning Process

# In[8]:


df.isna().sum()


# In[9]:


df.duplicated()


# In[10]:


df.duplicated().sum()


# In[11]:


df_remove_duplicates=df.drop_duplicates()
df_remove_duplicates


# In[12]:


original_row_count = df.shape[0]
print("Original number of rows:", original_row_count)


# In[13]:


duplicate_rows = df_remove_duplicates[df_remove_duplicates.duplicated()]
print("Number of remaining duplicate rows:", duplicate_rows.shape[0])


# In[14]:


removed_rows = original_row_count - df_remove_duplicates.shape[0]
print("Number of rows removed:", removed_rows)


# In[15]:


print("no.of rows in dataframe after removing:",df_remove_duplicates.shape[0])


# In[16]:


df_remove_duplicates.describe()#describe before removing outlier


# In[17]:


import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df_remove_duplicates['Churn']=le.fit_transform(df_remove_duplicates.Churn)


# In[18]:


df1=df_remove_duplicates.drop('Phone Number',axis=1)
df1.sample(20)


# In[19]:


df1.quantile(0.25)


# In[20]:


df1.quantile(0.75)


# In[21]:


numeric_cols=df1.select_dtypes(include=['number']).columns
numeric_cols


# In[22]:


# Assuming df1 is your DataFrame after removing duplicates
# and numeric_cols contains the numeric column names
 # Replace with your actual numeric columns

# Function to remove outliers based on IQR
def remove_outliers_iqr(df1, numeric_cols):
    df_cleaned = df1.copy()
    for col in numeric_cols:
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        # Define the outlier range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Filter out the outliers
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]
    return df_cleaned

# Remove outliers from df1 and store it in a new DataFrame
df_cleaned = remove_outliers_iqr(df1, numeric_cols)

# Save the cleaned DataFrame to a new CSV file if needed
df_cleaned.to_csv('df_cleaned.csv', index=False)

# Print the shape of the cleaned DataFrame to verify
print("Original DataFrame shape:", df1.shape)
print("Cleaned DataFrame shape:", df_cleaned.shape)


# In[23]:


data=df_cleaned


# In[78]:


data.describe()#describe after removing outlier


# In[24]:


data.info()


# In[25]:


import pandas as pd

# Assuming df1 is your DataFrame
# Replace spaces with underscores in column names
data.columns = data.columns.str.replace(' ', '_')

# Verify the updated column names
print(data.columns)


# In[26]:


import matplotlib.pyplot as plt
# List of feature pairs 
feature_pairs = [
    ('Day_Mins', 'Day_Charge'),
    ('Eve_Mins', 'Eve_Charge'),
    ('Night_Mins', 'Night_Charge'),
    ('Intl_Mins', 'Intl_Charge'),
    # ...
]

# scatter plots for each pair
for x, y in feature_pairs:
    plt.figure(figsize=(8, 6))
    plt.scatter(data[x], data[y], alpha=0.5)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(f'Scatter Plot of {x} vs {y}')
    plt.show()


# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# In[28]:


x=df1.drop('Churn',axis=1)
y=df1['Churn']


# In[29]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[30]:


model=RandomForestClassifier()


# In[31]:


model.fit(x_train,y_train)


# In[32]:


model.score(x_test,y_test)


# In[33]:


accuracy = round(model.score(x_test, y_test) * 100, 2)
print("Accuracy of the model is:", accuracy, "%")


# In[34]:


import warnings
warnings.filterwarnings('ignore')
model.predict([['222','0','262.2','116','211.22','222.11','112','211.090','229.20','2112','22.220','22.2','2','11.112','2']])


# In[35]:


import warnings
warnings.filterwarnings('ignore')
model.predict([['96','20','2011.6','90','211.26','206.20','2112','22.112','226.11','2211','11.611','211.2','2','11.620','0']])


# In[36]:


from sklearn.metrics import confusion_matrix


# In[37]:


y_predicted=model.predict(x_test)
cm=confusion_matrix(y_test,y_predicted)
cm


# In[38]:


import seaborn as sns
plt.figure(figsize=(8,8))
sns.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()


# TN , FP 
# 
# FN , TP 
# 
# True Negatives (TN): 10460
# 
# Correctly predicted as not churned (0).
# False Positives (FP): 234
# 
# Incorrectly predicted as churned (1) when they did not churn (0).
# False Negatives (FN): 500
# 
# Incorrectly predicted as not churned (0) when they actually churned (1).
# True Positives (TP): 895
# 

# In[ ]:




