#!/usr/bin/env python
# coding: utf-8

# # Name:Denster Joseph Frank   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp;   Student ID: s3894695                                                    

# ##  Importing all required Libraries 

# In[1]:


#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy


# ##  To read given csv file 

# In[2]:


# To read each individual dataframe
df_life=pd.read_csv("Data_Set.csv", delimiter=',')


# ## To check First Five Element

# In[3]:


# To see first 5 items of the dataframe
df_life.head()


# Working on copy of data,  original data will be safe

# In[4]:


# Working on copy of data,  original data will be safe
df_life_test=df_life.copy()


# In[5]:


# To find shape of dataframe
df_life_test.shape


# #### This data consist of 2071 rows and 24 columns.

# # Data Cleaning

# In[6]:


#To check is there any null value
df_life_test.isnull().values.any()


#  The above code checks is there any null value in code, as we can see its gives as false that means that there are no null values in data

# In[7]:


#To check percentage of null values in each column
df_life_test.isnull().sum()*100/len(df_life_test)


# To check percentage of null values in each column and we can see that all columns shows 0.0 % that means that there are no null values in our dataset

# In[8]:


# Checking for duplicate rows
df_life_test.duplicated().any()


#  The code above shows that is there any duplicated rows in our dataset, as we can see that it gives false that means that there are no duplicat rows in our dataset

# In[9]:


#To check datatypes of given datafram
df_life_test.info()


# #### All attributes consist of int or float as datatypes.

#  As there are no missing values and duplicat values we can  move further to do Exploratory data analysis of dataset(EDA)

# # Random splitting

# In[10]:


from sklearn.model_selection import train_test_split

with pd.option_context('mode.chained_assignment', None):
    df_life_TrainFrame, df_life_TestFrame = train_test_split(df_life_test, test_size=0.2, shuffle=True)


# In[11]:


print("Nunber of instances in the original dataset is {}. After spliting Train has {} instances and test has {} instances."
      .format(df_life_test.shape[0], df_life_TrainFrame.shape[0], df_life_TestFrame.shape[0]))


# In[12]:


df_life_TrainFrame.drop("ID",axis=1,inplace=True)


# In[13]:


df_life_TrainFrame.shape


# In[14]:


df_id=df_life_TestFrame['ID']


# In[15]:


df_life_TestFrame.drop("ID",axis=1,inplace=True)


# In[16]:


df_life_TestFrame.shape


# The data is split into train and test before doing EDA so that ML engineer doesn't see test data so that we can prevent data leakage and model can perform well on unkwon data.

# # Exploratory Data Analysis (EDA)

# ## Discriptive Statistics of Data

# - To get a deep understanding of the data Descriptive summary statistics and graphical techniques 
# on each column are carried out:
# - To check Descriptive Summary Statistic like mean, median, mode and standard deviation of all 
# column

# In[17]:


df_life_TrainFrame.describe()


# As this data has continious data we will plot several graphs and check how distubution of data is done on each atrribute.

# ##### From metadat.txt we found that there are continous and nominal data in our dataset:
# ##### Nominal Data is converted to numeric :
# - Company_Status
# - company unique id
# 
# #### Continous Data:
# - All other attributes are continous.
# 

# #### To change column name "Country" to "company's unique id" as per metadata.txt

# #### To change Training data

# In[18]:


df_life_TrainFrame.rename(columns = {'Country':'company unique id'}, inplace = True)


# In[19]:


df_life_TrainFrame.head()


# #### To change Testing data

# In[20]:


df_life_TestFrame.rename(columns = {'Country':'company unique id'}, inplace = True)


# In[21]:


df_life_TestFrame.head()


# ## Data Distribution

# ### Histogram Plot

# In[22]:


plt.figure(figsize=(20,20))
for i, col in enumerate(df_life_TrainFrame.columns):
    plt.subplot(5,5,i+1)
    plt.hist(df_life_TrainFrame[col], alpha=0.3, color='g', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# ##### Let us see some important attribute of dataset and check its distribution

# - ### Company_Confidence

# In[23]:


sns.distplot(df_life_TrainFrame['Company_Confidence'],kde=True).set(title="Distrubution Graph of Company_Confidence")


# In[24]:


# To check wheather mean and median is same .
df_life_TrainFrame['Company_Confidence'].mean()


# In[25]:


# To check wheather mean and median is same .
df_life_TrainFrame['Company_Confidence'].median()


# In[26]:


df_life_TrainFrame['Company_Confidence'].describe()


# In[27]:


# Box Plot for 
df_life_TrainFrame['Company_Confidence'].plot(kind='box')
plt.title("Box Plot for Company_Confidence")
plt.ylabel("Company_Confidence")
plt.show()


# ##### Four main aspects of this graph are:
# ###### Shape:
# -   Right-Skewed Unimodal Distribution Curve
# -  That means in the dataset number of devices having an Company_Confidence less than the mean (164.71) is  comparatively greater than the number of devices below mean.
# 
# ###### Center:
# - Mean=164.71
# - Median=145.5
# - As it is right-skewed median is lesser than the mean.
# 
# ###### Spread:
# - Range=max-min=693-1=692 
# 
# ###### Outliers:
# - In this data, all points are not pretty concise and together, So there are some apparent outliers but all the values are valid as they are within 1000 as per metadata
# - So the distribution of Company_Confidence is  positively skewed with a centre of about145.5, a range of 
# 692 (1 to 693), and with apparent outliers.
# 

# - ### Company_Status

# In[28]:


# Pie chart of Company_Status
Company_Status=["0:Developed","1:Developing status"]
df_life_TrainFrame['Company_Status'].value_counts().plot(kind='pie',figsize=(15, 6),autopct='%1.1f%%')
plt.title("Pie Chart of Anaemia")
plt.legend(labels=Company_Status, loc='upper right')
plt.show()


# In[29]:


#Count PLot of Company_Status
sns.countplot(x="Company_Status",data=df_life_TrainFrame,hue="Company_Status").set(title="Count PLot of Company_Status")


# As we can see from above plot the Company_Status data is completly imbalanced as developed percentage is almost 4 times that of Developing Status which can make mode to higly bais on developed status. 

# - ### Company_device_confidence

# In[30]:


sns.distplot(df_life_TrainFrame['Company_device_confidence'],kde=True).set(title="Distrubution Graph of Company_device_confidence")


# In[31]:


# To check wheather mean and median is same .
df_life_TrainFrame['Company_device_confidence'].mean()


# In[32]:


# To check wheather mean and median is same .
df_life_TrainFrame['Company_device_confidence'].median()


# In[33]:


df_life_TrainFrame['Company_device_confidence'].describe()


# In[34]:


# Box Plot for 
df_life_TrainFrame['Company_device_confidence'].plot(kind='box')
plt.title("Box Plot for Company_device_confidence")
plt.ylabel("Company_device_confidence")
plt.show()


# ##### Four main aspects of this graph are:
# ###### Shape:
# -   Right-Skewed Bimodal Distribution Curve
# -  That means in the dataset number of devices having an Company_device_confidence less than the mean (161.09) is  comparatively greater than the number of devices below mean.
# 
# ###### Center:
# - Mean=161.09
# - Median=142.0
# - As it is right-skewed median is lesser than the mean.
# 
# ###### Spread:
# - Range=max-min=696-0=696 
# 
# ###### Outliers:
# - In this data, all points are not pretty concise and together, So there are some apparent outliers but all the values are valid as they are within 1000 as per metadata
# - So the distribution of Company_device_confidence is  positively skewed with a centre of about 142.0, a range of 
# 696 (0 to 696), and with apparent outliers.

# - ### Device_confidence

# In[35]:


sns.distplot(df_life_TrainFrame['Device_confidence'],kde=True).set(title="Distrubution Graph of Device_confidence")


# In[36]:


# To check wheather mean and median is same .
df_life_TrainFrame['Device_confidence'].mean()


# In[37]:


# To check wheather mean and median is same .
df_life_TrainFrame['Device_confidence'].median()


# In[38]:


df_life_TrainFrame['Device_confidence'].describe()


# In[39]:


# Box Plot for 
df_life_TrainFrame['Device_confidence'].plot(kind='box')
plt.title("Box Plot for Device_confidence")
plt.ylabel("Device_confidence")
plt.show()


# ##### Four main aspects of this graph are:
# ###### Shape:
# -   Right-Skewed Bimodal Distribution Curve
# -  That means in the dataset number of devices having an Device_confidence less than the mean (163.12) is  comparatively greater than the number of devices below mean.
# 
# ###### Center:
# - Mean=163.12
# - Median=145.0
# - As it is right-skewed median is lesser than the mean.
# 
# ###### Spread:
# - Range=max-min=702-2=700
# 
# ###### Outliers:
# - In this data, all points are not pretty concise and together, So there are some apparent outliers but all the values are valid as they are within 1000 as per metadata
# - So the distribution of Device_confidence is  positively skewed with a centre of about 145.0, a range of 
# 700 (0 to 702), and with apparent outliers.

#  - ### Device_returen

# In[40]:


sns.distplot(df_life_TrainFrame['Device_returen'],kde=True).set(title="Distrubution Graph of Device_returen")


# In[41]:


# To check wheather mean and median is same .
df_life_TrainFrame['Device_returen'].mean()


# In[42]:


# To check wheather mean and median is same .
df_life_TrainFrame['Device_returen'].median()


# In[43]:


df_life_TrainFrame['Device_returen'].describe()


# In[44]:


# Box Plot for 
df_life_TrainFrame['Device_returen'].plot(kind='box')
plt.title("Box Plot for Device_returen")
plt.ylabel("Device_returen")
plt.show()


# ##### Four main aspects of this graph are:
# ###### Shape:
# -   Right-Skewed Bimodal Distribution Curve
# -  That means in the dataset number of devices having an Device_returen less than the mean (32.52) is  comparatively greater than the number of devices below mean.
# 
# ###### Center:
# - Mean=32.52
# - Median=3.0
# - As it is right-skewed median is lesser than the mean.
# 
# ###### Spread:
# - Range=max-min=1800-0=1800
# 
# ###### Outliers:
# - In this data, all points are not not pretty concise and together, So there are some apparent outliers and the values are not valid as they are more than  1000 as per metadata
# - So the distribution of Device_returen is  positively skewed with a centre of about 3.0, a range of 
# 1800 (0 to 1800), and with apparent outliers.

# ### This attribute has several outliers so has to replace value with mean value or can drop the data, but has dataset is not so big we are gonna replace the outliers with mean value

# -  ### PercentageExpenditure

# In[45]:


sns.distplot(df_life_TrainFrame['PercentageExpenditure'],kde=True).set(title="Distrubution Graph of  PercentageExpenditure")


# In[46]:


# To check wheather mean and median is same .
df_life_TrainFrame['PercentageExpenditure'].mean()


# In[47]:


# To check wheather mean and median is same .
df_life_TrainFrame['PercentageExpenditure'].median()


# In[48]:


df_life_TrainFrame['PercentageExpenditure'].describe()


# In[49]:


# Box Plot 
df_life_TrainFrame['PercentageExpenditure'].plot(kind='box')
plt.title("Box Plot for  PercentageExpenditure")
plt.ylabel(" PercentageExpenditure")
plt.show()


# ##### Four main aspects of this graph are:
# ###### Shape:
# -   Right-Skewed Bimodal Distribution Curve
# -  That means in the dataset number of devices having an PercentageExpenditure less than the mean (781.03) is  comparatively greater than the number of devices below mean.
# 
# ###### Center:
# - Mean=781.03
# - Median=67.50
# - As it is right-skewed median is lesser than the mean.
# 
# ###### Spread:
# - Range=max-min=19479-0=19479
# 
# ###### Outliers:
# - In this data, all points are not not pretty concise and together, So there are some apparent outliers and the values are not valid as they are more than  100 % as per metadata
# - So the distribution of PercentageExpenditure is  positively skewed with a centre of about 67.50, a range of 
# 19479 (0 to 19479), and with apparent outliers.

# ### This attribute has several outliers so has to replace value with mean value or can drop the data, but has dataset is not so big we are gonna replace the outliers with mean value

# ## Relationship between variable

# - This dataset consist both continous and catagorical data
# - Scatter matrix or pair plot is best to view relation between each other attribute.
# - Corelation score is calculated between each attribute to check realtion between attribute
# - Pearson Correlation lies between -1 and +1, where
# - Correlation >0.5 and <- 0.5 is said to be highly correlated

# In[50]:


import seaborn as sns
plt.figure(figsize=(20,20))
for i, col in enumerate(df_life_TrainFrame.columns):
    plt.subplot(5,5,i+1)
    sns.scatterplot(data=df_life_TrainFrame, x=col, y='TARGET_LifeExpectancy')
    
    plt.title(col)


plt.xticks(rotation='vertical')
plt.show()


# In[51]:


# To Check Pearson Correlation between all attribute
df_life_TrainFrame.corr()


# In[52]:


#To plot Correlation Heatmap
corr = df_life_TrainFrame.corr()
plt.figure(figsize=(25,15))
sns.heatmap(corr, cmap="Blues",annot=True)
plt.show()


# - Positive values means two variables changes in the same direction.
# - Negative values means two variable changes in different direction.
# - All bright colour in above graph are highly coreelated compared  to other attributes.

# ### From above scatter plot and pair plot we can draw some Hypothesis

# - Plausible Hypothesis-1: As Comapany_Confidence increses Company_device_confidence also increses
# - That means devices that has higher Comapany_Confidence also trend to have higher Company_device_confidence .

# In[53]:


#Scatter Plot To justify Plausible Hypothesis-1
plt.scatter(df_life_TrainFrame['Company_Confidence'],df_life_TrainFrame['Company_device_confidence'])
plt.xlabel("Comapany_Confidence")
plt.ylabel("Company_device_confidence")
plt.title("Scatter Plot To justify Plausible Hypothesis-1")


# In[54]:


sns.jointplot(x = "Company_Confidence", y = "Company_device_confidence",
              kind = "hex", data = df_life_TrainFrame)


# - From the above graphs its clearly seen that devices with lower Company_Confidence as also lower Company_device_confidence and vice versa,It is clearly seen that there is linear relation between these attribute with pearon ratio as 0.99 closely near to 1 which is higly corelated to each other .

# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# - Plausible Hypothesis-2: As Comapany_Confidence increses Device_confidence also increses
# - That means devices that has higher Comapany_Confidence also trend to have higher Device_confidence .

# In[55]:


#Scatter Plot To justify Plausible Hypothesis-2
plt.scatter(df_life_TrainFrame['Company_Confidence'],df_life_TrainFrame['Device_confidence'])
plt.xlabel("Comapany_Confidence")
plt.ylabel("Device_confidence")
plt.title("Scatter Plot To justify Plausible Hypothesis-2")


# In[56]:


sns.jointplot(x = "Company_Confidence", y = "Device_confidence",
              kind = "hex", data = df_life_TrainFrame)


# - From the above graphs its clearly seen that devices with lower Company_Confidence as also lower Device_confidence and vice versa,It is clearly seen that there is linear relation between these attribute with pearon ratio as 0.99 closely near to 1 which is higly corelated to each other .

# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# - Plausible Hypothesis-3: As Device_returen increses Obsolescence also increses
# - That means devices that has higher  Device_returen  also trend to have higher Obsolescence .

# In[57]:


#Scatter Plot To justify Plausible Hypothesis-3
plt.scatter(df_life_TrainFrame['Device_returen'],df_life_TrainFrame['Obsolescence'])
plt.xlabel("Device_returen")
plt.ylabel("Obsolescence")
plt.title("Scatter Plot To justify Plausible Hypothesis-3")


# In[58]:


sns.jointplot(x = "Device_returen", y = "Obsolescence",
              kind = "hex", data = df_life_TrainFrame)


# - From the above graphs its clearly seen that devices with lower Device_returen as also lower Obsolescence and vice versa,It is clearly seen that there is linear relation between these attribute with pearon ratio as 0.99 closely near to 1 which is higly corelated to each other .

# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# - Plausible Hypothesis-4: As GDP increses PercentageExpenditure also increases
# - That means devices that has higher  GDP   also trend to have higher PercentageExpenditure .

# In[59]:


#Scatter Plot To justify Plausible Hypothesis-4
plt.scatter(df_life_TrainFrame['GDP'],df_life_TrainFrame['PercentageExpenditure'])
plt.xlabel("GDP")
plt.ylabel("PercentageExpenditure")
plt.title("Scatter Plot To justify Plausible Hypothesis-4")


# In[60]:


sns.jointplot(x = "GDP", y = "PercentageExpenditure",
              kind = "hex", data = df_life_TrainFrame)


# - From the above graphs its clearly seen that devices with lower GDP as also lowerPercentageExpenditure and vice versa,It is clearly seen that there is linear relation between these attribute with pearon ratio as 0.99 closely near to 1 which is higly corelated to each other .
# - As we can see that the data is more at beginning and becomes lesser and lesser as the values increases.

# #### --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# - Plausible Hypothesis-5: As TARGET_LifeExpectancy increases Comapany_Confidence  Decreases
# - That means devices that has higher TARGET_LifeExpectancy  trend to have lower Comapany_Confidence.

# In[61]:


#Scatter Plot To justify Plausible Hypothesis-5
plt.scatter(df_life_TrainFrame['Company_Confidence'],df_life_TrainFrame['TARGET_LifeExpectancy'])
plt.xlabel("Company_Confidence")
plt.ylabel("TARGET_LifeExpectancy")
plt.title("Scatter Plot To justify Plausible Hypothesis-5")


# In[62]:


sns.jointplot(x = "Company_Confidence", y = "TARGET_LifeExpectancy",
              kind = "hex", data = df_life_TrainFrame)


# - As we can see most of the data is concentrated at end, we can say that there is very speed decrease in TARGET_LifeExpectancy until the age of 60 then the TARGET_LifeExpectancy starts decreasing less.

# # BaseLine Model

# ## Spliting of training data

# In[63]:


# selecting the predictors and targets for training data
attribute_train =df_life_TrainFrame.drop(['TARGET_LifeExpectancy'], axis=1) 
target_train = df_life_TrainFrame["TARGET_LifeExpectancy"]


# In[64]:


#To check all attributes values for model
attribute_train.head(5)


# In[65]:


#To check all targets values for model
target_train.head(5)


# ## Spliting of testing data

# In[66]:


# selecting the predictors and targets for testing data
attribute_test =df_life_TestFrame.drop(['TARGET_LifeExpectancy'], axis=1) 
target_test = df_life_TestFrame["TARGET_LifeExpectancy"]


# In[67]:


#To check all attributes values for model
attribute_test .head(5)


# In[68]:


#To check all targets values for model
target_test.head(5)


# ## Linear Regression

# In[69]:


regression_model = LinearRegression()
regression_model.fit(attribute_train ,target_train)


# #### To find coefficient of model

# In[70]:


for idx, col_name in enumerate(attribute_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))


# #### To find Intercept of model

# In[71]:


intercept = regression_model.intercept_
print("The intercept for our model is {}".format(intercept))


# #### To find R2 Score for training dataset

# In[72]:


regression_model.score(attribute_train, target_train)


# #### To find R2 Score for testing dataset

# In[73]:


regression_model.score(attribute_test, target_test)


# In[74]:


model_prediction= regression_model.predict(attribute_test)


# In[75]:


from sklearn.metrics import r2_score

r2_lr = r2_score( target_test,model_prediction)
print('The R^2 score for the linier regression model is: {:.3f}'.format(r2_lr))


# In[76]:


from sklearn import metrics
#print result of MAE
print(metrics.mean_absolute_error(target_test,model_prediction))


# In[77]:


#print result of MSE
print(metrics.mean_squared_error(target_test,model_prediction))


# In[78]:


#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(target_test,model_prediction)))


# ## Ridge Regression

# In[79]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=.3)
ridge.fit(attribute_train,target_train)


# #### To find coefficient of model

# In[80]:


for idx, col_name in enumerate(attribute_train.columns):
    print("The coefficient for {} is {}".format(col_name, ridge.coef_[idx]))


# In[81]:



print ("Ridge model:", (ridge.score(attribute_test,  target_test)))


# ## Lasso Regression

# In[82]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(attribute_train,target_train)


# #### To find coefficient of model

# In[83]:


for idx, col_name in enumerate(attribute_train.columns):
    print("The coefficient for {} is {}".format(col_name, lasso.coef_[idx]))


# In[84]:


print ("Lasso model:", (lasso .score(attribute_test,  target_test)))


# # Data Pre-processing

# ## Removing outliers from Training DataSet

# - From the baove EDA we have seen that attributes like Device_returen, PercentageExpenditure, Obsolescence and Engine_Cooling has outliers and that outliers has to be replaced or removed.
# - In our solution we are replacing upper outliers with upper whisker and lower outliers with lower whisker.
# - we are finding Inter quatile range range by finding diffrence between 1st quantile and 3rd quantile.

# In[85]:


#To create new dataframe with Device_returen and PercentageExpenditure
Outlier_df= pd.DataFrame([df_life_TrainFrame['Device_returen'],df_life_TrainFrame['PercentageExpenditure'],df_life_TrainFrame['Obsolescence'],df_life_TrainFrame['Engine_Cooling']]).T


# In[86]:


#To find 1st quantile and 2nd quantile
Q1 = Outlier_df.quantile(0.25)
Q3 = Outlier_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)  


# In[87]:


#To find outliers
np.where((Outlier_df < (Q1 - 1.5 * IQR)) | (Outlier_df > (Q3 + 1.5 * IQR)))


# In[88]:


#To find non outliers
Outlier_df_out = Outlier_df[~((Outlier_df < (Q1 - 1.5 * IQR)) |(Outlier_df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[89]:


#Replacing every outlier on the lower side by the lower whisker
for i, j in zip(np.where(Outlier_df < Q1 - 1.5 * IQR)[0], np.where(Outlier_df < Q1 - 1.5 * IQR)[1]): 
    
    whisker  = Q1 - 1.5 * IQR
    Outlier_df.iloc[i,j] = whisker[j] 
#Replacing every outlier on the upper side by the upper whisker
for i, j in zip(np.where(Outlier_df > Q3 + 1.5 * IQR)[0], np.where(Outlier_df > Q3 + 1.5 * IQR)[1]):
    
    whisker  = Q3 + 1.5 * IQR
    Outlier_df.iloc[i,j] = whisker[j]


# In[90]:


Outlier_df.head()


# In[91]:


Outlier_df.describe()


# In[92]:


Outlier_df.shape


# In[93]:


#To replace new power and price with original dataframe
df_life_TrainFrame['Device_returen']=Outlier_df['Device_returen']
df_life_TrainFrame['PercentageExpenditure']=Outlier_df['PercentageExpenditure']
df_life_TrainFrame['Obsolescence']=Outlier_df['Obsolescence']
df_life_TrainFrame['Engine_Cooling']=Outlier_df['Engine_Cooling']


# ### Box Plot to check outliers

# In[94]:


#To plot box plot to check
sns.boxplot(data=df_life_TrainFrame,x='Device_returen')


# In[95]:


#To plot box plot to check
sns.boxplot(data=df_life_TrainFrame,x='PercentageExpenditure')


# In[96]:


#To plot box plot to check
sns.boxplot(data=df_life_TrainFrame,x='Obsolescence')


# In[97]:


#To plot box plot to check
sns.boxplot(data=df_life_TrainFrame,x='Engine_Cooling')


# ## Removing outliers from Testing DataSet

# In[98]:


#To create new dataframe with Device_returen and PercentageExpenditure
Outlier_df= pd.DataFrame([df_life_TestFrame['Device_returen'],df_life_TestFrame['PercentageExpenditure'],df_life_TestFrame['Obsolescence'],df_life_TestFrame['Engine_Cooling']]).T


# In[99]:


#To find 1st quantile and 2nd quantile
Q1 = Outlier_df.quantile(0.25)
Q3 = Outlier_df.quantile(0.75)
IQR = Q3 - Q1
print(IQR)  


# In[100]:


#To find outliers
np.where((Outlier_df < (Q1 - 1.5 * IQR)) | (Outlier_df > (Q3 + 1.5 * IQR)))


# In[101]:


#To find non outliers
Outlier_df_out = Outlier_df[~((Outlier_df < (Q1 - 1.5 * IQR)) |(Outlier_df > (Q3 + 1.5 * IQR))).any(axis=1)]


# In[102]:


#Replacing every outlier on the lower side by the lower whisker
for i, j in zip(np.where(Outlier_df < Q1 - 1.5 * IQR)[0], np.where(Outlier_df < Q1 - 1.5 * IQR)[1]): 
    
    whisker  = Q1 - 1.5 * IQR
    Outlier_df.iloc[i,j] = whisker[j] 
#Replacing every outlier on the upper side by the upper whisker
for i, j in zip(np.where(Outlier_df > Q3 + 1.5 * IQR)[0], np.where(Outlier_df > Q3 + 1.5 * IQR)[1]):
    
    whisker  = Q3 + 1.5 * IQR
    Outlier_df.iloc[i,j] = whisker[j]


# In[103]:


Outlier_df.head()


# In[104]:


Outlier_df.describe()


# In[105]:


Outlier_df.shape


# In[106]:


#To replace new power and price with original dataframe
df_life_TestFrame['Device_returen']=Outlier_df['Device_returen']
df_life_TestFrame['PercentageExpenditure']=Outlier_df['PercentageExpenditure']
df_life_TestFrame['Obsolescence']=Outlier_df['Obsolescence']
df_life_TestFrame['Engine_Cooling']=Outlier_df['Engine_Cooling']


# ### Box Plot to check outliers

# In[107]:


#To plot box plot to check
sns.boxplot(data=df_life_TestFrame,x='Device_returen')


# In[108]:


#To plot box plot to check
sns.boxplot(data=df_life_TestFrame,x='PercentageExpenditure')


# In[109]:


#To plot box plot to check
sns.boxplot(data=df_life_TestFrame,x='Obsolescence')


# In[110]:


#To plot box plot to check
sns.boxplot(data=df_life_TestFrame,x='Engine_Cooling')


# # Feature Scaling

# As we have seen in EDA while checking data distribution most of the attributes in dataset has skewed distribution and are close to gaussian distribution, so we are gonna use non linear distriburtion on some features.
# 

# ### Power Transformer

# As we know that power transformer is the technique used to transform data by raising values of atrributes to its square by finding lamda, this transformer is usally used for data consisting of skewed distribution in dataset.

# - Power Transformer for Company_Confidence

# In[111]:


from sklearn.preprocessing import PowerTransformer

PowerTransformer_Company_Confidence = PowerTransformer(method='yeo-johnson', standardize=False).fit(df_life_TrainFrame[['Company_Confidence']])
Company_Confidence_power = PowerTransformer_Company_Confidence.transform(df_life_TrainFrame[['Company_Confidence']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['Company_Confidence'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist(Company_Confidence_power, alpha=0.3, color='r')
plt.title("Power scaling")


# In[112]:


df_life_TrainFrame['Company_Confidence'] = PowerTransformer_Company_Confidence.transform(df_life_TrainFrame[['Company_Confidence']])
df_life_TestFrame['Company_Confidence'] = PowerTransformer_Company_Confidence.transform(df_life_TestFrame[['Company_Confidence']])


# - Power Transformer for Company_Confidence

# In[113]:


from sklearn.preprocessing import PowerTransformer

PowerTransformer_Company_device_confidence = PowerTransformer(method='yeo-johnson', standardize=False).fit(df_life_TrainFrame[['Company_device_confidence']])
Company_device_confidence_power = PowerTransformer_Company_device_confidence.transform(df_life_TrainFrame[['Company_device_confidence']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['Company_device_confidence'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist(Company_device_confidence_power, alpha=0.3, color='r')
plt.title("Power scaling")


# In[114]:


df_life_TrainFrame['Company_device_confidence'] = PowerTransformer_Company_device_confidence.transform(df_life_TrainFrame[['Company_device_confidence']])
df_life_TestFrame['Company_device_confidence'] = PowerTransformer_Company_device_confidence.transform(df_life_TestFrame[['Company_device_confidence']])


# - Power Transformer for Engine_failure_Prevalence

# In[115]:


PowerTransformer_Engine_failure_Prevalence = PowerTransformer(method='yeo-johnson', standardize=False).fit(df_life_TrainFrame[['Engine_failure_Prevalence']])
Engine_failure_Prevalence_power = PowerTransformer_Engine_failure_Prevalence.transform(df_life_TrainFrame[['Engine_failure_Prevalence']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['Engine_failure_Prevalence'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist(Engine_failure_Prevalence_power, alpha=0.3, color='r')
plt.title("Power scaling")


# In[116]:


df_life_TrainFrame['Engine_failure_Prevalence'] = PowerTransformer_Engine_failure_Prevalence.transform(df_life_TrainFrame[['Engine_failure_Prevalence']])
df_life_TestFrame['Engine_failure_Prevalence'] = PowerTransformer_Engine_failure_Prevalence.transform(df_life_TestFrame[['Engine_failure_Prevalence']])


# - Power Transformer for Leakage_Prevalence

# In[117]:


PowerTransformer_Leakage_Prevalence = PowerTransformer(method='yeo-johnson', standardize=False).fit(df_life_TrainFrame[['Leakage_Prevalence']])
Leakage_Prevalence_power = PowerTransformer_Leakage_Prevalence.transform(df_life_TrainFrame[['Leakage_Prevalence']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['Leakage_Prevalence'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist( Leakage_Prevalence_power, alpha=0.3, color='r')
plt.title("Power scaling")


# In[118]:


df_life_TrainFrame['Leakage_Prevalence'] =PowerTransformer_Leakage_Prevalence .transform(df_life_TrainFrame[['Leakage_Prevalence']])
df_life_TestFrame['Leakage_Prevalence'] = PowerTransformer_Leakage_Prevalence .transform(df_life_TestFrame[['Leakage_Prevalence']])


# ### Logorithm Transformer

# Logorithm transformer works same as that of poer transformer but but involes natural logarathmic transformation which is very much good for data consisting of skewed distribution with larger range  

# - Logorithm Transformer for Device_returen

# In[119]:



from sklearn.preprocessing import FunctionTransformer   

log_transformer = FunctionTransformer(np.log1p)

log_transformer


# In[120]:


LogTransformer_Device_returen = log_transformer.fit(df_life_TrainFrame[['Device_returen']])
Device_returen_Log = LogTransformer_Device_returen.transform(df_life_TrainFrame[['Device_returen']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['Device_returen'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist( Device_returen_Log, alpha=0.3, color='r')
plt.title("Logorithm scaling")


# In[121]:


df_life_TrainFrame['Device_returen'] = LogTransformer_Device_returen.transform(df_life_TrainFrame[['Device_returen']])
df_life_TestFrame['Device_returen'] =LogTransformer_Device_returen.transform(df_life_TestFrame[['Device_returen']])


# - Logorithm Transformer for Engine_Cooling

# In[122]:


LogTransformer_Engine_Cooling = log_transformer.fit(df_life_TrainFrame[['Engine_Cooling']])
Engine_Cooling_Log = LogTransformer_Device_returen.transform(df_life_TrainFrame[['Engine_Cooling']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['Engine_Cooling'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist( Engine_Cooling_Log, alpha=0.3, color='r')
plt.title("Logorithm scaling")


# In[123]:


df_life_TrainFrame['Engine_Cooling'] = LogTransformer_Engine_Cooling.transform(df_life_TrainFrame[['Engine_Cooling']])
df_life_TestFrame['Engine_Cooling'] =LogTransformer_Engine_Cooling.transform(df_life_TestFrame[['Engine_Cooling']])


# - Logorithm Transformer for PercentageExpenditure

# In[124]:


LogTransformer_PercentageExpenditure = log_transformer.fit(df_life_TrainFrame[['PercentageExpenditure']])
PercentageExpenditure_Log = LogTransformer_PercentageExpenditure.transform(df_life_TrainFrame[['PercentageExpenditure']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['PercentageExpenditure'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist( PercentageExpenditure_Log, alpha=0.3, color='r')
plt.title("Logorithm scaling")


# In[125]:


df_life_TrainFrame['PercentageExpenditure'] = LogTransformer_PercentageExpenditure.transform(df_life_TrainFrame[['PercentageExpenditure']])
df_life_TestFrame['PercentageExpenditure'] =LogTransformer_PercentageExpenditure.transform(df_life_TestFrame[['PercentageExpenditure']])


# ### MinMaxScaler

# - MinMax for TotalExpenditure

# In[126]:


from sklearn.preprocessing import MinMaxScaler

MinMaxScaler_df_life_TrainFrame = MinMaxScaler().fit(df_life_TrainFrame[['TotalExpenditure']])
Min_TotalExpenditure= MinMaxScaler_df_life_TrainFrame.transform(df_life_TrainFrame[['TotalExpenditure']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['TotalExpenditure'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist( Min_TotalExpenditure, alpha=0.3, color='r')
plt.title("MinMax scaling")


# In[127]:


df_life_TrainFrame['TotalExpenditure']= MinMaxScaler_df_life_TrainFrame.transform(df_life_TrainFrame[['TotalExpenditure']])
df_life_TestFrame['TotalExpenditure'] = MinMaxScaler_df_life_TrainFrame.transform(df_life_TestFrame[['TotalExpenditure']])


# - MinMax for IncomeCompositionOfResources

# In[128]:


from sklearn.preprocessing import MinMaxScaler

MinMaxScaler_df_life_TrainFrame = MinMaxScaler().fit(df_life_TrainFrame[['IncomeCompositionOfResources']])
Min_IncomeCompositionOfResources=MinMaxScaler_df_life_TrainFrame.transform(df_life_TrainFrame[['IncomeCompositionOfResources']])

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.hist(df_life_TrainFrame['TotalExpenditure'], alpha=0.3, color='r', density=True)
plt.title("Original")

plt.subplot(1,2,2)
plt.hist( Min_IncomeCompositionOfResources, alpha=0.3, color='r')
plt.title("MinMax scaling")


# In[129]:


df_life_TrainFrame['IncomeCompositionOfResources']=MinMaxScaler_df_life_TrainFrame.transform(df_life_TrainFrame[['IncomeCompositionOfResources']])
df_life_TestFrame['IncomeCompositionOfResources'] = MinMaxScaler_df_life_TrainFrame.transform(df_life_TestFrame[['IncomeCompositionOfResources']])


# ### PowerTransformer Scaling

# For all the remaining attribute  we are gonna implement Polynomial Feaures scaling as most of the data is skewed and having no outliers Power transformer will be right fit

# In[130]:


from sklearn.preprocessing import PowerTransformer
attributes=["RD","Product_Quantity","GDP","Engine_failure","STRD_DTP","ISO_23","Obsolescence","Gas_Pressure","Test_Fail"]


# In[131]:


powertransformer = PowerTransformer(method='yeo-johnson', standardize=False).fit(df_life_TrainFrame.loc[:, attributes])
df_life_TrainFrame.loc[:, attributes] = powertransformer.transform(df_life_TrainFrame.loc[:, attributes])
df_life_TestFrame.loc[:, attributes] = powertransformer.transform(df_life_TestFrame.loc[:, attributes])


# ### Trainng data after Feature Scaling

# In[132]:


plt.figure(figsize=(20,20))
for i, col in enumerate(df_life_TrainFrame.columns):
    plt.subplot(5,5,i+1)
    plt.hist(df_life_TrainFrame[col], alpha=0.3, color='b', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# ### Testing data after Feature Scaling

# In[133]:


plt.figure(figsize=(20,20))
for i, col in enumerate(df_life_TestFrame.columns):
    plt.subplot(5,5,i+1)
    plt.hist(df_life_TestFrame[col], alpha=0.3, color='g', density=True)
    plt.title(col)
    plt.xticks(rotation='vertical')


# # Data Splitting after Data Pre Processing

# ## Spliting of training data

# In[134]:


# selecting the predictors and targets for training data
attribute_train =df_life_TrainFrame.drop(['TARGET_LifeExpectancy'], axis=1) 
target_train = df_life_TrainFrame["TARGET_LifeExpectancy"]


# In[135]:


#To check all attributes values for model
attribute_train.head(5)


# In[136]:


#To check all targets values for model
target_train.head(5)


# ## Spliting of testing data

# In[137]:


# selecting the predictors and targets for testing data
attribute_test =df_life_TestFrame.drop(['TARGET_LifeExpectancy'], axis=1) 
target_test = df_life_TestFrame["TARGET_LifeExpectancy"]


# In[138]:


#To check all attributes values for model
attribute_test .head(5)


# In[139]:


#To check all targets values for model
target_test.head(5)


# ## Linear Regression

# In[140]:


regression_model = LinearRegression()
model=regression_model.fit(attribute_train ,target_train)


# In[141]:


for idx, col_name in enumerate(attribute_train.columns):
    print("The coefficient for {} is {}".format(col_name, regression_model.coef_[idx]))


# In[142]:


intercept = regression_model.intercept_
print("The intercept for our model is {}".format(intercept))


# In[143]:


regression_model.score(attribute_train, target_train)


# In[144]:


regression_model.score(attribute_test, target_test)


# In[145]:


model_prediction= model.predict(attribute_test)


# In[146]:


from sklearn.metrics import r2_score

r2_lr = r2_score( target_test,model_prediction)
print('The R^2 score for the linier regression model is: {:.3f}'.format(r2_lr))


# In[147]:


from sklearn import metrics
#print result of MAE
print(metrics.mean_absolute_error(target_test,model_prediction))


# In[148]:


#print result of MSE
print(metrics.mean_squared_error(target_test,model_prediction))


# In[149]:


#print result of RMSE
print(np.sqrt(metrics.mean_squared_error(target_test,model_prediction)))


# As we can see that the model after data pre processing and features scaling has better r2 score and have less diffrence between actual and expected output

# ## Ridge Regression

# In[150]:


from sklearn.linear_model import Ridge
ridge = Ridge(alpha=.3)
ridge.fit(attribute_train,target_train)


# In[151]:


print ("Ridge model:", (ridge.coef_))


# In[152]:


print ("Ridge model:", (ridge.score(attribute_test,  target_test)))


# ## Lasso Regression

# In[153]:


from sklearn.linear_model import Lasso
lasso = Lasso(alpha=0.1)
lasso.fit(attribute_train,target_train)


# In[154]:


print ("Lasso model:", (lasso.coef_))


# In[155]:


print ("Lasso model:", (lasso .score(attribute_test,  target_test)))


# # K-Fold Cross Validation

# In[156]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

num_folds = 5
seed = 7
kfold = KFold(n_splits=num_folds,shuffle=True, random_state=seed)
regression_model = LinearRegression()    
results = cross_val_score(regression_model, attribute_test, target_test, cv=kfold)
print(results)
print("Accuracy: %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0))    


# The final model that is Linear Regression model is fed to K-Fold Cross validation so that model is built on diffrent types of inpt so that model can perform better during testing o unknown data.

# # Saving prediction file into csv

# In[157]:


print(df_id)


# In[158]:


df1 = pd.DataFrame(df_id, columns=['ID'])


# In[159]:


model_prediction= model.predict(attribute_test)


# In[160]:


print(model_prediction)


# In[161]:


df2 = pd.DataFrame(model_prediction, columns=['TARGET_LifeExpectancy'])


# In[162]:


result = pd.concat([df1, df2],axis=1,join='inner')


# In[163]:


result.to_csv('s3894695.csv',index=False)


# In[ ]:




