#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


my_file = pd.read_csv('1. Regression - Module - (Housing Prices).csv')


# # Data Preparation

# In[3]:


my_file.head(10)


# In[4]:


my_file.describe()


# In[5]:


my_file.describe(include = 'all')


# In[6]:


my_file.info()


# In[7]:


print(my_file['Sale Price'].mean())
print(my_file['Sale Price'].min())
print(my_file['Sale Price'].max())
print(my_file['Sale Price'].std())
print(my_file['Sale Price'].quantile(.25))


# In[8]:


plt.plot(my_file['Sale Price'], color ='green', marker='o', markerfacecolor='red', markersize=5, linewidth=5)
plt.xlabel("record number")
plt.ylabel("sale price")
plt.title("line graph of sales price")
plt.show()


# In[9]:


my_file.groupby('Condition of the House')['ID'].count()


# In[10]:


values = (30,1701,14031,5679,172)
labels = ('Bad', 'Excellent', 'Fair', 'Good', 'Okay')
plt.pie(values, labels = labels)


# In[11]:


plt.bar(labels, values)
plt.xlabel("condition of the house")
plt.ylabel("count of the house")
plt.title("my first bar graph")
plt.show()


# In[12]:


plt.scatter(my_file['Flat Area (in Sqft)'], my_file['Sale Price'], color='red')
plt.xlabel("area")
plt.ylabel("selling price")
plt.title("area vs selling price")
plt.show()


# In[13]:


plt.hist(my_file['Age of House (in Years)'], bins =10)


# In[14]:


plt.boxplot(my_file['Age of House (in Years)'])


# In[15]:


#initialising a new column
my_file['Condition Sale']=0
#calculating mean based on the condition of the house
for i in my_file['Condition of the House'].unique():
    my_file['Condition Sale'][my_file['Condition of the House']==str(i)] = my_file['Sale Price'][my_file['Condition of the House']==str(i)].mean()
    
#plotting the graph
plt.figure(dpi = 100)
plt.bar(my_file['Condition of the House'].unique(), my_file['Condition Sale'].unique())
plt.xlabel('Condition of the House')
plt.ylabel('Mean Sale Price')
plt.title('my graph for mean sale price')
plt.show()


# In[16]:


Zip_Condition_Sale1 = my_file.groupby(['Condition of the House', 'Zipcode'])['Sale Price'].mean()

Zip_Condition_Sale1


# In[17]:


Zipcode_Condition_Sale2 = pd.pivot_table(my_file, index=['Condition of the House','Zipcode'], values=['Sale Price'], aggfunc=np.mean)

Zipcode_Condition_Sale2


# In[18]:


Zipcode_Condition_Sale3 = pd.pivot_table(my_file, index=['Zipcode'], columns=['Condition of the House'], values=['Sale Price'], aggfunc=np.mean)

Zipcode_Condition_Sale3


# In[19]:


my_file['Condition of the House'] = my_file['Condition of the House'].map({'Good':'1', 'Excellent':'3', 'Bad':'0', 'Fair':'1', 'Okay':'0'})

my_file['Condition of the House'].unique()


# In[20]:


#to extract the year when the house was sold
year = []
for i in range(len(my_file['Date House was Sold'])):
    k = my_file['Date House was Sold'][i].split()[-1]
    year.append(k)
    
my_file['year_sold'] =  year
a=my_file['year_sold'].head()
a.unique()


# In[21]:


#defining a luxury home
my_file['luxury_home'] = 0

for i in range(len(my_file)):
    count = 0
    if my_file['Waterfront View'][i] == 'Yes':
        count = count +1
    if my_file['Condition of the House'][i] in ['Good', 'Excellent']:
        count = count +1
    if my_file['Overall Grade'][i] >= 8:
        count = count +1
    if count >=2:
        my_file['luxury_home'][i]='Yes'
    else:
        my_file['luxury_home'][i]='no'
        
my_file['luxury_home'].unique()


# In[22]:


def luxury_home(row):
    count = 0
    if row[0] == 'Yes':
        count = count+1
    if row[1] in ['Good', 'Excellent']:
        count += 1
    if row[2] >= 8:
        count += 1
    if count >=2:
        return 'Yes'
    else:
        return 'No'
    
my_file['luxury_home'] = my_file[['Waterfront View', 'Condition of the House', 'Overall Grade']].apply(luxury_home, axis =1)

my_file['luxury_home'].unique()


# In[23]:


my_file


# In[24]:


my_file = pd.read_csv('1. Regression - Module - (Housing Prices).csv')
sns.boxplot(my_file['Sale Price'])


# In[25]:


q1 = my_file['Sale Price'].quantile(0.25)
q3 = my_file['Sale Price'].quantile(0.75)
iqr = q3-q1
print(iqr)
upper_limit = q3+iqr*1.5
lower_limit = q1-iqr*1.5
print(upper_limit, lower_limit)


# In[26]:


def limit_imputer(value):
    if value > upper_limit:
        return upper_limit
    elif value < lower_limit:
        return lower_limit
    else:
        return value


# In[27]:


my_file['Sale Price'] = my_file['Sale Price'].apply(limit_imputer)


# In[28]:


sns.boxplot(my_file['Sale Price'])


# # Treating missing values
# 

# In[29]:


my_file.dropna(inplace = True, axis=0, subset=['Sale Price'])


# In[30]:


#these are the columns where we have missing values
numerical_columns = ['No of Bathrooms','Flat Area (in Sqft)', 'Area of the House from Basement (in Sqft)', 'Latitude', 'Longitude', 'Living Area after Renovation (in Sqft)']


# In[31]:


from sklearn.impute import SimpleImputer
#defining a variable named imputer 
#missing_values parameter represents how the missing values are present? 
#strategy specifies the way we want to impute missing values
imputer = SimpleImputer(missing_values = np.nan, strategy = 'median') 
my_file[numerical_columns] = imputer.fit_transform(my_file[numerical_columns])


# In[32]:


my_file.info()


# In[33]:


column = my_file['Zipcode'].values.reshape(-1, 1)
#here the (-1) signifies that this reshape function automatically adjusts the rows
column.shape


# In[34]:


column = my_file['Zipcode'].values.reshape(-1, 1)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
my_file['Zipcode'] = imputer.fit_transform(column)


# In[35]:


my_file.info()


# #  Variable transformation

# In[36]:


my_file['Zipcode'] = my_file['Zipcode'].astype(object)


# In[37]:


my_file['Ever Renovated'] = np.where(my_file['Renovated Year'] == 0, 'No', 'Yes')
my_file.head()


# In[38]:


#Fetching the year from the variable Date House was Sold
#purchase year is the new variable (column) created
my_file['Purchase Year'] = pd.DatetimeIndex(my_file['Date House was Sold']).year


# In[39]:


# creating the new column 'Year since renovation'
my_file['Year Since Renovation'] = np.where(my_file['Ever Renovated'] == 'yes', abs(my_file['Purchase Year'] - my_file['Renovated Year']), 0)


# # Dropping columns

# In[40]:


my_file.drop( columns = ['Purchase Year', 'Date House was Sold', 'Renovated Year'], inplace = True)


# In[41]:


my_file.head()


# # Correlation

# In[42]:


my_file['Sale Price'].corr(my_file['Flat Area (in Sqft)'])


# In[43]:


##Now to find the correlation between the target variable and all the independent variable##
my_file.drop( columns = ['ID']).corr()


# # Exploring the categorical variable

# In[44]:


#value_counts() function of python counts the number of unique values in that particular variable
my_file['Condition of the House'].value_counts()


# In[45]:


my_file.groupby('Condition of the House')['Sale Price'].mean().plot(kind = 'bar')


# In[46]:


#rearranging the bar graph in ascending order
my_file.groupby('Condition of the House')['Sale Price'].mean().sort_values().plot(kind = 'bar')


# # ANOVA

# In[47]:


#to use ANOVA we need to import two libraries
from statsmodels.formula.api import ols
import statsmodels.api as sm


# In[48]:


#renaming variable
my_file = my_file.rename(columns = {'Sale Price' : 'Sale_Price'})
my_file = my_file.rename(columns = {'Condition of the House' : 'Condition_of_the_House'})
my_file = my_file.rename(columns = {'Waterfront View' : 'Waterfront_view'})


# In[49]:


mod = ols('Sale_Price ~ Condition_of_the_House', data = my_file).fit()


# In[50]:


Anova_Table = sm.stats.anova_lm(mod, typ =2)


# In[51]:


print(Anova_Table)


# In[52]:


mod = ols('Sale_Price ~ Waterfront_view', data = my_file).fit()
sm.stats.anova_lm(mod, typ =2)


# In[53]:


my_file = pd.get_dummies(my_file, columns = ['Condition_of_the_House'], drop_first = True)


# In[54]:


my_file = pd.get_dummies(my_file, columns = [ 'Waterfront_view'], drop_first = True)


# # Binning

# In[55]:


#to get average sale price at each zipcode
Zip_Table = my_file.groupby('Zipcode').agg({'Sale_Price' : 'mean'}).sort_values('Sale_Price', ascending = True)


# In[56]:


Zip_Table.head()


# In[57]:


#now creating 10 bins from this table
Zip_Table['Zipcode_Group'] = pd.cut(Zip_Table['Sale_Price'], bins = 10, 
                                   labels = ['Zipcode_Group_0',
                                             'Zipcode_Group_1',
                                            'Zipcode_Group_2',
                                            'Zipcode_Group_3',
                                            'Zipcode_Group_4',
                                            'Zipcode_Group_5',
                                            'Zipcode_Group_6',
                                            'Zipcode_Group_7',
                                            'Zipcode_Group_8',
                                            'Zipcode_Group_9'], 
                                   include_lowest = True)


# In[58]:


Zip_Table = Zip_Table.drop(columns = 'Sale_Price')


# In[59]:


my_file = pd.merge(my_file, Zip_Table, 
                  left_on = 'Zipcode', how = 'left', 
                  right_index = True)


# In[60]:


my_file = my_file.drop(columns = 'Zipcode')


# In[61]:


my_file = pd.get_dummies(my_file, 
                        columns = ['Zipcode_Group'], drop_first = True)


# In[62]:


my_file.head()


# In[63]:


Y = my_file.iloc[:,[1]]
X = my_file.iloc[:,2:31]


# In[64]:


X.shape
Y.shape


# In[65]:


X.head(5)


# # Training Data

# In[66]:


from sklearn.model_selection import train_test_split


# In[67]:


X_train, X_test, Y_train, Y_test = train_test_split(Y, X, test_size = 0.3)
print(X_train.shape, X_test.shape, Y_train.shape, Y_test.shape)


# In[68]:


from sklearn import preprocessing
scale = preprocessing.StandardScaler()


# In[69]:


X_train = scale.fit_transform(X_train)
X_train


# In[70]:


X_test = scale.fit_transform(X_test)
X_test


# In[71]:


X.describe()


# In[72]:


X.to_csv('OutputExcel.csv',index = False)


# # MODEL BUILDING 

# # Mean regression model

# In[73]:


my_file['mean_sale'] = my_file['Sale_Price'].mean()
my_file['mean_sale'].head()


# In[74]:


plt.figure(dpi = 100)
k = range(0, len(my_file))
plt.scatter(k, my_file['Sale_Price'].sort_values(), color = 'red', label = 'Actual Sale Price')
plt.plot(k, my_file['mean_sale'].sort_values(), color = 'green', label = 'mean-price')
plt.xlabel('fitted points (asending)')
plt.ylabel('sale price')
plt.title('overall mean')
#with plt.show() we will not get the index box
plt.legend()


# In[75]:


#finding the mean sale price of houses with respect to overall grade of the house
grades_mean = my_file.pivot_table(values = 'Sale_Price', columns = 'Overall Grade', aggfunc = np.mean)
grades_mean


# In[76]:


#making a new column
my_file['grade_mean']= 0

#for every grade fill its mean price in new column
for i in grades_mean.columns :
    my_file['grade_mean'][my_file['Overall Grade'] == i] = grades_mean[i][0]
        
my_file['grade_mean'].head()


# In[77]:


gradewise_list = []
for i in range(1, 11):
    k = my_file['Sale_Price'][my_file['Overall Grade'] == i]
    gradewise_list.append(k)


# In[78]:


classwise_list = []
for i in range(1,11):
    k = my_file['Sale_Price'][my_file['Overall Grade'] == i]
    classwise_list.append(k)


# In[79]:


plt.figure(dpi = 120, figsize = (15,9))

###plotting "sale price" gradewise ###
#z variable is for x - axis
z = 0
for i in range(1,11):
    #defining x-axis using z
    points = [k for k in range(z, z+len(classwise_list[i-1]))]
    #plotting
    plt.scatter(points, 
               classwise_list[i-1].sort_values(),
               label = ('houses with overall grade', i), s = 4)
    
    #plotting gradewise mean
    plt.scatter(points, [classwise_list[i-1].mean() for q in range(len(classwise_list[i-1]))],
               s = 6, color = 'pink')
    z = max(points) + 1
    
###plotting overall mean###
plt.scatter([q for q in range(0, z)], 
           my_file['mean_sale'], 
           color = 'red', 
           label = 'Overall mean', 
           s = 6)

plt.xlabel('Fitted points (ascending)')
plt.ylabel('sale price')
plt.title('overall mean')
plt.legend(loc =4)


# # Residual plots

# In[80]:


mean_difference = my_file['mean_sale'] - my_file['Sale_Price']
grade_mean_difference = my_file['grade_mean'] - my_file['Sale_Price']


# In[81]:


#to create a list of indices fot the data points
k = range(0, len(my_file))
""" a list of zeros which will represent the residual of a perfect model where predictions are exactly the 
same as actuals and hence the residuals would be zero"""
l = [0 for i in range(len(my_file))]

plt.figure(figsize = (15,6), dpi = 100)

plt.subplot(1,2,1)
plt.scatter(k , mean_difference, color = 'red', label = 'residuals', s = 2)
plt.plot(k, l, color = 'green', label = 'mean regression', linewidth = 3)
plt.xlabel('fitted points')
plt.ylabel('residuals')
plt.title('residuals with respect to gradewise mean')

plt.subplot(1,2,2)
plt.scatter(k, grade_mean_difference, color = 'red', label = 'residuals', s = 2)
plt.plot(k, l, color = 'green', label = 'mean regression', linewidth = 3)
plt.xlabel('fitted points')
plt.ylabel('residuals')
plt.legend()
plt.title('residuals with respect to gradewise mean')

plt.legend()


# # MSE - Mean Squarred Error - How accurate our model is

# In[82]:


y = my_file['Sale_Price']
y_hat1 = my_file['mean_sale']
y_hat2 = my_file['grade_mean']
n = len(my_file)

len(y), len(y_hat1), len(y_hat2), n


# In[83]:


from sklearn.metrics import mean_squared_error as mse
cost_mean = mse(y_hat1, y)
cost_grade_mean = mse(y_hat2, y)
cost_mean, cost_grade_mean


# # Root Mean Squared Error (RMSE)

# In[84]:


from sklearn.metrics import mean_squared_error as mse
cost_mean = mse(y_hat1, y)**0.5
cost_grade_mean = mse(y_hat2, y)**0.5
cost_mean, cost_grade_mean


# # Model evaluation using R^2 method

# In[85]:


y = my_file['Sale_Price']
y_dash = my_file['mean_sale']
y_hat = my_file['grade_mean']
n= len(my_file)

len(y), len(y_dash), len(y_hat), n


# In[86]:


mse_mean = mse(y_dash, y)
mse_mean


# In[87]:


mse_model = mse(y_hat, y)
mse_model


# In[88]:


R2 = 1 - (mse_model / mse_mean)
R2


# # Linear Regression Model

# In[89]:


# we are just gonna use only two variables
sale_price = my_file['Sale_Price'].head(30)
flat_area = my_file['Flat Area (in Sqft)'].head(30)
sample_my_file = pd.DataFrame({'sale_price' : sale_price, 'flat_area' : flat_area})
sample_my_file


# In[90]:


plt.figure (dpi = 100)
plt.scatter(sample_my_file.flat_area, sample_my_file.sale_price, label = 'sale price', color = 'red')
plt.xlabel('flat_area')
plt.ylabel('sale_price')
plt.title('sale_price/flat_area')
plt.legend()
plt.show


# In[91]:


sample_my_file['mean_sale_price'] = sample_my_file.sale_price.mean()

plt.figure(dpi = 150)
plt.scatter(sample_my_file.flat_area, sample_my_file.sale_price, color = 'red', label = 'sale price')
plt.plot(sample_my_file.flat_area, sample_my_file.mean_sale_price, color = 'yellow', label = 'mean sale price')
plt.xlabel('flat_area')
plt.ylabel('sale_price')
plt.title('sale_price/flat_area')
plt.legend()
plt.show()


# In[92]:


## Using COST FUNCTION CURVE
c = 0
m = 0

line = []

for i in range(len(sample_my_file)):
    line.append(sample_my_file.flat_area[i]*m + c)
    
plt.figure(dpi = 130)
plt.scatter(sample_my_file.flat_area, sample_my_file.sale_price)
plt.plot(sample_my_file.flat_area, line, label = 'm = 0; c = 0')
plt.xlabel('flat_area')
plt.ylabel('sale_price')
plt.legend()
MSE = mse(sample_my_file.sale_price, line)
plt.title('slope '+str(m)+' with mse '+str(mse))


# In[93]:


c = 0
m = 50

line = []

for i in range(len(sample_my_file)):
    line.append(sample_my_file.flat_area[i]*m + c)
    
plt.figure(dpi = 130)
plt.scatter(sample_my_file.flat_area, sample_my_file.sale_price)
plt.plot(sample_my_file.flat_area, line, label = 'm = 0; c = 0')
plt.xlabel('flat_area')
plt.ylabel('sale_price')
plt.legend()
MSE = mse(sample_my_file.sale_price, line)
plt.title('slope '+str(m)+' with mse '+str(mse))


# In[94]:


def slope_Error(slope, intercept, sample_my_file):
    sale = []
    for i in range(len(sample_my_file.flat_area)):
        tmp = sample_my_file.flat_area[i] * slope + intercept
        sale.append(tmp)
    MSE = mse(sample_my_file.sale_price, sale)
    return MSE


# In[95]:


slope = [i/10 for i in range(0,5000)]
Cost = []
for i in slope:
    cost = slope_Error(slope = i, intercept = 0, sample_my_file = sample_my_file)
    Cost.append(cost)


# In[96]:


# Arranging in DataFrame
Cost_table = pd.DataFrame({
    'slope' : slope,
    'Cost' : Cost
})
Cost_table.tail()


# In[97]:


# Plotting the cost values corresponding to every value of Beta
plt.plot(Cost_table.slope, Cost_table.Cost, label = 'Cost Function Curve')
plt.xlabel('Value of slope')
plt.ylabel('Cost')
plt.legend


# In[98]:


def intercept_Error(slope, intercept, sample_my_file):
    sale = []
    for i in range(len(sample_my_file.flat_area)):
        tmp = sample_my_file.flat_area[i] * slope + intercept
        sale.append(tmp)
    MSE = mse(sample_my_file.sale_price, sale)
    return MSE


# In[99]:


intercept = [i for i in range(5000,50000)]
Cost = []
for i in intercept:
    cost = intercept_Error(slope = 234, intercept = i, sample_my_file = sample_my_file)
    Cost.append(cost)


# In[100]:


Cost_table = pd.DataFrame({
    'intercept' : intercept,
    'Cost' : Cost
})


# In[101]:


# Plotting the cost values corresponding to every value of Beta
plt.plot(Cost_table.intercept, Cost_table.Cost, label = 'Cost Function Curve')
plt.xlabel('Value of intercept')
plt.ylabel('Cost')
plt.legend()


# In[ ]:




