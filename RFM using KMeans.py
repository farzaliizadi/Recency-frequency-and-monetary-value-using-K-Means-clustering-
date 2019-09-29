# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 22:48:48 2019

@author: Izadi
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
os.chdir(r'D:\desktop\Python_DM_ML_BA\Data camp\Customer Segmentation in Python')
df = pd.read_excel('Online.xlsx')
df.shape
'''As df has 70864 rows, I only select a sample of 0.2 to work with. '''
df = df.sample(frac=0.2, replace=True, random_state=1)
df.shape
df.head()
df.columns
df=df.drop(['IndexNo'],axis=1)
t = set(df['Country'])
t
r = len((df.CustomerID).unique())
r
s = len((df.InvoiceNo).unique())
s
'''1. Create a new column InvoicePeriod from InvoiceDate'''
df['InvoicePeriod'] = df.InvoiceDate.apply(lambda x: x.strftime('%Y-%m'))
df.head()

'''2. Determine the user's cohort group (based on their first order)
Create a new column called CohortGroup, which is the year and month in 
which the user's first purchase occurred.'''

df.set_index('CustomerID', inplace=True)
df['CohortGroup'] = df.groupby(level=0)['InvoiceDate'].min().apply(lambda x: x.strftime('%Y-%m'))
df.reset_index(inplace=True)
df.head()

'''3. Rollup data by CohortGroup & OrderPeriod
Since we're looking at monthly cohorts, we need to aggregate users,
 orders, and amount spent by the CohortGroup within the month (OrderPeriod).'''
 
grouped = df.groupby(['CohortGroup', 'InvoicePeriod'])
# count the unique users, orders, and total revenue per CohortGroup + OrderPeriod
cohorts = grouped.agg({'CustomerID': pd.Series.nunique,
                       'InvoiceNo': pd.Series.nunique,
                       'UnitPrice': np.sum})

# make the column names more meaningful
cohorts.rename(columns={'CustomerID': 'TotalUsers',
                        'InvoiceNo': 'TotalOrders',
                         'UnitPrice':'TotalCharges'}, inplace=True)
cohorts.head()

 '''4. Creates a `CohortPeriod` column, which is the Nth period 
 based on the user's first purchase.'''
    
def cohort_period(df):
    df['CohortPeriod'] = np.arange(len(df)) + 1
    return df

cohorts = cohorts.groupby(level=0).apply(cohort_period)
cohorts.head()
# reindex the DataFrame
cohorts.reset_index(inplace=True)
cohorts.set_index(['CohortGroup','CohortPeriod'], inplace=True)     
# create a Series holding the total size of each CohortGroup
cohort_group_size = cohorts['TotalUsers'].groupby(level=0).first()
cohort_group_size.head()
cohorts['TotalUsers'].head()
cohorts['TotalUsers'].unstack(0).head()
user_retention = cohorts['TotalUsers'].unstack(0).divide(cohort_group_size, axis=1)
s = user_retention.head(13).T # T transpose the matrix
s
''' 5.Use the heat map to see the cohorts with retentions'''
sns.set(style='white')
plt.figure(figsize=(12, 8))
# Add a title
plt.title('Retention Rates')                        #best
# Create the heatmap
sns.heatmap(s, annot=True, cmap='BuGn',fmt='.0%')
plt.show()

'''6. To see the 4 cohots with Cohort period on x-axis and Cohort purchasing on y-axis'''
s.T[['2011-04', '2011-01', '2011-02', '2011-03']].plot(figsize=(10,5),colors=['red', 'purple', 'blue', 'green'])
plt.title('Cohorts: User Retention')
plt.xticks(np.arange(1, 12.1, 1))
plt.xlim(1, 12)
plt.ylabel('% of Cohort Purchasing')

# get a profile_report()
import pandas_profiling as pp
profile = pp.ProfileReport(df)
profile.to_file("report.html")
profile_report()



''' 7. We  now create daily cohorts based
on the day each customer has made first purchase
 we will create 6 variables that capture the integer value of years, 
months and days for Invoice and Cohort Date '''

import datetime as dt
import time as tm

 def get_month(x):
    return dt.datetime(x.year, x.month,1)

# Create InvoiceDay column
df['InvoiceMonth'] = df['InvoiceDate'].apply(get_month)    
 # Group by CustomerID and select the InvoiceDay value
grouping = df.groupby('CustomerID')['InvoiceMonth'] 
# Assign a minimum InvoiceDay value to the dataset
df['CohortMonth'] = grouping.transform('min')

''' 8. Now, we have six different data sets with year, month and 
day values for Invoice and Cohort dates - invoice_year, cohort_year, 
invoice_month, cohort_month, invoice_day, and cohort_day.
Now we will compute the difference between the Invoice 
and Cohort dates in years, months and days separately and then compute
the total days difference between the two. '''

def get_date_int(df, column):
    year = df[column].dt.year
    month = df[column].dt.month
    return year, month
    
# Get the integers for date parts from the InvoiceDaycolumn
invoice_year, invoice_month = get_date_int(df, 'InvoiceMonth')
# Get the integers for date parts from the CohortDay column
cohort_year, cohort_month = get_date_int(df, 'CohortMonth')

# Calculate difference in years
years_diff = invoice_year - cohort_year
# Calculate difference in months
months_diff = invoice_month - cohort_month
# Extract the difference in days from all previous values
df['CohortIndex'] = years_diff * 12 + months_diff  + 1
df.head()
df['CohortMonth'] = df['CohortMonth'].apply(dt.datetime.date)
df.head()
grouping = df.groupby(['CohortMonth', 'CohortIndex'])
# count the unique users, orders, and total revenue per CohortGroup + OrderPeriod
cohorts = grouping['CustomerID'].apply(pd.Series.nunique).reset_index()
cohortCounts = cohorts.pivot(index = 'CohortMonth', columns = 'CohortIndex', values = 'CustomerID')
cohortSizes = cohortCounts.iloc[:, 0]
retention = cohortCounts.divide(cohortSizes, axis = 0) * 100
retention.round(2)
# Showing cohorts groups by months on 11th
month_list = ["Dec '10", "Jan '11", "Feb '11", "Mar '11", 
              "Apr '11", "May '11", "Jun '11", "Jul '11",
              "Aug '11", "Sep '11", "Oct '11", "Nov '11", "Dec '11"]

plt.figure(figsize = (20,10))
plt.title('Retention by Monthly Cohorts')
sns.heatmap(retention.round(2), annot = True, cmap = "BuGn", 
            vmax = list(retention.max().sort_values(ascending = False))[1]+3,
            fmt = '.1f', linewidth = 0.3, yticklabels=month_list)
plt.show()


''' 9. RFM= Recency, Frequency and Monetary values''' 
# First set a time stamp
snapshot = pd.Timestamp('2011-12-10')
df['TotalSum'] = df['Quantity'] * df['UnitPrice']
# Calculate Recency, Frequency and Monetary value for each customer 
datamart = df.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Rename the columns 
datamart.rename(columns={'InvoiceDate': 'Recency',
                         'InvoiceNo': 'Frequency',
                         'TotalSum': 'MonetaryValue'}, inplace=True)

# Print top 5 rows
datamart.head()

''' 10. we divide the customers in 3 different groups for Recency and Frequency
and then on the MonetaryValue and finally calculate and RFM_Score.'''

# Create labels for Recency and Frequency
r_labels = range(3, 0, -1); f_labels = range(1, 4)
# Assign these labels to three equal percentile groups 
r_groups = pd.qcut(datamart['Recency'], q=3, labels=r_labels)
# Assign these labels to three equal percentile groups 
f_groups = pd.qcut(datamart['Frequency'], q=3, labels=f_labels)
# Create new columns R and F
datamart = datamart.assign(R=r_groups.values, F=f_groups.values)
datamart.head()

''' 10. We will now divide customers to 
three different groups based on the MonetaryValue percentiles and then 
calculate an RFM_Score which is a sum of the R, F, and M values.''' 
# Create labels for MonetaryValue 
m_labels = range(1, 4)
# Assign these labels to three equal percentile groups
m_groups = pd.qcut(datamart['MonetaryValue'], q=3, labels=m_labels)
m_groups.head()
datamart.head()
# Create new column M
datamart = datamart.assign(M=m_groups.values)
datamart.head()
datamart[['R','F','M']].head()
type(datamart[['R','F','M']].head())
# Calculate RFM_Score
datamart['RFM_Score'] = datamart[['R','F','M']].sum(axis=1)
datamart['RFM_Score'].head()
''' 11. Now we do customer segmentation based on RFM_Score values.'''
# Define rfm_level function
def rfm_level(df):
    if df['RFM_Score'] >= 10:
        return 'Top'
    elif ((df['RFM_Score'] >= 6) and (df['RFM_Score'] < 10)):
        return 'Middle'
    else:
        return 'Low'

# Create a new variable RFM_Level
datamart['RFM_Level'] = datamart.apply(rfm_level, axis=1)
datamart.head()

''' 12. Analyzing  average values of Recency, Frequency and 
MonetaryValue for the above segmentation. '''
# Calculate average values for each RFM_Level, and return a size of each segment 
rfm_level_agg = datamart.groupby('RFM_Level').agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
}).round(1)

# Print the aggregated dataset
rfm_level_agg
datamart.head()

'''13. In the rest of the work we will use some small data sets to 
explain how one describe the customer segmentation by KMeans clustering algorithms
. However, we need to first normalized the data '''

from sklearn.preprocessing import StandardScaler
# Print summary statistics to make sure average is zero and standard deviation is one
datamart_rfm = pd.read_csv('datamart_rfm.csv')
datamart_rfm.head()
#datamart_rfm = datamart_rfm.drop(['CustomerID'], axis=1)
# Initialize a scaler
scaler = StandardScaler()
# Fit the scaler
scaler.fit(datamart_rfm)
# Scale and center the data
data_normalized = scaler.transform(datamart_rfm)
data_normalized[0:5,]
# Create a pandas DataFrame
data_normalized = pd.DataFrame(data_normalized, index=datamart_rfm.index, columns=datamart_rfm.columns)
data_normalized.head()
# Print summary statistics
data_normalized.describe().round(2)
# Plot recency distribution
plt.subplot(3, 1, 1); sns.distplot(datamart_rfm['Recency'])
# Plot frequency distribution
plt.subplot(3, 1, 2); sns.distplot(datamart_rfm['Frequency'])
# Plot monetary value distribution
plt.subplot(3, 1, 3); sns.distplot(datamart_rfm['MonetaryValue'])
# Show the plot
plt.show()

# Unskew the data
datamart_log = np.log(datamart_rfm)
# Initialize a standard scaler and fit it
scaler = StandardScaler()
scaler.fit(datamart_log)
# Scale and center the data
datamart_normalized = scaler.transform(datamart_log)
# Create a pandas DataFrame
datamart_normalized = pd.DataFrame(data=datamart_normalized, index=datamart_rfm.index, columns=datamart_rfm.columns)
datamart_normalized.describe()

# Plot recency distribution
plt.subplot(3, 1, 1); sns.distplot(datamart_normalized['Recency'])
# Plot frequency distribution
plt.subplot(3, 1, 2); sns.distplot(datamart_normalized['Frequency'])
# Plot monetary value distribution
plt.subplot(3, 1, 3); sns.distplot(datamart_normalized['MonetaryValue'])
# Show the plot
plt.show()

'''method to define number of clusters
 Visual methods - elbow criterion
'''

   #import model
X = datamart_normalized  
from sklearn.cluster import KMeans
 
wcss =[]
for k in range(1,11):
    model = KMeans(n_clusters=k)
    model.fit(X)
    wcss.append(model.inertia_)
    
plt.plot(range(1,11), wcss)          
plt.xlabel('No of clusters')   
plt.ylabel('WCSS') 
plt.title('The Elbow Method')     
plt.legend()   
plt.show()
#import model
model = KMeans(n_clusters=3)
from sklearn.cluster import KMeans
model.fit(X)
# Cluster centers
plt.scatter(model.cluster_centers_[:, 0], model.cluster_centers_[:, 1], s = 300, c = 'red', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

model.inertia_

# Import KMeans 
from sklearn.cluster import KMeans
# Initialize KMeans
kmeans = KMeans(n_clusters=3, random_state=1) 
# Fit k-means clustering on the normalized data set
kmeans.fit(datamart_normalized)
# Extract cluster labels
cluster_labels = kmeans.labels_
# Create a DataFrame by adding a new cluster label column
datamart_rfm_k3 = datamart_rfm.assign(Cluster=cluster_labels)
# Group the data by cluster
grouped = datamart_rfm_k3.groupby(['Cluster'])
# Calculate average RFM values and segment sizes per cluster value
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': ['mean', 'count']
  }).round(1)

''' calculating the sum of squared errors
 for different number of clusters ranging from 1 to 20.'''

sse={}
# Fit KMeans and calculate SSE for each k
for k in range(1, 21):
    kmeans = KMeans(n_clusters=k, random_state=1)
    # Fit KMeans on the normalized dataset
    kmeans.fit( datamart_rfm_k3)
    # Assign sum of squared distances to k element of dictionary
    sse[k] = kmeans.inertia_
    
# Add the plot title "The Elbow Method"
plt.title('The Elbow Method')
# Add X-axis label "k"
plt.xlabel('k')
# Add Y-axis label "SSE"
plt.ylabel('SSE')
# Plot SSE values for each key in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()  

''' Preparin  data for the snake plot'''

datamart_rfm_k3.head()

datmart_normalized=pd.concat([datamart_rfm_k3.CustomerID, datamart_rfm_k3], axis=1)
# Melt the normalized dataset and reset the index
datamart_melt = pd.melt(
  					datamart_rfm_k3.reset_index(), 
                        # Assign CustomerID and Cluster as ID variables                  
                    id_vars=['CustomerID', 'Cluster'],
# Assign RFM values as value variables
                    value_vars=['Recency', 'Frequency', 'MonetaryValue'], 
                        # Name the variable and value
                    var_name='Metric', value_name='Value'
					)
  
datamart_melt.head()     
    
#We will see thhat 2 of cluters are close to each other
# Add the plot title
plt.title('Snake plot of normalized variables')
# Add the x axis label
plt.xlabel('Metric')
# Add the y axis label
plt.ylabel('Value')
# Plot a line for each value of the cluster variable
sns.lineplot(data=datamart_melt, x='Metric', y='Value', hue='Cluster')
plt.show()

#relative importance of the RFM values within each cluster. 
# Calculate average RFM values for each cluster
cluster_avg = datamart_rfm_k3.groupby(['Cluster']).mean() 
# Calculate average RFM values for the total customer population
population_avg = datamart_rfm.mean()
# Calculate relative importance of cluster's attribute value compared to population
relative_imp = cluster_avg / population_avg - 1
# Print relative importance score rounded to 2 decimals
relative_imp.round(2)


# Initialize a plot with a figure size of 8 by 2 inches 
plt.figure(figsize=(8, 2))
# Add the plot title
plt.title('Relative importance of attributes')
# Plot the heatmap
sns.heatmap(data=relative_imp, annot=True, fmt='.2f', cmap='RdYlGn')
plt.show()

datamart_rfmt = pd.read_csv('datamart_rfmt_tenure.csv')
# Import StandardScaler 
from sklearn.preprocessing import StandardScaler
# Apply log transformation
datamart_rfmt_log = np.log(datamart_rfmt)
# Initialize StandardScaler
scaler = StandardScaler(); scaler.fit(datamart_rfmt_log)
# Transform and store the scaled data as datamart_rfmt_normalized
datamart_rfmt_normalized = scaler.transform(datamart_rfmt_log)

'''sum of squared errors for different number 
of clusters ranging from 1 to 10. '''

 
# Fit KMeans and calculate SSE for each k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=1).fit(datamart_rfmt)
    # Assign sum of squared distances to k element of the sse dictionary
    sse[k] = kmeans.inertia_ 

# Add the plot title, x and y axis labels
plt.title('The Elbow Method'); plt.xlabel('k'); plt.ylabel('SSE')
# Plot SSE values for each k stored as keys in the dictionary
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show() 
 
'''WE see that the recommended number of clusters is 
somewhere between 3 and 4.'''

# Import KMeans 
from sklearn.cluster import KMeans
# Initialize KMeans
kmeans = KMeans(n_clusters=4, random_state=1) 
# Fit k-means clustering on the normalized data set
kmeans.fit(datamart_rfmt_normalized)
# Extract cluster labels
cluster_labels = kmeans.labels_
cluster_labels 


# Create a new DataFrame by adding a cluster label column to datamart_rfmt
datamart_rfmt_k4 = datamart_rfmt.assign(Cluster=cluster_labels)
# Group by cluster
grouped = datamart_rfmt_k4.groupby(['Cluster'])

# Calculate average RFMT values and segment sizes for each cluster
grouped.agg({
    'Recency': 'mean',
    'Frequency': 'mean',
    'MonetaryValue': 'mean',
    'Tenure': ['mean', 'count']
  }).round(1)
























