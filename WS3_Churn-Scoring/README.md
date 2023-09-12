# Churn Scoring
A process of assigning a score to each customer based on their likelihood of churning (cancelling their subscription or service). This score can be used to identify customers who are at risk of churning and take steps to prevent them from leaving.
# Dataset : Customertravel
CLick here 👉 [Customertravel.csv](https://github.com/ginga924/MADT8101_Customer-Analytics999/files/12580554/Customertravel.1.csv)
# Google Colab Notebook
CLick here 👉 [Google Colab](https://colab.research.google.com/drive/1SI4ZPn9EWAEs1ZAkmw7tYwln7ANIosH3?usp=sharing)

# Data Dictionary
- Age - This represents the age of an individual, typically in years, and is used as a demographic variable in the dataset.
- FrequentFlyer - This binary variable indicates whether a person is a frequent flyer, providing information about their travel behavior.
- AnnualIncomeClass - This categorical variable classifies individuals into income groups or classes, helping to analyze their economic status.
- ServicesOpted - This variable tracks the specific services or products that individuals have opted for, offering insights into their preferences or purchase behavior.
- AccountSyncedToSocialMedia - A binary variable indicating whether a user's account is linked to a social media platform (1 for yes, 0 for no), potentially reflecting their online engagement.
- BookedHotelOrNot - This binary variable shows whether a person has booked a hotel (1 for yes, 0 for no), which is relevant for studying travel-related behavior.
- Target - This variable is the target or dependent variable used in predictive modeling, representing the outcome or response that the analysis seeks to predict or explain.

# 1) Import & Load Dataset
### Import Libraries
```
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
```
### Import Data
```
df = pd.read_csv("Customertravel.csv")
```
# 2) Exploratory Data Analysis
### Check info
```
df.info
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/6f165198-f27d-4a12-95ac-3437b6c67804)
### Check Null Values
```
df.isna().sum()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/ee209e9e-5bcf-4c84-a55b-75c9135649e6)
### Check description of the data in the DataFrame
```
df.describe()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/96f1e014-161b-40a8-9a9f-816f3950ad95)
### Explore avg Age by Target
```
sns.boxplot(data=df, x='Target', y='Age')
plt.show()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/681d3d18-cb19-4c08-a01a-daf29fc8b8a6)
### Explore avg Age by Target
```
df_by_target = df.groupby(by=['Target']).agg(
          avgAge=('Age','mean')).reset_index()

sns.barplot(data=df_by_target, x='Target', y='avgAge')
plt.show()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/578b34b3-b127-46c9-aa54-12177c7ca4fe)
### Explore ServicesOpted by Target
```
df_by_target = df.groupby(by=['Target']).agg(
               ServicesOpted=('ServicesOpted','sum')).reset_index()

sns.barplot(data=df_by_target, x='Target', y='ServicesOpted')
plt.show()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/3c586863-e285-423f-ba4b-c99416e33b45)
### Explore avg ServicesOpted by Target
```
df_by_target = df.groupby(by=['Target']).agg(
               avgServicesOpted=('ServicesOpted','mean')).reset_index()

sns.barplot(data=df_by_target, x='Target', y='avgServicesOpted')
plt.show()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/7ae9454d-2316-47cb-b2fc-44085200516d)
### Explore No. of customers by Target
```
df_noCust_by_target = df.groupby(by=['Target']).agg(
          no_customers=('Target','count')).reset_index()

sns.barplot(data=df_noCust_by_target, x='Target', y='no_customers')
plt.show()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/f473f9ab-438e-4f96-928b-b46b4f422eef)
### Explore FrequentFlyer by Target
```

```
# 3)
