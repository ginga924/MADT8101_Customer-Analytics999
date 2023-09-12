# Churn Scoring
A process of assigning a score to each customer based on their likelihood of churning (cancelling their subscription or service). This score can be used to identify customers who are at risk of churning and take steps to prevent them from leaving.
# Dataset : Customertravel
[Customertravel.csv](https://github.com/ginga924/MADT8101_Customer-Analytics999/files/12580554/Customertravel.1.csv)
# Google Colab Notebook

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
### Explore avg Age by Target
```
sns.boxplot(data=df, x='Target', y='Age')
plt.show()
```

# 3)
