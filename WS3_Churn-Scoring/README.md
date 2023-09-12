![Static Badge](https://img.shields.io/badge/Python-EADF03) ![Static Badge](https://img.shields.io/badge/Classification-EC910E) ![Static Badge](https://img.shields.io/badge/XGBoost-57CE05) ![Static Badge](https://img.shields.io/badge/Google_Colab-0EDBEC) ![Static Badge](https://img.shields.io/badge/Novice-B60BB8)
# Churn Scoring 😣
A process of assigning a score to each customer based on their likelihood of churning (cancelling their subscription or service). This score can be used to identify customers who are at risk of churning and take steps to prevent them from leaving.
## Dataset 📚 : Customertravel 
CLick here 👉 [Customertravel.csv](https://github.com/ginga924/MADT8101_Customer-Analytics999/files/12580554/Customertravel.1.csv)
## Google Colab Notebook ![Static Badge](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)
CLick here 👉 [Google Colab](https://colab.research.google.com/drive/1SI4ZPn9EWAEs1ZAkmw7tYwln7ANIosH3?usp=sharing)

# Let's roll ⛏️
### Data Dictionary 📗
- Age - This represents the age of an individual, typically in years, and is used as a demographic variable in the dataset.
- FrequentFlyer - This binary variable indicates whether a person is a frequent flyer, providing information about their travel behavior.
- AnnualIncomeClass - This categorical variable classifies individuals into income groups or classes, helping to analyze their economic status.
- ServicesOpted - This variable tracks the specific services or products that individuals have opted for, offering insights into their preferences or purchase behavior.
- AccountSyncedToSocialMedia - A binary variable indicating whether a user's account is linked to a social media platform (1 for yes, 0 for no), potentially reflecting their online engagement.
- BookedHotelOrNot - This binary variable shows whether a person has booked a hotel (1 for yes, 0 for no), which is relevant for studying travel-related behavior.
- Target - This variable is the target or dependent variable used in predictive modeling, representing the outcome or response that the analysis seeks to predict or explain.

## 1) Import & Load Dataset
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
## 2) Exploratory Data Analysis
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
df_FrequentFlyer_by_target = df.groupby(by=['FrequentFlyer', 'Target']).agg(
                              no_customers=('Target','count')).reset_index().pivot('FrequentFlyer', 'Target', 'no_customers')

sns.heatmap(df_FrequentFlyer_by_target, annot=True, fmt='.0f')
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/4f1a899f-06cb-4278-8d2f-2efec0b6867c)
### Explore AnnualIncomeClass by Target
```
df_FrequentFlyer_by_target = df.groupby(by=['AnnualIncomeClass', 'Target']).agg(
                              no_customers=('Target','count')).reset_index().pivot('AnnualIncomeClass', 'Target', 'no_customers')
sns.heatmap(df_FrequentFlyer_by_target, annot=True, fmt='.0f')
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/b7362444-e374-4254-932e-648255a696c5)
### Explore AccountSyncedToSocialMedia by Target
```
df_FrequentFlyer_by_target = df.groupby(by=['AccountSyncedToSocialMedia', 'Target']).agg(
                              no_customers=('Target','count')).reset_index().pivot('AccountSyncedToSocialMedia', 'Target', 'no_customers')
sns.heatmap(df_FrequentFlyer_by_target, annot=True, fmt='.0f')
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/217838b9-a442-4c3b-9a9d-98c86daba68f)
### Explore BookedHotelOrNot by Target
```
df_FrequentFlyer_by_target = df.groupby(by=['BookedHotelOrNot', 'Target']).agg(
                              no_customers=('Target','count')).reset_index().pivot('BookedHotelOrNot', 'Target', 'no_customers')
sns.heatmap(df_FrequentFlyer_by_target, annot=True, fmt='.0f')
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/2aaebe44-aa30-4111-bac7-88cd98389a42)

## 3) Data Processing
```
cat_data = pd.DataFrame()

cat_data = pd.concat([cat_data, pd.get_dummies(df['FrequentFlyer'], prefix='FrequentFlyer')], axis=1)
cat_data = pd.concat([cat_data, pd.get_dummies(df['AnnualIncomeClass'], prefix='AnnualIncomeClass')], axis=1)

df['AnnualIncomeClass'] = df['AnnualIncomeClass'].map({'Low Income':0,
                             'Middle Income':1,
                             'High Income':2})

cat_data = pd.concat([cat_data, df['AnnualIncomeClass']], axis=1)
cat_data
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/9494f641-c9ee-416a-a0ba-d060306f8987)
```
X = pd.concat([df[['Age', 'ServicesOpted']], cat_data], axis=1)
y = df['Target']
print(df['Target'])
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/031a559b-a6e0-49c3-856f-8919d3e6ca83)
## 4) Model Creation and Evaluation (XGBoost)
### Import XGBoost, model selection and evaluation
```
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report, roc_curve, auc
```
### Define
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=42)
```
### Create Generic function to fit data and display results/predictions
```
def fit_evaluate(clf, X_train, X_test, y_train, y_test):
    # fit model to training data
    clf.fit(X_train, y_train)

    # make predictions for train data
    y_pred_train = clf.predict(X_train)

    # make predictions for test data
    y_pred_test = clf.predict(X_test)
    # print evaluation
    print(classification_report(y_test, y_pred_test))
    print('\nConfusion Matrix: \n')
    s = sns.heatmap(confusion_matrix(y_test, y_pred_test), annot=True, fmt='g', cmap='YlGnBu');
    s.set(xlabel='Predicted class', ylabel='True class')
    plt.show()

    fpr_train, tpr_train, _ = roc_curve(y_pred_train,  y_train)
    auc_train = roc_auc_score(y_pred_train, y_train)
    plt.plot(fpr_train,tpr_train, color='Blue', label='train: auc='+f'{auc_train:.2f}')

    fpr_test, tpr_test, _ = roc_curve(y_pred_test,  y_test)
    auc_test = roc_auc_score(y_pred_test, y_test)
    plt.plot(fpr_test,tpr_test, color='Red', label='test: auc='+f'{auc_test:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.legend(loc=4)
    plt.show()
```
### XGBoost
```
modelXGB = xgb.XGBClassifier(objective='binary:logistic', eval_metric="auc")
print('* XGBoost Classifier * \n')
fit_evaluate(modelXGB, X_train, X_test, y_train, y_test)
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/9f57c9c4-b1ec-49c1-9527-2faa8f01a1fe)
## 5) Hyperparameter Tuning (XGBoost)
### Building pipeline for hyperparameter tuning
```
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2

pipe = Pipeline([
  ('fs', SelectKBest()),
  ('clf', xgb.XGBClassifier(objective='binary:logistic', scale_pos_weight=9))
])
```
### Hyper parameter tuning - grid search
```
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
```
### Define search space for grid search
```
search_space = [
  {
    'clf__n_estimators': [50, 100, 150, 200],
    'clf__learning_rate': [0.01, 0.1],
    'clf__max_depth': range(2, 4),
    'clf__colsample_bytree': [i/10.0 for i in range(2, 5)],
    'clf__gamma': [i/10.0 for i in range(3)],
    'fs__score_func': [chi2],
    'fs__k': [2],
  }
]
```
### Define cross validation
```
kfold = KFold(n_splits=5)
```
### AUC and accuracy as score
```
scoring = {'AUC':'roc_auc', 'Accuracy':make_scorer(accuracy_score), 'F1 score': 'f1_micro'}
```
### Define grid search
```
grid = GridSearchCV(
  pipe,
  param_grid=search_space,
  cv=kfold,
  scoring=scoring,
  refit='AUC',
  verbose=1,
  n_jobs=-1
)
```
### Fit grid search
```
xgb_model_clv_GS = grid.fit(X_train, y_train)
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/765ee325-ca5f-41eb-ad60-afe65fd2f631)
## 6) XGBoost equipped with Hyper parameter tuning
```
modelXGB = xgb.XGBClassifier(
 learning_rate =0.01,
 n_estimators=100,
 max_depth=2,
 min_child_weight=1,
 gamma=0,
 subsample=0.8,
 colsample_bytree=0.4,
 objective= 'binary:logistic',
 eval_metric="auc")
print('* XGBoost Classifier * \n')
fit_evaluate(modelXGB, X_train, X_test, y_train, y_test)
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/8bd7ffbd-6f32-4e41-886f-553c44329c31)
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/5fdee6f1-072f-4920-8dbb-ec3ef8b02ef6)
### Feature Importance
```
feature_importances = modelXGB.feature_importances_
feature_names = X_train.columns  # Assuming X_train is a DataFrame
feature_importance_dict = dict(zip(feature_names, feature_importances))
sorted_feature_importance = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
sorted_feature_names, sorted_importance_values = zip(*sorted_feature_importance)
plt.figure(figsize=(12, 6))
plt.bar(sorted_feature_names, sorted_importance_values)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.xticks(rotation=90)  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()
```
![image](https://github.com/ginga924/MADT8101_Customer-Analytics999/assets/136943349/e77bc972-7566-4956-bfbc-51a08cccb050)

# Contact ✉️
If you have any inquiries or recommendations concerning this project, do not hesitate to reach out to me at https://www.linkedin.com/in/phaninthorn-swanyawatthaga-93467725a/

