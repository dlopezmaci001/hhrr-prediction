# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 16:51:13 2020

@author: dlopezmacias

https://www.kaggle.com/latencys/hr-viz-classifier
"""

# =============================================================================
# Set path
# =============================================================================
import os 

os.chdir(r'C:\Users\dlopezmacias\Documents\GitHub\HHRR')

# =============================================================================
# Load libraries
# =============================================================================

import pandas as pd 
import numpy as np
import datetime as dt
from datetime import date
from dateutil.relativedelta import relativedelta
import matplotlib.pyplot as plt
import seaborn as sns
from progressbar import *  

# =============================================================================
# Start code 
# =============================================================================

# Load df

df = pd.read_csv('HHRR_dataset.csv',sep=';',decimal='.',encoding='latin-1')

# Transform data 

df.EmpID = df.EmpID.astype(str)

date_column = ['DOB','DateofHire','DateofTermination',
                   'LastPerformanceReview_Date']

for i in date_column:
    df[i] = pd.to_datetime(df[i])
    for index,value in df.iterrows():
        if df.loc[index,i].year > dt.date.today().year:
            df.loc[index,i] = df.loc[index,i] - relativedelta(years = 100)
    
# =============================================================================
# Create variables     
# =============================================================================
            
df['Age'] = (pd.to_datetime(date.today()) - df.DOB) / np.timedelta64(1, 'Y')

df.Age = df.Age.round(0)

df['years_working'] = (pd.to_datetime(date.today()) - df.DateofHire)\
    / np.timedelta64(1, 'Y')
        
df.years_working = df.years_working.round(0)
    
df['days_since_eval'] = (pd.to_datetime(date.today())\
                         - df.LastPerformanceReview_Date).dt.days

df['days_since_eval'] = df['days_since_eval'].replace(np.nan,0)    

df = df.loc[df['DateofTermination'].isnull()]

# =============================================================================
# Drop Columns I don't need     
# =============================================================================

df = df.drop(['DOB','DateofHire','DateofTermination',
                   'LastPerformanceReview_Date','Termd','Sex',
                   'EmpStatusID','EmploymentStatus','DeptID',
                   'DaysLateLast30','PerformanceScore',
                   'PositionID','FromDiversityJobFairID',
                   'ManagerID','Department'], axis = 1) 

# =============================================================================
# EDA
# =============================================================================

# # Age
# df.groupby(['GenderID']).mean().Age.plot(kind='bar')

# # Pay Rate
# df.groupby(['GenderID']).mean().PayRate.plot(kind='bar')

# # Score
# df.groupby(['GenderID', 'PerfScoreID']).size().unstack().plot(kind='bar', 
#                                                               stacked=True,
#                                                               figsize=(10,10))

# # Grouped
# plt.figure(figsize=(15,15))
# sns.countplot(x='GenderID', data=df, hue = 'PerfScoreID', palette="Set1")
# plt.title('Age vs. Sex')

# =============================================================================
# Subset df based on those who haven't started 
# =============================================================================

df = df.loc[df.TermReason == 'N/A - still employed',]

# Drop columsn not needed 

df = df.drop('TermReason', axis = 1)

# =============================================================================
# Create new categories to work
# =============================================================================

# Define the bins to group by age

df['Age'].quantile(q=[0.25,0.50,0.75])

bins = [18, 34, 40, 45, 120]
labels = ['18-34', '35-40', '41-45', '+45']
df['agerange'] = pd.cut(df.Age, bins, labels = labels,include_lowest = True)

# Define the bins to group by years working

df['years_working'].quantile(q=[0.25,0.50,0.75])

bins = [0, 5, 6, 8, 120]
labels = ['junior', 'experience', 'manager', 'director']
df['wk_category'] = pd.cut(df.years_working, bins, labels = labels,include_lowest = True)

# Define the bins to group by special projects

df['SpecialProjectsCount'].quantile(q=[0.25,0.50,0.75])

bins = [0,1,120]
labels = ['no_special_project', 'in_special_project']
df['special_project'] = pd.cut(df.SpecialProjectsCount, bins, labels = labels,include_lowest = True)

 # Define the bins to group by emp satisfaction

df['EmpSatisfaction'].quantile(q=[0.25,0.50,0.75])

bins = [0,3,4,120]
labels = ['Low_satis', 'Medium_satis','High_satis']
df['emp_satisfaction'] = pd.cut(df.EmpSatisfaction, bins, labels = labels,include_lowest = True)
   
 # Define the bins to group by engagement 

df['EngagementSurvey'].quantile(q=[0.25,0.50,0.75])

bins = [0,2.1,3.6,120]
labels = ['Low_eng', 'Medium_eng','High_eng']
df['eng_satisfaction'] = pd.cut(df.EngagementSurvey, bins, labels = labels,include_lowest = True)

 # Define the bins to group by days since eval
   
df['days_since_eval'].quantile(q=[0.25,0.50,0.75])

bins = [0,385,401,414,1000]

labels = ['370-385 eval', '385-401 eval','402-414 eval','+ 415 eval']
df['days_since_evaluation'] = pd.cut(df.days_since_eval, bins, labels = labels,include_lowest = True)

 # Define the bins to group by payrate
   
df['PayRate'].quantile(q=[0.25,0.50,0.75])

bins = [0,20,26,53,1000]

labels = ['0-20 p.rate', '21-26 p.rate','27-53 p.rate','+ 53 p.rate']
df['pay_rate'] = pd.cut(df.PayRate, bins, labels = labels,include_lowest = True)
    
# Drop columns not needed

df = df.drop(['Age','years_working','SpecialProjectsCount','EmpSatisfaction',
              'EngagementSurvey','days_since_eval','PayRate'],axis = 1)

# # Add manager ID to each column val

# for index,value in df.iterrows():
#     df.loc[index,'ManagerID'] = str(df.loc[index,'ManagerID']) +'_' + 'ManagerID'

# Set employee ID as index

df.set_index('EmpID',inplace = True)

# =============================================================================
# Convert columns into dummy variables 
# =============================================================================

df1 = df.filter(like="PerfScoreID")

df = df.loc[:, df.columns != 'PerfScoreID']

df = pd.get_dummies(df.apply(pd.Series).stack()).sum(level=0)

df = df.rename(columns = {0.0:'is_male'})

df = df.drop(1.0,axis = 1)

df = pd.merge(df, df1, left_index=True, right_index=True)

# =============================================================================
# Check corr between vars
# =============================================================================

corr = df.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    horizontalalignment='right',
    fontsize=5
)

"""
There only seems to be high corr between job and payrate
"""

# =============================================================================
# Divide into test and train 
# =============================================================================

from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(df, test_size=0.5)

y_train = X_train.filter(like="PerfScoreID")

X_train = X_train.drop(columns = 'PerfScoreID')

y_test = X_test.filter(like="PerfScoreID")

X_test = X_test.drop(columns = 'PerfScoreID')

# =============================================================================
# Decision Tree
# =============================================================================
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn import tree
from sklearn.metrics import accuracy_score

# define model
weights = {1:20.25, 2:13.5,3:1,4:6.6}

dt = DecisionTreeClassifier(criterion = 'gini',random_state = 0,
                            max_depth = 10, min_samples_leaf = 5,
                            class_weight = weights)

# Train 

dt.fit(X_train, y_train)

dotfile = open(r'C:\Users\dlopezmacias\Documents\GitHub\HHRRdtree2.dot', 'w')

tree.export_graphviz(dt, out_file = dotfile, feature_names = X_train.columns)

dotfile.close()

y_pred = dt.predict(X_train)

confusion_matrix(y_train,y_pred)

score = accuracy_score(y_train, y_pred)
print('The score for this iteration was: %s' % score)

# Test 

y_pred_test = dt.predict(X_test)

confusion_matrix(y_test,y_pred_test)

score = accuracy_score(y_test, y_pred_test)
print('The score for this iteration was: %s' % score)

# Train: 0.78 
# Test: 0.8

# =============================================================================
# KNN
# =============================================================================

from sklearn.neighbors import KNeighborsClassifier

methods = ['auto', 'ball_tree', 'kd_tree', 'brute']

for i in methods:
    for k in range(1,10):
    
        knn = KNeighborsClassifier(algorithm = i, n_neighbors = k)
    
        knn.fit(X_train, y_train)
        
        y_pred = knn.predict(X_train)
        
        confusion_matrix(y_train,y_pred)
        
        score = accuracy_score(y_train, y_pred)
        
        print('The score for using %s and %s neigh this train iteration was: %s' % (i,k,score))
        
        y_pred_test = knn.predict(X_test)
              
        score = accuracy_score(y_test, y_pred_test)
        
        print('The score for using %s and %s neigh this test iteration was: %s' % (i,k,score))
        
# Apply results
        
# Use Brute and K=3 
        
knn = KNeighborsClassifier(algorithm = 'brute', n_neighbors = 3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_train)

confusion_matrix(y_train,y_pred)

score = accuracy_score(y_train, y_pred)

print('The score for using this train iteration was: %s' % score)

y_pred_test = knn.predict(X_test)
  
score = accuracy_score(y_test, y_pred_test)

print('The score for using this test iteration was: %s' % score)

# Train 0.8
# Test: 0.825

# =============================================================================
# XG Boost 
# =============================================================================

from xgboost import XGBClassifier

# fit model no training data
model = XGBClassifier()

model.fit(X_train, y_train)

print(model)

y_pred = model.predict(X_train)

confusion_matrix(y_train,y_pred)

score = accuracy_score(y_train, y_pred)

print('The score for using this train iteration was: %s' % score)

y_pred_test = model.predict(X_test)
  
score = accuracy_score(y_test, y_pred_test)

print('The score for using this test iteration was: %s' % score)

confusion_matrix(y_test,y_pred_test)

# Plot 

from xgboost import plot_importance

plot_importance(model, max_num_features=10) # top 10 most important features

# =============================================================================
# Naive Bayes 
# =============================================================================

# training a Naive Bayes classifier 
from sklearn.naive_bayes import GaussianNB 

gnb = GaussianNB().fit(X_train, y_train) 

y_pred = gnb.predict(X_train) 

confusion_matrix(y_train,y_pred)

score = accuracy_score(y_train, y_pred)
print('The score for using this train iteration was: %s' % score)

y_pred_test = gnb.predict(X_test)
  
score = accuracy_score(y_test, y_pred_test)
print('The score for using this test iteration was: %s' % score)

confusion_matrix(y_test,y_pred_test)

# Train 0.3
# Test 0.3
# =============================================================================
# Random Forest 
# =============================================================================

from sklearn.ensemble import RandomForestClassifier

# Fitting Random Forest Classification to the Training set
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_train) 

confusion_matrix(y_train,y_pred)

score = accuracy_score(y_train, y_pred)
print('The score for using this train iteration was: %s' % score)

y_pred_test = classifier.predict(X_test)
  
score = accuracy_score(y_test, y_pred_test)
print('The score for using this test iteration was: %s' % score)

confusion_matrix(y_test,y_pred_test)

# =============================================================================
# Iterative Decision tree
# =============================================================================
from sklearn.metrics import precision_score,f1_score


def check_model(trainX,trainy,testX,testy):
    # define model
    balance = [{1:20.25, 2:13.5,3:1,4:6.6},
               {1:12.25, 2:27.5,3:1,4:27.5},{1:100, 2:1,3:1,4:100},
               {1:1, 2:1,3:1,4:1}]
    
    crit = ['gini','entropy']
    
    depth = range(1,5)
    
    leaves = range(1,20)
    
    min_wight = range(10,50)
    results = pd.DataFrame({'criteria':[],'weight':[],
                        'depth': [], 'leaf': [], 'balance': [],'score':[],
                        'precision':[],'f1':[]})
    for i in balance:
            for leaf in leaves:
                for deep in depth:  
                    for criteria in crit:
                        for weight in min_wight:
                            weight = weight/100
                            dt = DecisionTreeClassifier(criterion = criteria,
                                                        random_state = 0,
                                                        max_depth = deep, 
                                                        min_samples_leaf = leaf,
                                                        class_weight = i,
                                                        min_weight_fraction_leaf = weight)          
                            # Train 
                            
                            dt.fit(trainX, trainy)
                            
                            # Test 
                            
                            y_pred_test = dt.predict(testX)
                            
                            confusion_matrix(testy,y_pred_test)
                            
                            score = accuracy_score(testy, y_pred_test)
                            
                            precision = precision_score(testy,y_pred_test,
                                                        average = 'weighted',
                                                        labels=np.unique(y_pred_test))
                            f1 = f1_score(testy,y_pred_test, average = 'weighted',
                                          labels=np.unique(y_pred_test)) 
                            
                            results = results.append({'criteria':criteria,'weight':weight,
                                                      'depth': deep, 'leaf': leaf, 
                                                      'balance': i,'score':score,
                                                      'precision': precision,
                                                      'f1': f1}, 
                                                      ignore_index=True)       
                            
        results = results.sort_values(['precision','f1','score'],ascending = False)
            
# Validate results

dt = DecisionTreeClassifier(criterion = 'entropy',random_state = 0,
                                        max_depth = 4, 
                                        min_samples_leaf = 1,
                                        class_weight = {1: 20.25, 2: 13.5, 3: 1, 4: 6.6},
                                        min_weight_fraction_leaf = 0.11)
            
# Train 

dt.fit(X_train, y_train)
            
y_pred_test = dt.predict(X_test)

confusion_matrix(y_test, y_pred_test)

precision_score(y_test,y_pred_test, average = 'weighted',labels=np.unique(y_pred))

f1_score(y_test,y_pred_test, average = 'macro',labels=np.unique(y_pred))  
        
dotfile = open(r'C:\Users\dlopezmacias\Documents\GitHub\HHRRdtree2.dot', 'w')

tree.export_graphviz(dt, out_file = dotfile, feature_names = X_train.columns)

dotfile.close()

# =============================================================================
# SMOTE
# =============================================================================

from imblearn.over_sampling import SMOTE
from sklearn.utils import resample

oversample =SMOTE(k_neighbors=2)

# concatenate our training data back together
X = pd.concat([X_train, y_train], axis=1)

# separate minority and majority classes
bajo = X[X.PerfScoreID==1]
mejora = X[X.PerfScoreID==2]
bien = X[X.PerfScoreID==3]
excede = X[X.PerfScoreID==4]

# upsample minority
bajo_upsampled = resample(bajo,
                          replace=True, # sample with replacement
                          n_samples=len(bien), # match number in majority class
                          random_state=27) # reproducible results

mejora_upsampled = resample(mejora,
                          replace=True, # sample with replacement
                          n_samples=len(bien), # match number in majority class
                          random_state=27) # reproducible results

excede_upsampled = resample(excede,
                          replace=True, # sample with replacement
                          n_samples=len(bien), # match number in majority class
                          random_state=27) # reproducible results

# combine majority and upsampled minority
upsampled = pd.concat([bajo_upsampled, mejora_upsampled,excede_upsampled,bien])

y_train_up = upsampled.PerfScoreID
X_train_up = upsampled.drop('PerfScoreID', axis=1)

for i in balance:
        for leaf in leaves:
            for deep in depth:  
                for criteria in crit:
                    for weight in min_wight:
                        weight = weight/100
                        dt = DecisionTreeClassifier(criterion = criteria,
                                                    random_state = 0,
                                                    max_depth = deep, 
                                                    min_samples_leaf = leaf,
                                                    class_weight = i,
                                                    min_weight_fraction_leaf = weight)          
                        # Train 
                        
                        dt.fit(X_train_up, y_train_up)
                        
                        # Test 
                        
                        y_pred_test = dt.predict(X_test)
                        
                        confusion_matrix(y_test,y_pred_test)
                        
                        score = accuracy_score(y_test, y_pred_test)
                        
                        precision = precision_score(y_test,y_pred_test,
                                                    average = 'weighted',
                                                    labels=np.unique(y_pred_test))
                        f1 = f1_score(y_test,y_pred_test, average = 'weighted',
                                      labels=np.unique(y_pred_test)) 
                        
                        results = results.append({'criteria':criteria,'weight':weight,
                                                  'depth': deep, 'leaf': leaf, 
                                                  'balance': i,'score':score,
                                                  'precision': precision,
                                                  'f1': f1}, 
                                                  ignore_index=True)       
                        
results = results.sort_values(['precision','f1','score'],ascending = False)

# Validate results

dt = DecisionTreeClassifier(criterion = 'gini',random_state = 0,
                                        max_depth = 2, 
                                        min_samples_leaf = 1,
                                        class_weight = {1: 20.25, 2: 13.5, 3: 1, 4: 6.6},
                                        min_weight_fraction_leaf = 0.12)
            
# Train 

dt.fit(X_train_up, y_train_up)
            
y_pred_test = dt.predict(X_test)

confusion_matrix(y_test, y_pred_test)

# =============================================================================
# function
# =============================================================================

def check_model(trainX,trainy,testX,testy):
    print('1')
    # define model
    global results
    balance = [{1:20.25, 2:13.5,3:1,4:6.6},
               {1:12.25, 2:27.5,3:1,4:27.5},{1:100, 2:1,3:1,4:100},
               {1:1, 2:1,3:1,4:1}]
    
    crit = ['gini','entropy']
    
    depth = range(1,5)
    
    leaves = range(1,20)
    
    min_wight = range(10,50)
    
    results = pd.DataFrame({'criteria':[],'weight':[],
                        'depth': [], 'leaf': [], 'balance': [],'score':[],
                        'precision':[],'f1':[]})
    print('2')
    for i in balance:
            for leaf in leaves:
                for deep in depth:  
                    for criteria in crit:
                        for weight in min_wight:
                            weight = weight/100
                            dt = DecisionTreeClassifier(criterion = criteria,
                                                        random_state = 0,
                                                        max_depth = deep, 
                                                        min_samples_leaf = leaf,
                                                        class_weight = i,
                                                        min_weight_fraction_leaf = weight)          
                            # Train 
                            
                            dt.fit(trainX, trainy)
                            
                            # Test 
                            
                            y_pred_test = dt.predict(testX)
                            
                            confusion_matrix(testy,y_pred_test)
                            
                            score = accuracy_score(testy, y_pred_test)
                            
                            precision = precision_score(testy,y_pred_test,
                                                        average = 'weighted',
                                                        labels=np.unique(y_pred_test))
                            f1 = f1_score(testy,y_pred_test, average = 'weighted',
                                          labels=np.unique(y_pred_test)) 
                            
                            results = results.append({'criteria':criteria,'weight':weight,
                                                      'depth': deep, 'leaf': leaf, 
                                                      'balance': i,'score':score,
                                                      'precision': precision,
                                                      'f1': f1}, 
                                                      ignore_index=True)       
    print('3')                        
    results = results.sort_values(['precision','f1','score'],ascending = False)
            
    
check_model(X_train,y_train,X_test,y_test)
