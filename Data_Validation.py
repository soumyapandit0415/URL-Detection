# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:26:07 2024

@author: indur
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("darkgrid")
urldata = pd.read_csv("C:\\Users\\indur\Downloads\\url_genie-main\\Research_Notebooks\\Url_Processed.csv")

# Drop the 'Unnamed: 0' column if it's not needed
urldata.drop("Unnamed: 0", axis=1, inplace=True)

# Check for missing values
print("Missing values in urldata:")
print(urldata.isnull().sum())

# Remove rows with missing values, or impute them if needed
urldata.dropna(inplace=True)

#Number of benign and malicious urls
i = urldata["label"].value_counts()
print(i)

# Check correlation between numeric columns
numeric_urldata = urldata.select_dtypes(include=['number'])
corrmat = numeric_urldata.corr()
print(corrmat)

# Plot correlation heatmap
f, ax = plt.subplots(figsize=(25, 19))
sns.heatmap(corrmat, square=True, annot=True, annot_kws={'size': 10})
plt.show()

plt.figure(figsize=(13,5))
sns.countplot(x='label',data=urldata)
plt.title("Count Of URLs",fontsize=20)
plt.xlabel("Type Of URLs",fontsize=18)
plt.ylabel("Number Of URLs",fontsize=18)
plt.show()


# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming `urldata` is your DataFrame and `label` is a Series
# Exclude non-numeric columns before computing correlation matrix
numeric_urldata = urldata.select_dtypes(include=['number'])

from matplotlib import rcParams
rcParams['figure.figsize'] = 15,10

# plotting distrubutions
features = list(urldata.columns) 
features.remove("url")
features.remove("result")

hist_features = ["url_length","hostname_length","path_length","fd_length"]
value_names = ["URL Length", "Hostname Length", "Path Length", "FD Length"]

for idx, i in enumerate(hist_features):
    sns.histplot(data=urldata,x=i,bins=100,hue='label')
    plt.xlabel(value_names[idx],fontsize=18)
    plt.ylabel("Number Of Urls",fontsize=18)
    plt.xlim(0,150)
    plt.show()


features = list(urldata.columns) # list of feature names
features.remove("url")

rcParams['figure.figsize'] = 12,8

for i in features:
   
    if i in hist_features:
      continue
   
    sns.countplot(x=i,data=urldata)
    plt.xlabel(i,fontsize=18)
    plt.ylabel("Number Of Urls",fontsize=18)
    plt.show()