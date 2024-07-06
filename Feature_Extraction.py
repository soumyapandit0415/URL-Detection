# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 12:33:19 2024

@author: indur
"""

# Check if GPU is being used.

import pandas as pd
# Loading the downloaded dataset
df = pd.read_csv(r"C:\Users\indur\Downloads\url_genie-main\Research_Notebooks\Url_Processed.csv")
#Removing the unnamed columns as it is not necesary.
df = df.drop('Unnamed: 0',axis=1)
print(df.head(5))
print(df.info())
# Printing number of legit and fraud domain urls
print(df["label"].value_counts())

'''
## **Extracting Length Features**
#### Length features of the following properties can be extracted for relevant data analysis
- Length Of Url
- Length of Hostname
- Length Of Path
- Length Of First Directory
- Length Of Top Level Domain
'''
#Importing dependencies
from urllib.parse import urlparse
import os.path

# changing dataframe variable
urldata = df

#Length of URL (Phishers can use long URL to hide the doubtful part in the address bar)
urldata['url_length'] = urldata['url'].apply(lambda i: len(str(i)))

#Hostname Length
urldata['hostname_length'] = urldata['url'].apply(lambda i: len(urlparse(i).netloc))

#Path Length
urldata['path_length'] = urldata['url'].apply(lambda i: len(urlparse(i).path))

#First Directory Length
def fd_length(url):
    urlpath= urlparse(url).path
    try:
        return len(urlpath.split('/')[1])
    except:
        return 0

urldata['fd_length'] = urldata['url'].apply(lambda i: fd_length(i))
print(urldata.head(5))

'''
## **Occurrence Count Features**
Occurrences of specific characters within malicious domains can be a relevant indicator for malicious domains
- Count Of '-'
- Count Of '@'
- Count Of '?'
- Count Of '%'
- Count Of '.'
- Count Of '='
- Count Of 'http'
- Count Of 'www'
- Count Of Digits
- Count Of Letters
- Count Of Number Of Directories
'''


# Count of how many times a special character appearsin url

urldata['count-'] = urldata['url'].apply(lambda i: i.count('-'))

urldata['count@'] = urldata['url'].apply(lambda i: i.count('@'))

urldata['count?'] = urldata['url'].apply(lambda i: i.count('?'))

urldata['count%'] = urldata['url'].apply(lambda i: i.count('%'))

urldata['count.'] = urldata['url'].apply(lambda i: i.count('.'))

urldata['count='] = urldata['url'].apply(lambda i: i.count('='))

urldata['count-http'] = urldata['url'].apply(lambda i : i.count('http'))

urldata['count-https'] = urldata['url'].apply(lambda i : i.count('https'))

urldata['count-www'] = urldata['url'].apply(lambda i: i.count('www'))

def digit_count(url):
    digits = 0
    for i in url:
        if i.isnumeric():
            digits = digits + 1
    return digits
urldata['count-digits']= urldata['url'].apply(lambda i: digit_count(i))


def letter_count(url):
    letters = 0
    for i in url:
        if i.isalpha():
            letters = letters + 1
    return letters
urldata['count-letters']= urldata['url'].apply(lambda i: letter_count(i))

def no_of_dir(url):
    urldir = urlparse(url).path
    return urldir.count('/')
urldata['count_dir'] = urldata['url'].apply(lambda i: no_of_dir(i))

print(urldata.head(5))

'''
## **Binary Features**

The following binary features can also be extracted from the dataset
- Use of IP or not
- Use of Shortening URL or not

#### **IP Address in the URL**

Checks for the presence of IP address in the URL. URLs may have IP address instead of domain name. If an IP address is used as an alternative of the domain name in the URL, we can be sure that someone is trying to steal personal information with this URL.
'''

import re

#Use of IP or not in domain
def having_ip_address(url):
    match = re.search(
        '(([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\.'
        '([01]?\\d\\d?|2[0-4]\\d|25[0-5])\\/)|'  # IPv4
        '((0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\.(0x[0-9a-fA-F]{1,2})\\/)' # IPv4 in hexadecimal
        '(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}', url)  # Ipv6
    if match:
        # print match.group()
        return -1
    else:
        # print 'No matching pattern found'
        return 1
urldata['use_of_ip'] = urldata['url'].apply(lambda i: having_ip_address(i))

'''
#### **Using URL Shortening Services “TinyURL”**

URL shortening is a method on the “World Wide Web” in which a URL may be made considerably 
smaller in length and still lead to the required webpage. This is accomplished by means of an 
“HTTP Redirect” on a domain name that is short, which links to the webpage that has a long URL.
'''
# use of url shortening service
def shortening_service(url):
    match = re.search('bit\.ly|goo\.gl|shorte\.st|go2l\.ink|x\.co|ow\.ly|t\.co|tinyurl|tr\.im|is\.gd|cli\.gs|'
                      'yfrog\.com|migre\.me|ff\.im|tiny\.cc|url4\.eu|twit\.ac|su\.pr|twurl\.nl|snipurl\.com|'
                      'short\.to|BudURL\.com|ping\.fm|post\.ly|Just\.as|bkite\.com|snipr\.com|fic\.kr|loopt\.us|'
                      'doiop\.com|short\.ie|kl\.am|wp\.me|rubyurl\.com|om\.ly|to\.ly|bit\.do|t\.co|lnkd\.in|'
                      'db\.tt|qr\.ae|adf\.ly|goo\.gl|bitly\.com|cur\.lv|tinyurl\.com|ow\.ly|bit\.ly|ity\.im|'
                      'q\.gs|is\.gd|po\.st|bc\.vc|twitthis\.com|u\.to|j\.mp|buzurl\.com|cutt\.us|u\.bb|yourls\.org|'
                      'x\.co|prettylinkpro\.com|scrnch\.me|filoops\.info|vzturl\.com|qr\.net|1url\.com|tweez\.me|v\.gd|'
                      'tr\.im|link\.zip\.net',
                      url)
    if match:
        return -1
    else:
        return 1
urldata['short_url'] = urldata['url'].apply(lambda i: shortening_service(i))

print(urldata.head(5))
urldata.to_csv("Url_Processed.csv")




'''
Code explanation
In this code, we are performing several data preprocessing steps and feature extraction from a dataset containing URLs. Here's a breakdown of what each section of the code does:

1. **Loading the Dataset**: We start by loading a dataset from a CSV file using pandas.

2. **Removing Unnamed Column**: If the dataset contains an "Unnamed" column, we remove it as it is not necessary.

3. **Displaying Dataset Information**: We print the first 5 rows of the dataset and its information (such as column names, data types, and non-null counts) using the `head()` and `info()` functions.

4. **Counting Legitimate and Malicious URLs**: We print the count of legitimate and malicious URLs in the dataset using the `value_counts()` function on the "label" column.

5. **Extracting Length Features**: We extract various length-related features from the URLs, such as the length of the URL, hostname, path, and first directory. These features provide insights into the structure and complexity of the URLs.

6. **Extracting Occurrence Count Features**: We count the occurrences of specific characters and substrings within the URLs, such as '-', '@', '?', '%', '.', '=', 'http', 'https', and 'www'. These features can be indicative of suspicious or malicious URLs.

7. **Extracting Binary Features**: We extract binary features indicating the presence of an IP address in the URL and the usage of URL shortening services. These features help identify URLs with potentially malicious intent.

8. **Saving Processed Dataset**: Finally, we save the processed dataset with extracted features to a new CSV file named "Url_Processed.csv" using the `to_csv()` function.

Overall, this code preprocesses the dataset and extracts various features from the URLs to prepare the data for further analysis or machine learning model training.
'''