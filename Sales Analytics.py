#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 17:11:00 2020

@author: Heqing Sun
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Get current working directory
os.getcwd()

# Read csv files
trades = pd.read_csv("./data/trades.csv")
maps = pd.read_csv("./data/partnership_map.csv")
tps = pd.read_csv("./data/touchpoints.csv")

# Define some functions for EDA
def print_dataframe_description(df, col):
    print('Column Name:', col)
    print('Number of Rows:', len(df.index))
    print('Number of Missing Values:', df[col].isnull().sum())
    print('Percent Missing:', df[col].isnull().sum()/len(df.index)*100, '%')
    print('Number of Unique Values:', len(df[col].unique()))
    print('\n')

# For continuous variables    
def print_descriptive_stats(df, col):
    print('Column Name:', col)
    print('Mean:', np.mean(df[col]))
    print('Median:', np.nanmedian(df[col]))
    print('Standard Deviation:', np.std(df[col]))
    print('Minimum:', np.min(df[col]))
    print('Maximum:', np.max(df[col]))

# Plotting countplots   
def plot_counts(df, col):
    sns.set(style='darkgrid')
    ax = sns.countplot(x=col, data=df)
    plt.xticks(rotation=90)
    plt.title('Count Plot')
    plt.show()
    
# Plotting distribution plots 
def plot_distribution(df, col):
    sns.set(style='darkgrid')
    ax = sns.distplot(df[col].dropna())
    plt.xticks(rotation=90)
    plt.title('Distribution Plot')
    plt.show()
    
#############################################################################
########################### EDA and Data Cleaning ###########################
#############################################################################
for col in list(trades.columns.values):
    print_dataframe_description(trades, col)
# Column Name: TRAN_ID
# Number of Rows: 10357
# Number of Missing Values: 0
# Percent Missing: 0.0 %
# Number of Unique Values: 8747
    
## Weird because this transcation identifier should be unique for each transaction

print_descriptive_stats(trades, 'TRAN_ID')
# Column Name: TRAN_ID
# Mean: 40167988.51588298
# Median: 48223425.0
# Standard Deviation: 45623596.47707455
# Minimum: -99955463
# Maximum: 99999396

## This column has negative value, which doesn't make sense, and abs(min) = max

for col in list(maps.columns.values):
    print_dataframe_description(maps, col)

for col in list(tps.columns.values):
    print_dataframe_description(tps, col)

# Subset a new dataframe when TRAN_ID variable is negative
trades_neg = trades.loc[trades['TRAN_ID'] < 0]

TRAN_ID_neg = trades_neg['TRAN_ID']
TRAN_AMT_neg = trades_neg['TRAN_AMT']
TRAN_ID_neg_abs = abs(TRAN_ID_neg)
TRAN_AMT_neg_abs = abs(TRAN_AMT_neg)

len(TRAN_ID_neg.unique()) # 1150 unique values, 1380 obs
len(TRAN_AMT_neg.unique()) # 1150 unique values

# Subset the original trades dataset by TRAN_ID equals the absolute value of the negatve TRAN_ID
trades_id_equal_absneg = trades.loc[trades['TRAN_ID'].isin(TRAN_ID_neg_abs)]
trades_id_equal_absneg

len(trades_id_equal_absneg['TRAN_ID'].unique()) # 1150 unique values
len(trades_id_equal_absneg['TRAN_AMT'].unique()) # 1150 unique values

## Due to reversal trades, need to take care of the negatvie TRAN_ID

# Remove the negative TRAN_ID obs
trades_orig = trades.copy()
trades = trades[trades['TRAN_ID'] > 0] # 8977 obs

# Ordered by transaction date time descending and keep the first one
trades = trades.sort_values(by='TRAN_DATE', ascending = False)
trades_new = trades.drop_duplicates(subset='TRAN_ID', keep="first") # 7597 obs

for col in list(trades_new.columns.values):
    print_dataframe_description(trades_new, col)
# Column Name: TRAN_ID
# Number of Rows: 7597
# Number of Missing Values: 0
# Percent Missing: 0.0 %
# Number of Unique Values: 7597
## Looks good now

# Filling missing values with 0 in ELITE_FIRM column
trades_new['ELITE_FIRM'].fillna(0, inplace=True)
trades_new['ELITE_FIRM'].value_counts()
# Now trades_new has 7597 obs, should be clean now

# Write cleaned trades_new table to csv and pickle file
trades_new.to_csv('trades_new.csv')
trades_new.to_pickle('trades_new.pkl')

#######################################################################
########################### 1. Top Advisors ###########################
#######################################################################
# Subset trades_new dataset by trade type
trades_new['TRADE_TYPE'].value_counts()
# I    5177
# P    2420

trades_new_I = trades_new[trades_new['TRADE_TYPE']=='I']
trades_new_P = trades_new[trades_new['TRADE_TYPE']=='P']

# Calculate the unique number of individual and partnership entity
len(trades_new_I['ID'].unique()) # 100 unique individuals
len(trades_new_P['ID'].unique()) # 50 unique partnerships

#####################################################################################
########################### 1. Top Advisors - Individuals ###########################
#####################################################################################
# Get top ten frequent trading individuals
freq_I = trades_new_I['ID'].value_counts()
freq_I = pd.DataFrame({'ID':freq_I.index, 'Trades_Counts_I':freq_I.values})
freq_I.head(10)
#            ID  Trades_Counts_I
# 0  3XO893HUB7              100
# 1  827HZMYB0T              100
# 2  FO8YWV23ZK               99
# 3  DQX3E7JVFZ               99
# 4  SCQ0XCQ7O1               98
# 5  2A9X4KHXZI               94
# 6  TPJKXTM0CB               94
# 7  SXS9F6CSY1               92
# 8  R89BBWNCD8               92
# 9  DF49LAD02I               92

# Calculate the sum of each individual TRAN_AMT
high_value_I = trades_new_I.groupby('ID').TRAN_AMT.sum()
high_value_I = pd.DataFrame({'ID':high_value_I.index, 'TRAN_AMT_SUM':high_value_I.values})

# Get the top ten high value trading individuals
high_value_I = high_value_I.sort_values(by='TRAN_AMT_SUM', ascending = False)
high_value_I.head(10)
#         ID      TRAN_AMT_SUM
# 97  ZEKBKU4CMP  2.719290e+08
# 83  TXW929QG7V  9.182620e+06
# 67  OMJMSPEXOA  8.282166e+06
# 35  AYGKZW5YPG  7.376097e+06
# 30  A4J0YU0L11  7.342080e+06
# 29  A06WP1H62W  7.303549e+06
# 32  AM03D1XUAP  6.471512e+06
# 84  U9EL7D4ERT  6.208968e+06
# 50  FO8YWV23ZK  5.909487e+06
# 66  NTEGEVCM9X  5.761956e+06

# Check if the 100 individuals' ID in trades dataset are exactly same as in touchopoints dataset
trades_I_ID = trades_new_I['ID'].unique()
trades_I_ID.sort()
tps_I_ID = tps['ID'].unique()
tps_I_ID.sort()
sum(tps_I_ID == trades_I_ID) # Sum = 100. Therefore, these two datasets contain same individual IDs

# maps dataset - since each row can contain multiple individual IDs, need to reformat it
maps_orig = maps.copy()

import ast
maps.INDIVIDUAL_IDS = maps.INDIVIDUAL_IDS.apply(ast.literal_eval)
part = []
ind = []
for _,row in maps.iterrows():
     part_id = row.PARTNERSHIP_ID
     for ind_id in row.INDIVIDUAL_IDS:
             part.append( part_id )
             ind.append( ind_id )
maps_new = pd.DataFrame({'PARTNERSHIP_ID':part, 'INDIVIDUAL_IDS':ind})

# Calculate the unique number of individual and partnership entity in maps dataset
len(maps_new['PARTNERSHIP_ID'].unique()) # 50 unique partnerships
len(maps_new['INDIVIDUAL_IDS'].unique()) # 99 unique individuals
## Therefore, one individual does not belong to any partnership and some individuals belong to several partnerships

######################################################################################
########################### 1. Top Advisors - Partnerships ###########################
######################################################################################
# Get top ten frequent trading partnerships - not standardized by partnership size!
freq_P = trades_new_P['ID'].value_counts()
freq_P = pd.DataFrame({'ID':freq_P.index, 'Trades_Counts_P':freq_P.values})
freq_P.head(10)
#   ID       Counts
# K2HI7Z6A4L    100
# MMF0ILGGR0     95
# 20B9HH7W8T     90
# ADY9EZRG5S     88
# U7BJKAE83O     87
# DNXZLTM3AA     81
# 6RUCM9BPJ9     81
# GSITSYMDWU     79
# MUJWN4XFML     79
# 08JLESLKWL     77

# Since different partnerships have different numbers of individuals, need to standardize the partnership by 'size'
maps_counts = maps_new.groupby('PARTNERSHIP_ID').INDIVIDUAL_IDS.count()
maps_counts = pd.DataFrame({'ID':maps_counts.index, 'Size':maps_counts.values}) 
## Here 'Size' means how many individuals that each partnership has

# Left join maps_counts table with the freq_P table so that can create a standardized frequency column
freq_P_merged = freq_P.merge(maps_counts, on='ID', how='left')
freq_P_merged['Trades_Counts_P/Size'] = freq_P_merged['Trades_Counts_P']/freq_P_merged['Size']

# Get top ten frequent trading partnerships - standardized by partnership size
freq_P_merged = freq_P_merged.sort_values(by=['Trades_Counts_P/Size'], ascending=False)
freq_P_merged.head(10)
#        ID_P        Trades_Counts_P Size         Trades_Counts_P/Size
# 2   20B9HH7W8T               90     2                  45.0
# 5   DNXZLTM3AA               81     2                  40.5
# 6   6RUCM9BPJ9               81     2                  40.5
# 10  RAO5TAG1BD               76     2                  38.0
# 12  8UD4S22K70               74     2                  37.0
# 13  7VBKY5B1PM               74     2                  37.0
# 11  AHTDNYP4A3               74     2                  37.0
# 14  ZW0J9NZAS3               73     2                  36.5
# 16  EQF1X3CNKZ               71     2                  35.5
# 18  N3GUJVQNZ8               67     2                  33.5

# Calculate the sum of each partnership TRAN_AMT
high_value_P = trades_new_P.groupby('ID').TRAN_AMT.sum()
high_value_P = pd.DataFrame({'ID':high_value_P.index, 'TRAN_AMT_SUM':high_value_P.values})

# Get the top ten high value trading partnership - not standardized by partnership size!
high_value_P = high_value_P.sort_values(by='TRAN_AMT_SUM', ascending = False)
high_value_P.head(10)
#             ID  TRAN_AMT_SUM
# 14  9KPSVHG8YA  2.705462e+08
# 15  A1Y388YRKE  2.931162e+07
# 39  RAO5TAG1BD  8.396714e+06
# 28  J7GOCJMT4X  5.508838e+06
# 16  ADY9EZRG5S  5.282480e+06
# 29  J8D2BETXMA  4.972868e+06
# 42  T9AD15KSTV  3.055455e+06
# 3   20B9HH7W8T  1.021609e+06
# 30  K2HI7Z6A4L  1.020800e+06
# 32  MMF0ILGGR0  9.577002e+05

# Same here, need to standardize the sum of TRAN_AMT by size
freq_P_merged = freq_P_merged.merge(high_value_P, on='ID', how='left')

# Calculate share for each individual in the same partnership (aka standarize the sum of TRAN_AMT by size)
freq_P_merged['TRAN_AMT_SUM/Size'] = freq_P_merged['TRAN_AMT_SUM']/freq_P_merged['Size']

# Get top ten high value trading partnerships - standardized by partnership size
freq_P_merged = freq_P_merged.sort_values(by=['TRAN_AMT_SUM/Size'], ascending=False)
freq_P_merged.head(10)
#    ID  Trades_Counts_P  ...  TRAN_AMT_SUM  TRAN_AMT_SUM/Size
# 38  9KPSVHG8YA               18  ...  2.705462e+08       2.705462e+07
# 42  A1Y388YRKE                2  ...  2.931162e+07       1.465581e+07
# 3   RAO5TAG1BD               76  ...  8.396714e+06       4.198357e+06
# 11  J7GOCJMT4X               54  ...  5.508838e+06       2.754419e+06
# 12  T9AD15KSTV               36  ...  3.055455e+06       1.527728e+06
# 0   20B9HH7W8T               90  ...  1.021609e+06       5.108045e+05
# 24  J8D2BETXMA               45  ...  4.972868e+06       4.972868e+05
# 2   6RUCM9BPJ9               81  ...  8.448986e+05       4.224493e+05
# 1   DNXZLTM3AA               81  ...  8.431994e+05       4.215997e+05
# 6   8UD4S22K70               74  ...  7.945603e+05       3.972801e+05

# Try to get the share of each individual comes from the partnership
temp1 = freq_P_merged[['ID','TRAN_AMT_SUM/Size']] # Here ID is the partnership ID
# Rename two columns name
temp1.columns = ['PARTNERSHIP_ID', 'Shares']

# Left join with the maps_new table
maps_new = maps_new.merge(temp1, on='PARTNERSHIP_ID', how='left')
temp2 = maps_new.copy()
temp2 = temp2.drop(columns=['PARTNERSHIP_ID'])

# Rolled up to individual level
temp3 = temp2.groupby('INDIVIDUAL_IDS').Shares.sum()
temp3 = pd.DataFrame({'ID':temp3.index, 'Shares':temp3.values}) 

# Left join with high_value_I
trades_I_all = high_value_I.merge(temp3, on='ID', how='left')
trades_I_all = trades_I_all.fillna(0) # filling missing values with 0
trades_I_all['TRAN_AMT_all'] = trades_I_all['TRAN_AMT_SUM'] + trades_I_all['Shares']
trades_I_all.head(10)
#            ID  TRAN_AMT_SUM        Shares  TRAN_AMT_all
# 0  ZEKBKU4CMP  2.719290e+08  5.784165e+05  2.725074e+08
# 1  TXW929QG7V  9.182620e+06  1.966045e+05  9.379224e+06
# 2  OMJMSPEXOA  8.282166e+06  7.383449e+05  9.020511e+06
# 3  AYGKZW5YPG  7.376097e+06  1.245506e+05  7.500648e+06
# 4  A4J0YU0L11  7.342080e+06  8.970757e+05  8.239156e+06
# 5  A06WP1H62W  7.303549e+06  2.751575e+07  3.481930e+07
# 6  AM03D1XUAP  6.471512e+06  1.748690e+05  6.646381e+06
# 7  U9EL7D4ERT  6.208968e+06  6.479852e+05  6.856953e+06
# 8  FO8YWV23ZK  5.909487e+06  5.681998e+05  6.477687e+06
# 9  NTEGEVCM9X  5.761956e+06  2.750626e+05  6.037018e+06

## Now trades_I_all dataset not only contains all individual trades, but also contains the share that comes from the partnership that the individual belongs to
# Write cleaned trades_I_all table to csv and pickle file
trades_I_all.to_csv('./data/clean/trades_I_all.csv')
trades_I_all.to_pickle('./data/clean/trades_I_all.pkl')

##########################################################################################
########################### 1. Top Advisors - High Engagements ###########################
##########################################################################################
# Get top ten individuals by IN_PERSON_MEETING
tps_eng_meeting = tps.groupby('ID').IN_PERSON_MEETING.sum()
tps_eng_meeting = pd.DataFrame({'ID':tps_eng_meeting.index, 'Meeting_Sum':tps_eng_meeting.values})
tps_eng_meeting = tps_eng_meeting.sort_values(by=['Meeting_Sum'], ascending=False)
tps_eng_meeting.head(10)
#         ID           Meeting_Sum
# 97  ZEKBKU4CMP          220
# 38  C3FC972N4Q          212
# 50  FO8YWV23ZK          210
# 61  LMPLLW4ENR          210
# 11  3B7BHD6UEU          208
# 80  SXS9F6CSY1          207
# 34  AWZPLRZ289          205
# 33  ARN5GMCI89          201
# 30  A4J0YU0L11          201
# 24  701CU2R5WE          197

# Get top ten individuals by WEBINAR
tps_eng_webinar = tps.groupby('ID').WEBINAR.sum()
tps_eng_webinar = pd.DataFrame({'ID':tps_eng_webinar.index, 'Webinar_Sum':tps_eng_webinar.values})
tps_eng_webinar = tps_eng_webinar.sort_values(by=['Webinar_Sum'], ascending=False)
tps_eng_webinar.head(10)
#             ID  Webinar_Sum
# 97  ZEKBKU4CMP         4401
# 77  RG0G1D96G3         4193
# 11  3B7BHD6UEU         4169
# 34  AWZPLRZ289         4002
# 7   1ZEAIO3U5N         3983
# 61  LMPLLW4ENR         3964
# 30  A4J0YU0L11         3827
# 50  FO8YWV23ZK         3827
# 87  UI7EJIY2KI         3764
# 21  5KO6XTG3NZ         3749

# Get top ten individuals by MAIL
tps_eng_mail = tps.groupby('ID').MAIL.sum()
tps_eng_mail = pd.DataFrame({'ID':tps_eng_mail.index, 'Mail_Sum':tps_eng_mail.values})
tps_eng_mail = tps_eng_mail.sort_values(by=['Mail_Sum'], ascending=False)
tps_eng_mail.head(10)
#             ID  Mail_Sum
# 7   1ZEAIO3U5N       114
# 30  A4J0YU0L11       107
# 80  SXS9F6CSY1       104
# 34  AWZPLRZ289       102
# 38  C3FC972N4Q        99
# 91  WL69WBRD1G        98
# 77  RG0G1D96G3        97
# 24  701CU2R5WE        97
# 21  5KO6XTG3NZ        97
# 33  ARN5GMCI89        96

################################################################################################################################
########################### 2. Explore Relationship between touchpoint and trades - Data Preparation ###########################
################################################################################################################################
tps_orig = tps.copy()
len(tps['ID'].unique())  # 100 unique values and they are all individuals

# Drop DATE column
tps = tps.drop(columns=['DATE'])

# Rolled up the tps dataset to individual level using sum metric
aggregations = dict()
keys = ['EMAIL_SENT', 'EMAIL_OPENED', 'IN_PERSON_MEETING', 'PHONE_ATTEMPT', 'PHONE_SUCCESS', 'WEBINAR', 'MAIL']
for ele in keys:
    aggregations[str(ele)] = {}
    aggregations[str(ele)][ele + '_sum'] = 'sum'
tps_agg = tps.groupby('ID').agg(aggregations)
tps_agg.columns = ['EMAIL_SENT_sum', 'EMAIL_OPENED_sum', 'IN_PERSON_MEETING_sum', 'PHONE_ATTEMPT_sum', 'PHONE_SUCCESS_sum', 'WEBINAR_sum', 'MAIL_sum']

# Feature Engineering - create two new percentage features
tps_agg['EMAIL_pct'] = tps_agg['EMAIL_OPENED_sum']/tps_agg['EMAIL_SENT_sum']
tps_agg['PHONE_pct'] = tps_agg['PHONE_SUCCESS_sum']/(tps_agg['PHONE_ATTEMPT_sum'] + tps_agg['PHONE_SUCCESS_sum'])
# WARNING: Some individuals' EMAIL_SENT_sum < EMAIL_OPENED_sum !!!!
## Change these individuals' EMAIL_pct variable to be 1
tps_agg.loc[(tps_agg.EMAIL_pct > 1),'EMAIL_pct'] = 1
tps_agg.head(10)
#             EMAIL_SENT_sum  EMAIL_OPENED_sum  ...  EMAIL_pct  PHONE_pct
# ID                                            ...                      
# 007AS3ESRJ             227               208  ...   0.916300   0.394737
# 086N3Y4VVM             162               143  ...   0.882716   0.307692
# 0ESA3CXA5N            1237              1037  ...   0.838319   0.400000
# 0FIT0NJYNV            1468              1127  ...   0.767711   0.421348
# 13Y2YC97H6             593               608  ...   1.000000   0.440476
# 166FJV65R6             470               330  ...   0.702128   0.386667
# 17PCTFAZUP            1073               976  ...   0.909599   0.390728
# 1ZEAIO3U5N            1778              1593  ...   0.895951   0.387500
# 2A9X4KHXZI            1292               964  ...   0.746130   0.418301
# 2F99RMVRIZ             302               258  ...   0.854305   0.428571

# Write cleaned tps_agg table to csv and pickle file
tps_agg.to_csv('./data/clean/tps_agg.csv')
tps_agg.to_pickle('./data/clean/tps_agg.pkl')

# Using the trades_I_all table to do the left join
tps_tranamt_merged = tps_agg.merge(trades_I_all, on='ID', how='left')
tps_tranamt_merged = tps_tranamt_merged.drop(columns=['TRAN_AMT_SUM', 'Shares'])

# Set Individual's ID to be the index
tps_final = tps_tranamt_merged.copy()
tps_final = tps_final.set_index('ID')
list(tps_final)
# ['EMAIL_SENT_sum',
#  'EMAIL_OPENED_sum',
#  'IN_PERSON_MEETING_sum',
#  'PHONE_ATTEMPT_sum',
#  'PHONE_SUCCESS_sum',
#  'WEBINAR_sum',
#  'MAIL_sum',
#  'EMAIL_pct',
#  'PHONE_pct',
#  'TRAN_AMT_all']

tps_final.head(10)
#             EMAIL_SENT_sum  EMAIL_OPENED_sum  ...  PHONE_pct  TRAN_AMT_all
# ID                                            ...                         
# 007AS3ESRJ             227               208  ...   0.394737  6.404138e+05
# 086N3Y4VVM             162               143  ...   0.307692  2.186225e+06
# 0ESA3CXA5N            1237              1037  ...   0.400000  2.756422e+07
# 0FIT0NJYNV            1468              1127  ...   0.421348  6.219391e+05
# 13Y2YC97H6             593               608  ...   0.440476  3.869896e+06
# 166FJV65R6             470               330  ...   0.386667  2.932927e+05
# 17PCTFAZUP            1073               976  ...   0.390728  1.270846e+06
# 1ZEAIO3U5N            1778              1593  ...   0.387500  3.250560e+06
# 2A9X4KHXZI            1292               964  ...   0.418301  1.126040e+06
# 2F99RMVRIZ             302               258  ...   0.428571  3.143708e+05

# Write cleaned tps_final table to csv and pickle file
tps_final.to_csv('./data/clean/tps_final.csv')
tps_final.to_pickle('./data/clean/tps_final.pkl')

# Split the dataset to training and validation
from sklearn.model_selection import train_test_split
X = tps_final.iloc[:, :-1]
y = tps_final.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
X_train.head(10)
#            EMAIL_SENT_sum  EMAIL_OPENED_sum  ...  EMAIL_pct  PHONE_pct
# ID                                            ...                      
# AYGKZW5YPG             721               718  ...   0.995839   0.481928
# UYPHDND4FA            1553              1254  ...   0.807469   0.435484
# 4ODNGNJKKP             676               638  ...   0.943787   0.404762
# A06WP1H62W             988               893  ...   0.903846   0.432432
# ADFYY3NPQS             206               156  ...   0.757282   0.470588
# WL69WBRD1G            1690              1620  ...   0.958580   0.423237
# UBRJS7UAWJ             453               418  ...   0.922737   0.415584
# 2F99RMVRIZ             302               258  ...   0.854305   0.428571
# H9NSEIPUYE            1223              1235  ...   1.000000   0.420382
# CDGKL7HHXF             497               385  ...   0.774648   0.440000
X_train.shape # (80, 9)

#######################################################################################################################
########################### 2. Explore Relationship between touchpoint and trades - XGBoost ###########################
#######################################################################################################################
import xgboost as xgb
from sklearn.metrics import mean_squared_error

model_xgb = xgb.XGBRegressor(colsample_bytree = 0.3, learning_rate = 0.1, max_depth = 5, alpha = 10, n_estimators = 10)

model_xgb.fit(X_train, y_train)
preds = model_xgb.predict(X_test)

# Calculate the rmse
rmse = np.sqrt(mean_squared_error(y_test, preds))
print("RMSE: %f" % (rmse))
# RMSE: 6216384.130652

# Plot feature importance
xgb.plot_importance(model_xgb)
plt.rcParams['figure.figsize'] = [10, 40]
plt.show()

#############################################################################################################################
########################### 2. Explore Relationship between touchpoint and trades - Random Forest ###########################
#############################################################################################################################
from sklearn.ensemble import RandomForestRegressor
# Instantiate model with 1000 decision trees
model_rf = RandomForestRegressor(n_estimators = 100, random_state = 42, n_jobs = -2)
# Train the model on training data
model_rf.fit(X_train, y_train);

# Use the forest's predict method on the test dataj
preds_rf = model_rf.predict(X_test)
# Calculate the absolute errors
errors = abs(preds_rf - y_test)
# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2))
# Mean Absolute Error: 5282775.79

########################################################################################################
########################### 3.Identify Different Types of Advisors - k-means ###########################
########################################################################################################
from sklearn.cluster import KMeans

# Find the approporiate cluster number - Elbow Plot
tps_final_kmeans = tps_final.copy()

wcss = []
for i in range(1, 30):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(tps_final_kmeans)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 30), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
## Looks like n=3 is the elbow

# k-means Clustering
# Fitting k-means to the dataset
kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(tps_final_kmeans)
# Beginning of the cluster numbering with 1 instead of 0
y_kmeans = y_kmeans + 1

# Adding cluster column to the dataset
tps_final_kmeans['cluster'] = y_kmeans
tps_final_kmeans['cluster'].value_counts()
# Cluster# Counts
# 1         89
# 3         10
# 2         1

tps_final_kmeans[tps_final_kmeans['cluster'] == 2]
# Individual ID: ZEKBKU4CMP 

tps_final_kmeans[tps_final_kmeans['cluster'] == 3]
# Individual ID
# 0ESA3CXA5N
# 2MVH9N44OT
# 4ODNGNJKKP
# A06WP1H62W
# K3M3GSSAN2
# RG0G1D96G3
# UI7EJIY2KI
# UYPHDND4FA
# W1Y175DVPO
# ZK6QSE5CWV

# Mean of clusters - Description Matrix
kmeans_mean_cluster = pd.DataFrame(round(tps_final_kmeans.groupby('cluster').mean(), 1))
kmeans_mean_cluster
## Looks like Cluster 2 has significantly higher values for all 10 variables compared to the other two clusters