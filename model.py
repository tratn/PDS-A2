import pandas as pd
import numpy as np
import seaborn as sns
# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
import xgboost as xg
import pickle


from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_curve
from catboost import CatBoostClassifier, Pool, cv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, LogisticRegression

url = 'survey_results_public.csv'
survey = pd.read_csv(url, sep=',')

preprocess = survey.copy()

# List of mandatory columns.
# Respodent ID is assigned automatically, hence not included in this list
mandatory_cols = ['MainBranch', 'Hobbyist', 'Country',
                  'CurrencyDesc', 'CurrencySymbol', 'JobSeek']
preprocess.dropna(subset=mandatory_cols, inplace=True)


compensation_cols = ['CompTotal', 'CompFreq', 'ConvertedComp']
# drop missing values for CompFreq
preprocess.dropna(subset=['CompFreq', 'CompTotal'], axis=0, inplace=True)


dkk_to_usd = 138936.0/80000.0
converted_comp = preprocess.loc[preprocess.index.isin(
    [47224]), 'CompTotal']*dkk_to_usd

# fill the missing converted compensation
preprocess.loc[preprocess.index.isin(
    [47224]), 'ConvertedComp'] = converted_comp

# while we're at it, let's fix the currency symbol too
preprocess.loc[preprocess.index.isin(
    [47224]), 'CurrencyDesc'] = 'Faroese krona'
preprocess.loc[preprocess.index.isin([47224]), 'CurrencySymbol'] = 'KR'


# drop missing values for Cook Island dollar
cols_to_drop = preprocess[preprocess.loc[:,
                                         'ConvertedComp'].isnull() == True].index.tolist()
preprocess.drop(index=cols_to_drop, inplace=True)

# fill missing age and working hours with median
med_age = preprocess['Age'].median()
med_hour = preprocess['WorkWeekHrs'].median()
preprocess.fillna({'Age': med_age, 'WorkWeekHrs': med_hour}, inplace=True)


cat_cols = preprocess.select_dtypes(include=[object]).columns.tolist()
preprocess[cat_cols] = preprocess[cat_cols].fillna(value='NotMentioned')

# remove whitespace
preprocess = preprocess.apply(
    lambda x: x.str.strip() if x.dtype == "object" else x)
# transform letter to lowercase
preprocess = preprocess.apply(
    lambda x: x.str.lower() if x.dtype == "object" else x)


# Remove extreme values for Age and WorkWeekHrs
preprocess.drop(preprocess.loc[preprocess['Age'] == 279].index, inplace=True)
preprocess.drop(
    preprocess.loc[preprocess['WorkWeekHrs'] > (24*7)].index, inplace=True)


# Create categories for compensation
preprocess["ConvertedComp"] = pd.cut(preprocess["ConvertedComp"],
                                     bins=[0, 2.5e4, 5e4, 1e5, np.inf],
                                     labels=['<25k', '<50k', '<100k', '>100k'],
                                     include_lowest=True)

# Modeling
df_train = pd.DataFrame(data=preprocess, columns=['Respondent', 'MainBranch', 'Age', 'Hobbyist', 'CompFreq', 'Country',
                                                  'Employment', 'ConvertedComp', 'DevType', 'EdLevel', 'JobSat', 'WorkWeekHrs', 'OpSys',
                                                  'YearsCodePro', 'YearsCode'])
df_train['Age'] = df_train['Age'].astype('int')
df_train.loc[df_train['Age'] < 16, 'Age'] = 16
df_train.loc[df_train['Age'] > 83, 'Age'] = 83


df_train['DevType'] = df_train['DevType'].str.split(';')
df_train = df_train.explode('DevType')
df_train = df_train.reset_index(drop=True)

# Catboost model
right_df = pd.DataFrame(data=preprocess, columns=['Respondent', 'MainBranch',
                                                  'Hobbyist', 'CompFreq', 'Country', 'JobSat', 'Age', 'Employment', 'EdLevel', 'ConvertedComp',
                                                  'WorkWeekHrs', 'OpSys', 'YearsCodePro', 'YearsCode'])
left_df = pd.crosstab(df_train['Respondent'], df_train['DevType']).rename_axis(
    None, axis=1).add_prefix('DevType_').reset_index()
df_train_new = pd.merge(right_df, left_df, on="Respondent")

df_train_new['WorkWeekHrs'] = df_train_new['WorkWeekHrs'].astype('int')
df_train_new['YearsCodePro'] = df_train_new['YearsCodePro'].astype('category')
df_train_new['YearsCode'] = df_train_new['YearsCode'].astype('category')
df_train_new['OpSys'] = df_train_new['OpSys'].astype('category')
df_train_new['Employment'] = df_train_new['Employment'].astype('category')
df_train_new['Country'] = df_train_new['Country'].astype('category')
df_train_new['CompFreq'] = df_train_new['CompFreq'].astype('category')
df_train_new['Hobbyist'] = df_train_new['Hobbyist'].astype('category')
df_train_new['MainBranch'] = df_train_new['MainBranch'].astype('category')
df_train_new['EdLevel'] = df_train_new['EdLevel'].astype('category')
df_train_new['JobSat'] = df_train_new['JobSat'].astype('category')
# df_train_new['CompTotal'] = df_train_new['CompTotal'].astype('category')


df_train_new = df_train_new.drop(['Respondent'], axis=1)


y = df_train_new['ConvertedComp']
train_data = df_train_new.drop(['ConvertedComp'], axis=1)

cat_features = np.where(train_data.dtypes == 'category')[0]

X_train, X_test, y_train, y_test = train_test_split(
    train_data, y, test_size=0.2, random_state=42)

# X_test.to_csv('X_test.csv', index=False)
# y_test.to_csv('y_test.csv', index=False)

train_pool = Pool(X_train, y_train, cat_features)


# catboost_model = CatBoostClassifier(iterations=250,
#                                     learning_rate=0.7,
#                                     depth=6,
#                                     random_seed=42,
#                                     loss_function='MultiClass')

# Fit CatBoost model
# catboost_model.fit(train_pool)

# pickle.dump(catboost_model, open('model.pkl', 'wb'))
