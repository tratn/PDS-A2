import os
import re
import pickle
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.model_selection import KFold
from flask import Flask, jsonify, request, render_template
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

MODEL_PATH = "pickle_model.pkl"
with open(MODEL_PATH, "rb") as rf:
    pkl = pickle.load(rf)

# Preprocessing data

def preprocessing_data():
    ### READ DATA
    url = 'survey_results_public.csv'
    preprocess = pd.read_csv(url, sep=',')

    ### MISSING VALUE
    # list of mandatory columns.
    # 'RespodentID' is assigned automatically, hence not included in this list
    mandatory_cols = ['MainBranch', 'Hobbyist', 'Country',
                      'CurrencyDesc', 'CurrencySymbol', 'JobSeek']
    preprocess.dropna(subset=mandatory_cols, inplace=True)

    # drop missing values for CompFreq
    preprocess.dropna(subset=['CompFreq', 'CompTotal'], axis=0, inplace=True)
    
    # find the exchange rate for Danish krone 
    dkk_to_usd = 138936.0/80000.0
    converted_comp = preprocess.loc[preprocess.index.isin(
        [47224]), 'CompTotal']*dkk_to_usd

    # fill the missing converted compensation
    preprocess.loc[preprocess.index.isin(
        [47224]), 'ConvertedComp'] = converted_comp

    # fix the currency symbol
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
    
    # fill other columns with 'NotMentioned'
    cat_cols = preprocess.select_dtypes(include=[object]).columns.tolist()
    preprocess[cat_cols] = preprocess[cat_cols].fillna(value='NotMentioned')

    ### WHITESPACES/STRING MANIPULATION
    # remove whitespace
    preprocess = preprocess.apply(
        lambda x: x.str.strip() if x.dtype == "object" else x)
    # transform letter to lowercase
    preprocess = preprocess.apply(
        lambda x: x.str.lower() if x.dtype == "object" else x)
    
    # Replace highest/lowest values with the corresponding float value
    # Replace 'notmentioned' values with the MODE value
    preprocess['Age1stCode'] = preprocess['Age1stCode'].replace("younger than 5 years", "4")
    preprocess['Age1stCode'] = preprocess['Age1stCode'].replace("older than 85", "86")
    preprocess['Age1stCode'] = preprocess['Age1stCode'].replace("notmentioned", "14")

    preprocess['YearsCode'] = preprocess['YearsCode'].replace("less than 1 year", "0.5")
    preprocess['YearsCode'] = preprocess['YearsCode'].replace("more than 50 years", "51")
    preprocess['YearsCode'] = preprocess['YearsCode'].replace("notmentioned", "10")

    preprocess['YearsCodePro'] = preprocess['YearsCodePro'].replace("less than 1 year", "0.5")
    preprocess['YearsCodePro'] = preprocess['YearsCodePro'].replace("more than 50 years", "51")
    preprocess['YearsCodePro'] = preprocess['YearsCodePro'].replace("notmentioned", "3")
    
    # Cast 'Age1stCode', 'YearsCode', 'YearsCodePro' to float type
    for col in ['Age1stCode', 'YearsCode', 'YearsCodePro']:
        preprocess[col] = preprocess[col].astype('float64')

    ## EXTREME VALUE AND OUTLIERS
    preprocess.drop(preprocess[preprocess['Age1stCode'] > preprocess['Age']].index, inplace = True)
    preprocess.drop(preprocess[preprocess['YearsCodePro'] > preprocess['YearsCode']].index, inplace = True)
   
    # Outliers
    outliers_df = preprocess.loc[(preprocess['Age'] > 80) | (preprocess['Age1stCode'] > 75)]
    preprocess.drop(outliers_df.index, inplace = True)
    
    # Impossible value
    preprocess[preprocess['YearsCode'] > preprocess['Age']]
    preprocess.loc[preprocess['YearsCode'] > preprocess['Age'], ['Age']] = preprocess['YearsCode'] + preprocess['Age1stCode']
    preprocess.drop(preprocess.loc[preprocess['WorkWeekHrs'] > (24*7)].index, inplace=True)
    
    # generate copy of ConvertedComp column
    preprocess["ConvertedComp_Copy"] = preprocess["ConvertedComp"]
    
    # Create categories for compensation
    preprocess["ConvertedComp"] = pd.cut(preprocess["ConvertedComp"],
                                         bins=[0, 24000, 48000,
                                               96000, np.inf],
                                         labels=['0-24k', '24k-48k', '48k-96k', '>96k'],
                                         include_lowest=True)

    ### CREATE NEW FEATURES 
    def count_unique_value(row):
        values = re.split(';', row)   
        return len(values)

    preprocess['Database_Count'] = preprocess['DatabaseWorkedWith'].apply(lambda x: count_unique_value(x))
    preprocess['Lang_Count'] = preprocess['LanguageWorkedWith'].apply(lambda x: count_unique_value(x))
    preprocess['Misc_Count'] = preprocess['MiscTechWorkedWith'].apply(lambda x: count_unique_value(x))
    preprocess['Platform_Count'] = preprocess['PlatformWorkedWith'].apply(lambda x: count_unique_value(x))
    preprocess['Webframework_Count'] = preprocess['WebframeWorkedWith'].apply(lambda x: count_unique_value(x))
    preprocess['Total_Count'] = preprocess['Database_Count'] + preprocess['Lang_Count'] + preprocess['Misc_Count'] + preprocess['Platform_Count'] + preprocess['Webframework_Count']
    
    # numerical features
    num_feats = ['Age', 'Age1stCode', 'WorkWeekHrs', 'YearsCode', 'YearsCodePro', 'Lang_Count', 'Misc_Count', 'Platform_Count', 'Total_Count']
    
    # categorical features
    cate_feats = ['MainBranch', 'Hobbyist', 'Country', 'EdLevel', 'Employment', 'JobSat',
                 'JobSeek', 'NEWEdImpt', 'NEWLearn', 'NEWOffTopic', 'NEWOtherComms', 'NEWOvertime',
                 'OpSys', 'OrgSize', 'PurchaseWhat', 'SOAccount', 'SOComm', 'SOPartFreq',
                 'SOVisitFreq', 'UndergradMajor', 'WelcomeChange']
    
    # Cast some features to category types 
    for col in cate_feats:
        preprocess[col] = pd.Categorical(preprocess[col])

    # others
    othe_cols = ['DevType', 'JobFactors', 'LanguageWorkedWith', 'PlatformWorkedWith', 'MiscTechWorkedWith', 'DatabaseWorkedWith']
    # merge all features to one dataframe
    df_data = pd.concat([preprocess['ConvertedComp'], preprocess[num_feats], preprocess[cate_feats], preprocess[othe_cols]], axis=1)


    ### EXTRACT INFORMATION AND ENCODING
    def split_value(dataframe, col):
        values_dict = set()

        for index, row in dataframe.iterrows():
            values = re.split(';', row[col])
            for value in values:
                values_dict.add(value)
        return values_dict

    ohe_cols = df_data.select_dtypes(include=['object']).columns.to_list()
    
    for col in ohe_cols:
        ohe_col = split_value(df_data, col)
        for val in ohe_col:
            df_data[col + '_' + val] = np.where(df_data[col].str.contains(val, regex=False), 1, 0)
    
    df_data.drop(columns=ohe_cols, inplace=True)
    
    ### SPLIT THE DATASET FOR TESTING API
    # split X and y
    X = df_data.iloc[:, 1:]
    y = df_data["ConvertedComp"]
    
    return X, y


def train_test_split(X, y):
    # create splitter function
    splitter = StratifiedShuffleSplit(n_splits=1, random_state=42)
    # get train and set dataset
    for train,test in splitter.split(X, y):
        X_train_SS = X.iloc[train]
        y_train_SS = y.iloc[train]
        X_test_SS = X.iloc[test]
        y_test_SS = y.iloc[test]
    return X_test_SS, y_test_SS


# Init the app
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

# predict function


def predict_function(pkl):
    y_pred = pkl.predict(X_test_SS)

    # make sure there are no "empty" dimensions since y_pred return 2d array
    y_pred = np.squeeze(y_pred)
    y_pred.ndim

    return y_pred


# Predict function api
@app.route("/predict", methods=['POST'])
def predict():
    predictions = predict_function(pkl)    
    result = {
        'prediction': list(predictions)
    }

    return jsonify(result)


# evaluate function
def evaluate_function(model):
    # predict
    y_pred = pkl.predict(X_test_SS)

    # evaluate
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import precision_score
    accuracy = accuracy_score(y_test_SS, y_pred)
    precision = precision_score(y_test_SS, y_pred, average='macro')
    report = classification_report(y_test_SS, y_pred, output_dict=True)
    matrix = confusion_matrix(y_test_SS, y_pred)

    return accuracy, precision, report, matrix


# evaluate function api
@app.route("/evaluate", methods=['POST'])
def evaluate():
    accuracy, precision, report, matrix = evaluate_function(pkl)
    matrix = pd.DataFrame(matrix).to_json(orient='values')
    result = {
        "accuracy": accuracy,
        "precision": precision,
        "classification_report": report,
        "confusion matrix": matrix
    }
    return jsonify(result)


# main
if __name__ == '__main__':
    X, y = preprocessing_data()
    X_test_SS, y_test_SS = train_test_split(X, y)
    app.run(debug=True)