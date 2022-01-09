import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
import json

# PAGE SIDEBAR
st.sidebar.title('Select the page to display visualisation')
app_mode = st.sidebar.selectbox('Select Page', ['Exploration', 'Prediction'])

# PAGE 1: EXPLORATION
if app_mode is 'Exploration':
    # BODY
    st.title('Exploration')
    st.subheader('Dataset')
    st.caption('The dataset contains responses from the 2020 Stack Overflow Developer survey, which is among the largest and most comprehensive survey of software developers. Below the first few rows of the dataset, together with the attributes that correspond to the questions included in the survey.')

    # DATAFRAME
    data = pd.read_csv('cleaned_survey.csv')
    st.write(data.head())

    # CHART FOR DEMOGRAPHIC FACTOR
    st.subheader('Demographic of survey respondents')
    demo_factor = st.selectbox('Select demographic factor to explore', [
        'Age', 'Employment', 'Education', 'Undergraduate Major'])

    # plot the data
    if demo_factor == 'Age':
        fig = go.Figure(
            data=[go.Histogram(
                x=data['Age'],
            )])
        xlabel = 'Age'
        ylabel = 'Frequency'
        title = 'Age distribution among survey respondents'

    elif demo_factor == 'Employment':
        fig = go.Figure(data=[go.Bar(
            x=data['Employment'].value_counts(), y=data['Employment'].value_counts().index.tolist(), orientation='h',
        )])
        xlabel = 'Frequency'
        ylabel = 'Employment status'
        title = 'Employment status among survey respondents'

    elif demo_factor == 'Undergraduate Major':
        fig = go.Figure(data=[go.Bar(
            x=data['UndergradMajor'].value_counts(), y=data['UndergradMajor'].value_counts().index.tolist(), orientation='h',
        )])
        xlabel = 'Frequency'
        ylabel = 'Major'
        title = 'Undergraduate major among survey respondents'

    elif demo_factor == 'Education':
        fig = go.Figure(data=[go.Pie(labels=list(data['EdLevel'].value_counts().index), values=data['EdLevel'].value_counts(), hoverinfo="label+percent",
                                     )])
        xlabel = None
        ylabel = None
        title = 'Education level among survey respondents'

    # plot styling
    fig.update_layout(
        title=title,
        xaxis_title=xlabel,
        yaxis_title=ylabel,
        margin=dict(
            pad=10,
        ),
        yaxis=dict(
            showgrid=False
        ),
        xaxis=dict(
            showgrid=False
        ),
        showlegend=False
    )
    # display plot
    st.plotly_chart(fig)

    # CHART FOR COLUMN PAIR 1
    st.subheader('Working hours and Job satisfaction')
    # get the data
    workhrs_jobsat = data.loc[:, ['WorkWeekHrs', 'JobSat']]
    # plot the data
    fig = go.Figure()
    jobsat = ['slightly dissatisfied', 'very satisfied', 'very dissatisfied',
              'slightly satisfied', 'neither satisfied nor dissatisfied', 'notmentioned']
    for jobsat_type in jobsat:
        fig.add_trace(go.Box(
            x=data['JobSat'][data['JobSat'] == jobsat_type],
            y=data['WorkWeekHrs'][data['JobSat'] == jobsat_type])
        )
    # plot styling
    fig.update_layout(
        title_text='Total working hours per week and job satisfaction level',
        width=700,
        height=900,
        margin=dict(
            pad=10
        ),
        yaxis=dict(
            showgrid=False
        ),
        xaxis=dict(
            showgrid=False
        ),
        showlegend=False,
        violingap=0.2)
    # display the plot
    st.plotly_chart(fig)

    # CHART FOR COLUMN PAIR 2
    st.subheader('Job Factors and Gender')
    # get the data
    jobfact = data['JobFactors'].str.get_dummies(sep=';')
    jobfact_gender = jobfact.join(data['Gender'])
    jobfact_gender = jobfact_gender.drop(
        jobfact_gender[(jobfact.notmentioned == 1)].index)
    jobfact_gender.drop(columns='notmentioned', inplace=True)

    # transform the data
    def counts_per_group(df, col, group, options):
        values = df[df.loc[:, col] == group]
        d = {}
        for option in options:
            count = values[option].sum(axis=0)
            d[option] = count
            d = dict(sorted(d.items(), key=lambda item: item[1]))
        results = pd.DataFrame(list(d.items())[:])
        results = results.set_index(0)
        return results

    factors = jobfact_gender.columns.tolist()
    factors.remove("Gender")
    jobfactors_man = counts_per_group(jobfact_gender, 'Gender', 'man', factors)
    jobfactors_woman = counts_per_group(
        jobfact_gender, 'Gender', 'woman', factors)
    jobfactors_nonbinary = counts_per_group(
        jobfact_gender, 'Gender', 'non-binary, genderqueer, or gender non-conforming', factors)

    # plot the data based on selected option
    gender_options = st.selectbox('View by gender type', [
        'Man', 'Woman', 'Non-binary, genderqueer, or gender non-conforming'])

    if gender_options == 'Man':
        fig = go.Figure(data=[go.Bar(
            x=jobfactors_man.iloc[:, 0].values, y=jobfactors_man.iloc[:, 0].index.tolist(), orientation='h',
        )])

    elif gender_options == 'Woman':
        fig = go.Figure(data=[go.Bar(
            x=jobfactors_woman.iloc[:, 0].values, y=jobfactors_woman.iloc[:, 0].index.tolist(), orientation='h',
        )])
    elif gender_options == 'Non-binary, genderqueer, or gender non-conforming':
        fig = go.Figure(data=[go.Bar(
            x=jobfactors_nonbinary.iloc[:, 0].values, y=jobfactors_nonbinary.iloc[:, 0].index.tolist(), orientation='h',

        )])

    # plot styling
    fig.update_layout(
        title_text='Most important job factors by gender',
        width=1000,
        height=500,
        margin=dict(
            pad=10,
        ),
        yaxis=dict(
            showgrid=False
        ),
        xaxis=dict(
            showgrid=False
        ),
        showlegend=False
    )
    # display the plot
    st.plotly_chart(fig)

# PAGE 2: PREDICTiON
elif app_mode is 'Prediction':
    st.title('Multi-label classification')

    # FEATURES
    st.subheader('Features')

    # PREDICTION
    st.subheader('Prediction')
    # get the data from flask api
    pred_url = 'http://127.0.0.1:5000/predict'
    pred_res = requests.post(pred_url)
    pred_data = pred_res.json()['prediction']
    pred_df = pd.DataFrame(pred_data, columns=['Predicted income category'])

    # display the prediction
    st.write(pred_df)

    # EVALUATION
    st.subheader('Evaluation')
    # get the data from flask api
    eval_url = 'http://127.0.0.1:5000/evaluate'
    eval_res = requests.post(eval_url)
    eval_data = eval_res.json()
    # display the prediction
    st.write('Model Accuracy: {accuracy}'.format(
        accuracy=eval_data['accuracy']))
    st.write('Precision: {precision}'.format(precision=eval_data['precision']))
    st.write('Confusion matrix: {cfmatrix}'.format(
        cfmatrix=eval_data['confussion matrix']))

    # CHART
    st.subheader('Chart')
    pred_options = st.selectbox('Select chart option to display', [
                                'Years of coding', 'Developer type', 'Languages/frameworks', 'Education Level'])
