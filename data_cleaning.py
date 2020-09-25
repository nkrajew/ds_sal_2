# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 18:35:23 2020

@author: nkraj
"""

import pandas as pd

df = pd.read_csv('glassdoor_jobs.csv')

# SALARY PARSING

#  flag estimates that are provided per hour
df['hourly'] = df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)

# drop hourly jobs (researching showed these are not real salary estimates)
df = df[df['hourly'] == 0]
df.drop('hourly', axis=1, inplace=True)

# isolate salary (remove Glassdoor est. text) 
salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])
minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

min_hr = minus_Kd.apply(lambda x: x.lower().replace('per hour', ''))

# create columns to split min and max and calc avg
df['min_salary'] = min_hr.apply(lambda x: int(x.split('-')[0]))
df['max_salary'] = min_hr.apply(lambda x: int(x.split('-')[1]))
df['avg_salary'] = (df.min_salary + df.max_salary)/2

# COMPANY NAME
df['company_txt'] = df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3], axis=1)

# STATE FIELD
# parse out state code
df['job_state'] = df['Location'].apply(lambda x: x.split(',')[1] if ',' in x.lower() else x) 

# fix states that aren't state code
state_map = {'New Jersey': 'NJ', 'Virginia': 'VA', 'Missouri': 'MO', 'Utah': 'UT', 'United States': 'US'}
df['job_state'].replace(state_map, inplace=True)
# remove extra white space
df['job_state'] = df['job_state'].apply(lambda x: x.strip())
df.job_state.value_counts()

# drop England entries
df = df[df['job_state'] != 'East of England']

# is job at company HQ?
# HQ for my query returned all -1 so commented out this code
# df['same_state'] = df.apply(lambda x: 1 if x.Location == x.Headquarters else 0)

# COMPANY AGE
df['age'] = df.Founded.apply(lambda x: x if x < 0 else 2020 - x)

# parsing of job description (pthon, etc.)

# parse out top data science tools

# python
df['python_yn'] = df['Job Description'].apply(lambda x: 1 if 'python' in x.lower() else 0)
# r studio
df['R_yn'] = df['Job Description'].apply(lambda x: 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
# spark
df['spark_yn'] = df['Job Description'].apply(lambda x: 1 if 'spark' in x.lower() else 0)
# aws
df['aws_yn'] = df['Job Description'].apply(lambda x: 1 if 'aws' in x.lower() else 0)
# excel
df['excel_yn'] = df['Job Description'].apply(lambda x: 1 if 'excel' in x.lower() else 0)
# SQL
df['SQL_yn'] = df['Job Description'].apply(lambda x: 1 if 'sql' in x.lower() else 0)
# tableau
df['tableau_yn'] = df['Job Description'].apply(lambda x: 1 if 'tableau' in x.lower() else 0)

# drop blank columns (competitors and headquarters)
df.drop(['Headquarters','Competitors'], axis=1, inplace=True)
df.reset_index(inplace=True, drop=True)

# save csv
df.to_csv('salary_data_cleaned.csv', index = False)

# test save
pd.read_csv('salary_data_cleaned.csv')