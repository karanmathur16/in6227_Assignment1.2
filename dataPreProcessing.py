import pandas as pd
import numpy as np

traindata_df = pd.read_csv('adult.data.csv',skipinitialspace = True,header=None)
traindata_df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','captial-loss','hours-per-week','native-country','label']
traindata_df = traindata_df.replace("?",np.nan)
traindata_df = traindata_df.dropna()

testdata_df = pd.read_csv('adult.test.csv',skipinitialspace = True,header=None,skiprows=1)
testdata_df.columns = ['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','captial-loss','hours-per-week','native-country','label']
print(testdata_df)
testdata_df = testdata_df.replace("?",np.nan)
testdata_df = testdata_df.dropna()
print(testdata_df)
testdata_df['label'] = testdata_df['label'].str.rstrip('.') #remove the "." char from the end of the labels in test data only