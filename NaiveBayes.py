import pandas as pd
import numpy as np
import time
import math
from sklearn import metrics, preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from dataPreProcessing import testdata_df,traindata_df

#normalize string values for naive bayes
features = ['workclass','education-num','marital-status','occupation','relationship','race','sex','native-country']
df_combined = pd.concat([traindata_df[features], testdata_df[features]])

le = preprocessing.LabelEncoder()
for feature in features:
    le = le.fit(df_combined[feature])
    traindata_df[feature] = le.transform(traindata_df[feature])
    testdata_df[feature] = le.transform(testdata_df[feature])

x_features = traindata_df.drop(['label','education'],axis=1) # education-num already exists
y_label = traindata_df['label']

seed_number = 1234 # for reproducible output across multiple function calls

X_train, X_test, y_train, y_test = train_test_split(x_features, y_label, test_size=0.2, 
                                                            random_state=seed_number, stratify=y_label)

startTime = time.time()

clf = GaussianNB()
clf = clf.fit(X_train,y_train)

endTime = time.time()

#Time to train the model
trainTime = endTime - startTime
print(f"Time to train the model NB: {trainTime:.3f} seconds")

startPredictTime = time.time()
prediction = clf.predict(X_test)
endPredictTime = time.time()

predictTime = endPredictTime - startPredictTime
print(f"Time to apply the model NB: {predictTime:.3f} seconds")


print("Test-Train split Accuracy NB:",metrics.accuracy_score(y_test, prediction))
error_rate = 1 - (metrics.accuracy_score(y_test, prediction))
print(metrics.classification_report(y_test, prediction))  

cm = metrics.confusion_matrix(y_test, prediction)
cm_df = pd.DataFrame(cm.T, index=["<=50K", ">50K"], columns=["<=50K", ">50K"])
print(cm_df)

y_test_num = le.fit_transform(y_test)
prediction_num = le.transform(prediction)

recall = metrics.recall_score(y_test_num, prediction_num, average="weighted")
precision = metrics.precision_score(y_test_num, prediction_num, average="weighted")
fmeasure = metrics.f1_score(y_test_num, prediction_num, average="weighted")

print("Test-Train split Recall NB:", recall)
print("Test-Train split Precision NB:", precision)
print("Test-Train split F-Measure NB:", fmeasure)
print(f"Test-Train split Error rate NB: {error_rate:.2f}")

#improve accuracy using Grid Search with var smoothing params
nb_grid_params= {
    'var_smoothing': np.logspace(0,-9, num=100)
}

newNBGridModel = GridSearchCV(estimator=GaussianNB(), param_grid=nb_grid_params, verbose=1, cv=10, n_jobs=-1)

startGridTime = time.time()
newNBGridModel.fit(X_train,y_train)
endGridTime = time.time()
gridTrainTime = endGridTime - startGridTime
print(f"Time to train GridSearch NB Model: {gridTrainTime:.3f} seconds")

print(newNBGridModel.best_params_)

startGridTimePredict = time.time()
newNBGridModelPredict = newNBGridModel.predict(X_test)
endGridTimePredict = time.time()
gridPredictTime = endGridTimePredict - startGridTimePredict

print(f"Time to apply gridSearch NB Model: {gridPredictTime:.3f} seconds")

print("Test-Train split GridSearch Accuracy:",metrics.accuracy_score(y_test, newNBGridModelPredict))
error_rate = 1 - (metrics.accuracy_score(y_test, prediction))


numberOfSamples = len(X_test.index)
confidence = 0.95
zscore = 1.96

confidenceInterval = zscore * math.sqrt((error_rate * (1 - error_rate)) / numberOfSamples)
print("Confidence Interval = " + str(round(error_rate,3)) + " \u00B1 " + str(round(confidenceInterval,3)))

# #Test Data code fragment
test_data = testdata_df.drop(['education'],axis=1) # education-num already exists, same with train set
test_data_features = test_data.drop(['label'],axis=1)
test_data_labels = test_data['label']

prediction_test = newNBGridModel.predict(test_data_features)

print("Test Data Accuracy NB:",metrics.accuracy_score(test_data_labels, prediction_test))
error_rate = 1 - (metrics.accuracy_score(test_data_labels, prediction_test))
print(metrics.classification_report(test_data_labels, prediction_test))  

cm = metrics.confusion_matrix(test_data_labels, prediction_test)
cm_df = pd.DataFrame(cm.T, index=["<=50K", ">50K"], columns=["<=50K", ">50K"])
print(cm_df)

y_test_num = le.fit_transform(test_data_labels)
prediction_num = le.transform(prediction_test)

recall = metrics.recall_score(y_test_num, prediction_num, average="weighted")
precision = metrics.precision_score(y_test_num, prediction_num, average="weighted")
fmeasure = metrics.f1_score(y_test_num, prediction_num, average="weighted")

print("Test Data Recall NB:", recall)
print("Test Data Precision NB:", precision)
print("Test Data F-Measure NB:", fmeasure)
print(f"Test Data Error rate NB: {error_rate:.3f}")
