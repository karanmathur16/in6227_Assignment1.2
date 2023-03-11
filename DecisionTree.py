import pandas as pd
import time
import math
from sklearn import metrics, preprocessing
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeClassifier
from dataPreProcessing import testdata_df,traindata_df

#normalize string values for decision tree
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
# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

endTime = time.time()

#Time to train model
train_time = endTime - startTime

print(f"Time to train the model DT: {train_time:.3f} seconds")

#Predict the response for test dataset
startPredictTime = time.time()
prediction = clf.predict(X_test)
endPredictTime = time.time()

predictTime = endPredictTime - startPredictTime
print(f"Time to apply the model DT: {predictTime:.3f} seconds")

print("Test Train Split Accuracy DT:",metrics.accuracy_score(y_test, prediction))


#GridSearch for tuning max depth
param_grid_dt = {'max_depth': range(1, 11)}
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(), param_grid = param_grid_dt, verbose=1,cv=10,n_jobs=-1)
startTime = time.time()
# Create Decision Tree classifer object
grid_search = grid_search.fit(X_train,y_train)
endTime = time.time()
#Time to train model
train_time = endTime - startTime
print(f"Time to train the model GridSearchDT: {train_time:.3f} seconds")

startPredictTime = time.time()
prediction = grid_search.predict(X_test)
endPredictTime = time.time()

predictTime = endPredictTime - startPredictTime
print(f"Time to apply the model GridSearchDT: {predictTime:.3f} seconds")

print("Test Train Split Accuracy GridSearch DT:",metrics.accuracy_score(y_test, prediction))
print(grid_search.best_params_)

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

print("Test Train Split Recall GridSearchDT:", recall)
print("Test Train Split Precision GridSearchDT:", precision)
print("Test Train Split F-Measure GridSearchDT:", fmeasure)

numberOfSamples = len(X_test.index)
confidence = 0.95
zscore = 1.96

confidenceInterval = zscore * math.sqrt((error_rate * (1 - error_rate)) / numberOfSamples)
print("Confidence Interval = " + str(round(error_rate,3)) + " \u00B1 " + str(round(confidenceInterval,3)))

##Test Data fragment
test_data = testdata_df.drop(['education'],axis=1) # education-num already exists, same with train set
test_data_features = test_data.drop(['label'],axis=1)
test_data_labels = test_data['label']

prediction_test = grid_search.predict(test_data_features)

print("Actual Test Data Accuracy DT:",metrics.accuracy_score(test_data_labels, prediction_test))
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

print("Actual Test Data Recall DT:", recall)
print("Actual Test Data Precision DT:", precision)
print("Actual Test Data F-Measure DT:", fmeasure)
print(f"Actual Test Data Error rate DT: {error_rate:.3f}")
