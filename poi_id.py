#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data 
import pandas as pd
import numpy as np

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#Note that this list is created based of the results of SelectKBest of our final
#model.
features_list = ['poi','salary', 'exercised_stock_options', 'total_stock_value']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

enron_df = pd.DataFrame.from_dict(data_dict, orient='index')
#We will remove email_address from the list of features as it does not have any value as a feature.
enron_df = enron_df.drop(['email_address'], axis = 1)

### Task 2: Remove outliers
#There are two entries that are not really a person, so we should remove them.
enron_df = enron_df.drop(['TOTAL','THE TRAVEL AGENCY IN THE PARK'], axis = 0)

all_features = enron_df.columns.difference(['poi'])
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                      'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 
                      'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 
                      'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                  'from_this_person_to_poi', 'shared_receipt_with_poi']

#According to the project forum, financial data that have 'NaN' are actually '0' value 
#whereas email data that have 'NaN' are missing data.
for feature in financial_features:
    enron_df[feature] = enron_df[feature].replace('NaN', 0.0)
for feature in email_features:
    enron_df[feature] = enron_df[feature].replace('NaN', np.nan)
enron_df[all_features] = enron_df[all_features].apply(pd.to_numeric)

#remove all rows that are all 0.0/NaN
enron_features_df = enron_df[all_features].copy()
mask = np.all(np.isnan(enron_features_df) | np.equal(enron_features_df, 0), axis=1)
enron_df = enron_df[~mask]

#We change the NaN back to 0.0 here as during our analysis, the NaN values were all came from
#email features, and accounts for 57 entries.  Removing these from our training data would be
#too reduce our dataset significantly.
for feature in email_features:
      enron_df[feature] = enron_df[feature].replace(np.nan, 0.0)

### Task 3: Create new feature(s)
#Note the +1 is to ensure we're not dividing by zero.  Given that a $1 raise to everyone's
#salary is miniscule compared to their annual salary, the should not affect the results.
enron_df['bonus_salary_ratio'] = enron_df['bonus'] / (enron_df['salary'] + 1)
enron_df['exercised_stock_salary_ratio'] = enron_df['exercised_stock_options'] / \
    (enron_df['salary'] + 1)

#We will also drop some of the features that provide little to no information
remove_features = ['director_fees', 'restricted_stock_deferred', 'loan_advances'] 
enron_cleaned_df = enron_df.drop(remove_features, axis = 1)

### Store to my_dataset for easy export below.
my_dataset = enron_cleaned_df.to_dict('index')

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.cross_validation import StratifiedShuffleSplit, train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from tester import test_classifier

#Baselines
baseline_features = enron_cleaned_df.drop(['poi'], axis = 1)
baseline_labels = enron_cleaned_df[['poi']]
baseline_columns = ['poi'] + list(baseline_features)

X_train, X_test, y_train, y_test = train_test_split(
    baseline_features.values, baseline_labels.values, test_size=0.3, 
		random_state=42, stratify=baseline_labels.values)

lr_clf = LogisticRegression()
#lr_clf.fit(X_train, y_train)
#test_classifier(lr_clf, my_dataset, baseline_columns)

rf_clf = RandomForestClassifier(random_state = 42)
#rf_clf.fit(X_train, y_train)
#test_classifier(rf_clf, my_dataset, baseline_columns)

svm_clf = SVC()
#svm_clf.fit(min_max_scaler.fit_transform(X_train), y_train)
#test_classifier(svm_clf, my_dataset, baseline_columns)

dt_clf = DecisionTreeClassifier(random_state = 42)
#dt_clf.fit(X_train, y_train)
#test_classifier(dt_clf, my_dataset, baseline_columns)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

sss = StratifiedShuffleSplit(baseline_labels.values, 100, random_state = 42)
kbest = SelectKBest(f_classif)

#Logistic Regression:
pipeline = Pipeline([('kbest', kbest), ('lr', LogisticRegression())])
grid_search_lr = GridSearchCV(pipeline, {'kbest__k': np.arange(1, len(X_train[0]) + 1),
                                      'lr__C': np.logspace(-2, 2, 10)},
                              scoring="f1", cv = sss, n_jobs = 4)
#grid_search_lr.fit(baseline_features.values, 
#                   np.reshape(baseline_labels.values,[len(baseline_labels.values),]))
#clf_lr = grid_search_lr.best_estimator_
#test_classifier(clf_lr, my_dataset, baseline_columns)

#SVM:
combined_features = FeatureUnion([("pca", PCA()), ("kbest", kbest)])
pipeline = Pipeline([('scale', MinMaxScaler(feature_range=(0, 1))),
                     ('features', combined_features),
                     ('svm', SVC())])
grid_search_svm = GridSearchCV(pipeline, {'features__kbest__k': np.arange(5, len(X_train[0]) + 1),
                                          'features__pca__n_components': np.arange(2,15),
                                          'svm__C': np.logspace(-2, 2, 5),
                                          'svm__gamma': np.logspace(-4, -1, 10),
                                          'svm__kernel': ['linear', 'rbf']},
                              scoring="f1", cv = sss, n_jobs = 4)
#grid_search_svm.fit(baseline_features.values, 
#                    np.reshape(baseline_labels.values,[len(baseline_labels.values),]))
#clf_svm = grid_search_svm.best_estimator_
#test_classifier(clf_svm, my_dataset, baseline_columns)

#Random Forest:
pipeline = Pipeline([('kbest', kbest), ('rf', RandomForestClassifier(random_state = 42))])
grid_search_rf = GridSearchCV(pipeline, {'kbest__k': np.arange(5, len(X_train[0]) + 1),
                                         'rf__n_estimators': np.arange(10, 20),
                                         'rf__max_features': ['sqrt', 'log2', None],
                                         'rf__min_samples_split': np.arange(2,6)
                                        },
                             scoring="f1", cv = sss, n_jobs = 4)
#grid_search_rf.fit(baseline_features.values,
#                   np.reshape(baseline_labels.values,[len(baseline_labels.values),]))
#clf_rf = grid_search_rf.best_estimator_
#test_classifier(clf_rf, my_dataset, baseline_columns)

#Decision Tree
pipeline = Pipeline([('kbest', kbest), ('dt', DecisionTreeClassifier(random_state = 42))])
grid_search_dt = GridSearchCV(pipeline, {'kbest__k': np.arange(1, len(X_train[0]) + 1),
                                         'dt__min_samples_split': np.arange(2,10),
                                         'dt__max_depth': [None, 2, 4, 6, 10],
                                         'dt__min_samples_leaf': np.arange(1,5),
                                         'dt__max_leaf_nodes': [None, 2, 4, 6, 10, 20]
                                        },
                             scoring="f1", cv = sss, n_jobs = 4)
#grid_search_dt.fit(baseline_features.values,
#                   np.reshape(baseline_labels.values,[len(baseline_labels.values),]))
#clf_dt = grid_search_dt.best_estimator_
#test_classifier(clf_dt, my_dataset, baseline_columns)

#Final Decision Tree based oned the training above.
clf = DecisionTreeClassifier(random_state = 42,
                             max_depth = 6,
                             max_leaf_nodes = 20,
                             min_samples_leaf = 1,
                             min_samples_split = 2)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
