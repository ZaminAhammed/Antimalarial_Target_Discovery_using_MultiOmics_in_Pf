#!/usr/bin/python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tabpfn import TabPFNClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay

def main():
    print (f"Training Set supplied : '{sys.argv[1]}'")
    print (f"Test Set supplied : '{sys.argv[2]}'")

    #Models to be evaluated (hyperamaters are set based on GridSearch performed using TrainingSet_1)
    model_1 = LogisticRegression(C=2.0, max_iter=50, penalty='l1', solver='liblinear')
    model_2 = SVC(kernel='linear', probability=True)
    model_3 = RandomForestClassifier(random_state=1)
    model_4 = xgb.XGBClassifier(objective="binary:logistic", seed=1, learning_rate=0.75, n_estimators=None, max_depth=None)
    model_5 = TabPFNClassifier()
    models = [model_1, model_2, model_3, model_4, model_5]
    
    #Data importing
    train = pd.DataFrame(sys.argv[1])
    test = pd.DataFrame(sys.argv[2])

    #Cleaning data
    train_X, train_Y = clean_data(train)
    test_X, test_Y = clean_data(test)

    #Training
    best_model = model_eval(models, train_X, train_Y)   

    #Classifying unlabelled test set
    results = test(best_model, train_X, train_Y, test_X)
    results.to_csv(sys.argv[3], index=False)

    print (f"The final results have been saved to : '{sys.argv[3]}'\n")    

def clean_data(data, standardizer = StandardScaler()):
    data.dropna(inplace=True)
    X = data.select_dtypes({'float64','int64'})
    if 'Target' in X.columns:
        X.drop(columns={'Target'})
        Y = data.Target
    else:
        Y = {}
    X_stand = standardizer.fit_transform(X)
    X_df = pd.DataFrame(X_stand, columns = X.columns)
    return X_df, Y

def model_eval(models, X, Y, cv = KFold(n_splits=10, random_state=1, shuffle=True),
               scoring = ['accuracy', 'f1', 'precision', 'recall', 'roc_auc'], n_jobs = -1, error_score = 'raise'):
    result = pd.DataFrame({'Model': ['LogisticRegression', 'SVM', 'RandomForest', 'XGBoost', 'TabFPN']})
    f1_scores = []
    for y in scoring:
        kfold = []
        for x in models:
            score = cross_val_score(x, X, Y, scoring=y, cv=cv, n_jobs=n_jobs, error_score=error_score)
            kfold.append('%.3f (%.3f)' % (np.mean(score), np.std(score)))
            if y == 'f1':
                f1_scores.append(np.mean(score))
        result[y] = kfold
    print(f'Performance of Models:')
    print(result.to_string())
    best_f1 = max(f1_scores)
    for i in range(5):
        if f1[i] == best_f1:
            best_model = models[i]
    return best_model
        
def test(model, train_X, train_Y, test):
    model.fit(train_X_stand, train_Y)
    probs = model.predict_proba(test_X_stand)
    yes, no = [], []
    for x in probs:
        no.append(x[0])
        yes.append(x[1])
    test_results = pd.DataFrame({'Gene':test.Gene, 'Yes':yes, 'No':no})
    return test_results

if __name__ == "__main__":
    main()
