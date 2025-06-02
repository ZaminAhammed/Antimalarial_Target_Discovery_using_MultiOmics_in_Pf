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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV, learning_curve, KFold
from sklearn.metrics import roc_curve, auc, roc_auc_score, RocCurveDisplay
from sklearn.metrics import precision_recall_curve, confusion_matrix
import shap


def main():
    print (f"Training Set supplied : '{sys.argv[1]}'")
    
    #Models to be evaluated (hyperamaters are set based on GridSearch performed using TrainingSet_1)
    model_1 = LogisticRegression(C=2.0, max_iter=50, penalty='l1', solver='liblinear')
    model_2 = SVC(kernel='linear', probability=True)
    model_3 = RandomForestClassifier(random_state=1)
    model_4 = xgb.XGBClassifier(objective="binary:logistic", seed=1, learning_rate=0.75, n_estimators=None, max_depth=None)
    model_5 = TabPFNClassifier()
    models = [model_1, model_2, model_3, model_4, model_5]
    
    #Data importing and cleaning
    train = pd.DataFrame(sys.argv[1])
    train_X, train_Y = clean_data(train)

    #Evaluating different models
    best_model = model_eval(models, train_X, train_Y)   
    shap_eval(model_4, train_X, train_Y)
    PR_curves(models, train_X, train_Y)
    learning_curves(models, train_X, train_Y)

    print (f"Best-performing model was identified as : {best_model}\n")    

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
        

def shap_eval(model, X, Y):
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
    model.fit(X_train, y_train)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values, X_test, max_display=X.shape[1])

def PR_curves(models, X, Y):
    fig, axis = plt.subplots(2, 3, figsize=(15, 10))
    mods = ["LR", "SVC", "RF", "XGB", "TabFPN"]
    axes = [axis[0, 0], axis[0, 1], axis[0,2], axis[1, 0], axis[1, 1]]
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        models[i].fit(X_train, y_train)
        y_scores = models[i].predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_scores)
        auc_score = auc(recall, precision)
        axes[i].plot(recall, precision, label=f'PR Curve (AUC = {auc_score:.2f})')
        axes[i].set_xlabel('Recall')
        axes[i].set_ylabel('Precision')
        axes[i].set_title(f'PR Curve of {mods[i]} for')
        axes[i].legend()
    fig.delaxes(ax=axis[1, 2])
    fig.savefig(f'PR_Curves.jpeg')
    fig.show()

def learning_curves(models, X, Y):
    fig, axis = plt.subplots(2, 3, figsize=(15, 10))
    mods = ["LR", "SVC", "RF", "XGB", "TabFPN"]
    axes = [axis[0, 0], axis[0, 1], axis[0, 2], axis[1, 0], axis[1, 1]]
    for i in range(5):
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=42)
        sizes, training_scores, testing_scores = learning_curve(models[i], X_train, y_train, cv=10, scoring='accuracy', random_state=13,
                                                                train_sizes=np.linspace(0.1, 1.0, 20), error_score='raise', n_jobs=-1)
        mean_training = np.mean(training_scores, axis=1)
        std_training = np.std(training_scores, axis=1)
        mean_testing = np.mean(testing_scores, axis=1)
        std_testing = np.std(testing_scores, axis=1)

        axes[i].plot(sizes, mean_training, '--', color="b", label="Training score")
        axes[i].fill_between(sizes, mean_training + std_training, mean_training - std_training, color="b", alpha=0.2)
        axes[i].plot(sizes, mean_testing, color="g", label="Testing score")
        axes[i].fill_between(sizes, mean_testing + std_testing, mean_testing - std_testing, color="g", alpha=0.2)
        axes[i].set_xlabel("Training set size")
        axes[i].set_ylabel("Accuracy score")
        axes[i].set_title(f'Learning Curve of {mods[i]}')
        axes[i].legend(loc="best")
    fig.delaxes(ax=axis[1, 2])
    fig.savefig(f'Learning_Curves.jpeg')
    fig.show()

if __name__ == "__main__":
    main()
