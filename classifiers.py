from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from pandas_ml import ConfusionMatrix
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd

def data_classification(train_x, train_y, test_x = None ,test_y = None):
    names = ["NB", "RT", "KNN", "LDA"]

    classifiers = [
        GaussianNB(),
        DecisionTreeRegressor(criterion='mae'),
        KNeighborsClassifier(n_neighbors=10),
        LinearDiscriminantAnalysis()
    ]
    model_return = {}
    for name, clf in zip(names, classifiers):
        
        clf.fit(train_x, np.ravel(train_y))
        print(clf)
        
        # TRAIN
        #probs = clf.predict_proba(train_x)
        pred_ytrain = clf.predict(train_x)
        score_ytrain = clf.score(train_x,np.ravel(train_y))
        
        #TEST
        #probs_t = clf.predict_proba(test_x)
        pred_ytest = clf.predict(test_x)
        score_ytest = clf.score(test_x,np.ravel(test_y))
        
        model = {'Name' : name,
                'Classes': ['cancer', 'no cancer'],
                'Date': '2020-10-01',
                'Datasets': ['madelon', 'musk'],
                'Training_Score' : score_ytrain,
                 'Test_Score' : score_ytest
                }
        cl_report = classification_report(np.ravel(test_y), pred_ytest, digits = 4)
        #print("classifier %s, train score %.5f \n" %(name,score_ytrain))
        #print("classifier %s, test score %.5f \n" %(name,score_ytest))
        print("classifier %s " %(name), cl_report)
        #print("probabilty of test", probs)
        model_return[name] = cl_report #pred_ytrain
    return model_return