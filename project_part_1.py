import warnings
import numpy as np
import pandas as pd

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
import project_part_2 as p2

warnings.filterwarnings('ignore')

class feature_optimization:
    def __init__(self, trainX, trainY, testX, testY, population, obj_function, s_function, binary_function):
        self.trainX = trainX
        self.trainY = trainY
        self.testX  = testX
        self.testY  = testY
        self.P = population
        if obj_function == 'ER':
            self.objective_function = fx_ER
            self.jaya_function = self.jaya_ER
        elif obj_function == 'AUC':
            self.objective_function = fx_AUC
            self.jaya_function = self.jaya_AUC
        else:
            print('please select the correct fx function-- ER [Error rate] or AUC [Area under the curve]')
        
        self.s_function = p2.sigmoid_function(s_function)
        self.binary_function = p2.jaya_binary(binary_function)
        self.dsp_binary_data = generate_binary_dataset(self.P, self.objective_function)
        
        
    def __call__(self):
        self.X = self.dsp_binary_data(self.trainX, self.testX, self.trainY, self.testY)
        optimized_population = self.jaya_function(self.X)
        #print('OP_Features ', optimized_population.shape)
        trainOP, testOP = self.optimized_training_test_sets(optimized_population)
        return trainOP, testOP
        
    def __objective_function(self, X_new): 
        FV = []
        for n in range(len(X_new)):
            #print('xnew ', X_new.loc[n])
            idx ,  = np.where(X_new.loc[n]==1)
            
            temp_train = self.trainX[idx]
            temp_test = self.testX[idx]
            
           
            FV.append(self.objective_function(X_train = temp_train,
                          x_test  = temp_test,
                          Y_train = self.trainY,
                          y_test  = self.testY)
                     )

        return FV
    
    def optimized_training_test_sets(self, X):
        idx,  = np.where(X.loc[0]==1)
        #print("feature IDX ", idx)
        temp_train = self.trainX[idx]
        temp_test = self.testX[idx]
        
        return temp_train, temp_test
   
    

    def binary_x_new(self,X_new):
        '''
        PURPOSE: to convert x_new value into binary [0, 1] 
        INPUT : output of Jaya algorithm line 36
        OUTPUT: binary value of x_new
        '''
        #probability_jaya_x  = 1/(1 + np.exp(-2 * X_new) - 0.25)   # OUR PROPOSED ONE
        
        probability_jaya_x = (self.s_function(X_new))   # OUR PROPOSED ONE 
        
        prob = []
        for p in probability_jaya_x:
            prob.append(self.binary_function(p))
            #if p > random_r:
                #prob.append(1.0)
            #else:
            #    prob.append(0.0)
        
           
        return prob    

    def jaya_ER(self,x):
        '''
        INPUT:
        bextX is with the smaller fx value, 
        worstX, 
        r1 and r2 are two random numbers between 0 and 1 
        '''

        MaxIter = 100
        print('Processing .....')
        for s in range(MaxIter):      # line 6
            if len(x) >1:
                
                #print('=================    ITERATION ==============', s)
                X_new = pd.DataFrame()
                bestX = x['fx'].idxmin(axis=1)
                worstX = x['fx'].idxmax(axis=1)
                #print('BEST ', bestX, 'WORST ', worstX)
                #print('INITIAL f(x) BEST ', x['fx'].min())
                #print('INITIAL f(x) WORST ', x['fx'].max())
                for i in range(len(x.columns)-1):
                    r1 = np.round(np.random.random(1),2)
                    r2 = np.round(np.random.random(1),2)

                    jaya_X = (x[x.columns[i]] + r1 * (bestX - abs(x[x.columns[i]])) - r2*(worstX - abs(x[x.columns[i]])))

                    X_new[x.columns[i]] = self.binary_x_new(jaya_X)

                X_new['fx'] = self.__objective_function(X_new)
                #print(X_new)

                for d in range(len(x)):
                    remove_index = []
                    if(X_new['fx'].iloc[d] < x['fx'].iloc[d]):
                        x['fx'].iloc[d] = X_new['fx'].iloc[d]
                    
                    else:
                                              
                        remove_index.append(d)
                        
                
                x =x.drop(index=x.index[[remove_index]])
            else:
                print('Only one population left')
                break;
                #print(x)
        return x

    def jaya_AUC(self,x):
        '''
        INPUT:
        bextX is with the larger fx value, 
        worstX, 
        r1 and r2 are two random numbers between 0 and 1 
        '''

        MaxIter = 100
        print('Processing ........')
        for s in range(MaxIter):      # line 6
            # print("# of Populations ", len(x))
            if len(x) >1:
               
                #print('=================    ITERATION ==============', s)
                X_new = pd.DataFrame()
                bestX = x['fx'].idxmax(axis=1)
                worstX = x['fx'].idxmin(axis=1)
                #print('BEST ', bestX, 'WORST ', worstX)
                #print('INITIAL f(x) BEST ', x['fx'].max())
                #print('INITIAL f(x) WORST ', x['fx'].min())
                for i in range(len(x.columns)-1):
                    r1 = np.round(np.random.random(1),2)
                    r2 = np.round(np.random.random(1),2)

                    jaya_X = (x[x.columns[i]] + r1 * (bestX - abs(x[x.columns[i]])) - r2*(worstX - abs(x[x.columns[i]])))

                    X_new[x.columns[i]] = self.binary_x_new(jaya_X)

                X_new['fx'] = self.__objective_function(X_new)
                #print(X_new)

                for d in range(len(x)):
                    remove_index = []
                    if(X_new['fx'].iloc[d] > x['fx'].iloc[d]):

                        x['fx'].iloc[d] = X_new['fx'].iloc[d]
                    else:
                                              
                        remove_index.append(d)
                        
                
                x =x.drop(index=x.index[[remove_index]])
            else:
                print('Only one population left')
                break;
                
        return x 
    
def error_rate(pred, orig):
    '''
    PURPOSE: Calculate the rate of accuracy of a classifier. 
    INPUT: pred = predicted class by a classifier orig: original class. [type binary array]. 
    OUTPUT: float error rate. 
    
    '''
    error = np.mean((pred != orig).astype(float))
    return error


## NOTE: Covert fx_ER and fx_AUC into classes, so we can transfer the classifier as well. 
'''
class fx_function_selection:
    def __init__(self, model_selected, function_selected):
        
        
        self.function_selected = function
'''   
def fx_ER(X_train, x_test, Y_train, y_test):
    # We can choose any classifier here. 
    clf = GaussianNB()
    #clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, np.ravel(Y_train))
    pred_y = clf.predict(x_test)
    return error_rate(pred_y, y_test)

def fx_AUC(X_train, x_test, Y_train, y_test):
    # We can choose any classifier here.  
    clf = GaussianNB()
    #clf = KNeighborsClassifier(n_neighbors=10)
    clf.fit(X_train, np.ravel(Y_train))
    pred_y = clf.predict_proba(x_test)
    # keep probabilities for the positive outcome only
    pred_y  = pred_y [:, 1]
    lr_auc = roc_auc_score(y_test, pred_y )

    return lr_auc

class binary_sets:
    def __init__(self, D):
        self.D = D
    def __call__(self, population):
        idx = (population.keys())
        binary = np.zeros(self.D)
        np.put(binary, idx,1)
        return binary
    
        
class feature_subset:
    def __init__(self, list_of_features):
        self.features = list_of_features
        
        
    def __call__(self, dataframe):
        #columns = [f  for f in self.args]
                
        #print(self.features)  
        new_df = dataframe[self.features]
        return new_df 
    
class generate_binary_dataset:
    def __init__(self, population, fx):
        self.P = population
        self.f_x = fx
        
        return None
    def __call__(self,X_train, x_test, Y_train, y_test):
        D = len(X_train.columns)
        col_names = X_train.columns
        binary_dsp = binary_sets(D)
        min_features = int(D/3)
        
        X = {}
        FV = []
        
        
        for i in range(self.P):
            #print('=================================  Population: ', i, "===============================")
            random_D = np.arange(2,(np.random.randint(min_features, D)))
            #print(i, '# of features: ', len(random_D))
            feature_sub = feature_subset(random_D)
            
            tem_train_X = feature_sub(X_train)
            tem_test_X  = feature_sub(x_test)
            
            
                       
            FV.append(self.f_x(X_train = tem_train_X,
                          x_test  = tem_test_X,
                          Y_train = Y_train,
                          y_test  = y_test)
                     )
            
            
            X[i] =  binary_dsp(tem_test_X)
        Xx = pd.DataFrame(X.values(), columns = col_names ,index = X.keys())
        
        Xx['fx'] = FV
        
        #print(Xx.head(10))
        return Xx
    
    
    