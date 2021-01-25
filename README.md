# Feature Selection Optimization using Binary Jaya Algorithm
## Objective function: Area Under the Curve (AUC) Score

## ABSTRACT
The purpose of this document is to prepare a step by step guide for Ajay. The algorithm includes binary Jaya with the Area Under the Curve (AUC) score as a feature function. These algorithms are used to optimized the feature selection process and improve the overall accuracy of classification results up to 6- 10 % using Naive Bayes (NB), K-Nearest Neighbour (KNN), LDA, and Regression Tree (RT). 

## Introduction
The document contains the following:

- Datasets Musk and Madelon.
- Classifiers: NB, KNN, LDS, RT.
- Our Approach: Binary Jaya algorithm with AUC score as a function.
- Results

## Dataset
Two datasets are used for classification purposes. Both datasets are available at https://archive.ics.uci.edu/ml/datasets    
### Dataset-1: Musk
- Musk dataset statistics are shown in the table: 1. This dataset is available at https://archive.ics.uci.edu/ml/datasets/Musk+(Version+2).
### Dataset-2: Madelon
- Madelon dataset statistics are shown in the table: 2: This dataset is available at https://archive.ics.uci.edu/ml/datasets/Madelon


Table 1: Musk Dataset Statistics

Content	                            Training Dataset	Test Dataset

Number of Features  [F_mu]	             168	          168

Number of Measurements [M_mu^Tr]	       6598	

Number of Measurements [M_mu^Te]		                    476

Table 2: Madelon Dataset Statistics

Content	                            Training Dataset	Test Dataset

Number of Features [F_ma]	               500	          500

Number of Measurements [M_ma^Tr]	       2000	

Number of Measurements [M_ma^Te]		                    1800

## Algorithm Rationale

This algorithm's ultimate purpose is to select the best features without compromising the classification results in terms of overall accuracy and individual class detection accuracy. Jaya algorithm is an optimization algorithm. The selection of its parameters and objective function plays a vital role in a classification problem.  In the proposed algorithm, parameters and objective functions are selected based on maintaining the classifier's performance for overall accuracy and individual classes. 

-	The minimum number of features: it is essential to select at least a specific feature set, which covers 80-90% of data variance in a hyperspace. This is based upon the dimensionality reduction algorithm. 

-	Area Under the Curve: This objective function focused on a balanced ratio between True Positive Class detection versus True Negative Class detection. This is important as this function makes the complete feature selection process independent of any one-class impact. The value of the AUC score is higher when both classes detection rate is comparable. AUC score is a better option when the datasets are not balanced. 

-	S-Function (Sigmoid Function): S-function values are finalized using exhaustive search. 
The result table above shows a definite improvement in overall classifiers performance with the above parameters and AUC score as an objective function compared to the classifier's performance with Error rate.

## PART - 2 
### S-Fucntion
5 differnet s-functions (sigmoid fucntions) added. S- function graphical representation is added as well in a form of an initial report. 
A variety of S-functions are used as one of the parameters. Please refer to `s_function_analysis.pdf` to see the functional graph of the selective sigmoid(s) functions. 


### Binary conversion of JAYA algorithm. 
Instead of comparing binary conversion by comparing its value with a random number, now, it will be done using a combinations of one or two S-functions. 
In the project part one, `random_r` binary conversion of JAYA algorithm is used. According to this function, if the output of JAYA function is greater than the random float number (between 0 - 1), it will be **1** otherwise **0**. 

As we have proposed f(x) function based on AUC score. The higher AUC score refers to better classification. Based on this logic, the binary conversion is proposed as if the vlaue of JAYA algorithm is higher than 0.5 then it is 1, otherwise 0 as compared to random float point. 


### MODEL EXECUTION
#### Hyper-parameters

By the end of part-1 and part -2 of the project, the following hyper-parameters are created:

1- S Fucntions 
  - default : proposed in part-1 of the project
  - logistic function
  - arctangnet 
  - hyperbolic_function
  - gen_logistic_function
  - algebraic_function
  - gudermannian_function
  
2- Objective Functions
  - Error Function 
  - AUC score based function
  
3 - Binary Conversion Functions
  - random_r
  - default_r
  
4 - Population size (p)


#### Code files 

1- Test_Cases.py : few test cases are created with various combinations of hyper-parameters. 
2- Classifiers.py : have all the classifiers
3- project_part_1.py  : jaya algorithms 
4- project_part_2.py : hyper-parameters functions 
5- algorithm.py : execute this file to run the complete code. 
