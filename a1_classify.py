  #  * This code is provided solely for the personal and private use of students
#  * taking the CSC401 course at the University of Toronto. Copying for purposes
#  * other than this use is expressly prohibited. All forms of distribution of
#  * this code, including but not limited to public repositories on GitHub,
#  * GitLab, Bitbucket, or any other online platform, whether as given or with
#  * any changes, are expressly prohibited.
#  
#  * All of the files in this directory and all subdirectories are:
#  * Copyright (c) 2020 Frank Rudzicz

import argparse
import os
from scipy.stats import ttest_rel
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier  

# set the random state for reproducibility 
import numpy as np
np.random.seed(401)

# global variables
classifiers = ["SGDClassifier", "GaussianNB", "RandomForestClassifier", "MLPClassifier", "AdaBoostClassifier"]
parameters = ["()", "()", "(max_depth=5, n_estimators=10)","(alpha=0.05)", "()"]

def accuracy(C):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    # print ('TODO')
    return C.trace() / C.sum()



def recall(C):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    # print ('TODO')
    return C.diagonal()/C.sum(axis=1)


def precision(C):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    # print ('TODO')
    return C.diagonal()/C.sum(axis=0)
    

def class31(output_dir, X_train, X_test, y_train, y_test):
    ''' This function performs experiment 3.1
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes

    Returns:      
       i: int, the index of the supposed best classifier
    '''
    # print('TODO Section 3.1')
    
    iBest = None
    max_accuracy = -float("inf")
    with open(f"{output_dir}/a1_3.1.txt", "w") as outf:
        # For each classifier, compute results and write the following output:
        for i in range(len(classifiers)):
            classifier = eval(classifiers[i] + parameters[i])
            classifier_name = classifier.__class__.__name__
            # train classifier
            classifier.fit(X_train, y_train)

            y_predictions = classifier.predict(X_test)
            conf_matrix = confusion_matrix(y_test, y_predictions)
            acc = accuracy(conf_matrix)
            if acc > max_accuracy:
                iBest = i
                max_accuracy = acc
            recall_val = recall(conf_matrix)
            precision_val = precision(conf_matrix)

            outf.write(f'Results for {classifier_name}:\n')  # Classifier name
            outf.write(f'\tAccuracy: {acc:.4f}\n')
            outf.write(f'\tRecall: {[round(item, 4) for item in recall_val]}\n')
            outf.write(f'\tPrecision: {[round(item, 4) for item in precision_val]}\n')
            outf.write(f'\tConfusion Matrix: \n{conf_matrix}\n\n')
        # pass

    return iBest


def class32(output_dir, X_train, X_test, y_train, y_test, iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       iBest: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    # print('TODO Section 3.2')
    num_samples = [1000, 5000, 10000, 15000, 20000]
    with open(f"{output_dir}/a1_3.2.txt", "w") as outf:
        # For each number of training examples, compute results and write
        # the following output:
        #     outf.write(f'{num_train}: {accuracy:.4f}\n'))

        # pass
        X_1k, y_1k = None, None
        for num_train in num_samples:
            selected_indices = np.random.choice(X_train.shape[0], num_train, replace=False)
            x_training = X_train[selected_indices]
            y_training = y_train[selected_indices]
            classifier = eval(classifiers[iBest] + parameters[iBest])

            classifier.fit(x_training, y_training)
            y_predictions = classifier.predict(X_test)
            conf_maxtrix = confusion_matrix(y_test, y_predictions)

            arr = accuracy(conf_maxtrix)
            outf.write(f'{num_train}: {arr:.4f}\n')

            if num_train == 1000:
                X_1k, y_1k = x_training, y_training

    return (X_1k, y_1k)

# helper for class33
def get_top_5_features_accuracy(X_train, y_train, X_test, y_test, i):
    classifier = eval(classifiers[i] + parameters[i])
    selector = SelectKBest(f_classif, k=5)
    X_new = selector.fit_transform(X_train, y_train)

    feature_indices = selector.get_support(indices=True)

    x_new_test = np.take(X_test, feature_indices, axis=1)
    classifier.fit(X_new, y_train)
    y_predictions = classifier.predict(x_new_test)
    conf_matrix = confusion_matrix(y_test, y_predictions)

    acc = accuracy(conf_matrix)
    return acc, feature_indices


def class33(output_dir, X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    # print('TODO Section 3.3')
    
    with open(f"{output_dir}/a1_3.3.txt", "w") as outf:
        # Prepare the variables with corresponding names, then uncomment
        # this, so it writes them to outf.
        # for each number of features k_feat, write the p-values for
        # that number of features:
        for k_feat in [5, 50]:
            selector = SelectKBest(f_classif, k=k_feat)
            X_new = selector.fit_transform(X_train, y_train)
            p_values = selector.pvalues_
            outf.write(f'{k_feat} p-values: {[round(pval, 4) for pval in p_values]}\n')

        accuracy_1k, top_features_1k = get_top_5_features_accuracy(X_1k, y_1k, X_test, y_test, i)
        accuracy_full, top_features_full = get_top_5_features_accuracy(X_train, y_train, X_test, y_test, i)

        feature_intersection = np.intersect1d(top_features_1k, top_features_full)
        top_5 = top_features_full

        outf.write(f'Accuracy for 1k: {accuracy_1k:.4f}\n')
        outf.write(f'Accuracy for full dataset: {accuracy_full:.4f}\n')
        outf.write(f'Chosen feature intersection: {feature_intersection}\n')
        outf.write(f'Top-5 at higher: {top_5}\n')
        


def class34(output_dir, X_train, X_test, y_train, y_test, i):
    ''' This function performs experiment 3.4
    
    Parameters
       output_dir: path of directory to write output to
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    # print('TODO Section 3.4')
    X = np.concatenate((X_train, X_test))
    Y = np.concatenate((y_train, y_test))
    print("start processing")
    with open(f"{output_dir}/a1_3.4.txt", "w") as outf:
        # Prepare kfold_accuracies, then uncomment this, so it writes them to outf.
        # for each fold:
        #     outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')
        # outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')
        # pass
        Kf = KFold(shuffle=True)
        classifier_accuracies = []
        print("Calc accuracy")
        for train_index, test_index in Kf.split(X):
            x_train, y_train = X[train_index], Y[train_index]
            x_test, y_test = X[test_index], Y[test_index]

            kfold_accuracies = np.zeros(5)

            for j in range(len(classifiers)):
                classifier = eval(classifiers[i] + parameters[i])
                classifier.fit(x_train, y_train)
                y_predictions = classifier.predict(x_test)
                conf_matrix = confusion_matrix(y_test, y_predictions)

                acc = accuracy(conf_matrix)
                kfold_accuracies[i] = acc
            classifier_accuracies.append(kfold_accuracies)
            outf.write(f'Kfold Accuracies: {[round(acc, 4) for acc in kfold_accuracies]}\n')

        print("Pval...")
        p_values = []
        classifier_accuracies = np.array(classifier_accuracies)
        classifier_accuracies = classifier_accuracies.transpose()
        for j in range(len(classifiers)):
            if j == i:
                continue
            S = ttest_rel(classifier_accuracies[j], classifier_accuracies[i])
            p_values.append(S.pvalue)
        outf.write(f'p-values: {[round(pval, 4) for pval in p_values]}\n')


            
        
            




    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    parser.add_argument(
        "-o", "--output_dir",
        help="The directory to write a1_3.X.txt files to.",
        default=os.path.dirname(os.path.dirname(__file__)))
    args = parser.parse_args()
    
    # TODO: load data and split into train and test.
    data = np.load(args.input)["arr_0"] # Need to be tested
    # print(data["arr_0"].shape)
    x = data[:, :173]
    y = data[:, 173]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
    # TODO : complete each classification experiment, in sequence.

    output_dir = args.output_dir
    
    iBest = class31(output_dir, X_train, X_test, y_train, y_test)
    X_1k, y_1k = class32(output_dir, X_train, X_test, y_train, y_test, iBest)

    class33(output_dir, X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    class34(output_dir, X_train, X_test, y_train, y_test, iBest)
