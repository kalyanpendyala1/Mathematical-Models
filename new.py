## Ricardo A. Calix, 2016 
## getting started with scikit-learn
#######################################################

#imports

import sklearn
import numpy as np
#numpy its a linear algebra
import pandas as pd
#pandas a lib for excels

from sklearn import datasets
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score,f1_score
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

#####################################
## set parameters




#######################################
## load problem data
## find a csv file and load it here
#f_numpy = open("iris.csv",'r')
#Matrix_data = np.loadtxt(f_numpy, delimiter = ',', skiprows = 1)

#x = Matrix_data[:, [0, 1, 2, 3]]
#y = Matrix_data[:, 4]
#print(x)
#print(y)


#######################################
#######################################
##load iris data for testing purposes
#iris = datasets.load_iris()
#x = iris.data
#y = iris.target



#######################################
## load breast cancer data

#df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", header = None)
#print(df)
#x = df.loc[:, 2:].values
#y = df.loc[:, 1].values
#print(x)
#print(y)

#le = LabelEncoder()
#y = le.fit_transform(y)
#print(y)
#######################################
mnist=fetch_mldata('MNIST original')
x=mnist.data
y=mnist.target

#######################################
## plotting function

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(),xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z,alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test_set')

###################################################

def print_stats_10_fold_crossvalidation(algo_name, model, X_train, y_train ):
    from sklearn.cross_validation import StratifiedKFold
    from sklearn.metrics import roc_curve, auc
    from scipy import interp
    kfold = StratifiedKFold(y=y_train,
                        n_folds=10,
                        random_state=1)
    print "----------------------------------------------"
    print "Start of 10 fold crossvalidation results"
    print "the algorithm is: ", algo_name
    #################################
    #roc
    fig = plt.figure(figsize=(7,5))
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    all_tpr = []
    ################################
    scores = []
    f_scores = []
    for k, (train, test) in enumerate(kfold):
        model.fit(X_train[train], y_train[train])
        y_pred = model.predict(X_train[test])
        ########################
        #roc
        probas = model.predict_proba(X_train[test])
        #pos_label in the roc_curve function is very important. it is the value
        #of your classes such as 1 or 2, for versicolor or setosa
        fpr, tpr, thresholds = roc_curve(y_train[test],probas[:,1], pos_label=2)
        mean_tpr += interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=1,
                 label='ROC fold %d (area = %0.2f)' % (k+1, roc_auc))

        ########################
        ## print results
        print('Accuracy: %.2f' % accuracy_score(y_train[test], y_pred))
        confmat = confusion_matrix(y_true=y_train[test], y_pred=y_pred)
        print "confusion matrix"
        print(confmat)
        print('Precision: %.3f' % precision_score(y_true=y_train[test], y_pred=y_pred))
        print('Recall: %.3f' % recall_score(y_true=y_train[test], y_pred=y_pred))
        f_score = f1_score(y_true=y_train[test], y_pred=y_pred)
        print('F1-measure: %.3f' % f_score)
        f_scores.append(f_score)
        score = model.score(X_train[test], y_train[test])
        scores.append(score)
        print('fold : %s, Accuracy: %.3f' % (k+1, score))
        print("######################################################")
    ######################################
    #roc
    mean_tpr /= len(kfold)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plot_ROC_curve(plt, mean_fpr, mean_tpr, mean_auc )
    ######################################
    print('overall accuracy: %.3f and +/- %.3f' % (np.mean(scores), np.std(scores)))
    print('overall f1_score: %.3f' % np.mean(f_scores))

###################################################

def print_stats_percentage_train_test(algorithm_name, y_test, y_pred):
    print("########################################################")
    print('algorithm is:', algorithm_name)
    print('accuracy: %.2f' % accuracy_score(y_test, y_pred))
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print "confusion matrix"
    print(confmat)
    print('Precision: %.3f' % precision_score(y_true=y_test, y_pred=y_pred))
    print('Recall: %.3f' % recall_score(y_true=y_test, y_pred=y_pred))
    f_score = f1_score(y_true=y_test, y_pred=y_pred)
    print('F1-measure: %.3f' % f_score)


##################################################
## plot ROC curve

def plot_ROC_curve(plt, mean_fpr, mean_tpr, mean_auc ):
    #fig = plt.figure(figsize=(7,5))
    plt.plot( [0,1],
              [0,1],
              linestyle='--',
              color=(0.6, 0.6, 0.6),
              label='random guessing')
    plt.plot(mean_fpr, mean_tpr, 'k--',
             label='mean ROC (area = %0.2f)' % mean_auc, lw=2)
    plt.plot( [0,0,1],
              [0,1,1],
              lw=2,
              linestyle=':',
              color='black',
              label='perfect performance')
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.title('Receiver Operator Characterstics')
    plt.legend(loc="lower right").set_visible(True)
    plt.show()




###################################################
## plot 2d graphs

def plot_2d_graph_model(model,start_idx_test, end_idx_test, X_train_std, X_test_std, y_train, y_test ):
    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))
    # first 70% is train, last 30% is test
    #so in y_combined from 1 to 941 is train data. from 942 to 1345 is test
    plot_decision_regions(X=X_combined_std,
                      y=y_combined,
                      classifier=model,
                      test_idx=range(start_idx_test,end_idx_test)) #942,1344
    plt.xlabel('cnt')
    plt.ylabel('sum')
    plt.legend(loc='lower left')
    plt.show()

###################################################
## print stats train % and test percentage (i.e. 70% train
## and 30% test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.30, random_state=42)
#x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.01, random_state=42)

print(y_test)
print(x_test)

sc = StandardScaler()
sc.fit(x_train)

x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)


###################################################
## knn

def knn_kp(x_train_std, y_train, x_test_std, y_test):
    from sklearn.neighbors import KNeighborsClassifier

    knn = KNeighborsClassifier(n_neighbors =5, metric = 'minkowski')
    knn.fit(x_train_std, y_train)
    y_pred = knn.predict(x_test_std)
    #print "Predicted values are", y_pred,  y_test
    #print ('Accuracy: %.2f'% accuracy_score(y_test, y_pred))
    print_stats_percentage_train_test('KNN', y_test, y_pred)

#######################################
## logistic regression
def logistic_regression(x_train_std, y_train, x_test_std, y_test):
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=1000.0, random_state=0, solver='lbfgs')
    lr.fit(x_train_std, y_train)
    y_pred = lr.predict(x_test_std)
    #lr_result = lr.predict_proba(x_test_std[0,:])
    #print(lr_result)
    #print_stats_10_fold_crossvalidation("logisticRegression", lr,x_train_std, y_train)# you need to put all the data without split
    print_stats_percentage_train_test("logistic_regression", y_test, y_pred)


#####################################################
## random forest

def random_forest_rc(X_train, y_train, X_test, y_test):
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    forest.fit(x_train, y_train)
    y_pred = forest.predict(X_test)
    print_stats_percentage_train_test("Random Forest", y_test, y_pred)
    #print_stats_10_fold_crossvalidation("Random Forest", forest ,x_train, y_train)



#######################################
## svm
"""
def svm_rc(X_train, y_train,X_test,y_test):
    from sklearn.svm import SVC
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10)
    svm.fit(x_train, y_train)
    y_pred = svm.predict(X_test)
    print_stats_percentage_train_test("SVM", y_test, y_pred)
"""

def svm_rc(X_train_std, y_train, X_test_std, y_test):
    from sklearn.svm import SVC
    #svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10, probability=True) # high precision, low recall, why?
    svm = SVC(kernel='rbf', random_state=0, gamma=0.10, C=10)
    svm.fit(X_train_std, y_train)
    y_pred = svm.predict(X_test_std)
    print_stats_percentage_train_test("svm (rbf)", y_test, y_pred)
    #print_stats_10_fold_crossvalidation("svm (rbf)",svm,X_train_std,y_train )

#######################################
# A perceptron




#######################################
## decision trees
## prints tree graph as well
def decision_trees_rc(X_train, y_train,X_test,y_test ):
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=30, random_state=0)
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    print_stats_percentage_train_test('decision trees', y_test, y_pred)
    from sklearn.tree import export_graphviz
    export_graphviz(tree, out_file='tree.dot')

###################################################
## create train and test sets, or put all data in train sets
## for k-fold cross validation
## also perform feature scaling



#######################################
## does not plot roc curve


#######################################
## ML_MAIN()
#random_forest_rc(x_train_std, y_train, x_test_std, y_test)
knn_kp(x_train_std, y_train, x_test_std, y_test)
#logistic_regression(x_train_std, y_train, x_test_std, y_test)
#decision_trees_rc(x_train_std, y_train, x_test_std, y_test)
#svm_rc(x_train_std, y_train, x_test_std, y_test)
#######################################

print "<<<<<<DONE>>>>>>>"









