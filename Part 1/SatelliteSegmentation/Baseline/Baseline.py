import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from matplotlib.legend_handler import HandlerLine2D
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

plt.style.use('ggplot')

# Takes a directory of spreadsheets for a given set of classes and combines them
# into one dataset
def clean_data_test(quads):

    classes = ['Amazon', 'Caatinga', 'Cerrado']
    classes = {'Amazon': 0, 'Caatinga': 1, 'Cerrado': 2}

    data, labels = [], []

    # Loop through the directory for each class and combine all CSV files
    # for each class into one dataframe

    start_biome_index = 0

    for cur_class in classes.keys():

        cur_folder_dir = 'Data/'+cur_class+' NDVI Quads'

        quad = quads[cur_class]

        quad_folder_dir = cur_folder_dir + '/Quad ' + str(quad)

        print (quad_folder_dir)

        # Sort the name of each spreadsheet by number
        files = os.listdir(quad_folder_dir)

        if '.DS_Store' in files: files.remove('.DS_Store')

        files.sort(key=lambda f: int(re.sub('\D', '', f)))

        prev_index = 0

        # Append each spreadsheet NDVI value to array of all NDVI values
        for file in files:

            cur_df = pd.read_csv(quad_folder_dir+'/'+file, sep=',',header=0, index_col =0, engine='python')

            array = list(np.array(cur_df['NDVI']))

            indices = np.arange(prev_index, prev_index+len(array))

            # Append NDVI values to array of all NDVI values
            data = np.hstack([data, array])

            # Extend array of class labels to reflect new NDVI values
            labels = np.hstack([labels, [classes[cur_class]]*len(array)])

            prev_index = prev_index+len(array)

        end_biome_index = start_biome_index+prev_index

        # Replace all NaN values with mean of current quad
        classMean = np.nanmean(data[start_biome_index:end_biome_index])

        data = np.nan_to_num(data, nan = classMean, posinf=classMean, neginf=classMean)

        start_biome_index = end_biome_index

    return np.array(data), np.array(labels)

# Takes a directory of spreadsheets for a given set of classes and combines them
# into one dataset
def clean_data(quads):

    classes = ['Amazon', 'Caatinga', 'Cerrado']
    classes = {'Amazon': 0, 'Caatinga': 1, 'Cerrado': 2}

    data, labels = [], []

    start_biome_index = 0

    # Loop through the directory for each class and combine all CSV files
    # for each class into one dataframe
    for cur_class in classes.keys():

        cur_folder_dir = 'Data/' + cur_class+' NDVI Quads'

        for quad in quads[cur_class]:

            quad_folder_dir = cur_folder_dir + '/Quad ' + str(quad)
            print (quad_folder_dir)
            # Sort the name of each spreadsheet by number
            files = os.listdir(quad_folder_dir)

            if '.DS_Store' in files: files.remove('.DS_Store')

            files.sort(key=lambda f: int(re.sub('\D', '', f)))

            prev_index = 0

            # Append each spreadsheet NDVI value to array of all NDVI values
            for file in files:

                cur_df = pd.read_csv(quad_folder_dir+'/'+file, sep=',',header=0, index_col =0, engine='python')

                array = list(np.array(cur_df['NDVI']))

                indices = np.arange(prev_index, prev_index+len(array))

                # Append NDVI values to array of all NDVI values
                data = np.hstack([data, array])

                # Extend array of class labels to reflect new NDVI values
                labels = np.hstack([labels, [classes[cur_class]]*len(array)])

                prev_index = prev_index+len(array)

            end_biome_index = start_biome_index+prev_index

            # Replace all NaN values with mean of current quad
            classMean = np.nanmean(data[start_biome_index:end_biome_index])

            data = np.nan_to_num(data, nan = classMean, posinf=classMean, neginf=classMean)

            start_biome_index = end_biome_index

    return np.array(data), np.array(labels)

# Takes a training set of data and their labels and returns
# the predictions of a multinomial linear regression
# model for a test set based on this data
def Regression(X_train, y_train, X_test):

    # Define the multinomial logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='saga')

    # Fit the model on the whole dataset
    model.fit(X_train, y_train)

    # Get predictions of model on test data
    lr_preds = model.predict(X_test)

    return lr_preds

# Trains a Decision Tree classifier on the train given training set and then
# Returns the performance on the given test set
# Code can be uncommented in order to apply hyper-parameter tuning
def DecisionTree(X_train, y_train, X_test, y_test):

    random_state = np.random.RandomState(0)

    decisionTree = DecisionTreeClassifier(random_state=0, max_depth = 11)

    decisionTree.fit(X_train, y_train)

    predictions = decisionTree.predict(X_test)

    # Uncomment below for best depth experiment
    """
    accs = []
    depths = list(range(1, 41))

    for depth in range(1,41):

        decisionTree = DecisionTreeClassifier(random_state=0, max_depth = depth)

        decisionTree.fit(X_train, y_train)

        predictions = decisionTree.predict(X_test)

        correct_preds = 0
        for i in range(0,len(predictions)):
            if predictions[i] == y_test[i]:
                correct_preds += 1

        accs.append(correct_preds/len(predictions))

    print (np.where(accs == np.amax(accs)))
    print (np.amax(accs))

    plt.plot(depths, accs) #adds the line
    plt.ylabel('Accuracy') #xlabel
    plt.xlabel('Depth') #ylabel
    plt.savefig('depthvsAccuracyDT') # Best = 7
    plt.show()
    """
    return predictions

# Trains a Random Forest on the given training set and then evaluates
# The performance on the test set. Depending on the passed `mode' the
# model will do either
# Train = Train on the given train set and return predictions for given test set
# estimators = Plot the performance of random forest with differing number of
#                estimators on the given test set`
# depth = Plot the performance of random forest with differing number of
#                max depth on the given test set`
def RandomForest(X_train, y_train, X_test, y_test, mode='Train'):

    if (mode=='Train'):
        rf = RandomForestClassifier(max_depth=8, n_estimators=16)

        rf.fit(X_train, y_train)

        y_pred = rf.predict(X_test)

        return y_pred

    # Determine optimal number of estimators for the random forest
    if (mode=='estimators'):

        accs = []
        n_estimators = [1, 2, 4, 8, 16, 32, 64, 100, 200]

        train_results = []
        test_results = []

        for estimator in n_estimators:

           rf = RandomForestClassifier(n_estimators=estimator, n_jobs=-1)

           rf.fit(X_train, y_train)

           predictions = rf.predict(X_test)

           correct_preds = 0
           for i in range(0,len(predictions)):
               if predictions[i] == y_test[i]:
                   correct_preds += 1

           accs.append(correct_preds/len(predictions))

        print (np.where(accs == np.amax(accs)))
        print (np.amax(accs))

        plt.plot(n_estimators, accs) #adds the line
        plt.ylabel('Accuracy') #xlabel
        plt.xlabel('Number of Estimators') #ylabel
        plt.savefig('EstimatorsvsAccuracyRF') # Best = 7
        plt.show()

        return []

    # Determine optimal number of estimators for the random forest
    if (mode=='depth'):

        max_depths = np.linspace(1, 41, 41, endpoint=True)

        accs = []

        train_results = []
        test_results = []

        for depth in max_depths:

           rf = RandomForestClassifier(max_depth=depth, n_jobs=-1)

           rf.fit(X_train, y_train)

           predictions = rf.predict(X_test)

           correct_preds = 0
           for i in range(0,len(predictions)):
               if predictions[i] == y_test[i]:
                   correct_preds += 1

           accs.append(correct_preds/len(predictions))

        print (np.where(accs == np.amax(accs)))
        print (np.amax(accs))

        plt.plot(max_depths, accs) #adds the line
        plt.ylabel('Accuracy') #xlabel
        plt.xlabel('Depth') #ylabel
        plt.savefig('DepthvsAccuracyRF') # Best = 7
        plt.show()

        return []

# Returns the evaluation metrics (accuracy, precision and recall)
# for the passed predicitions and actual values
def evaluate(y_test, preds):

        correct_preds = 0
        for i in range(0,len(preds)):
            if y_test[i] == preds[i]:
                correct_preds += 1

        accuracy = correct_preds/len(preds)

        precision, recall, fscore, _ = precision_recall_fscore_support(y_test, preds, average='macro')

        f1score = 2*((precision * recall)/(precision+recall))

        return accuracy, precision, recall, f1score

def printResults(accuracy, precision, recall, f1_score, classifier):
    print ("\n#########" + str(classifier) + "#########\n")
    print ("______MACRO EVALUATION______")
    print ("ACCURACY: {:.3f} ± {:.3f}".format(np.mean(accuracy), np.std(accuracy)))
    print ("PRECISION: {:.3f} ± {:.3f}".format(np.mean(precision), np.std(precision)))
    print ("RECALL: {:.3f} ± {:.3f}".format(np.mean(recall), np.std(recall)))
    print ("F1-SCORE: {:.3f} ± {:.3f}".format(np.mean(f1_score), np.std(f1_score)))

def main():

    biome_quads_test = {'Amazon': 2, 'Caatinga': 3, 'Cerrado': 4}

    X_test, y_test = clean_data_test(biome_quads_test)

    biome_quads_train = {'Amazon': [1,3,4], 'Caatinga': [1,2,4], 'Cerrado': [1,2,3]}

    X_train, y_train = clean_data(biome_quads_train)

    # Change shape of data
    X_train = np.array([[a] for a in X_train])
    X_test = np.array([[a] for a in X_test])

    print (X_train.shape)

    n_trials = 10

    lr_accs, lr_pres, lr_recs, lr_f1ss = [], [], [], []
    dt_accs, dt_pres, dt_recs, dt_f1ss = [], [], [], []
    rf_accs, rf_pres, rf_recs, rf_f1ss = [], [], [], []

    X_tst, X_val, y_tst, y_val = train_test_split(X_test, y_test, test_size=0.2)

    print (X_tst.shape)
    print (X_val.shape)

    for i in range(0, n_trials):

        lr_preds = Regression(X_train, y_train, X_tst)

        rf_preds = RandomForest(X_train, y_train, X_tst, y_tst, mode='Train')

        dt_preds = DecisionTree(X_train, y_train, X_tst, y_tst)

        acc, pre, rec, f1s = evaluate(y_tst, lr_preds)

        lr_accs.append(acc)
        lr_pres.append(pre)
        lr_recs.append(rec)
        lr_f1ss.append(f1s)

        acc, pre, rec, f1s = evaluate(y_tst, dt_preds)

        dt_accs.append(acc)
        dt_pres.append(pre)
        dt_recs.append(rec)
        dt_f1ss.append(f1s)

        acc, pre, rec, f1s = evaluate(y_tst, rf_preds)

        rf_accs.append(acc)
        rf_pres.append(pre)
        rf_recs.append(rec)
        rf_f1ss.append(f1s)

    printResults(lr_accs, lr_pres, lr_recs, lr_f1ss, "MULTINOMIAL LOGISTIC REGRESSION")
    printResults(dt_accs, dt_pres, dt_recs, dt_f1ss, "DECISION TREE")
    printResults(rf_accs, rf_pres, rf_recs, rf_f1ss, "RandomForest")


if __name__ == "__main__":
    main()
