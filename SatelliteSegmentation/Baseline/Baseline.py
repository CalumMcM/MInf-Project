import os
import pandas as pd
import numpy as np
import re
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression

# Takes a directory of spreadsheets for a given set of classes and combines them
# into one dataset
def clean_data(dir):

    classes = ['Amazonia', 'Caatinga', 'Cerrado']
    classes = {'Amazonia': 0, 'Caatinga': 1, 'Cerrado': 2}

    data, labels = [], []

    # Loop through the directory for each class and combine all CSV files
    # for each class into one dataframe
    for cur_class in classes.keys():

        cur_folder_dir = dir+'/'+cur_class+' NDVI'

        # Sort the name of each spreadsheet by number
        files = os.listdir(cur_folder_dir)

        if '.DS_Store' in files: files.remove('.DS_Store')

        files.sort(key=lambda f: int(re.sub('\D', '', f)))


        # Append each spreadsheet NDVI value to array of all NDVI values
        for file in files:

            cur_df = pd.read_csv(cur_folder_dir+'/'+file, sep=',',header=0, index_col =0)

            index = int(file.split('.')[0])

            indices = np.arange(index, index+950)

            array = list(np.array(cur_df['NDVI']))

            # Append NDVI values to array of all NDVI values
            data = np.hstack([data, array])

            # Extend array of class labels to reflect new NDVI values
            labels = np.hstack([labels, [classes[cur_class]]*len(array)])

        # Replace all NaN values with mean of current class
        start = index*classes[cur_class]
        end = index*(classes[cur_class]+1)

        classMean = np.nanmean(data[start:end])

        data = np.nan_to_num(data, nan = classMean, posinf=classMean, neginf=classMean)

    return np.array(data), np.array(labels)

def main():

    X_test, y_test = clean_data('Test')

    X_train, y_train = clean_data('Train')

    X, y = make_classification(n_samples=1000, n_features=10, n_informative=5, n_redundant=5, n_classes=3, random_state=1)

    X_train = np.array([[a] for a in X_train])

    # Define the multinomial logistic regression model
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs')

    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)

    # evaluate the model and collect the scores
    n_scores = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=cv, n_jobs=-1)

    # report the model performance
    print('Mean Accuracy: %.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

    # Fit the model on the whole dataset
    model.fit(X_train, y_train)



if __name__ == "__main__":
    main()
