"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import streamlit as st
from sklearn.model_selection import train_test_split
#Import svm model
from sklearn import svm
# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
# @st.cache()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Parkinson.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns = {"MDVP:Fo(Hz)": "AVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Fhi(Hz)": "MAVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Flo(Hz)": "MIVFF",}, inplace = True)

    return df

@st.cache()
def train_model(X, y):
    """This function trains the model and return the model and model score"""
    # Create the model
    model = DecisionTreeClassifier(
            ccp_alpha=0.0, class_weight=None, criterion='entropy',
            max_depth=4, max_features=None, max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_samples_leaf=1,
            min_samples_split=2, min_weight_fraction_leaf=0.0,
            random_state=42, splitter='best'
        )
    # Fit the data on model
    model.fit(X, y)
    # Get the model score
    score = model.score(X, y)

    # Return the values
    return model, score

def predictTree(features):
    my_models =[]
    my_scores =[]
    my_predictions =[]
    for i in range(1,50):
        """This function returns the preprocessed data"""

        # Load the Diabetes dataset into DataFrame.
        df = pd.read_csv('Parkinson.csv')

        # Rename the column names in the DataFrame.
        df.rename(columns={"MDVP:Fo(Hz)": "AVFF", }, inplace=True)
        df.rename(columns={"MDVP:Fhi(Hz)": "MAVFF", }, inplace=True)
        df.rename(columns={"MDVP:Flo(Hz)": "MIVFF", }, inplace=True)

        train, test = train_test_split(df, test_size=0.3, shuffle=True)

        # Perform feature and target split
        X_train = train[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                         "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA",
                         "D2", "PPE"]]
        y_train = train['status']
        X_test = test[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                       "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA", "D2",
                       "PPE"]]
        y_test = test['status']
     # Create the model
        model = DecisionTreeClassifier(
                ccp_alpha=0.0, class_weight=None, criterion='entropy',
                max_depth=None, max_features=None, max_leaf_nodes=None,
                min_impurity_decrease=0.0, min_samples_leaf=1,
                min_samples_split=2, min_weight_fraction_leaf=0.0,
                random_state=42, splitter='best'
            )
        # Fit the data on mode
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Get the model score
        score = metrics.accuracy_score(y_test, y_pred)
        # Predict the value
        prediction = model.predict(np.array(features).reshape(1, -1))
        my_models.append(model)
        my_scores.append(score)
        my_predictions.append(prediction)

    best_model = my_models[my_scores.index(max(my_scores))]
    best_prediction = my_predictions[my_models.index(best_model)]

    return best_prediction, max(my_scores)

def predictSVM(features):
    my_models =[]
    my_scores =[]
    my_predictions =[]
    for i in range(1,20):
        """This function returns the preprocessed data"""

        # Load the Diabetes dataset into DataFrame.
        df = pd.read_csv('Parkinson.csv')

        # Rename the column names in the DataFrame.
        df.rename(columns={"MDVP:Fo(Hz)": "AVFF", }, inplace=True)
        df.rename(columns={"MDVP:Fhi(Hz)": "MAVFF", }, inplace=True)
        df.rename(columns={"MDVP:Flo(Hz)": "MIVFF", }, inplace=True)

        train, test = train_test_split(df, test_size=0.3, shuffle=True)

        # Perform feature and target split
        X_train = train[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                         "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA",
                         "D2", "PPE"]]
        y_train = train['status']
        X_test = test[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                       "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA", "D2",
                       "PPE"]]
        y_test = test['status']
     # Create the model
        clf = svm.SVC(C=1.0, gamma='scale', kernel='linear')  # Linear Kernel
        # Fit the data on mode
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Get the model score
        score = metrics.accuracy_score(y_test, y_pred)
        # Predict the value
        prediction = clf.predict(np.array(features).reshape(1, -1))
        my_models.append(clf)
        my_scores.append(score)
        my_predictions.append(prediction)

    best_model = my_models[my_scores.index(max(my_scores))]
    best_prediction = my_predictions[my_models.index(best_model)]

    return best_prediction, max(my_scores)

def predictRandomF(features):

    my_models =[]
    my_scores =[]
    my_predictions =[]
    for i in range(1,20):
        """This function returns the preprocessed data"""

        # Load the Diabetes dataset into DataFrame.
        df = pd.read_csv('Parkinson.csv')

        # Rename the column names in the DataFrame.
        df.rename(columns={"MDVP:Fo(Hz)": "AVFF", }, inplace=True)
        df.rename(columns={"MDVP:Fhi(Hz)": "MAVFF", }, inplace=True)
        df.rename(columns={"MDVP:Flo(Hz)": "MIVFF", }, inplace=True)

        train, test = train_test_split(df, test_size=0.3, shuffle=True)

        # Perform feature and target split
        X_train = train[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                         "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA",
                         "D2", "PPE"]]
        y_train = train['status']
        X_test = test[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                       "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA", "D2",
                       "PPE"]]
        y_test = test['status']
        clf = RandomForestClassifier(n_estimators=100, random_state=0)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        # Get the model score
        score = metrics.accuracy_score(y_test, y_pred)
        # Predict the value
        prediction = clf.predict(np.array(features).reshape(1, -1))
        my_models.append(clf)
        my_scores.append(score)
        my_predictions.append(prediction)

    best_model = my_models[my_scores.index(max(my_scores))]
    best_prediction = my_predictions[my_models.index(best_model)]

    return best_prediction, max(my_scores)

def predictKNN(features):

    my_models =[]
    my_scores =[]
    my_predictions =[]
    for i in range(1,40):
        """This function returns the preprocessed data"""

        # Load the Diabetes dataset into DataFrame.
        df = pd.read_csv('Parkinson.csv')

        # Rename the column names in the DataFrame.
        df.rename(columns={"MDVP:Fo(Hz)": "AVFF", }, inplace=True)
        df.rename(columns={"MDVP:Fhi(Hz)": "MAVFF", }, inplace=True)
        df.rename(columns={"MDVP:Flo(Hz)": "MIVFF", }, inplace=True)

        train, test = train_test_split(df, test_size=0.3, shuffle=True)

        # Perform feature and target split
        X_train = train[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                         "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA",
                         "D2", "PPE"]]
        y_train = train['status']
        X_test = test[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                       "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA", "D2",
                       "PPE"]]
        y_test = test['status']
     # Create the model
        knn = KNeighborsClassifier(n_neighbors=7)
        # Fit the data on mode
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # Get the model score
        score = metrics.accuracy_score(y_test, y_pred)
        # Predict the value
        prediction = knn.predict(np.array(features).reshape(1, -1))
        my_models.append(knn)
        my_scores.append(score)
        my_predictions.append(prediction)

    best_model = my_models[my_scores.index(max(my_scores))]
    best_prediction = my_predictions[my_models.index(best_model)]

    return best_prediction, max(my_scores)

def predictADB(features):
    my_models =[]
    my_scores =[]
    my_predictions =[]
    for i in range(1,20):
        """This function returns the preprocessed data"""

        # Load the Diabetes dataset into DataFrame.
        df = pd.read_csv('Parkinson.csv')

        # Rename the column names in the DataFrame.
        df.rename(columns={"MDVP:Fo(Hz)": "AVFF", }, inplace=True)
        df.rename(columns={"MDVP:Fhi(Hz)": "MAVFF", }, inplace=True)
        df.rename(columns={"MDVP:Flo(Hz)": "MIVFF", }, inplace=True)

        train, test = train_test_split(df, test_size=0.3, shuffle=True)

        # Perform feature and target split
        X_train = train[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                         "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA",
                         "D2", "PPE"]]
        y_train = train['status']
        X_test = test[["AVFF", "MAVFF", "MIVFF", "Jitter:DDP", "MDVP:Jitter(%)", "MDVP:RAP", "MDVP:APQ", "MDVP:PPQ",
                       "MDVP:Shimmer", "Shimmer:DDA", "Shimmer:APQ3", "Shimmer:APQ5", "NHR", "HNR", "RPDE", "DFA", "D2",
                       "PPE"]]
        y_test = test['status']
     # Create the model
        abc = AdaBoostClassifier(n_estimators=50, learning_rate=1)
        # Train Adaboost Classifer
        abc.fit(X_train, y_train)

        y_pred = abc.predict(X_test)
        # Get the model score
        score = metrics.accuracy_score(y_test, y_pred)
        # Predict the value
        prediction = abc.predict(np.array(features).reshape(1, -1))
        my_models.append(abc)
        my_scores.append(score)
        my_predictions.append(prediction)

    best_model = my_models[my_scores.index(max(my_scores))]
    best_prediction = my_predictions[my_models.index(best_model)]

    return best_prediction, max(my_scores)