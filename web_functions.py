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
@st.cache()
def load_data():
    """This function returns the preprocessed data"""

    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('Parkinson.csv')

    # Rename the column names in the DataFrame.
    df.rename(columns = {"MDVP:Fo(Hz)": "AVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Fhi(Hz)": "MAVFF",}, inplace = True)
    df.rename(columns = {"MDVP:Flo(Hz)": "MIVFF",}, inplace = True)
    

    # Perform feature and target split
    X = df[["AVFF", "MAVFF", "MIVFF","Jitter:DDP","MDVP:Jitter(%)","MDVP:RAP","MDVP:APQ","MDVP:PPQ","MDVP:Shimmer","Shimmer:DDA","Shimmer:APQ3","Shimmer:APQ5","NHR","HNR","RPDE","DFA","D2","PPE"]]
    y = df['status']

    return df, X, y

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

def predictTree(X, y, features):
    # Get model and model score
    model, score = train_model(X, y)
    # Predict the value
    prediction = model.predict(np.array(features).reshape(1, -1))

    return prediction, score

def predictSVM(X,y,features):
    # Split dataset into training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=109)  # 70% training and 30% test
    clf = svm.SVC(C=1.0,gamma='scale',kernel='linear') # Linear Kernel
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score =  metrics.accuracy_score(y_test, y_pred)
    prediction = clf.predict([features])
    return prediction, score

def predictRandomF(X,y,features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
    clf=RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score  = metrics.accuracy_score(y_test, y_pred)
    prediction = clf.predict([features])
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return prediction , score

def predictKNN(X,y,features):
    res = train_test_split(X, y,train_size=0.8,test_size=0.2,random_state=1)
    train_data, test_data, train_labels, test_labels = res
    knn = KNeighborsClassifier(n_neighbors = 8)
    knn.fit(train_data, train_labels)
    y_predict = knn.predict(test_data)
    prediction = knn.predict([features])
    score = metrics.accuracy_score(test_labels, y_predict)
    return prediction,score

def predictADB(X,y,features):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # 70% training and 30% test
    # Create adaboost classifer object
    # svc = svm.SVC(probability=True, kernel='poly')


    abc = AdaBoostClassifier(n_estimators=50,learning_rate=1)
    # Train Adaboost Classifer
    model = abc.fit(X_train, y_train)

    # Predict the response for test dataset
    y_pred = model.predict(X_test)
    prediction  = model.predict([features])
    score = metrics.accuracy_score(y_test, y_pred)
    print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
    return prediction , score