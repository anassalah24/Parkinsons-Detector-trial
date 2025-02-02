"""This is the main module to run the app"""

# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from web_functions import load_data

# Import pages
from Tabs import home, data, predictTree, visualise , predictSVM ,predictKNN,predictRandomF,predictADB

# Configure the app
st.set_page_config(
    page_title = 'Parkinson\'s Disease Prediction',
    page_icon = 'raised_hand_with_fingers_splayed',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)

# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Decision Tree Prediction": predictTree,
    "SVM Prediction": predictSVM,
    "KNN Prediction": predictKNN,
    "Random Forest Prediction": predictRandomF,
    "Ada-boost Prediction": predictADB,
    "Visualisation": visualise
    
}

# Create a sidebar
# Add title to sidear
st.sidebar.title("Navigation")

# Create radio option to select the page
page = st.sidebar.radio("Pages", list(Tabs.keys()))

# Loading the dataset.
df = load_data()

# Call the app function of selected page to run
if page in ["Decision Tree Prediction", "Visualisation" , "SVM Prediction" , "KNN Prediction" , "Random Forest Prediction", "Ada-boost Prediction" ]:
    Tabs[page].app(df)
elif (page == "Data Info"):
    Tabs[page].app(df)
else:
    Tabs[page].app()
