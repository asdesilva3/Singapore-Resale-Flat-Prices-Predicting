# import
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import pickle
from PIL import Image

import warnings
warnings.filterwarnings("ignore")
#________________________________________________________________________

def predict_price(year, encoded_town, floor_area, flat_town, lease, storey_log):

    with open("C:/Users/Desilva/Documents/ASD_Python/Singapore Resale Flat Prices Predicting/Resale_Flat_Prices_Predictor.pkl","rb") as m:
        model = pickle.load(m)
    
    data = np.array([[year, encoded_town, floor_area, flat_town,lease,storey_log]])
    prediction = model.predict(data)
    price = np.exp(prediction[0])

    return round(price)

#_______________________________________________________________________


df_town = pd.read_csv("C:/Users/Desilva/Documents/ASD_Python/Singapore Resale Flat Prices Predicting/df_town.csv")
df_flat_model = pd.read_csv("C:/Users/Desilva/Documents/ASD_Python/Singapore Resale Flat Prices Predicting/df_flat_model.csv")

town_dict = dict(zip(df_town['town'], df_town['town_labeled']))
flat_model_dict = dict(zip(df_flat_model['flat_model'], df_flat_model['flat_model_labeled']))

#_________________________________________________________________________

st.set_page_config(page_title= "AIRBNB ANALYSIS",
                   layout= "wide",
                   menu_items={'About': "### This page is created by Desilva!"})

st.markdown("<h1 style='text-align: center; color: #fa6607;'>SINGAPORE RESALE FLAT PRICES PREDICTING</h1>", unsafe_allow_html=True)
st.write("")

select = option_menu(None,["Home", "Price Prediction"], 
                    icons =["house-door-fill","currency-dollar"], orientation="horizontal",
                    styles={"container": {"padding": "0!important", "background-color": "#fafafa"},
                            "icon": {"color": "#fdfcfb", "font-size": "20px"},
                            "nav-link": {"font-size": "20px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
                            "nav-link-selected": {"background-color": "#fa6607"}})

if select == "Home":
    st.title("Welcome to Singapore Resale Flat Prices Predictor!")

    st.write('''
#### About
Our web application aims to assist both potential buyers and sellers in estimating the resale value of flats in Singapore. 
             Using machine learning techniques, we predict resale prices based on various factors such as location, flat model, floor area, lease duration & storey.

#### How it Works
1. **Input Details:** Users can input details of a Year, town, flat area (sqm), flat model, lease commence date & number of storey.
2. **Prediction:** Our machine learning model processes the input data and predicts the resale price of the flat.
3. **Get Estimate:** Users receive an estimated resale price, helping them make informed decisions about buying or selling a flat.

#### Data Source
We collect data from the Singapore Housing and Development Board (HDB) for resale flat transactions spanning from 1990 to the present(2024).

#### Model Performance
Our model's predictive performance is evaluated using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.

#### Try It Out!
Use the navigation bar on the top to explore different features of our web application:
- **Home:** You're here!
- **Predict:** Input flat details and get a resale price estimate.

#### Get Started
Start predicting resale flat prices now! Simply head over to the "Predict" page and input the details of the flat you're interested in.
''')

elif select == "Price Prediction":

    st.write("")
    st.header("Fill all the details below to know the resale price prediction")
    st.write("")

    col1,col2,col3 = st.columns([5,1,5])
    with col1:
        year = st.selectbox('Select a Year:', range(1990, 2025))

        town = st.selectbox('Select a town:', list(town_dict.keys()))
        # Map the selected town to its encoded value
        encoded_town = town_dict[town]

        floor_area = st.selectbox('Select a Floor Area (sqm):', range(28, 174))

    with col3:
        flat = st.selectbox('Select a Flat Model:', list(flat_model_dict.keys()))
        # Map the selected town to its encoded value
        flat_town = flat_model_dict[flat]
        
        lease = st.selectbox('Select a Lease Commence Year:', range(1966, 2019))

        storey = st.selectbox('Select a Number of Storey:',  range(1, 50))
        storey_log = np.log(storey)

    st.write("")
    st.write("")

    col1,col2,col3 = st.columns([3,5,3])
    with col2:
        button = st.button(":red[PREDICT THE RESALE VALUE]",use_container_width= True)

        if button:
            price = predict_price(year, encoded_town, floor_area, flat_town, lease, storey_log)

            st.write("## :green[Predicted Resale Value is :]",price)


#__________________________________________END_____________________________________________________________