# Singapore Resale Flat Prices Predicting

## Table of Contents

- [Project Overview](#project-overview)
- [Problem Statement](#problem-statement)
- [Solution Steps](#solution-steps)
- [Approach](#approach)
- [Workflow](#workflow)
- [Data Description](#data-description)
- [Learning Outcomes](#learning-outcomes)
- [Technologies Used](#technologies-used)
- [How to Run](#how-to-run)

## Project Overview

This project aims to develop a machine learning model and deploy it as a user-friendly web application that predicts the resale prices of flats in Singapore. The predictive model will be based on historical data of resale flat transactions, assisting both potential buyers and sellers in estimating the resale value of a flat.

## Problem Statement

The resale flat market in Singapore is highly competitive, making it challenging to accurately estimate resale values. This project addresses this issue by utilizing machine learning techniques to develop a predictive model that considers factors such as location, flat type, floor area, and lease duration.

## Solution Steps

1. **Data Collection and Preprocessing**: Collect a dataset of resale flat transactions from the Singapore Housing and Development Board (HDB) and preprocess the data to clean and structure it for machine learning.
2. **Feature Engineering**: Extract relevant features from the dataset and create additional features to enhance prediction accuracy.
3. **Model Selection and Training**: Choose appropriate machine learning models for regression and train them on the historical data.
4. **Model Evaluation**: Evaluate the model's performance using regression metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R2 Score.
5. **Streamlit Web Application**: Develop a user-friendly web application using Streamlit, allowing users to input details of a flat and predict the resale price based on their inputs.


## Approach

- **Data Understanding**: Understand the structure and content of the dataset, identifying key variables and their distributions.
- **Data Preprocessing**: Clean the data by handling missing values, outliers, and encoding categorical variables.
- **Model Building**: Select suitable regression models and train them using the preprocessed data.
- **Web Development**: Utilize Streamlit to create an interactive web application for model deployment.

## Workflow



## Data Description

**Dataset Link**: [Singapore Resale Flat Transactions Dataset](https://beta.data.gov.sg/collections/189/view)

The dataset includes various columns such as 

* `month`: The month and year when the resale transaction occurred.
* `town`: The town where the flat is located.
* `flat_type`: The type of flat (e.g., 3-room, 4-room, etc.).
* `block`: The block number of the flat.
* `street_name`: The name of the street where the flat is located.
* `storey_range`: The range of storeys where the flat is situated.
* `floor_area_sqm`: The floor area of the flat in square meters.
* `flat_model`: The model or layout of the flat.
* `lease_commence_date`: The year when the lease commenced for the flat.
* `resale_price`: The price at which the flat was resold.


## Learning Outcomes

This project enhances proficiency in data science techniques, including data preprocessing, exploratory data analysis (EDA), machine learning modeling and web application development.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Streamlit

## How to Run

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the Streamlit application using `streamlit run app.py`.
