##Company-Bankruptcy-Prediction-Flask-Deployment
Our top priority in this application is to identify companies in bankruptcy using KNN Classifier.

##Project Structure
This project has four major parts :

model.py - This contains code for our Data Preprocessing (Data Cleaning - Data Balancing - Feature Selection) and Machine Learning model (KNN Classifier) to predict Company Bankruptcy based on training data in 'data.csv' file. 
app.py - This contains Flask APIs that receives company details through GUI, computes the precited value based on our model and returns it.
templates - This folder contains the HTML template to allow user to enter company details and displays the predicted value of Bankruptcy.

###The Dataset :  https://www.kaggle.com/fedesoriano/company-bankruptcy-prediction  
