import streamlit as st
import numpy as np
from model import train_model
model,scaler,mse,r2=train_model()
st.title("House Price Prediction")
st.write("This model predict house price using multiple feature . And the model being used is linear regression")
bathroom=st.number_input("Enter the number of bathroom")
bedroom=st.number_input("Enter the number of bedroom")
living=st.number_input("Enter the square ft for living")
lot=st.number_input("Enter the square ft for lot")
floor=st.number_input("Enter the number of floor")
st.write("MSE:-",mse)
st.write("R2:-",r2)
if st.button("predict"):
    userdata=np.array([[bathroom,bedroom,living,lot,floor]])
    userdata=scaler.transform(userdata)
    st.write("Expected price:- ",model.predict(userdata))
