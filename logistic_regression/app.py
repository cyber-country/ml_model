import streamlit as st
from model import model_train
import numpy as np
model,scaler,accuracy,c=model_train()
st.title("Exam prediction")
st.write("This model tell us pass/fail using multi feature and using logistic regression")
st.write("The model accuracy is :-",accuracy)
st.write("The model confusion matrix:-",c)
print(c)
hour=st.number_input("Enter study hour")
prev=st.number_input("Enter Previous marks")
if st.button("Predict"):
    data=np.array([[hour,prev]])
    data=scaler.transform(data)
    ouput=model.predict(data)
    if ouput==1:
        ouput="Pass"
    else:
        ouput="Fail"
    st.write("prediction:-",ouput)