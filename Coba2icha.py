import streamlit as st
from keras.models import load_model

#set title
st.title('Car Classification')

#set header
st.header('Please upload a car image')

#upload file
st.file_uploader('', type=['jpeg', 'jpg', 'png'])

#load classifier
model = load_model('./model/Annisa Humaira_Laporan 2.h5')
