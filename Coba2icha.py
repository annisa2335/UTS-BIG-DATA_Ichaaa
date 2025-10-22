import streamlit as st
from keras.models import load_model
from PIL import image
from util import classify

#set title
st.title('Car Classification')

#set header
st.header('Please upload a car image')

#upload file
st.file_uploader('', type=['jpeg', 'jpg', 'png'])

#load classifier
model = load_model('./model/Annisa Humaira_Laporan 2.h5')

#display image
if file is not None:
  image = image.open(file).convert('RGB')
  st.image(image, use_columnn_wodth=True)

#classify image
classify(image, model, class_names)
