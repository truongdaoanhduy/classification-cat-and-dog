import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from loadfile import path_data_train
from preprocessing import *
import numpy as np
from model import *
from loadfile import *

if __name__ == '__main__':
    st.title('DOG AND CAT')
    st.divider()
    file_upload = st.file_uploader('Choose a file',type=['jpg','png','jpeg'])
    if file_upload:
        col1,col2 = st.columns(2)
        img = Image.open(file_upload)
        col1.image(img)
        pre_img = preprocessing(img)
        new_img = (pre_img.detach().clone().permute(1,2,0).numpy()*255).astype(np.uint8)
        
        col2.image(new_img)
        train_loader,val_loader = get_data()
        if st.button('Training'):
            training().running(train_loader,val_loader,device,criterion,optimizer)
        if st.button('predict'):
            classification = training().predict(pre_img,device)
            print(classification)
            if classification == [0]:
                st.write('Cat')
            else: 
                st.write('Dog')

