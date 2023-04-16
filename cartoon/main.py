import streamlit as st
from PIL import Image

import numpy as np
import cv2 as cv



DEMO_IMAGE = 'index.jpg'


@st.cache
def sketch(img):
    img2grey = cv.cvtColor(img , cv.COLOR_BGR2GRAY)
    img2grey = cv.medianBlur(img2grey,5)
    edge  = cv.Laplacian(img2grey,cv.CV_8U,ksize =5)

    ret , threshold = cv.threshold(edge,70 , 255 , cv.THRESH_BINARY_INV)

    return threshold

def cartoon(img, grey_mode = False):
    threshold = sketch(img)

    filtered = cv.bilateralFilter(img,10,250,250)
    carto = cv.bitwise_and(filtered , filtered,mask = threshold)

    if grey_mode:
        return cv.cvtColor(carto , cv.COLOR_BGR2GRAY)
    
    return carto






st.title("cartoon the image")


img_file_buffer = st.file_uploader("Upload an image", type=[ "jpg", "jpeg",'png'])


if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))


st.image(image,caption = 'original_image' , use_column_width = True)



sketch_image = sketch(image)
cartoon_image = cartoon(image)
cartoon_image_grey = cartoon(image , True)

sketch_grey , sketch_color = cv.pencilSketch(image ,sigma_r = 0.1 , shade_factor = 0.1)
stylish_image = cv.stylization(image , sigma_s = 60 , sigma_r = 0.07)





st.subheader('Sketch Image')
st.image(sketch_image , caption = 'sketch_image' , use_column_width = True)


st.subheader('Cartoonized Image')
st.image(cartoon_image , caption = 'cartoon_image' , use_column_width = True)



st.subheader('Cartoonized Image grey')
st.image(cartoon_image_grey , caption = 'cartoon_image' , use_column_width = True)



st.subheader('Pencil Image')
st.image(sketch_grey , caption = 'sketch_grey' , use_column_width = True)


st.subheader('Stylized Image')
st.image(stylish_image , caption = 'stylish_image' , use_column_width = True)




