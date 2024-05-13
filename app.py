import streamlit as st
from ultralytics import YOLO
from PIL import Image

model = YOLO('yolov8n.pt')
with st.sidebar:
    st.text('Navigate Sidebar...')
    add_radio = st.radio(
        "Choose a resolution",
        ("Normal", "Hd")
    )
    threshold = st.slider("Threshold", min_value=0.4, max_value=0.99,value=0.5)
with st.expander("About this app"):
    st.text("This app was created in class for fun")

img_file = st.file_uploader("Upload Your Image", type = ['png', 'jpg'], help="This should be only images you nitwit")

if img_file:
    col1, col2 = st.columns(2)
    col1.image(img_file, caption="This is your image", use_column_width=True)

    # st.image(img_file, caption="This is your Image")
    Image.open(img_file).save(img_file.name)
    results = model(img_file.name, stream=False, conf=threshold)
    results[0].save(filename='result.jpg')
    col2.image('result.jpg', caption="This is your result", use_column_width=True)