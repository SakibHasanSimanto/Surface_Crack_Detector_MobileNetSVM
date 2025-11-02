import streamlit as st
import numpy as np
from tensorflow.keras.applications import MobileNetV3Small
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from PIL import Image
import joblib

# ==============================
# Load model and label mappings
# ==============================
@st.cache_resource
def load_models():
    mobilenetv3 = MobileNetV3Small(weights="imagenet", include_top=False, pooling="avg")
    clf = joblib.load('mobilenetv3_svm.pkl')
    class_indices = np.load('class_indices.npy', allow_pickle=True).item()
    inv_classes = {v: k for k, v in class_indices.items()}
    return mobilenetv3, clf, inv_classes

mobilenetv3, clf, inv_classes = load_models()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="MobileNet-SVM Image Classifier", layout="centered")
st.title("Surface Crack Detector (MobileNet-SVM)")
st.write("Upload an image below to classify it using MobileNet-SVM model.")

uploaded_file = st.file_uploader("📁 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = img.resize((160, 160))
    img_array = np.expand_dims(np.array(img_resized), axis=0)
    img_array = preprocess_input(img_array)

    # Feature extraction
    features = mobilenetv3.predict(img_array)

    # SVM prediction
    pred = clf.predict(features)[0]

    if pred == 1 : 
        st.subheader('⚠️ Crack Detected')
    else: 
        st.subheader('✅ No Crack')

   
 
