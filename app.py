import streamlit as st
import joblib
import pickle
import cv2
import numpy as np
from PIL import Image

# =====================================
# PAGE CONFIG
# =====================================

st.set_page_config(
    page_title="Apple vs Tomato Classifier",
    page_icon="🍎",
    layout="wide"
)

# =====================================
# LOAD MODEL
# =====================================

model = joblib.load("apple_tomato_best.pkl")

le = pickle.load(
    open("label_encoder.pkl", "rb")
)

# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.stApp{
    background:linear-gradient(
        135deg,
        #020617,
        #0f172a,
        #172554
    );
}

/* Main Title */

.main-title{
    text-align:center;
    font-size:60px;
    font-weight:900;
    color:white;
}

.sub-title{
    text-align:center;
    color:#cbd5e1;
    font-size:22px;
    margin-bottom:25px;
}

/* Upload Box */

[data-testid="stFileUploader"]{
    background:rgba(255,255,255,0.08);
    border:2px solid #38bdf8;
    border-radius:15px;
    padding:15px;
}

/* Predict Button */

.stButton > button{
    width:100%;
    height:70px;
    font-size:24px;
    font-weight:bold;
    color:white;
    border:none;
    border-radius:15px;
    background:linear-gradient(
        90deg,
        #2563eb,
        #06b6d4
    );
}

/* Footer */

.footer{
    text-align:center;
    color:white;
    margin-top:40px;
    padding:20px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# HEADER
# =====================================

st.markdown("""
<div class='main-title'>
🍎🍅 Apple vs Tomato Classifier
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
Machine Learning Image Classification using KNN
</div>
""", unsafe_allow_html=True)

# =====================================
# PROJECT OVERVIEW
# =====================================

st.info("""
📌 PROJECT OVERVIEW

This application classifies images into:

🍎 Apple

🍅 Tomato

🚀 Technologies Used

• Python

• OpenCV

• NumPy

• Scikit-Learn

• Streamlit

🎯 Model Details

• KNN Classifier

• K = 11

• Distance Weighted

• Accuracy = 75.95%
""")

# =====================================
# FILE UPLOAD
# =====================================

st.markdown("## 📤 Upload Image For Prediction")

uploaded_file = st.file_uploader(
    "Choose an Apple or Tomato Image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# PREDICTION
# =====================================

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns([2,1])

    with col1:

        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.markdown("### 📊 Image Details")

        st.info(f"""
Width : {image.size[0]} px

Height : {image.size[1]} px

Mode : {image.mode}
""")

        predict_btn = st.button(
            "🚀 Predict Image"
        )

    if predict_btn:

        img = np.array(
            image.convert("RGB")
        )

        img = cv2.resize(
            img,
            (100,100)
        )

        img = img.flatten().reshape(1,-1)

        prediction = model.predict(img)[0]

        result = le.inverse_transform(
            [prediction]
        )[0]

        st.markdown("---")

        if result == "apple":

            st.markdown("""
            <div style="
                background:#16a34a;
                padding:25px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-size:30px;
                font-weight:bold;
            ">
                🍎 APPLE DETECTED
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

        else:

            st.markdown("""
            <div style="
                background:#ea580c;
                padding:25px;
                border-radius:15px;
                text-align:center;
                color:white;
                font-size:30px;
                font-weight:bold;
            ">
                🍅 TOMATO DETECTED
            </div>
            """, unsafe_allow_html=True)

            st.balloons()

# =====================================
# FOOTER
# =====================================

st.markdown("""
<div class='footer'>

<h2>💻 Developed By Khaja Mainuddin</h2>

<h4>B.Tech Artificial Intelligence & Data Science</h4>

<p>
Python • OpenCV • Scikit-Learn • Streamlit
</p>

⭐ Thank You For Visiting ⭐

</div>
""", unsafe_allow_html=True)