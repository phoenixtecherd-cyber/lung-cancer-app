import streamlit as st
from pipeline import run_pipeline
import cv2
from PIL import Image

# 🎨 Custom Background & Styling
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
    color: white;
}

h1, h2, h3 {
    color: #00E5FF;
    text-align: center;
}

.stMetric {
    background-color: rgba(255,255,255,0.1);
    padding: 10px;
    border-radius: 10px;
    text-align: center;
}

.css-1d391kg {
    background-color: transparent;
}

</style>
""", unsafe_allow_html=True)

st.set_page_config(layout="wide")
#st.title("🫁 Lung Cancer Detection & Risk Analysis System")

st.markdown("<h1>🫁 Lung Cancer Detection & Risk Analysis System</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload CT Image", type=["jpg", "png"])

if uploaded_file:

    # Save uploaded file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.read())

    col1, col2 = st.columns(2)

    # ORIGINAL IMAGE
    with col1:
        st.subheader("Original CT Image")
        st.image("temp.jpg", width='stretch')

    # RUN PIPELINE
    result = run_pipeline("temp.jpg")

    # YOLO OUTPUT IMAGE (if saved)
    detect_img_path = "runs/detect/predict/temp.jpg"

    with col2:
        st.subheader("YOLO Detection")
        try:
            st.image(detect_img_path, width='stretch')
        except:
            st.warning("Detection image not found")

    st.markdown("---")

    # 📊 METRICS
    st.markdown("### 📊 Tumour Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("🔍 Nodules", result["nodules"])
    st.metric("📏 Max Diameter", round(result["max_diameter"], 2))

with col2:
    st.metric("📦 Avg Volume", round(result["avg_volume"], 2))
    st.metric("📊 Confidence", round(result["confidence"], 3))

with col3:
    st.metric("🧬 Prediction", result["prediction"])
    st.metric("⚠️ Risk Level", result["risk"])
    st.markdown("---")
    
    if result["risk"] == "High Risk":
      st.error("🚨 High Risk Detected")
    elif result["risk"] == "Medium Risk":
      st.warning("⚠️ Medium Risk")
    else:
      st.success("✅ Low Risk / Normal")

    
    

    # 🔥 GRAD-CAM / HEATMAP
# 🔥 GRAD-CAM / HEATMAP
# 🔥 GRAD-CAM / HEATMAP
# 🔥 GRAD-CAM / HEATMAP
st.subheader("🔥 Grad-CAM++ Heatmap")

col1, col2 = st.columns(2)

with col1:
    st.image(result["heatmap"], channels="BGR", caption="🔥 Heatmap", width='stretch')

with col2:
    st.image(result["overlay"], channels="BGR", caption="🧠 Overlay", width='stretch')
    st.markdown("---")

    # 🧠 FINAL SUMMARY
    

    st.markdown("### 🧠 Final Interpretation")

    st.info(f"""
Detected Nodules: {result['nodules']}  
Max Diameter: {result['max_diameter']:.2f}  
Volume: {result['avg_volume']:.2f}  
Prediction: {result['prediction']}  
Confidence: {result['confidence']:.2f}  
Risk: {result['risk']}
""")

    st.markdown("---")

    # 📌 PROJECT NOTE (for viva)
    with st.expander("📌 Important Notes (For Viva)"):
        st.write("""
        - Tumour volume is approximated using 2D bounding box area.
        - This is not a full 3D volumetric reconstruction.
        - The system is designed for screening-level risk assessment.
        - Risk is computed using:
            • Nodule count  
            • Size (diameter)  
            • Detection confidence  
            • Classification output  
        """)
