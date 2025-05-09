import streamlit as st
import numpy as np
import cv2
from PIL import Image
import folium
from streamlit_folium import st_folium
import torch
import torch.nn as nn
from torchvision import models, transforms
from fpdf import FPDF

# -------------------- Page Configuration -------------------- #
st.set_page_config(page_title="🛰️ Satellite Images Analysis 🛰️", layout="wide", page_icon="🌍")

st.markdown("""
    <style>
    .main { background-color: #f7f9fc; }
    .stTabs [role="tab"] {
        font-size: 18px;
        padding: 12px;
        color: #0e1117;
        border-radius: 8px 8px 0 0;
        border-bottom: 2px solid #ccc;
    }
    .stTabs [role="tab"]:hover {
        background-color: #dbeafe;
    }
    .st-emotion-cache-1y4p8pa {
        background-color: #f0f4f8;
        padding: 10px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 8px 16px;
    }
    .stDownloadButton>button {
        background-color: #10b981;
        color: white;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🛰️ Satellite Image Analysis Platform ")

tab1, tab2 = st.tabs(["📸 Image Analyzer", "🗺️ Interactive Geo-Map"])

# ---------------- Transformation & Model Loader ---------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_model(model_name, num_classes=2):
    if model_name == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "MobileNetV3":
        model = models.mobilenet_v3_small(pretrained=True)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
    elif model_name == "EfficientNet_B0":
        model = models.efficientnet_b0(pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    model.eval()
    return model

# ------------------- TAB 1: Image Upload & Analysis ------------------- #
with tab1:
    st.header("📷 Upload and Analyze Satellite Images")

    uploaded_file = st.file_uploader("Upload a satellite image", type=["jpg", "jpeg", "png", "tif", "tiff"])

    final_label = None

if uploaded_file:
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(uploaded_file, caption="🖼️ Original Image", use_column_width=True)

        with col2:
            st.subheader("Choose Image Processing")
            image = Image.open(uploaded_file).convert("RGB")
            opencv_image = np.array(image)

            analysis_option = st.radio("Select a method:", ["None", "Grayscale", "Edge Detection", "Simulated NDVI"], horizontal=True)

            if analysis_option == "Grayscale":
                gray = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY)
                st.image(gray, caption=" Grayscale Output", use_column_width=True, clamp=True)

            elif analysis_option == "Edge Detection":
                edges = cv2.Canny(cv2.cvtColor(opencv_image, cv2.COLOR_RGB2GRAY), 100, 200)
                st.image(edges, caption="⚡ Edge Detection", use_column_width=True, clamp=True)

            elif analysis_option == "Simulated NDVI":
                red = opencv_image[:, :, 0].astype(float)
                nir = opencv_image[:, :, 1].astype(float)
                ndvi = (nir - red) / (nir + red + 1e-5)
                ndvi_image = ((ndvi + 1) / 2 * 255).astype(np.uint8)
                st.image(ndvi_image, caption="🌱 Simulated NDVI", use_column_width=True, clamp=True)

        # CNN Classification and PDF generation must be inside this block
        st.subheader("🧠 CNN Classification")

        model_choices = ["MobileNetV2", "MobileNetV3", "EfficientNet_B0"]
        label_map = {0: "Healthy Vegetation", 1: "Deforested Area"}
        model_outputs = {}

        input_tensor = transform(image).unsqueeze(0)

        for model_name in model_choices:
            model = load_model(model_name)
            with torch.no_grad():
                outputs = model(input_tensor)
                predicted_class = torch.argmax(outputs, dim=1).item()
                prediction_label = label_map.get(predicted_class, "Unknown")
                model_outputs[model_name] = prediction_label

        model_option = st.selectbox("Choose model to display:", model_choices)
        st.success(f"✅ Final Classification: **{model_outputs[model_option]}** (Model: {model_option})")

        if model_outputs:
            class PDF(FPDF):
                def header(self):
                    self.set_font("Arial", "B", 12)
                    self.cell(0, 10, "Satellite Image Classification Report", 0, 1, "C")

            pdf = PDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, "Model Predictions Summary:", ln=True)
            pdf.ln(5)

            for model_name, prediction in model_outputs.items():
                pdf.cell(200, 10, f"{model_name}: {prediction}", ln=True)

            pdf_output_path = "classification_report.pdf"
            pdf.output(pdf_output_path)

            with open(pdf_output_path, "rb") as file:
                st.download_button(
                    label="📄 Download PDF Report",
                    data=file,
                    file_name="satellite_classification_report.pdf",
                    mime="application/pdf"
                )

# ------------------- TAB 2: Geo Map Interaction ------------------- #
with tab2:
    st.header("🗺️ Clickable Map for Coordinates")

    col_map, col_data = st.columns([2, 1])

    with col_map:
        st.markdown("Click anywhere on the map below to get latitude and longitude:")
        map_obj = folium.Map(location=[20.5937, 78.9629], zoom_start=4)
        map_obj.add_child(folium.LatLngPopup())
        st_data = st_folium(map_obj, height=600, width=800)

    with col_data:
        st.subheader("📍 Coordinates")
        if st_data and st_data["last_clicked"]:
            lat = st_data["last_clicked"]["lat"]
            lon = st_data["last_clicked"]["lng"]
            st.metric("Latitude", f"{lat:.5f}")
            st.metric("Longitude", f"{lon:.5f}")
        else:
            st.info("Click on the map to show coordinates.")

# ------------------- Footer ------------------- #
st.markdown("---")
st.markdown("<center>© 2025 Satellite Analysis Platform | Powered by MobileNet, EfficientNet</center>", unsafe_allow_html=True)

