import streamlit as st
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch

# Set page config
st.set_page_config(page_title="AI vs Real Image Detector", layout="wide")

@st.cache_resource
def load_model():
    model = AutoModelForImageClassification.from_pretrained("Hemg/AI-VS-REAL-IMAGE-DETECTION")
    feature_extractor = AutoFeatureExtractor.from_pretrained("Hemg/AI-VS-REAL-IMAGE-DETECTION")
    return model, feature_extractor

def predict_image(image, model, feature_extractor):
    inputs = feature_extractor(image, return_tensors="pt")
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs[0].tolist()

# Load model
model, feature_extractor = load_model()

# Title
st.title("AI vs Real Image Detection")
st.write("Upload an image to check if it's AI-generated or real!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with col2:
        st.write("Analyzing...")
        # Make prediction
        probabilities = predict_image(image, model, feature_extractor)
        real_prob = probabilities[1] * 100
        fake_prob = probabilities[0] * 100
        
        # Display results
        st.write("## Results")
        st.write(f"**{'REAL' if real_prob > fake_prob else 'AI-GENERATED'} IMAGE**")
        
        # Create progress bars
        st.write("Probability Breakdown:")
        st.progress(real_prob/100, text=f"Real: {real_prob:.2f}%")
        st.progress(fake_prob/100, text=f"AI-Generated: {fake_prob:.2f}%")