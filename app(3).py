
import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import numpy as np

# 1. Define the LungCancerClassifierFineTuned model architecture
# This class is identical to the one used for retraining
class LungCancerClassifierFineTuned(nn.Module):
    def __init__(self, num_classes=3):
        super(LungCancerClassifierFineTuned, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)

# 2. Define the image transformations for evaluation (val_test_transform)
val_test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. Define the label mapping
# This should match the label_map used during training
label_map = {'Normal': 0, 'Benign': 1, 'Malignant': 2}
inverse_label_map = {v: k for k, v in label_map.items()}
class_names = [inverse_label_map[i] for i in sorted(inverse_label_map.keys())]

# 4. Function to load the model weights
@st.cache_resource
def load_model(model_path='lung_cancer_classifier.pth', num_classes=3, device='cpu'):
    model = LungCancerClassifierFineTuned(num_classes=num_classes)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval() # Set model to evaluation mode
        st.success(f"Model loaded successfully from {model_path} to {device}.")
    except Exception as e:
        st.error(f"Error loading model: {e}. Make sure 'lung_cancer_classifier.pth' is in the same directory.")
        st.stop()
    return model

# 5. Prediction function
def predict_image(image_input, model, transform, device='cpu'):
    # image_input can be bytes data (from Streamlit UploadedFile)
    image = Image.open(io.BytesIO(image_input)).convert('RGB')

    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted_class_idx = torch.max(outputs, 1)

    predicted_label = inverse_label_map[predicted_class_idx.item()]
    predicted_probability = probabilities[0][predicted_class_idx.item()].item()

    return predicted_label, predicted_probability, probabilities[0].cpu().numpy()

# Streamlit Application Layout
st.set_page_config(page_title="Lung Cancer Prediction", page_icon="⚕️")
st.title("⚕️ Lung Cancer Prediction from CT Scan")
st.write("Upload a CT scan image to get a prediction for Normal, Benign, or Malignant lung conditions.")
st.write("Disclaimer: This model is for research and educational purposes only and should not be used for medical diagnosis or treatment decisions.")

# File uploader widget
uploaded_file = st.file_uploader("Choose a CT Scan Image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.success("Image uploaded successfully!")

    # Display the uploaded image
    col1, col2 = st.columns(2)
    with col1:
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

    # Load the model
    # Use 'cpu' as map_location because Streamlit Cloud generally doesn't have GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(num_classes=len(class_names), device=device)

    # Make prediction
    predicted_label, predicted_probability, all_probs = predict_image(uploaded_file.getvalue(), model, val_test_transform, device=device)

    with col2:
        st.subheader("Prediction Results")
        st.write(f"**Predicted Class:** <span style='font-size:24px; color:red;'>{predicted_label}</span>", unsafe_allow_html=True)
        st.write(f"**Confidence:** {predicted_probability*100:.2f}% Barton")

        st.subheader("All Class Probabilities:")
        for i, class_name in enumerate(class_names):
            st.write(f"- {class_name}: {all_probs[i]*100:.2f}%")

else:
    st.info("Please upload an image file to get started.")

st.markdown("--- Developed by Shreas Shivam for Colab DSA Project --- ")
