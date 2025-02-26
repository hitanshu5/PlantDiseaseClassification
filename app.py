import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import pandas as pd

# Set page configuration
st.set_page_config(page_title="Plant Disease Classification", layout="wide")

# Cache the model using the new caching command
@st.cache_resource
def load_trained_model():
    return load_model("model.h5")

model = load_trained_model()

# Define image size and class mapping
IMG_SIZE = 256
CLASS_NAMES = {
    0: 'Apple___Apple_scab',
    1: 'Apple___Black_rot',
    2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy',
    4: 'Blueberry___healthy',
    5: 'Cherry_(including_sour)___Powdery_mildew',
    6: 'Cherry_(including_sour)___healthy',
    7: 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    8: 'Corn_(maize)___Common_rust_',
    9: 'Corn_(maize)___Northern_Leaf_Blight',
    10: 'Corn_(maize)___healthy',
    11: 'Grape___Black_rot',
    12: 'Grape___Esca_(Black_Measles)',
    13: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    14: 'Grape___healthy',
    15: 'Orange___Haunglongbing_(Citrus_greening)',
    16: 'Peach___Bacterial_spot',
    17: 'Peach___healthy',
    18: 'Pepper,_bell___Bacterial_spot',
    19: 'Pepper,_bell___healthy',
    20: 'Potato___Early_blight',
    21: 'Potato___Late_blight',
    22: 'Potato___healthy',
    23: 'Raspberry___healthy',
    24: 'Soybean___healthy',
    25: 'Squash___Powdery_mildew',
    26: 'Strawberry___Leaf_scorch',
    27: 'Strawberry___healthy',
    28: 'Tomato___Bacterial_spot',
    29: 'Tomato___Early_blight',
    30: 'Tomato___Late_blight',
    31: 'Tomato___Leaf_Mold',
    32: 'Tomato___Septoria_leaf_spot',
    33: 'Tomato___Spider_mites Two-spotted_spider_mite',
    34: 'Tomato___Target_Spot',
    35: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    36: 'Tomato___Tomato_mosaic_virus',
    37: 'Tomato___healthy'
}

# Treatment advice for each disease
DISEASE_TREATMENT = {
    'Apple___Apple_scab': "Treat with appropriate fungicides, remove infected leaves, and ensure proper spacing for air circulation.",
    'Apple___Black_rot': "Remove and destroy infected fruits and branches; apply copper-based fungicides.",
    'Apple___Cedar_apple_rust': "Remove nearby alternate hosts (like juniper) if possible and apply fungicides.",
    'Apple___healthy': "The apple is healthy. No treatment required.",
    'Blueberry___healthy': "The blueberry plant is healthy. No treatment required.",
    'Cherry_(including_sour)___Powdery_mildew': "Improve air circulation, remove infected parts, and apply sulfur-based fungicides.",
    'Cherry_(including_sour)___healthy': "The cherry plant is healthy. No treatment required.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use fungicides and practice crop rotation to minimize spread.",
    'Corn_(maize)___Common_rust_': "Apply fungicides and remove infected leaves to manage the spread.",
    'Corn_(maize)___Northern_Leaf_Blight': "Use resistant varieties, apply fungicides, and rotate crops.",
    'Corn_(maize)___healthy': "The corn is healthy. No treatment required.",
    'Grape___Black_rot': "Prune infected areas, apply sulfur or copper-based fungicides, and maintain proper vineyard hygiene.",
    'Grape___Esca_(Black_Measles)': "Remove infected wood and consider fungicide treatments; consult a local expert.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Improve canopy airflow and apply appropriate fungicides to control the disease.",
    'Grape___healthy': "The grape is healthy. No treatment required.",
    'Orange___Haunglongbing_(Citrus_greening)': "Currently incurable; remove infected trees and manage the vector (psyllids) to slow spread.",
    'Peach___Bacterial_spot': "Apply copper sprays and remove infected foliage to prevent spread.",
    'Peach___healthy': "The peach is healthy. No treatment required.",
    'Pepper,_bell___Bacterial_spot': "Remove infected parts and apply bactericides; maintain field sanitation.",
    'Pepper,_bell___healthy': "The pepper plant is healthy. No treatment required.",
    'Potato___Early_blight': "Apply fungicides, use crop rotation, and remove affected leaves.",
    'Potato___Late_blight': "Remove and destroy infected plants immediately; use appropriate fungicides.",
    'Potato___healthy': "The potato plant is healthy. No treatment required.",
    'Raspberry___healthy': "The raspberry plant is healthy. No treatment required.",
    'Soybean___healthy': "The soybean is healthy. No treatment required.",
    'Squash___Powdery_mildew': "Apply fungicides, improve ventilation, and remove infected foliage.",
    'Strawberry___Leaf_scorch': "Remove affected leaves and apply fungicides if conditions worsen.",
    'Strawberry___healthy': "The strawberry plant is healthy. No treatment required.",
    'Tomato___Bacterial_spot': "Remove infected leaves, use copper-based sprays, and maintain field hygiene.",
    'Tomato___Early_blight': "Apply fungicides, practice crop rotation, and remove affected foliage.",
    'Tomato___Late_blight': "Remove infected plants immediately and apply fungicides; ensure proper field sanitation.",
    'Tomato___Leaf_Mold': "Improve air circulation around plants and apply appropriate fungicides.",
    'Tomato___Septoria_leaf_spot': "Remove affected leaves and use fungicides to prevent spread.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Control spider mites using appropriate acaricides and increase humidity if possible.",
    'Tomato___Target_Spot': "Remove infected parts and apply fungicide treatments as necessary.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Control the whitefly vector and consider using resistant varieties.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants, sanitize tools, and avoid mechanical transmission.",
    'Tomato___healthy': "The tomato is healthy. No treatment required."
}

def preprocess_image(image, target_size=(IMG_SIZE, IMG_SIZE)):
    """
    Preprocess the input image:
      - Convert to RGB if necessary.
      - Resize to target dimensions.
      - Normalize pixel values.
      - Add a batch dimension.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict_disease(image):
    """
    Predict the class of the input image and return both the label and raw prediction probabilities.
    """
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = CLASS_NAMES.get(predicted_class_index, "Unknown")
    return predicted_class_name, predictions

def plot_prediction_probabilities(predictions, top_n=5):
    """
    Generate a horizontal bar chart of the top prediction probabilities.
    """
    probabilities = predictions[0]
    data = [(CLASS_NAMES[i], float(probabilities[i])) for i in range(len(probabilities))]
    data.sort(key=lambda x: x[1], reverse=True)
    top_data = data[:top_n]
    labels = [item[0] for item in top_data]
    probs = [item[1] for item in top_data]
    
    fig = px.bar(
        x=probs,
        y=labels,
        orientation='h',
        labels={'x': 'Probability', 'y': 'Disease'},
        title='Top Prediction Probabilities',
        text=[f"{p:.2%}" for p in probs]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_tickformat='%')
    return fig

def plot_dominant_colors(image, n_colors=5):
    """
    Compute dominant colors in the image using KMeans clustering and display as a pie chart.
    This analysis can provide insight into the overall coloration of the leaf,
    which may be indicative of its health.
    """
    img_array = np.array(image)
    pixels = img_array.reshape(-1, 3)
    kmeans = KMeans(n_clusters=n_colors, random_state=42)
    kmeans.fit(pixels)
    colors = kmeans.cluster_centers_
    labels = kmeans.labels_
    label_counts = np.bincount(labels)
    percentages = label_counts / label_counts.sum()
    
    df = pd.DataFrame({
         'color': [f'Color {i+1}' for i in range(n_colors)],
         'percentage': percentages,
         'hex': [f'#{int(c[0]):02x}{int(c[1]):02x}{int(c[2]):02x}' for c in colors]
    })
    
    fig = px.pie(df, names='color', values='percentage', title='Dominant Colors in Image', 
                 color='color', color_discrete_map={f'Color {i+1}': df.loc[i, 'hex'] for i in range(n_colors)})
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def main():
    st.title("🌱 Plant Disease Classification")
    st.markdown("Upload an image of a plant leaf to classify its disease and receive actionable treatment insights.")
    
    # Sidebar settings
    st.sidebar.header("Settings")
    multi_upload = st.sidebar.checkbox("Allow Multiple Uploads", value=True)
    show_prediction_plot = st.sidebar.checkbox("Show Prediction Probabilities", value=True)
    show_dominant_colors = st.sidebar.checkbox("Show Dominant Colors Analysis", value=True)
    
    if multi_upload:
        uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    else:
        uploaded_files = [st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])]
    
    # Create tabs for different sections
    tabs = st.tabs(["Prediction", "Model Info", "About"])
    
    with tabs[0]:
        st.header("Prediction and Insights")
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file is not None:
                    image = Image.open(uploaded_file)
                    st.subheader(f"Image: {uploaded_file.name}")
                    st.image(image, caption="Original Image", width=300)
                    
                    if st.button(f"Predict for {uploaded_file.name}"):
                        with st.spinner("Classifying..."):
                            predicted_class_name, predictions = predict_disease(image)
                            st.success(f"Prediction: **{predicted_class_name}**")
                            
                            treatment_text = DISEASE_TREATMENT.get(predicted_class_name, "No treatment information available.")
                            st.markdown(f"**Treatment Advice:** {treatment_text}")
                            
                            if show_prediction_plot:
                                fig_pred = plot_prediction_probabilities(predictions, top_n=5)
                                st.plotly_chart(fig_pred, use_container_width=True)
                            
                            if show_dominant_colors:
                                fig_colors = plot_dominant_colors(image, n_colors=5)
                                st.plotly_chart(fig_colors, use_container_width=True)
                            st.markdown("---")
        else:
            st.info("Upload an image to get started.")
    
    with tabs[1]:
        st.header("Model Information")
        st.markdown(
            """
            ### Plant Disease Classification Model Report

            **Overview:**  
            This deep Convolutional Neural Network (CNN) model has been trained on thousands of plant leaf images to accurately identify 38 classes of plant diseases and distinguish healthy leaves. By leveraging data augmentation, dropout, and batch normalization, the model delivers robust performance under diverse field conditions.

            **Key Highlights:**  
            - **Objective:** To enable rapid and accurate diagnosis of plant diseases, facilitating timely intervention and minimizing crop losses.
            - **Dataset:** Comprising thousands of images, the dataset captures a wide range of plant conditions—from common diseases to healthy states.
            - **Performance & Impact:**  
              - Achieves high diagnostic accuracy, crucial for early detection and targeted treatment.
              - Early intervention based on accurate predictions can significantly reduce crop losses and support sustainable farming practices.
            - **Real-World Benefits:**  
              - Enhances food security by preventing large-scale crop failures.
              - Promotes precise, need-based treatment, reducing the reliance on broad-spectrum pesticides.
            """
        )
    
    with tabs[2]:
        st.header("About the App")
        st.markdown(
            """
            ### Why Plant Disease Detection is Crucial

            **Agricultural Impact:**  
            Early detection of plant diseases is essential for preventing widespread crop damage. This technology empowers farmers with real-time insights, enabling them to apply targeted treatments and save valuable resources.

            **Economic & Environmental Benefits:**  
            - **Economic:** Minimizes crop losses and maximizes yield, providing direct benefits to the farming community.
            - **Environmental:** Supports sustainable agriculture by reducing unnecessary chemical applications and promoting healthier ecosystems.

            **Technological Innovation:**  
            Leveraging advanced deep learning techniques, this app offers a user-friendly interface for disease diagnosis and actionable treatment advice, transforming traditional farming practices.

            **How This App Helps:**  
            - **Ease of Use:** Simply upload an image and receive instant predictions along with treatment recommendations.
            - **Actionable Insights:** Gain a comprehensive understanding through both prediction probabilities and an analysis of the dominant leaf colors, which can hint at underlying health issues.
            - **Future Prospects:** Continued enhancements aim to incorporate more sophisticated image analytics and extend treatment recommendations based on localized farming practices.

            **Developer:** Your Name or Organization  
            **Contact:** your.email@example.com
            """
        )

if __name__ == '__main__':
    main()