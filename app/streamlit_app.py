"""
Streamlit web application for Forest Fire Detection.

Upload a forest/landscape image and get a prediction (fire or no-fire)
with confidence score and optional Grad-CAM visualization.

Run with: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

# Add src/ to Python path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import streamlit as st
import torch
import numpy as np
from PIL import Image

from predict import load_model, predict_image, CLASS_NAMES
from gradcam import generate_gradcam


# Page configuration
st.set_page_config(
    page_title="Forest Fire Detection",
    page_icon="\U0001f525",
    layout="centered",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_cached_model(checkpoint_path, model_name, device_str):
    """Cache the model so it is not reloaded on every interaction."""
    device = torch.device(device_str)
    return load_model(checkpoint_path, model_name, device=device)


def get_device():
    """Get the best available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .prediction-fire {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 1rem;
        border-radius: 4px;
    }
    .prediction-safe {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
        padding: 1rem;
        border-radius: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.title("\U0001f525 Forest Fire Detection")
    st.markdown(
        "Upload a forest or landscape image to detect whether it contains fire. "
        "Powered by a ResNet18 model fine-tuned on the DeepFire dataset."
    )

    # Sidebar configuration
    st.sidebar.header("Settings")

    MODEL_DESCRIPTIONS = {
        "mobilenetv2": "MobileNetV2 (Transfer Learning | 99.0%)",
        "resnet18": "ResNet18 (Transfer Learning | 97.2%)",
        "improved": "Improved CNN (From Scratch | 95.8%)",
        "baseline": "Baseline CNN (From Scratch | 94.8%)"
    }
    
    model_name = st.sidebar.selectbox(
        "Model Architecture",
        ["mobilenetv2", "resnet18", "improved", "baseline"],
        index=0,
        format_func=lambda x: MODEL_DESCRIPTIONS[x]
    )

    BLURBS = {
        "mobilenetv2": "Lightweight, highly efficient pre-trained architecture with SOTA accuracy.",
        "resnet18": "Deep pre-trained architecture fine-tuned for high accuracy.",
        "improved": "Deeper 5-block CNN with batch normalization, trained entirely from scratch.",
        "baseline": "Simple 3-block CNN trained entirely from scratch."
    }
    st.sidebar.caption(BLURBS[model_name])

    models_dir = Path(__file__).parent.parent / "models"
    checkpoints = sorted(models_dir.glob("*.pth")) if models_dir.exists() else []

    if not checkpoints:
        st.error(
            "No model checkpoints found in the models/ directory. "
            "Please train a model first using `python src/train.py`."
        )
        return

    # Match the checkpoint based on model architecture
    search_term = "baseline_cnn" if model_name == "baseline" else model_name
    matching_checkpoints = [cp for cp in checkpoints if search_term in cp.name.lower()]

    if not matching_checkpoints:
        st.sidebar.error(f"No checkpoint found for {model_name}!")
        return

    checkpoint_path = str(matching_checkpoints[0])

    show_gradcam = st.sidebar.checkbox("Show Grad-CAM Visualization", value=True)
    gradcam_method = st.sidebar.selectbox(
        "Grad-CAM Method",
        ["gradcam", "eigencam"],
        index=0,
        help="GradCAM is standard; EigenCAM often produces cleaner visualizations"
    )

    # Load model
    device = get_device()
    st.sidebar.info(f"Device: {device}")

    try:
        model = load_cached_model(checkpoint_path, model_name, str(device))
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # File upload
    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Upload a forest or landscape image"
    )

    if uploaded_file is not None:
        # Display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # Make prediction
        with st.spinner("Analyzing image..."):
            # Save temp file for prediction functions
            temp_path = Path(__file__).parent / "temp_upload.jpg"
            image.save(temp_path)

            class_name, confidence, probabilities = predict_image(model, str(temp_path), device)

            # Grad-CAM
            overlay = None
            if show_gradcam:
                try:
                    rgb_img, heatmap, overlay, _ = generate_gradcam(
                        model, str(temp_path), model_name, device, gradcam_method
                    )
                except Exception as e:
                    st.warning(f"Could not generate Grad-CAM: {e}")

            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()

        # Display Grad-CAM
        with col2:
            if overlay is not None:
                st.image(overlay, caption="Grad-CAM Visualization", use_container_width=True)
            else:
                st.info("Enable Grad-CAM in the sidebar to see model attention.")

        # Results section
        st.markdown("---")

        # Prediction result with colored indicator
        if class_name == "fire":
            st.error(f"\U0001f525 **FIRE DETECTED** (Confidence: {confidence:.1%})")
        else:
            st.success(f"\U0001f332 **No Fire Detected** (Confidence: {confidence:.1%})")

        # Probability breakdown
        st.markdown("#### Prediction Probabilities")
        for name, prob in probabilities.items():
            label = "\U0001f525 Fire" if name == "fire" else "\U0001f332 No Fire"
            st.progress(prob, text=f"{label}: {prob:.1%}")

        # Additional info
        with st.expander("About this prediction"):
            st.markdown(f"""
            - **Model**: {model_name}
            - **Checkpoint**: {Path(checkpoint_path).name}
            - **Device**: {device}
            - **Image size**: {image.size[0]} x {image.size[1]}
            - **Grad-CAM method**: {gradcam_method if show_gradcam else "disabled"}

            **Note**: This model was trained on the DeepFire dataset for educational
            purposes. It should not be used as a sole fire detection system in
            production environments.
            """)

    else:
        # Show instructions when no image is uploaded
        st.info(
            "\U0001f4f7 Upload an image above to get started. "
            "The model works best with outdoor forest or landscape images."
        )

    # Footer
    st.markdown("---")
    st.markdown(
        "*Uses ResNet18 transfer learning on the "
        "[DeepFire dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset).*"
    )


if __name__ == "__main__":
    main()
