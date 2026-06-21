# Forest Fire Detection using Deep Learning

This project is a binary image classification system that detects whether a forest or landscape image contains fire.

The goal of this repository is to demonstrate an end-to-end computer vision workflow: data preparation, baseline modeling, transfer learning, comprehensive evaluation, model interpretability, and deployment.

## Project Structure

*   `src/`: Core Python modules.
    *   `dataset.py`: Handles data splitting (70/15/15), ImageNet normalization, and augmentations.
    *   `model.py`: Defines the BaselineCNN and transfer learning setups for ResNet18 and MobileNetV2.
    *   `train.py`: Contains the training loop, learning rate scheduling (ReduceLROnPlateau), and early stopping.
    *   `evaluate.py`: Generates classification reports, confusion matrices, ROC curves, and model comparison charts.
    *   `predict.py`: Single-image inference script.
    *   `gradcam.py`: Generates visual heatmaps to explain model predictions.
*   `app/`: Contains the Streamlit web application for real-time inference.
*   `notebooks/`: Contains `eda.py` for dataset analysis.
*   `data/`: Raw and processed dataset (downloaded via Kaggle API).
*   `models/`: Saved `.pth` checkpoint files for trained models.
*   `results/`: Evaluation charts, metrics, and Grad-CAM visualizations.

## Dataset

The model is trained on the [DeepFire Forest Fire Dataset](https://www.kaggle.com/datasets/alik05/forest-fire-dataset). 
The dataset contains 1,900 images perfectly balanced between two classes: `fire` (950) and `nofire` (950).

## Results

Four models were trained and evaluated on the test set:

1.  **Baseline CNN**: A simple 3-block convolutional network trained from scratch. (Accuracy: 94.76%)
2.  **Improved CNN**: A deeper 5-block convolutional network with Batch Normalization, trained from scratch. (Accuracy: 95.80%)
3.  **ResNet18**: A pre-trained model fine-tuned using transfer learning. (Accuracy: 97.20%)
4.  **MobileNetV2**: A lightweight pre-trained model fine-tuned using transfer learning. (Accuracy: 98.95%)

Transfer learning significantly reduced false negatives and false positives, with MobileNetV2 achieving the best overall performance. However, the Improved CNN demonstrated that architectural depth and regularization can substantially boost performance even when training entirely from scratch on a small dataset.

## Interpretability

To ensure the models are learning the correct visual features, we implemented **Grad-CAM** and **EigenCAM**. These techniques visualize where the model is "looking" when making a decision. The generated heatmaps confirm that the models focus accurately on the flames and smoke plumes rather than background artifacts. EigenCAM, in particular, produces very sharp, class-agnostic structural bounding maps.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Streamlit App:**
    ```bash
    streamlit run app/streamlit_app.py
    ```
