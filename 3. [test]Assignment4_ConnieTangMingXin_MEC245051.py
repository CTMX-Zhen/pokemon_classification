"""
ASSIGNMENT 4 Task Description:
## TASK:
1. Find one problem and one deep learning model from any platforms (Not limited to):
- GitHub (https://github.com)
- TensorFlow (https://www.tensorflow.org/tutorials/images/cnn)
- Kaggle (https://www.kaggle.com/models?query=cnn)

*Apply the CNN model to solve the selected problem.

## SUBMISSION: 
Report:
- Introduction to the selected problem and CNN model.
- Model setup and implementation explanation.
- Result analysis and discussion.
- Screenshot of implementation.
- Screenshot of results

NAME: Connie Tang Ming Xin
MATRIC NUMBER: MEC245051
"""

# Assignment 4 - DL/CNN
    # Dataset: Mall Customer Segmentation Data 
        # - Sourse downloaded from Kaggle: https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python?resource=download
        # - Description: This dataset contains demographic and spending behavior data of mall customers. 
        # - The columns include:
            # - CustomerID: Unique customer ID
            # - Gender: Gender of the customer
            # - Age: Age of the customer
            # - Annual Income (k$): Annual income of the customer
            # - Spending Score (1-100): Spending score of the customer on a scale of 1 to 100

    # Platform: offline AI tools & code your own solution
        # - Visual Studio Code
        # - Jupyter Notebook
        # - Python
        # - pandas, matplotlib, seaborn, scikit-learn
        
"""
    STUDENT'S OUTCOMES
"""
# pip install icrawler

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import sys
import logging
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from icrawler.builtin import GoogleImageCrawler

# == IMPORT TENSORFLOW LIBRARIES ==
tensorflow_callback = tf.keras.callbacks.Callback

# === Configuration ===
data_dir = "pokemon_dataset"
test_dir = "test_images"
output_dir = "predicted_result"
model_path = "pokemon_cnn_model_manual.keras"
images_per_class = 3
img_size = (180, 180)

# 1. Setup logger
def setup_logger(log_file="test_log.txt"):
    """
    Sets up a logger with two handlers: a file handler and a console handler.

    Args:
        log_file (str, optional): The file to write logs to. Defaults to "test_log.txt".

    Returns:
        logging.Logger: The logger object.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Formatter
    # The formatter is used to format the log messages.
    # It takes the log message as input and returns a string.
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # File handler
    # The file handler is used to write logs to a file.
    # It takes the log file path as an argument.
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    # The console handler is used to write logs to the console.
    # It takes the output stream as an argument.
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger

# 2. Load class names
def load_class_names(data_dir):
    """
    Loads the class names from a directory.

    Args:
        data_dir (str): The path to the dataset directory.

    Returns:
        list: A list of class names.
    """
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    logging.info(f"Found classes: {class_names}")
    return class_names

# 3. Download test images for each class
def download_test_images(class_names, test_dir, images_per_class):
    """
    Downloads a specified number of test images for each class from Google.

    Args:
        class_names (list): A list of class names to download images for.
        test_dir (str): The path to the directory where test images will be stored.
        images_per_class (int): The number of images to download for each class.

    Returns:
        None
    """
    # Create the test directory if it does not exist
    os.makedirs(test_dir, exist_ok=True)

    # Iterate through each class name
    for cls in class_names:
        # Create a directory for the current class if it does not exist
        cls_dir = os.path.join(test_dir, cls)
        os.makedirs(cls_dir, exist_ok=True)

        # Define the search term for the current class
        search_term = f"{cls} pokemon"

        # Initialize the GoogleImageCrawler with the storage directory
        crawler = GoogleImageCrawler(storage={"root_dir": cls_dir})

        # Crawl the web for images according to the search term and max number
        crawler.crawl(keyword=search_term, max_num=images_per_class)

        # Log the download completion for the current class
        logging.info(f"Downloaded {images_per_class} images for class: {cls}")

# 4. Load model
def load_model(model_path):
    """
    Loads a TensorFlow model from a file path.

    Args:
        model_path (str): The path to the model file.

    Returns:
        tensorflow.keras.models.Model: The loaded model.
    """
    # Load the model from the file path
    model = tf.keras.models.load_model(model_path)

    # Log a message to indicate that the model has been loaded
    logging.info("Model loaded.")

    # Return the loaded model
    return model

# 5. Predict and save
def predict_and_save(model, class_names, test_dir, output_dir, img_size):
    """
    Predicts the class labels for images in each class directory and saves the
    annotated images to the output directory.

    Args:
        model (tensorflow.keras.models.Model): The model to use for prediction.
        class_names (list[str]): The list of class names.
        test_dir (str): The directory containing the test images.
        output_dir (str): The directory to save the annotated images to.
        img_size (tuple): The size to resize the images to.

    Returns:
        pandas.DataFrame: A DataFrame containing the true class, predicted class,
            confidence, and filename for each image.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    # Iterate through each class directory
    for cls in class_names:
        cls_input_dir = os.path.join(test_dir, cls)
        cls_output_dir = os.path.join(output_dir, cls)
        os.makedirs(cls_output_dir, exist_ok=True)

        # Iterate through each image in the class directory
        for img_path in Path(cls_input_dir).glob("*"):
            img = cv2.imread(str(img_path))
            if img is None:
                logging.warning(f"Skipped unreadable image: {img_path}")
                continue

            # Resize the image to the specified size
            img_resized = cv2.resize(img, img_size)

            # Normalize the image
            img_norm = img_resized.astype(np.float32) / 255.0

            # Get the predicted class index and confidence
            logits = model.predict(np.expand_dims(img_norm, axis=0), verbose=0)[0]
            pred_idx = int(np.argmax(logits))
            pred_class = class_names[pred_idx]
            confidence = float(logits[pred_idx])

            # Annotate the image with the predicted class and confidence
            label = f"{pred_class} ({confidence * 100:.1f}%)"
            cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save the annotated image
            save_path = os.path.join(cls_output_dir, img_path.name)
            cv2.imwrite(save_path, img)

            # Append the result to the list
            results.append({
                "true_class": cls,
                "pred_class": pred_class,
                "confidence": confidence,
                "filename": str(img_path)
            })

    # Convert the results to a DataFrame and save it to a CSV file
    df = pd.DataFrame(results)
    df.to_csv("inference_results.csv", index=False)
    logging.info("Inference complete. Results saved to 'inference_results.csv'")
    return df

# 6. Summarize
def summarize_and_visualize(df):
    """
    Summarizes and visualizes the classification accuracy.

    Args:
        df (DataFrame): DataFrame containing columns 'pred_class' and 'true_class'.

    Returns:
        None
    """
    # Calculate and log overall accuracy
    overall_acc = (df['pred_class'] == df['true_class']).mean()
    logging.info(f"Overall Accuracy: {overall_acc:.2%}")

    # Group by true class and calculate total, correct counts, and accuracy
    summary = df.groupby("true_class").apply(
        lambda g: pd.Series({
            "total": len(g),
            "correct": (g['pred_class'] == g['true_class']).sum(),
            "accuracy": (g['pred_class'] == g['true_class']).mean()
        })
    ).reset_index()

    # Log per-class summary
    logging.info("Per-class Summary:")
    logging.info(summary)

    # Sort by class name for consistent display
    summary = summary.sort_values("true_class")

    # Create a bar chart for per-class accuracy
    plt.figure(figsize=(20, 6))  # Set figure size to fit all labels
    plt.bar(summary["true_class"], summary["accuracy"], color="skyblue")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.title("Per-Class Accuracy (All 150 Classes)")

    # Improve x-axis label readability
    plt.xticks(rotation=75, fontsize=6, ha='right')
    plt.tight_layout()

    # Save the bar chart as an image file
    plt.savefig("class_accuracy_full.png", dpi=300)
    plt.show()

    # Save summary as CSV
    summary.to_csv("class_accuracy_summary.csv", index=False)
    logging.info("Saved full per-class accuracy chart and CSV summary.")

# == START ==
if __name__ == "__main__":
    # Set up logging
    logger = setup_logger()

    # Load class names
    class_names = load_class_names(data_dir)

    # Download test images
    download_test_images(class_names, test_dir, images_per_class)

    # Load model
    model = load_model(model_path)

    # Predict and save
    df = predict_and_save(model, class_names, test_dir, output_dir, img_size)

    # Summarize
    summarize_and_visualize(df)

# == END ==