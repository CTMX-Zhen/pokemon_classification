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
# pip install pandas matplotlib seaborn scikit-learn
# pip install tensorflow
# pip install tqdm

# == IMPORT LIBRARIES ==
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import cv2
import sys
import logging
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split

# == IMPORT TENSORFLOW LIBRARIES ==
tensorflow_callback = tf.keras.callbacks.Callback
tensorflow_dataset = tf.data.Dataset
tensorflow_models = tf.keras.models
tensorflow_layers = tf.keras.layers
tensorflow_SparseCategoricalCrossentropy = tf.keras.losses.SparseCategoricalCrossentropy

# == LOGGING CALLBACK ==
class LoggingCallback(tensorflow_callback):
    def on_epoch_end(self, epoch, logs=None):
        """
        This method is called at the end of each epoch during training.
        
        Args:
            epoch (int): The index of the epoch.
            logs (dict, optional): A dictionary containing the training and validation metrics.
        
        Logs:
            The training and validation loss and accuracy for the epoch.
        """
        # Initialize logs if not provided
        logs = logs or {}
        
        # Log the epoch number
        logging.info(f"Epoch {epoch + 1}: ")
        
        # Log the training loss and accuracy
        logging.info(f"loss={logs.get('loss'):.4f}, accuracy={logs.get('accuracy'):.4f}, ")
        
        # Log the validation loss and accuracy
        logging.info(f"val_loss={logs.get('val_loss'):.4f}, val_accuracy={logs.get('val_accuracy'):.4f}")

# == TQDM CALLBACK ==
class TqdmToLogger(io.StringIO):
    def __init__(self, logger, level=logging.INFO):
        """
        Initialize a TqdmToLogger object.
        
        Args:
            logger (logging.Logger): The logger object to write to.
            level (int, optional): The logging level. Defaults to logging.INFO.
        """
        super().__init__()
        self.logger = logger
        self.level = level
        self.buffer = ""

    def write(self, buf):
        """
        Write buffer content to the logger if newline is found.

        Args:
            buf (str): The string buffer to write to the logger.
        """
        # Append buffer content
        self.buffer += buf
        
        # Check for newline character in the buffer
        if "\n" in self.buffer:
            # Split buffer into lines and log each non-empty line
            for line in self.buffer.splitlines():
                if line.strip():  # Log only if line is not empty
                    self.logger.log(self.level, line.strip())
            
            # Clear the buffer
            self.buffer = ""

    def flush(self):
        """
        Flushes the buffer content to the logger if it is not empty.
        
        This method checks if the buffer contains any non-whitespace characters.
        If so, it logs the content and then clears the buffer.
        """
        # Check if buffer contains non-whitespace characters
        if self.buffer.strip():
            # Log the buffer content
            self.logger.log(self.level, self.buffer.strip())
        
        # Clear the buffer
        self.buffer = ""

# == HELPER FUNCTIONS ==
# 1. Setup logger
def setup_logger(log_file="train_log.txt"):
    """
    Sets up a logger with two handlers: a file handler and a console handler.
    
    Args:
        log_file (str, optional): The file to write logs to. Defaults to "train_log.txt".
    
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

# 2. Load dataset
def load_dataset(data_dir, img_size):
    """
    Loads a dataset from a directory.

    Args:
        data_dir (str): The path to the dataset directory.
        img_size (tuple): The size to resize the images to.

    Returns:
        tuple: A tuple containing the input data, labels, and class names.
    """
    X = []  # List of input images
    y = []  # List of labels
    class_names = sorted(os.listdir(data_dir))  # sorted to keep label order consistent
    class_indices = {name: idx for idx, name in enumerate(class_names)}  # map class name to index

    # Iterate over each class and its images
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)

            # Read and resize image
            img = cv2.imread(img_path)
            if img is None:
                continue  # skip corrupted files

            # Resize image
            img = cv2.resize(img, img_size)

            # Add image and label to the lists
            X.append(img)
            y.append(class_indices[class_name])

    # Convert to numpy arrays
    X = np.array(X, dtype=np.float32) / 255.0  # Normalize to [0, 1]
    y = np.array(y)

    logging.info(f"Loaded dataset with shape: {X.shape}, Labels shape: {y.shape}")
    logging.info(f"Classes found: {class_names}")

    return X, y, class_names

# 3. Prepare TensorFlow datasets
def prepare_datasets(X, y, batch_size=32, val_split=0.2):
    """
    Prepares the dataset for training.

    Args:
        X (numpy.array): The input data.
        y (numpy.array): The labels.
        batch_size (int, optional): The batch size. Defaults to 32.
        val_split (float, optional): The validation split. Defaults to 0.2.

    Returns:
        tuple: A tuple containing the training and validation datasets.
    """
    # Split data into train and validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_split, random_state=123, stratify=y)

    # Create TensorFlow datasets
    train_ds = tensorflow_dataset.from_tensor_slices((X_train, y_train))
    val_ds = tensorflow_dataset.from_tensor_slices((X_val, y_val))

    # Preprocess data
    AUTOTUNE = tf.data.AUTOTUNE
    # Shuffle the training dataset
    train_ds = train_ds.shuffle(1000)
    # Batch the datasets
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)
    # Prefetch the datasets
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # Log the number of batches
    logging.info(f"Training dataset batches: {len(list(train_ds))}")
    logging.info(f"Validation dataset batches: {len(list(val_ds))}")

    return train_ds, val_ds

# 4. Build and compile model
def build_model(input_shape, num_classes):
    """
    Builds a convolutional neural network model.

    Args:
        input_shape (tuple): The shape of the input data.
        num_classes (int): The number of classes in the dataset.

    Returns:
        tensorflow.keras.models.Sequential: A compiled convolutional neural network model.
    """
    # Create a convolutional neural network
    model = tensorflow_models.Sequential([
        # Convolutional layer with 16 filters, kernel size 3, padding same, and ReLU activation
        tensorflow_layers.Conv2D(16, 3, padding='same', activation='relu', input_shape=input_shape),
        # Max pooling layer with pool size 2
        tensorflow_layers.MaxPooling2D(),
        # Convolutional layer with 32 filters, kernel size 3, padding same, and ReLU activation
        tensorflow_layers.Conv2D(32, 3, padding='same', activation='relu'),
        # Max pooling layer with pool size 2
        tensorflow_layers.MaxPooling2D(),
        # Convolutional layer with 64 filters, kernel size 3, padding same, and ReLU activation
        tensorflow_layers.Conv2D(64, 3, padding='same', activation='relu'),
        # Max pooling layer with pool size 2
        tensorflow_layers.MaxPooling2D(),
        # Flatten the output of the convolutional layers
        tensorflow_layers.Flatten(),
        # Dense layer with 128 neurons and ReLU activation
        tensorflow_layers.Dense(128, activation='relu'),
        # Dense layer with num_classes neurons and softmax activation
        tensorflow_layers.Dense(num_classes, activation='softmax')
    ])
    # Compile the model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 5. Train model
def plot_history(history):
    """
    Plots training and validation metrics from a history object.

    Args:
        history (dict): A dictionary containing the training and validation metrics.
    """
    # Get metrics from history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    # Plot metrics
    plt.figure(figsize=(8, 6))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')

    # Save and show plot
    plt.savefig("training_result.png")
    plt.show()

# == START ==
if __name__ == "__main__":
    # Set up logging
    logger = setup_logger()

    # Hyperparameters
    data_dir = "pokemon_dataset"
    img_size = (180, 180)
    batch_size = 32
    epochs = 10

    # Load dataset manually
    logging.info("Starting dataset loading...")
    X, y, class_names = load_dataset(data_dir, img_size)

    # Prepare TensorFlow datasets
    logging.info("Preparing datasets...")
    train_ds, val_ds = prepare_datasets(X, y, batch_size)

    # Build and compile model
    logging.info("Building model...")
    model = build_model(
        input_shape=(img_size[0], img_size[1], 3), 
        num_classes=len(class_names)
    )
    model.compile(
        optimizer='adam',
        loss=tensorflow_SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    # Print model summary
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    summary_str_ascii = summary_str.encode('ascii', errors='ignore').decode('ascii')
    logging.info("\n" + summary_str_ascii)

    # Train model
    logging.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[
            LoggingCallback(),
            TqdmCallback(file=TqdmToLogger(logger))
        ],
    )

    # Save model
    logging.info("Training completed. Saving model...")
    model.save("pokemon_cnn_model_manual.keras")

    # Plot training history
    logging.info("Plotting training history...")
    plot_history(history)

    logging.info("All done!")

# == END ==