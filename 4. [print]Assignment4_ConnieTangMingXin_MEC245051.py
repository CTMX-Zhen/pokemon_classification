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

import sys
import logging
import pandas as pd

# 1. Setup logger
def setup_logger(log_file="print_log.txt"):
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

# == START ==
if __name__ == "__main__":

    # Set up logging
    logger = setup_logger()

    # Load the CSV data
    df = pd.read_csv('class_accuracy_summary.csv')

    # Filter the data to include only rows where correct is 0, 1, 2, or 3
    filtered_df = df[df['correct'].isin([0, 1, 2, 3])]

    # Group the data by correct and calculate the total number of classes for each
    total_number_of_each_correct = filtered_df.groupby('correct')['true_class'].count().reset_index()

    # logger.info the result
    logger.info(f"\n{total_number_of_each_correct}\n")

    # Group the data by correct and get the list of true class names
    name_list_of_each_correct = filtered_df.groupby('correct')['true_class'].apply(lambda x: ', '.join(x)).reset_index()

    # Display the full contents of the true_class column
    pd.set_option('display.max_colwidth', None)

    # logger.info the result
    logger.info(f"\n{name_list_of_each_correct}\n")

# == END ==