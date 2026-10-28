CS-433 project 1

# ML Project 1: Cardiovascular Disease Prediction

##### Team members:

* András Horkay
* Mathilde Aymon
* Mélusine Plumart

##### Project Description

This project focuses on predicting cardiovascular disease risk using the BRFSS (Behavioral Risk Factor Surveillance System) dataset. We implemented six machine learning algorithms (Gradient Descent, Stochastic Gradient Descent, Least Squares, Ridge Regression, Logistic Regression, and Regularized Logistic Regression) to perform binary classification.

##### Project Structure

* `run.py` : Main script that executes the data loading, preprocessing, model training, and generates the final submission file.
* `implementations.py`: Contains the implementations of the 6 mandatory machine learning algorithms, loss functions, gradient computations, and cross-validation logic.
* `preprocessing.py`: Contains the data cleaning and preprocessing pipeline (imputation, encoding, standardizing, and bias addition).
* `categories.py`: Defines lists and dictionaries for feature categorization and specific BRFSS encoding rules used by the preprocessing script.
* `helpers.py`: Utility functions for loading the CSV data and generating the submission files.
* `project1_analysis.ipynb`: A Jupyter Notebook containing our Exploratory Data Analysis (EDA), hyperparameter tuning experiments, and visualization of the results.

##### Requirements

The code relies only on standard Python libraries. No external machine learning frameworks are used.

* Python 3.x
* NumPy
* Matplotlib (to run the analysis notebook)

##### How to Run

1. To generate the final predictions, execute the `run.py` script from your terminal.
2. The script will output a file named `submission.csv` containing the final predictions for the test set.
3. To explore the data and visualize the results, open the `project1_analysis.ipynb`. It contains the EDA, hyperparameter tuning experiments, and visualizations of the model performance.
