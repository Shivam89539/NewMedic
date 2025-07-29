ML Project: Solubility Prediction with Linear Regression
Overview
This project focuses on predicting the solubility of molecules using a linear regression model. The dataset used contains molecular descriptors such as MolLogP, MolWt, NumRotatableBonds, and AromaticProportion, which are used to predict the logS (log of solubility) values.

Dataset
The dataset is sourced from the following URL:
Delaney Solubility Dataset

It contains 1144 rows and 5 columns:

MolLogP: Molecular LogP (partition coefficient)

MolWt: Molecular weight

NumRotatableBonds: Number of rotatable bonds

AromaticProportion: Proportion of aromatic atoms

logS: Target variable (log of solubility)

Project Structure
Data Loading: The dataset is loaded using pandas from the provided URL.

Data Preparation: The data is split into features (X) and target (y).

Data Splitting: The dataset is divided into training (80%) and testing (20%) sets using train_test_split from sklearn.

Model Building: A linear regression model is trained on the training data.

Model Evaluation: Predictions are made on both training and test sets to evaluate performance.

Key Steps
Feature Selection: All columns except logS are used as features.

Model Training: The linear regression model is trained using LinearRegression from sklearn.linear_model.

Predictions: The model is used to predict logS values for both training and test datasets.

Usage
Clone the repository:

bash
git clone <repository-url>
Open the Jupyter Notebook:

bash
jupyter notebook NewMedic.ipynb
Run the cells sequentially to load the data, train the model, and make predictions.

Dependencies
Python 3

pandas

scikit-learn

Install the required packages using:

bash
pip install pandas scikit-learn
Results
The model's performance can be evaluated using metrics such as Mean Squared Error (MSE) or R-squared score (not shown in the notebook but can be added for further analysis).

Future Work
Add performance metrics (e.g., MSE, R-squared).

Experiment with other regression models (e.g., Random Forest, Gradient Boosting).

Perform feature engineering to improve model accuracy.
