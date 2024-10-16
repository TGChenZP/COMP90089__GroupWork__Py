# ICU Mortality Prediction Project

## Overview
This project aims to predict mortality in Intensive Care Unit (ICU) patients using machine learning techniques. We utilize data from the MIMIC-IV dataset and employ various feature selection and modeling approaches to develop accurate prediction models.

## Project Structure
- `data/`: Contains raw and processed datasets
  - `raw/`: Original MIMIC-IV data
  - `curated/`: Cleaned and preprocessed data
- `notebooks/`: Jupyter notebooks for data analysis and model development
  - `data_preprocessing_data_select.ipynb`: Initial data selection and preprocessing
  - `data_preprocessing_dropna.ipynb`: Handling missing values
  - `EDA_feature_selection.ipynb`: Exploratory Data Analysis and feature selection
  - `feature_selection_iter2.ipynb`: Second iteration of feature selection
  - `data_preprocessing_for_modelling.ipynb`: Final data preparation for modeling
  - `tuning/`: Notebooks for hyperparameter tuning of various models
  - `FinalResults.ipynb`: Evaluation of final models and results presentation
- `plots/`: Generated visualizations and plots
- `models/`: Saved model files and feature importance data

## Key Steps

1. **Data Preprocessing**: 
   - Initial data selection from MIMIC-IV
   - Handling missing values
   - Feature engineering
   Relevant notebooks:
   - `data_preprocessing_data_select.ipynb`
   - `data_preprocessing_dropna.ipynb`
   - `data_preprocessing_for_modelling.ipynb`


2. **Feature Selection**:
   - Visualizing data distributions
   - Analyzing correlations between features
   - Mann-Whitney U Test for continuous variables
   - Chi-Square Test for categorical variables
   - Spearman Correlation analysis for identifying highly correlated features
   Relevant notebooks:
   - `EDA_feature_selection.ipynb`
   - `feature_selection_iter2.ipynb`

3. **Model Development**:
   - Training various models including XGBoost, Logistic Regression, SVM, Neural Networks
   - Hyperparameter tuning using cross-validation
   Relevant notebooks:
   - Notebooks in the `tuning/` directory for different models

4. **Model Evaluation**:
   - Performance metrics calculation (e.g., F1 score, AUC-ROC)
   - Feature importance analysis
   - SHAP (SHapley Additive exPlanations) analysis for model interpretability
   Relevant notebook:
   - `FinalResults.ipynb`

## Results
The final results, including model performance metrics and feature importance, are presented in the `FinalResults.ipynb` notebook.

## Usage
1. Clone the repository
2. Install required packages: `pip install -r requirements.txt`
3. Run the notebooks in the order mentioned in the "Key Steps" section

## Contributors
Jiayuan Li
Lang Chen
Xinyi Wang
Zixuan Guo

## License
[Specify the license under which this project is released]