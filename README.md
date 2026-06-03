# Credit Card Fraud Detection

This project focuses on building a machine learning model to detect credit card fraud using a transactional dataset. It involves extensive data exploration, feature engineering, model training with various algorithms, and comprehensive evaluation, including geospatial visualization of fraud rates.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Key Steps and Analysis](#key-steps-and-analysis)
  - [Data Loading and Initial Exploration](#data-loading-and-initial-exploration)
  - [Outlier Handling](#outlier-handling)
  - [Feature Engineering](#feature-engineering)
  - [Model Preparation and Training](#model-preparation-and-training)
  - [Model Evaluation and Visualization](#model-evaluation-and-visualization)
  - [Exploratory Data Visualizations](#exploratory-data-visualizations)
- [Results](#results)
- [Technologies Used](#technologies-used)

## Project Overview
Credit card fraud is a significant concern for financial institutions and consumers alike. This project aims to develop a predictive model that can identify fraudulent transactions, thereby minimizing financial losses and enhancing security. We leverage a dataset containing various transaction details, customer information, and geographical data to build a robust detection system.

## Dataset
The dataset used is `credit_card_fraud.csv`, which includes the following key columns:
- `trans_date_trans_time`: Transaction date and time
- `merchant`: Merchant name
- `category`: Transaction category
- `amt`: Transaction amount
- `city`, `state`, `lat`, `long`: Transaction location details
- `city_pop`: City population
- `job`: Customer's job
- `dob`: Customer's date of birth
- `trans_num`: Unique transaction number
- `merch_lat`, `merch_long`: Merchant location details
- `is_fraud`: Target variable (1 for fraud, 0 for legitimate)

## Key Steps and Analysis

### Data Loading and Initial Exploration
- Loaded the `credit_card_fraud.csv` dataset into a pandas DataFrame.
- Performed initial data checks including `head()`, `info()`, `shape`, `describe()`, and `isnull().sum()` to understand the data structure, types, and identify missing values.

### Outlier Handling
- Identified outliers in the `amt` (transaction amount) column using the Interquartile Range (IQR) method.
- Visualized the distribution of `amt` before and after outlier detection using box plots.

### Feature Engineering
- Converted `trans_date_trans_time` and `dob` columns to datetime objects.
- Extracted new time-based features: `hour`, `dow` (day of week), and `month` from `trans_date_trans_time`.
- Calculated `age` from `trans_date_trans_time` and `dob`.
- Implemented a `haversine` function to compute the straight-line distance (`dist_miles`) between transaction and merchant locations based on their latitudes and longitudes.

### Model Preparation and Training
- Separated features (`X`) and the target variable (`y = is_fraud`).
- One-hot encoded categorical features (e.g., `category`) using `pd.get_dummies`.
- Split the data into training and testing sets using `train_test_split` with `stratify=y` to maintain class balance and `random_state=42` for reproducibility.
- Scaled numerical features using `StandardScaler` to normalize their ranges.
- Applied SMOTE (Synthetic Minority Over-sampling Technique) to the training data to address class imbalance, oversampling the minority class (`is_fraud`) with a sampling strategy of 0.2.
- Trained the following models:
  - **Linear Regression:** Evaluated its intercept, coefficients, and R2 score.
  - **Logistic Regression:** Trained a baseline model and a primary model, addressing `ConvergenceWarning` by increasing `max_iter`.
  - **Random Forest Classifier:** Configured with `class_weight='balanced'`, `n_estimators=200`, `max_depth=15`, and `min_samples_leaf=2`.

### Model Evaluation and Visualization
- Generated classification reports for Logistic Regression and Random Forest models.
- Calculated and reported ROC-AUC, Recall, Precision, and F1-Score for the Random Forest model, emphasizing Recall for fraud detection by setting a prediction threshold of 0.30.
- Plotted a Confusion Matrix for the Random Forest model.
- Generated a plot of Top 10 Feature Importances for the Random Forest model.
- Plotted an ROC curve for the Logistic Regression model, yielding an AUC of 0.65.

### Exploratory Data Visualizations
- **Correlation Analysis:** Created a heatmap to visualize correlations between numerical features.
- **Fraud by Category:** Bar plot showing 'grocery_pos' and 'shopping_net' as categories with the highest fraud incidence.
- **Fraud by City:** Bar plot identifying top cities with the most fraudulent transactions (e.g., 'Albuquerque', 'Aurora', 'Fort Washakie', 'Glendale', 'Mesa').
- **Fraud by Age Group:** Created `age_group` bins and visualized fraudulent transactions by age, indicating higher fraud in '46-55' and '56-65' age groups.
- **Fraud by Merchant:** Bar plot displaying the top 10 merchants associated with the most fraudulent transactions.
- **Geospatial Plot of Fraud Rates by State:** Calculated fraud rates per state and visualized them using a choropleth map. This involved downloading a US states GeoJSON file, mapping full state names to abbreviations, and merging the fraud data with the geospatial data to highlight states with higher fraud rates like Alaska, Oregon, and Nebraska.

## Results

The Random Forest model achieved strong performance in fraud detection, particularly in recall, which was prioritized due to the nature of fraud detection (minimizing false negatives). The model's ROC-AUC was **0.9936**, and recall for fraud was **0.9270** at a threshold of 0.30.

Key insights from EDA include:
- Certain transaction categories (e.g., grocery_pos, shopping_net) and specific cities exhibit higher fraud occurrences.
- Middle-aged to older customer segments (46-55, 56-65) appear to be more frequently targeted by fraud.
- Geospatial analysis revealed specific states with elevated fraud rates, suggesting potential regional vulnerabilities.

## Technologies Used
- Python 3.x
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imblearn (for SMOTE)
- GeoPandas (for geospatial analysis)
- `wget` (for downloading GeoJSON data)

