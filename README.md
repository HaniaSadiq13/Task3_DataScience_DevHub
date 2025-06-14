# Task 3: Customer Churn Prediction (Bank Customers)

## 🎯 Objective
Predict whether a bank customer is likely to **churn** (i.e., leave the bank) using supervised classification techniques.

## 📁 Dataset
- **Name**: Churn Modelling Dataset  
- **Source**: Available on Kaggle  
- **Target Variable**: `Exited` (1 = customer left the bank, 0 = customer stayed)

## 🛠️ Libraries Used
- `pandas` and `numpy` – for data manipulation  
- `matplotlib.pyplot` and `seaborn` – for visualization  
- `scikit-learn` – for preprocessing, model training, and evaluation  

## 🔍 Steps Performed

### 1. Load and Inspect Dataset
- Read the dataset using `pd.read_csv()`
- Displayed shape, columns, and initial rows to understand the structure

### 2. Data Cleaning & Preprocessing
- Removed irrelevant columns: `RowNumber`, `CustomerId`, and `Surname`
- Encoded:
  - `Gender` using **Label Encoding**
  - `Geography` using **One-Hot Encoding** (drop_first=True)
- Scaled numerical features using `StandardScaler`

### 3. Feature and Target Selection
- **Features (X)**: All variables except `Exited`  
- **Target (y)**: `Exited`

### 4. Data Splitting
- Split the data into training and test sets using `train_test_split`  
- 80% training, 20% testing

### 5. Model Training
- Trained a **Random Forest Classifier** (`RandomForestClassifier` from sklearn)

### 6. Model Evaluation
- Used `accuracy_score` to measure performance
- Created a **confusion matrix** and a **classification report** to assess precision, recall, and F1-score

### 7. Feature Importance Analysis
- Extracted feature importance from the Random Forest model
- Visualized top contributing features using a horizontal bar chart

## 📊 Results

| Metric       | Value   |
|--------------|---------|
| Accuracy     | ~85%    |
| Top Features | CreditScore, Age, Balance, Geography_Germany, NumOfProducts |

## 📌 Key Skills Demonstrated
- ✅ Label and One-Hot Encoding  
- ✅ Binary classification modeling  
- ✅ Feature scaling with StandardScaler  
- ✅ Model evaluation using accuracy and confusion matrix  
- ✅ Feature importance interpretation

## 🧠 Possible Improvements
- Experiment with other models (e.g., XGBoost, Logistic Regression)
- Perform cross-validation
- Hyperparameter tuning for better accuracy
- Handle class imbalance if present (e.g., SMOTE, class weights)
