# Titanic Survival Prediction Using Machine Learning

## **Problem Statement**

The goal of this project is to predict whether a passenger survived or perished in the Titanic disaster based on features such as age, gender, ticket class, and family size. This is a binary classification problem where the target variable is `Survived` (1 if the passenger survived, 0 otherwise).

The project uses a machine learning ensemble approach to improve prediction accuracy by combining the strengths of multiple models.

---

## **Methodology**

### **1. Data Preprocessing and Feature Engineering**

#### **Feature Engineering**
Several new features are engineered to enhance the model's predictive power:
- **Title Extraction**: Titles are extracted from the `Name` field (e.g., `Mr`, `Mrs`, `Miss`). Rare titles are grouped into an "Other" category.
- **Family Features**:
  - `FamilySize`: Combines `SibSp` (siblings/spouses aboard) and `Parch` (parents/children aboard) with the passenger to indicate family size.
  - `IsAlone`: A binary feature indicating whether the passenger was traveling alone.
- **Fare Binning**: `Fare` is divided into quartiles (e.g., Low, Mid, Mid-High, High).
- **Age Binning**: `Age` is categorized into bins (e.g., Child, Young, Adult, Middle, Senior).
- **Cabin Presence**: `HasCabin` is a binary feature indicating whether cabin information is available.

#### **Handling Missing Values**
- **Age**: Imputed using the median.
- **Fare**: Missing values are replaced with the median.
- **Embarked**: Missing values are filled with the most frequent value (mode).

#### **Categorical Encoding**
Categorical variables (`Sex`, `Embarked`, `Title`, `AgeBin`) are label-encoded to convert them into numerical formats suitable for machine learning.

#### **Scaling Numerical Features**
Continuous features (`Age`, `FamilySize`, `Fare`) are standardized using `StandardScaler` for consistency across feature magnitudes.

#### **Selected Features**
Key features used for modeling:
- Original: `Pclass`, `Fare`, `SibSp`, `Parch`
- Engineered: `FamilySize`, `IsAlone`, `Title`, `AgeBin`, `HasCabin`

---

### **2. Ensemble Model Creation**

An ensemble approach combines multiple models to improve predictive accuracy. The following classifiers are included in the ensemble:

- **Random Forest Classifier**:
  - A tree-based model that builds multiple decision trees and outputs their average prediction.
  - Hyperparameters like `n_estimators` (number of trees) and `max_depth` are tuned to prevent overfitting.

- **XGBoost Classifier**:
  - Gradient-boosted trees, optimized for handling imbalanced data and capturing complex patterns.
  - Parameters such as `learning_rate` (step size) and `n_estimators` control learning.

- **Logistic Regression**:
  - A linear model for binary classification that is simple and interpretable.

The ensemble is implemented using a **VotingClassifier**:
- **Voting Strategy**: Soft voting, where the final prediction is based on the average of predicted probabilities from each model.

---

### **3. Training and Prediction**

The process consists of the following steps:

1. **Training**:
   - The training dataset is preprocessed (imputation, feature scaling, and encoding).
   - The ensemble model is trained using the preprocessed data.
   - Model performance is evaluated using 5-fold cross-validation to ensure robustness and prevent overfitting.

2. **Prediction**:
   - The test dataset undergoes similar preprocessing steps as the training data.
   - The trained ensemble model predicts the survival of passengers in the test set.

---

### **4. Output**

The predictions are saved in a CSV file (`ensemble_submission.csv`) containing:
- `PassengerId`: The unique ID of each passenger.
- `Survived`: Predicted survival status (1 for survived, 0 for perished).

---

### **Advantages of the Approach**
1. **Feature Engineering**: Incorporates domain knowledge to improve model interpretability.
2. **Robust Preprocessing**: Handles missing values, encodes categories, and scales features effectively.
3. **Ensemble Learning**: Combines multiple classifiers for improved accuracy and robustness.
4. **Cross-Validation**: Evaluates model performance across multiple folds to prevent overfitting.

This approach balances predictive power with model interpretability, providing reliable predictions for Titanic survival.

