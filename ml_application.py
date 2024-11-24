import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

class TitanicEnsemble:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.imputer = SimpleImputer(strategy='median')
        
    def extract_title(self, name):
        """Extract title from name."""
        title = name.split(',')[1].split('.')[0].strip()
        if title in ['Mr', 'Mrs', 'Miss', 'Master']:
            return title
        return 'Other'
    
    def preprocess_data(self, df, is_training=True):
        """Preprocess the data with feature engineering."""
        # Create a copy to avoid modifying original data
        data = df.copy()
        
        # Feature Engineering
        # Extract titles from names
        data['Title'] = data['Name'].apply(self.extract_title)
        
        # Create family size feature
        data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
        
        # Create is_alone feature
        data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
        
        # Create fare bins
        data['FareBin'] = pd.qcut(data['Fare'], 4, labels=['Low', 'Mid', 'Mid-High', 'High'])
        
        # Create age bins
        data['Age'] = self.imputer.fit_transform(data[['Age']])
        data['AgeBin'] = pd.cut(data['Age'], 5, labels=['Child', 'Young', 'Adult', 'Middle', 'Senior'])
        
        # Handle cabin information
        data['HasCabin'] = data['Cabin'].notna().astype(int)
        
        # Encode categorical variables
        categorical_features = ['Sex', 'Embarked', 'Title', 'FareBin', 'AgeBin']
        for feature in categorical_features:
            if is_training:
                data[feature] = self.label_encoder.fit_transform(data[feature].fillna('Missing'))
            else:
                data[feature] = self.label_encoder.transform(data[feature].fillna('Missing'))
        
        # Select features for modeling
        features = ['Pclass', 'Sex', 'Age', 'FamilySize', 'Fare', 'IsAlone', 
                   'HasCabin', 'Title', 'Embarked', 'AgeBin', 'FareBin']
        
        X = data[features]
        
        # Scale numerical features
        numerical_features = ['Age', 'FamilySize', 'Fare']
        if is_training:
            X[numerical_features] = self.scaler.fit_transform(X[numerical_features])
        else:
            X[numerical_features] = self.scaler.transform(X[numerical_features])
        
        if is_training:
            y = data['Survived']
            return X, y
        return X
    
    def create_ensemble(self):
        """Create the ensemble model."""
        # Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42
        )
        
        # XGBoost
        xgb_model = xgb.XGBClassifier(
            max_depth=3,
            learning_rate=0.1,
            n_estimators=100,
            objective='binary:logistic',
            random_state=42
        )
        
        # Logistic Regression
        lr = LogisticRegression(
            C=0.1,
            random_state=42,
            max_iter=1000
        )
        
        # Create ensemble
        ensemble = VotingClassifier(
            estimators=[
                ('rf', rf),
                ('xgb', xgb_model),
                ('lr', lr)
            ],
            voting='soft'
        )
        
        return ensemble
    
    def fit_predict(self, train_data, test_data):
        """Fit the model and make predictions."""
        # Preprocess training data
        X_train, y_train = self.preprocess_data(train_data, is_training=True)
        
        # Create and train ensemble
        ensemble = self.create_ensemble()
        ensemble.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Average CV score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Preprocess test data
        X_test = self.preprocess_data(test_data, is_training=False)
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        
        return predictions

# Example usage
if __name__ == "__main__":
    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    
    # Initialize and run ensemble
    model = TitanicEnsemble()
    predictions = model.fit_predict(train_data, test_data)
    
    # Create submission file
    submission = pd.DataFrame({
        'PassengerId': test_data['PassengerId'],
        'Survived': predictions
    })
    submission.to_csv('ensemble_submission.csv', index=False)