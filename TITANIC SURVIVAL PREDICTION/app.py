from flask import Flask, render_template, request
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess the dataset, then train the model
def train_model():
    # Load dataset
    data = pd.read_csv("C:\\Users\\HP\\Downloads\\Titanic-Dataset.csv")
    
    # Preprocess data
    data = data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1)
    data['Age'].fillna(data['Age'].median(), inplace=True)
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'].fillna(data['Fare'].median(), inplace=True)
    
    X = data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    y = data['Survived']
    
    # Define preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['Sex', 'Embarked']),
            ('num', StandardScaler(), ['Age', 'Fare'])
        ],
        remainder='passthrough'
    )
    
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    # Split data and train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    
    return model

# Train the model
model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        Pclass = int(request.form['Pclass'])
        Sex = request.form['Sex']
        Age = float(request.form['Age'])
        SibSp = int(request.form['SibSp'])
        Parch = int(request.form['Parch'])
        Fare = float(request.form['Fare'])
        Embarked = request.form['Embarked']
        
        # Create input DataFrame for the model
        input_data = pd.DataFrame([{
            'Pclass': Pclass,
            'Sex': Sex,
            'Age': Age,
            'SibSp': SibSp,
            'Parch': Parch,
            'Fare': Fare,
            'Embarked': Embarked
        }])
        
        # Predict survival
        prediction = model.predict(input_data)
        survival = 'Survived' if prediction[0] == 1 else 'Did not survive'
        
        return render_template('result.html', survival=survival)

if __name__ == '__main__':
    app.run(debug=True)
