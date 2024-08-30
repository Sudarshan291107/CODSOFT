import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request
import numpy as np

# Load the dataset
file_path = r"C:\Users\HP\Downloads\advertising.csv"
df = pd.read_csv(file_path)

# Data Visualization
sns.pairplot(df)
plt.show()

# Interactive plot using Plotly
fig = px.scatter_matrix(df, dimensions=df.columns[:-1], color='Sales')
fig.show()

# Modeling for Sales Prediction
X = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Flask Web Application
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    tv = float(request.form['TV'])
    radio = float(request.form['Radio'])
    newspaper = float(request.form['Newspaper'])
    features = np.array([[tv, radio, newspaper]])
    prediction = model.predict(features)[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
