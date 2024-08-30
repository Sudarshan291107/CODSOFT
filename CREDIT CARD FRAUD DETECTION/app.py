from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

app = Flask(__name__)

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6,6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(cm, annot=True, fmt='.2f' if normalize else 'd', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train_model():
    df = pd.read_csv(r'C:\Users\HP\Downloads\creditcard.csv')

    df_sample = df.sample(n=10000, random_state=42)

    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    report = classification_report(y_test, y_pred, output_dict=True)
    
    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, classes=['Genuine', 'Fraudulent'], title='Confusion Matrix')
    
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    
    feature_importances = model.feature_importances_
    features = X.columns
    importance_dict = dict(zip(features, feature_importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)

    return render_template('results.html', precision=precision, recall=recall, f1=f1, report=report, plot_url=plot_url, feature_importances=sorted_importance)

if __name__ == '__main__':
    app.run(debug=True)
