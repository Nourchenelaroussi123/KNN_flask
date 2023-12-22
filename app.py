# Import des bibliothèques
from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

# Initialisation de l'application Flask
app = Flask(__name__, template_folder='template')

# Chargement du modèle KNN
model_path = "model/knn_model.joblib"
knn = joblib.load(model_path)

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template('index.html')

# Route pour la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Récupérer les données du formulaire
        features = [
            float(request.form['High']),
            float(request.form['Low']),
            float(request.form['Open_Price']),
            float(request.form['Adj_Close']),
            float(request.form['Volume'])
        ]

        # Créer un DataFrame avec les caractéristiques
        input_data = pd.DataFrame([features], columns=['High', 'Low', 'Open_Price', 'Adj_Close', 'Volume'])

        # Prédire avec le modèle KNN
        prediction = knn.predict(input_data)

        # Rendre le résultat à la page HTML
        return render_template('index.html', prediction=prediction[0])

    except Exception as e:
        # Gérer les erreurs et les afficher dans la page HTML
        error_message = f"Error: {str(e)}"
        return render_template('index.html', error=error_message)

# Lancer l'application Flask
if __name__ == '__main__':
    app.run(debug=True)
