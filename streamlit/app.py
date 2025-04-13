import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import numpy as np

# Charger le modèle
model_data = joblib.load("REGLOG.pkl")
model = model_data['model']
threshold = model_data['threshold']

# Page config
st.set_page_config(page_title="Prédicteur de Solvabilité", layout="centered")

st.title("🔍 Prédiction de Solvabilité Bancaire")
st.markdown("Entrez les caractéristiques d'un client pour prédire s'il est **solvable** ou **non solvable**.")

# Fonction de prédiction
def predict(data):
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    proba = model.predict_proba(scaled)[0][1]
    prediction = int(proba >= threshold)
    return prediction, proba

# Entrée manuelle
with st.form("client_form"):
    age = st.slider("Âge", 18, 100, 35)
    status = st.selectbox("Statut marital", options=[1, 2, 3], format_func=lambda x: {1: "Célibataire", 2: "Marié(e)", 3: "Autre"}[x])
    expenses = st.number_input("Dépenses mensuelles", min_value=0.0, value=100.0)
    income = st.number_input("Revenu mensuel", min_value=0.0, value=200.0)
    amount = st.number_input("Montant de crédit", min_value=0.0, value=800.0)
    price = st.number_input("Prix de l'achat", min_value=0.0, value=1000.0)
    
    submitted = st.form_submit_button("Prédire")

    if submitted:
        input_data = {
            'Age': age,
            'Status': status,
            'Expenses': expenses,
            'Income': income,
            'Amount': amount,
            'Price': price
        }
        prediction, proba = predict(input_data)
        st.success(f"Résultat : {'Non Solvable 🔴' if prediction == 1 else 'Solvable ✅'}")
        st.info(f"Probabilité de non solvabilité : {proba:.2%}")

# Chargement d'un fichier CSV
st.markdown("---")
st.header("📄 Prédictions en Lot")
file = st.file_uploader("Téléversez un fichier CSV", type="csv")

if file:
    data_csv = pd.read_csv(file)
    scaled_data = scaler.transform(data_csv)
    probas = model.predict_proba(scaled_data)[:, 1]
    predictions = (probas >= threshold).astype(int)
    
    data_csv["Prédiction"] = predictions
    data_csv["Probabilité_Non_Solvabilité"] = probas
    
    st.write("🔎 Résultats :")
    st.dataframe(data_csv)
    
    csv_output = data_csv.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Télécharger les résultats", data=csv_output, file_name="predictions.csv", mime="text/csv")

# Initialisation du scaler (le même que pendant l'entraînement)
scaler = StandardScaler()
# IMPORTANT : Redéfinir l’ordre des colonnes utilisé à l'entraînement
columns = ['Age', 'Status', 'Expenses', 'Income', 'Amount', 'Price']
scaler.fit(pd.DataFrame(columns=columns, data=np.zeros((1, len(columns)))))  # dummy fit to enable transform

