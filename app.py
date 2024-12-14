import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Charger le dataset
@st.cache_data
def load_data():
    return pd.read_csv('DatasetmalwareExtrait.csv')

# Fonction pour entraîner les modèles
def train_model(algorithm, X_train, Y_train, X_test, Y_test):
    if algorithm == 'Decision Tree':
        model = DecisionTreeClassifier(random_state=42)
    elif algorithm == 'K-Nearest Neighbors':
        model = KNeighborsClassifier()
    elif algorithm == 'Support Vector Machine':
        model = SVC(random_state=42)
    
    model.fit(X_train, Y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)
    report = classification_report(Y_test, y_pred, output_dict=True)
    return accuracy, report

# Charger les données
st.title("Détection de Malware avec ML")
st.sidebar.title("Options")
data = load_data()

# Aperçu des données
if st.sidebar.checkbox("Afficher le dataset"):
    st.write("### Aperçu du dataset :")
    st.dataframe(data)

# Séparation des données
X = data.drop(['legitimate', 'AddressOfEntryPoint'], axis=1)
Y = data['legitimate']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, random_state=42)

# Choisir l'algorithme
algorithm = st.sidebar.selectbox(
    "Choisissez un algorithme :",
    ['Decision Tree', 'K-Nearest Neighbors', 'Support Vector Machine']
)

if st.sidebar.button("Entraîner le modèle"):
    st.write(f"### Résultats pour l'algorithme : {algorithm}")
    accuracy, report = train_model(algorithm, X_train, Y_train, X_test, Y_test)
    
    # Afficher les résultats
    st.write(f"**Accuracy Score:** {accuracy}")
    st.write("### Rapport de classification :")
    st.json(report)
