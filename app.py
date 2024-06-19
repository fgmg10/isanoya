import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import io

# Función para cargar datos
@st.cache_data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    return data

# Función para mostrar información general del dataset
def show_data_info(data):
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("Información de las columnas:")
        st.text(info_str)
    with col2:
        st.write("Número de nulos por columna:")
        st.write(data.isnull().sum())

# Función para visualizar la matriz de correlación
def show_correlation_matrix(data):
    st.subheader("Matriz de correlación")
    corr = data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    st.pyplot(plt.gcf())
    st.write("Este gráfico muestra la correlación entre las diferentes características del dataset.")

# Función para entrenar y evaluar modelos
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    return accuracy, conf_matrix

# Título de la aplicación
st.title("Visualización y Análisis de Datos con Algoritmos de Machine Learning")

# Subir archivo CSV
uploaded_file = st.file_uploader("Sube tu archivo CSV", type=["csv"])

# Barra lateral
st.sidebar.title("Isaias Noya")

if uploaded_file is not None:
    data = load_data(uploaded_file)
    
    # Comprobar si hay columnas no numéricas
    non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric_cols:
        st.error(f"Las siguientes columnas no son numéricas: {non_numeric_cols}. Por favor, sube un archivo que contenga solo datos numéricos.")
    else:
        st.write("Datos cargados correctamente!")
        
        # Mostrar información del dataset
        show_data_info(data)
        
        # Normalizar datos
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(scaler.fit_transform(data), columns=data.columns)
        
        # Mostrar matriz de correlación
        show_correlation_matrix(data)
        
        # Seleccionar columna objetivo
        target_column = st.selectbox("Selecciona la columna objetivo", data.columns)
        
        # Asegurar que la columna objetivo sea categórica
        if data[target_column].nunique() > 10:
            st.error("La columna objetivo tiene demasiados valores únicos, lo que indica que podría ser continua en lugar de categórica.")
        else:
            # Separar características y columna objetivo
            X = data_scaled.drop(columns=[target_column])
            y = data[target_column].astype('category').cat.codes
            
            # Separar datos en conjunto de entrenamiento y prueba
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Selección de algoritmo
            classifier_name = st.sidebar.selectbox(
                "Selecciona el algoritmo de Machine Learning",
                ("Random Forest", "XGBoost", "Logistic Regression", "Support Vector Classifier", 
                 "Decision Tree", "Naive Bayes", "Stochastic Gradient Descent", "K Nearest Neighbor")
            )
            
            # Inicializar el modelo basado en la selección
            if classifier_name == "Random Forest":
                model = RandomForestClassifier()
            elif classifier_name == "XGBoost":
                model = XGBClassifier()
            elif classifier_name == "Logistic Regression":
                model = LogisticRegression()
            elif classifier_name == "Support Vector Classifier":
                model = SVC()
            elif classifier_name == "Decision Tree":
                model = DecisionTreeClassifier()
            elif classifier_name == "Naive Bayes":
                model = GaussianNB()
            elif classifier_name == "Stochastic Gradient Descent":
                model = SGDClassifier()
            elif classifier_name == "K Nearest Neighbor":
                model = KNeighborsClassifier()
            
            # Entrenar y evaluar el modelo
            if st.sidebar.button("Entrenar y evaluar modelo"):
                accuracy, conf_matrix = train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
                
                # Mostrar resultados
                st.subheader(f"Exactitud del modelo: {accuracy:.2f}")
                
                st.subheader("Matriz de confusión")
                fig, ax = plt.subplots()
                ConfusionMatrixDisplay(conf_matrix).plot(ax=ax)
                st.pyplot(fig)
                st.write("La matriz de confusión muestra el rendimiento del modelo en términos de verdaderos positivos, falsos positivos, verdaderos negativos y falsos negativos.")
                
                # Mostrar distribución de la columna objetivo
                st.subheader(f"Distribución de la columna objetivo: {target_column}")
                plt.figure(figsize=(10, 6))
                sns.histplot(data[target_column], kde=True)
                st.pyplot(plt.gcf())
                st.write("Este gráfico muestra la distribución de los valores en la columna objetivo.")
        
        # Botón para limpiar el contenido de los algoritmos de Machine Learning
        if st.sidebar.button("Limpiar contenido de algoritmos ML"):
            st.caching.clear_cache()
            st.experimental_rerun()

        # Filtro de columnas
        column_filter = st.sidebar.selectbox("Selecciona una columna para graficar", data.columns)
        
        # Filtro de tipos de gráficos
        graph_type = st.sidebar.selectbox(
            "Selecciona el tipo de gráfico",
            ("Histograma", "Boxplot", "Scatterplot", "Barras")
        )
        
        # Generar gráficos basados en la selección
        if column_filter and graph_type:
            st.subheader(f"Gráfico de tipo {graph_type} para la columna {column_filter}")
            plt.figure(figsize=(10, 6))
            
            if graph_type == "Histograma":
                sns.histplot(data[column_filter], kde=True)
            elif graph_type == "Boxplot":
                sns.boxplot(y=data[column_filter])
            elif graph_type == "Scatterplot":
                other_column = st.sidebar.selectbox("Selecciona otra columna para el eje X", data.columns)
                sns.scatterplot(x=data[other_column], y=data[column_filter])
            elif graph_type == "Barras":
                sns.countplot(x=data[column_filter])
            
            st.pyplot(plt.gcf())
            st.write(f"Este gráfico muestra un {graph_type.lower()} de la columna {column_filter}.")
else:
    st.write("Por favor, sube un archivo CSV.")
