import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, mean_absolute_error
import openpyxl  # Asegúrate de tener esta librería si trabajas con archivos Excel

# Función para cargar y preprocesar los datos
def load_and_preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file)

    # Seleccionar solo las variables específicas para X
    features = df[['Sector', 'SubSector', 'Pais', 'RecenciaDesembolso']]

    # Preparar la variable 'Año' como categoría
    df['Años'] = pd.Categorical(df['Años'])
    df['Año_Cat'] = df['Años'].cat.codes
    encoder = OneHotEncoder(sparse_output=False)
    encoded_year = encoder.fit_transform(df[['Año_Cat']])

    # Escalar 'RecenciaDesembolso'
    scaler = StandardScaler()
    recencia_desembolso_scaled = scaler.fit_transform(df[['RecenciaDesembolso']])

    # Codificar variables categóricas
    categorical_features = ['Sector', 'SubSector', 'Pais']
    encoded_features = encoder.fit_transform(df[categorical_features])

    # Combinar características
    X_scaled = np.concatenate([encoded_features, recencia_desembolso_scaled], axis=1)
    y_regression = df['Porcentaje Acumulado'].values

    # Dividir los datos
    X_train, X_test, y_year_train, y_year_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, encoded_year, y_regression, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_year_train, y_year_test, y_reg_train, y_reg_test, encoder

# Función para construir y entrenar el modelo
def build_and_train_model(X_train, y_year_train, y_reg_train, encoder):
    input_layer = Input(shape=(X_train.shape[1],))
    hidden1 = Dense(128, activation='relu')(input_layer)
    hidden2 = Dense(64, activation='relu')(hidden1)
    output_year = Dense(encoder.categories_[0].size, activation='softmax', name='year_output')(hidden2)
    output_reg = Dense(1, name='regression_output')(hidden2)

    model = Model(inputs=input_layer, outputs=[output_year, output_reg])

    model.compile(optimizer=Adam(),
                  loss={'year_output': 'categorical_crossentropy', 'regression_output': 'mean_squared_error'},
                  metrics={'year_output': 'accuracy', 'regression_output': 'mae'})
    
    # Convertir y_year_train a una representación de una sola columna
    y_year_train_single_column = np.argmax(y_year_train, axis=1).reshape(-1, 1)

    history = model.fit(X_train, {'year_output': y_year_train_single_column, 'regression_output': y_reg_train},
                        epochs=200,  # Ajusta este número según tus necesidades
                        validation_split=0.2,
                        verbose=0)  # Cambia a verbose=1 para ver la salida del entrenamiento

    return model



# Función para realizar predicciones y evaluar el modelo
def evaluate_model(model, X_test, y_year_test, y_reg_test, encoder):
    predictions = model.predict(X_test)
    y_year_pred, y_reg_pred = predictions

    y_year_pred_classes = np.argmax(y_year_pred, axis=1)
    y_year_test_classes = np.argmax(y_year_test, axis=1)

    year_accuracy = accuracy_score(y_year_test_classes, y_year_pred_classes)
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred.flatten())

    return year_accuracy, reg_mae

# Aplicación Streamlit
def main():
    st.title("Aplicación de Streamlit para Modelo de Redes Neuronales")

    uploaded_file = st.file_uploader("Carga tu archivo Excel", type="xlsx")
    if uploaded_file is not None:
        with st.spinner('Cargando y preprocesando datos...'):
            X_train, X_test, y_year_train, y_year_test, y_reg_train, y_reg_test, encoder = load_and_preprocess_data(uploaded_file)
            st.success('¡Datos cargados y preprocesados con éxito!')

        if st.button('Entrenar Modelo'):
            with st.spinner('Entrenando el modelo...'):
                model = build_and_train_model(X_train, y_year_train, y_reg_train, encoder)
                st.success('Modelo entrenado')

            year_accuracy, reg_mae = evaluate_model(model, X_test, y_year_test, y_reg_test, encoder)
            st.write(f"Exactitud de la predicción de año: {year_accuracy}")
            st.write(f"Error Absoluto Medio (MAE) para Porcentaje Acumulado: {reg_mae}")

if __name__ == "__main__":
    main()

