import pandas as pd
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, r2_score  # Añadimos la importación de r2_score

# Función para cargar y preprocesar los datos
def load_and_preprocess_data(uploaded_file):
    df = pd.read_excel(uploaded_file)

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
    X_train, X_test, y_reg_train, y_reg_test = train_test_split(
        X_scaled, y_regression, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_reg_train, y_reg_test, encoder


# Función para construir y entrenar el modelo
def build_and_train_model(X_train, y_reg_train):
    # Diseñar la red neuronal con TensorFlow y Keras
    input_layer = Input(shape=(X_train.shape[1],))
    hidden1 = Dense(128, activation='relu')(input_layer)
    hidden2 = Dense(64, activation='relu')(hidden1)
    output_reg = Dense(1, name='regression_output')(hidden2)

    model = Model(inputs=input_layer, outputs=output_reg)

    model.compile(optimizer=Adam(),
                loss='mean_squared_error',
                metrics=['mae'])

    # Entrenar el modelo
    history = model.fit(X_train, y_reg_train,
                        epochs=450,
                        validation_split=0.2)
    
    return model

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_reg_test):
    # Realizar predicciones
    y_reg_pred = model.predict(X_test)

    # Calcular el Error Absoluto Medio (MAE) para Porcentaje Acumulado
    reg_mae = mean_absolute_error(y_reg_test, y_reg_pred.flatten())
    
    # Calcular el coeficiente de determinación (R^2)
    r2 = r2_score(y_reg_test, y_reg_pred)
    
    return reg_mae, r2


# Aplicación Streamlit
def main():
    st.title("Aplicación de Streamlit para Modelo de Redes Neuronales")

    uploaded_file = st.file_uploader("Carga tu archivo Excel", type="xlsx")
    if uploaded_file is not None:
        with st.spinner('Cargando y preprocesando datos...'):
            X_train, X_test, y_reg_train, y_reg_test, encoder = load_and_preprocess_data(uploaded_file)
            st.success('¡Datos cargados y preprocesados con éxito!')

        if st.button('Entrenar Modelo'):
            with st.spinner('Entrenando el modelo...'):
                model = build_and_train_model(X_train, y_reg_train)
                st.success('Modelo entrenado')

            reg_mae, r2 = evaluate_model(model, X_test, y_reg_test)
            st.write(f"Error Absoluto Medio (MAE) para Porcentaje Acumulado: {reg_mae}")
            st.write(f"Coeficiente de Determinación (R^2): {r2}")
            

if __name__ == "__main__":
    main()


