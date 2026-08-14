import streamlit as st
from utils import cargar_modelo, predecir

st.set_page_config(page_title="Predicción de Diabetes", page_icon="🩺")


@st.cache_resource
def get_modelo():
    return cargar_modelo()


model, scaler, metadata = get_modelo()
feature_names = metadata['feature_names']

st.title("Predicción de Diabetes")
st.markdown("Modelo de Logistic Regression entrenado sobre el dataset Pima Indians Diabetes.")

st.subheader("Datos del paciente")

# Un input numérico por cada feature, organizados en 2 columnas
col1, col2 = st.columns(2)
valores = {}
for i, feat in enumerate(feature_names):
    columna = col1 if i % 2 == 0 else col2
    valores[feat] = columna.number_input(feat, min_value=0.0, step=1.0, format="%.3f")

if st.button("Predecir", type="primary"):
    resultado = predecir(valores, model, scaler, metadata)

    if resultado['prediccion'] == 1:
        st.error(f"**Riesgo de diabetes detectado**  \nProbabilidad estimada: {resultado['probabilidad']*100:.2f}%")
    else:
        st.success(f"**Riesgo de diabetes no detectado**  \nProbabilidad estimada: {resultado['probabilidad']*100:.2f}%")

    st.caption(f"Umbral de decisión usado: {resultado['umbral_usado']}")