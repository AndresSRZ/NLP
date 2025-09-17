import streamlit as st
from transformers import pipeline
import pandas as pd

# -----------------------
# 1. Carga eficiente del modelo
# -----------------------
@st.cache_resource
def load_zero_shot_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# -----------------------
# 2. Interfaz Streamlit
# -----------------------
st.title("🧠 Clasificador de Tópicos Flexible (Zero-Shot)")
st.write("""
Esta aplicación clasifica un texto en las categorías que tú elijas, **sin necesidad de reentrenamiento**.
Usa el modelo `facebook/bart-large-mnli` para realizar inferencia de lenguaje natural (NLI).
""")

# Entrada de texto
text_input = st.text_area("✏️ Ingresa el texto que deseas analizar:", height=200)

# Entrada de etiquetas
labels_input = st.text_input("🏷️ Ingresa las etiquetas (separadas por comas):", value="deportes, política, salud")

# Botón para clasificar
if st.button("📊 Clasificar"):

    if text_input.strip() == "" or labels_input.strip() == "":
        st.warning("Por favor ingresa un texto y al menos una etiqueta.")
    else:
        with st.spinner("Analizando con el modelo..."):

            # Convertir etiquetas a lista
            labels = [label.strip() for label in labels_input.split(",") if label.strip() != ""]

            # Cargar modelo
            classifier = load_zero_shot_model()

            # Ejecutar clasificación
            result = classifier(text_input, candidate_labels=labels)

            # Mostrar resultados
            st.subheader("🎯 Resultados del modelo:")
            scores = result['scores']
            labels = result['labels']

            # Crear DataFrame para mostrar en gráfico
            df = pd.DataFrame({
                "Etiqueta": labels,
                "Puntaje": [round(score * 100, 2) for score in scores]
            })

            st.dataframe(df)

            # Mostrar gráfico de barras
            st.bar_chart(df.set_index("Etiqueta"))
