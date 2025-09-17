# app.py
import streamlit as st
from transformers import pipeline
import torch
import pandas as pd

st.set_page_config(page_title="Clasificador de Tópicos (Zero-Shot)", layout="centered")

st.title("Clasificador de Tópicos Flexible (Zero-Shot)")
st.write(
    "Ingresa un texto y una lista de etiquetas separadas por comas. "
    "El modelo evaluará la afinidad entre el texto y cada etiqueta usando NLI."
)

# Cachar la carga del pipeline para que sólo se descargue/instancie una vez
@st.cache_resource
def cargar_pipeline(model_name: str = "facebook/bart-large-mnli"):
    # usar GPU si está disponible
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("zero-shot-classification", model=model_name, device=device)

pipe = cargar_pipeline()

with st.form("formulario_clasificacion"):
    texto = st.text_area("Texto (premisa) a analizar", height=160, placeholder="Escribe aquí el texto...")
    etiquetas_raw = st.text_input("Etiquetas (separadas por comas)", placeholder="deportes, política, tecnología, salud")
    submit = st.form_submit_button("Clasificar")

if submit:
    if (not texto) or (not etiquetas_raw.strip()):
        st.warning("Por favor ingresa tanto el texto como al menos una etiqueta.")
    else:
        # Procesar etiquetas: limpiar y eliminar vacíos
        etiquetas = [e.strip() for e in etiquetas_raw.split(",") if e.strip()]
        if len(etiquetas) == 0:
            st.warning("No se detectaron etiquetas válidas. Revisa la entrada.")
        else:
            with st.spinner("Evaluando..."):
                # Llamada al pipeline (por defecto hypothesis_template puede estar bien)
                resultado = pipe(texto, etiquetas, multi_label=True)
                # resultado tiene 'labels' y 'scores' (si multi_label=True)
                labels = resultado["labels"]
                scores = resultado["scores"]

                # Crear DataFrame ordenado
                df = pd.DataFrame({"Etiqueta": labels, "Puntaje": scores})
                df_sorted = df.sort_values("Puntaje", ascending=False).reset_index(drop=True)

            st.subheader("Resultados")
            st.write("Tabla de afinidad (ordenada):")
            st.dataframe(df_sorted.style.format({"Puntaje": "{:.4f}"}), use_container_width=True)

            st.subheader("Gráfico de barras")
            # st.bar_chart funciona con dataframe indexado por etiqueta
            chart_df = df_sorted.set_index("Etiqueta")
            st.bar_chart(chart_df)

            st.markdown("**Etiqueta más probable:**")
            mejor = df_sorted.iloc[0]
            st.markdown(f"- **{mejor['Etiqueta']}** con puntaje **{mejor['Puntaje']:.4f}**")

            # Mostrar salida completa del pipeline (opcional)
            with st.expander("Ver salida cruda del modelo"):
                st.json(resultado)

