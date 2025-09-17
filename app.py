# app.py
import streamlit as st
import pandas as pd

st.set_page_config(page_title="Clasificador Zero-Shot (Demo)", layout="centered")
st.title("Clasificador de Tópicos Flexible (Zero-Shot)")

st.write("Ingresa un texto y etiquetas separadas por comas. Intentaremos usar `transformers` (zero-shot). Si no está disponible, caeremos a un demo por palabras clave.")

# Intentar cargar transformers lazy (evitar fallo inmediato al importar en entornos sin deps)
def cargar_pipeline_seguro():
    try:
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)
        return ("hf", pipe)
    except Exception as e:
        # Retornar None + excepción para mostrar al usuario
        return ("fallback", str(e))

modo, recurso = cargar_pipeline_seguro()

with st.form("form"):
    texto = st.text_area("Texto a analizar", height=160, placeholder="Escribe el texto...")
    etiquetas_raw = st.text_input("Etiquetas (separadas por comas)", placeholder="deportes, politica, tecnologia")
    submit = st.form_submit_button("Clasificar")

def clasificador_fallback(texto, etiquetas):
    # Demo simple: puntaje por coincidencias de palabras (normalizado)
    txt = texto.lower()
    puntajes = []
    for e in etiquetas:
        e_low = e.lower()
        matches = sum(1 for tok in e_low.split() if tok in txt)
        # puntaje básico: coincidencias / (1 + numero_palabras_etiqueta)
        score = matches / (1 + len(e_low.split()))
        puntajes.append(score)
    # normalizar a [0,1]
    mx = max(puntajes) if puntajes else 1
    if mx > 0:
        puntajes = [p/mx for p in puntajes]
    return etiquetas, puntajes

if submit:
    if not texto or not etiquetas_raw.strip():
        st.warning("Por favor ingresa texto y al menos una etiqueta.")
    else:
        etiquetas = [e.strip() for e in etiquetas_raw.split(",") if e.strip()]
        if not etiquetas:
            st.warning("No se detectaron etiquetas válidas.")
        else:
            if modo == "hf":
                st.info("Usando transformers (HF) para clasificación zero-shot.")
                pipe = recurso
                with st.spinner("Evaluando con modelo HF... (esto puede tardar)"):
                    try:
                        resultado = pipe(texto, etiquetas, multi_label=True)
                        labels = resultado["labels"]
                        scores = resultado["scores"]
                        df = pd.DataFrame({"Etiqueta": labels, "Puntaje": scores}).sort_values("Puntaje", ascending=False).reset_index(drop=True)
                        st.dataframe(df.style.format({"Puntaje": "{:.4f}"}), use_container_width=True)
                        st.subheader("Gráfico de barras")
                        st.bar_chart(df.set_index("Etiqueta"))
                        st.markdown(f"**Etiqueta más probable:** **{df.iloc[0]['Etiqueta']}** ({df.iloc[0]['Puntaje']:.4f})")
                        with st.expander("Salida cruda del modelo (HF)"):
                            st.json(resultado)
                    except Exception as e:
                        st.error("Ocurrió un error al evaluar el modelo HF:")
                        st.exception(e)
            else:
                st.warning("`transformers` o `torch` no están disponibles en este entorno. Usando demo por palabras clave.")
                etiquetas_out, puntajes = clasificador_fallback(texto, etiquetas)
                df = pd.DataFrame({"Etiqueta": etiquetas_out, "Puntaje": puntajes}).sort_values("Puntaje", ascending=False).reset_index(drop=True)
                st.dataframe(df.style.format({"Puntaje": "{:.4f}"}), use_container_width=True)
                st.subheader("Gráfico de barras")
                st.bar_chart(df.set_index("Etiqueta"))
                st.markdown("**Nota:** Este es un fallback para demo. Para usar el clasificador real en la nube, añade `transformers` y `torch` a `requirements.txt` y redepliega.")

                st.caption("Detalles del intento de carga de transformers:")
                st.code(recurso)
