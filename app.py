# app.py
import streamlit as st
import pandas as pd
import requests
import json
from typing import List, Tuple

st.set_page_config(page_title="Clasificador Zero-Shot (HF Inference)", layout="centered")
st.title("Clasificador de Tópicos Flexible (Zero-Shot)")

st.write(
    "Ingresa un texto y etiquetas separadas por comas. "
    "La app intentará usar la Hugging Face Inference API (facebook/bart-large-mnli). "
    "Si no se configura la API key, se usará un fallback por palabras clave para demo."
)

# ---------- Config de la API ----------
# Streamlit Cloud: colocar secret en Settings > Secrets con la clave HUGGINGFACE_API_TOKEN
hf_token = None
# Primero intentar leer st.secrets (recomendado en Streamlit Cloud)
try:
    hf_token = st.secrets["HUGGINGFACE_API_TOKEN"]
except Exception:
    hf_token = None

# También permitimos pegar temporalmente la key (no recomendado en producción)
entrada_token = st.text_input("Hugging Face API Token (opcional, pegalo si quieres usar el modelo real)", type="password")
if entrada_token:
    hf_token = entrada_token.strip()

MODEL = "facebook/bart-large-mnli"
HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL}"

# ---------- Helpers ----------
def call_hf_zero_shot(premise: str, candidate_labels: List[str], multi_class: bool = True, timeout: int = 30):
    """
    Llama a la HF Inference API para zero-shot classification.
    Retorna (labels, scores) en orden.
    """
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
    payload = {
        "inputs": premise,
        "parameters": {"candidate_labels": candidate_labels, "multi_class": multi_class},
        "options": {"wait_for_model": True}
    }
    resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()

def clasificador_fallback(texto: str, etiquetas: List[str]) -> Tuple[List[str], List[float]]:
    txt = texto.lower()
    puntajes = []
    for e in etiquetas:
        e_low = e.lower()
        matches = sum(1 for tok in e_low.split() if tok in txt)
        score = matches / (1 + len(e_low.split()))
        puntajes.append(score)
    mx = max(puntajes) if puntajes else 1
    if mx > 0:
        puntajes = [p / mx for p in puntajes]
    return etiquetas, puntajes

# ---------- Form ----------
with st.form("form_clas"):
    texto = st.text_area("Texto (premisa) a analizar", height=180, placeholder="Escribe el texto que quieres clasificar...")
    etiquetas_raw = st.text_input("Etiquetas (separadas por comas)", placeholder="deportes, politica, tecnologia, salud")
    multi_label = st.checkbox("Permitir múltiples etiquetas (multi-label)", value=True)
    submit = st.form_submit_button("Clasificar")

if submit:
    if (not texto) or (not etiquetas_raw.strip()):
        st.warning("Por favor ingresa texto y al menos una etiqueta.")
    else:
        etiquetas = [e.strip() for e in etiquetas_raw.split(",") if e.strip()]
        if not etiquetas:
            st.warning("No se detectaron etiquetas válidas.")
        else:
            if hf_token:
                st.info("Usando Hugging Face Inference API (modelo remoto).")
                with st.spinner("Consultando el modelo (puede tardar unos segundos)..."):
                    try:
                        raw = call_hf_zero_shot(texto, etiquetas, multi_class=multi_label)
                        # La respuesta típica tiene campos 'labels' y 'scores' o un dict con 'error'
                        if isinstance(raw, dict) and raw.get("error"):
                            raise RuntimeError(raw.get("error"))
                        # Algunos endpoints devuelven {'labels':[...], 'scores':[...]}
                        if isinstance(raw, dict) and "labels" in raw and "scores" in raw:
                            labels = raw["labels"]
                            scores = raw["scores"]
                        # Otros modelos devuelven una lista de dicts (cada label -> score)
                        elif isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
                            # intentar extraer labels y scores
                            labels = [d.get("label") or d.get("role") or str(d) for d in raw]
                            scores = [float(d.get("score", 0)) for d in raw]
                        else:
                            # fallback: convertir a texto y mostrar
                            st.error("Respuesta inesperada de la API. Se muestra el contenido crudo.")
                            with st.expander("Respuesta cruda de HF"):
                                st.json(raw)
                            labels, scores = clasificador_fallback(texto, etiquetas)
                        # construir DataFrame y mostrar
                        df = pd.DataFrame({"Etiqueta": labels, "Puntaje": scores}).sort_values("Puntaje", ascending=False).reset_index(drop=True)
                        st.subheader("Resultados (ordenados)")
                        st.dataframe(df.style.format({"Puntaje": "{:.4f}"}), use_container_width=True)
                        st.subheader("Gráfico de barras")
                        st.bar_chart(df.set_index("Etiqueta"))
                        st.markdown(f"**Etiqueta más probable:** **{df.iloc[0]['Etiqueta']}** con puntaje **{df.iloc[0]['Puntaje']:.4f}**")
                        with st.expander("Ver respuesta cruda de la API"):
                            st.json(raw)
                    except Exception as e:
                        st.error("Ocurrió un error al llamar la Inference API de Hugging Face:")
                        st.exception(e)
                        st.info("Se aplicará el fallback por palabras clave.")
                        etiquetas_out, puntajes = clasificador_fallback(texto, etiquetas)
                        df = pd.DataFrame({"Etiqueta": etiquetas_out, "Puntaje": puntajes}).sort_values("Puntaje", ascending=False).reset_index(drop=True)
                        st.dataframe(df.style.format({"Puntaje": "{:.4f}"}), use_container_width=True)
                        st.bar_chart(df.set_index("Etiqueta"))
            else:
                st.warning("No se detectó Hugging Face API token. Usando fallback por palabras clave.")
                etiquetas_out, puntajes = clasificador_fallback(texto, etiquetas)
                df = pd.DataFrame({"Etiqueta": etiquetas_out, "Puntaje": puntajes}).sort_values("Puntaje", ascending=False).reset_index(drop=True)
                st.dataframe(df.style.format({"Puntaje": "{:.4f}"}), use_container_width=True)
                st.subheader("Gráfico de barras")
                st.bar_chart(df.set_index("Etiqueta"))

st.sidebar.markdown("### Instrucciones de despliegue")
st.sidebar.write(
    "- Añade `requirements.txt` (ver instrucciones). "
    " - En Streamlit Cloud configura `HUGGINGFACE_API_TOKEN` en Settings → Secrets para usar el modelo real."
)
