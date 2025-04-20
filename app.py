import streamlit as st
import preprocessing as prep
import deceleration as dc
import os
import matplotlib.pyplot as plt
import tempfile
import zipfile
import requests
from pathlib import Path

st.set_page_config(page_title="Visualización de CTG", layout="wide")
st.title("Visualizador de señales CTG")

# === CONFIGURACIÓN ===
DB_FOLDER = "ctu-chb-database"
DROPBOX_URL = "https://www.dropbox.com/scl/fo/8k6fx7usnfwhfo7u5wsqy/AN5KRLH-zYU-8A0bqD6l7x0?rlkey=6bxc88pby3oexj5gfl6gq9h9q&st=jt11gh8d&dl=1" 

# === FUNCIONES ===
def download_and_extract_dropbox_zip(url, extract_to=DB_FOLDER, zip_name="ctg_data.zip"):
    if not Path(extract_to).exists():
        st.info("Descargando base de datos desde Dropbox...")
        r = requests.get(url)
        with open(zip_name, "wb") as f:
            f.write(r.content)
        with zipfile.ZipFile(zip_name, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_name)
        st.success("Base de datos descargada y lista.")

# === LÓGICA PRINCIPAL ===

# Descarga si es necesario
download_and_extract_dropbox_zip(DROPBOX_URL)

# Entrada de usuario
record_name = st.text_input("Escribe el nombre del registro (por ejemplo, '2012')", value="2012")

if st.button("Mostrar gráficos"):
    with st.spinner("Procesando..."):
        try:
            # Procesamiento de señales
            fhr, uc, fs, metadata_df = prep.preprocess_ctg_pipeline(
                record_name, DB_FOLDER, tolerance=1, interpolation_method='linear', plot=True
            )

            # Mostrar gráfico generado por matplotlib (del preprocesamiento)
            st.pyplot(plt.gcf())
            plt.clf()

            # Clasificación de desaceleraciones y eventos
            early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, dBaseline = dc.get_classified_decelerations(
                fhr, uc, fs, verbose=True
            )

            # Mostrar animación de eventos emparejados
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                gif_path = dc.animate_paired_events(fhr, uc, fs, dBaseline, decelerations, contractions, paired_events)
                st.image(gif_path, caption="Animación de eventos emparejados")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
