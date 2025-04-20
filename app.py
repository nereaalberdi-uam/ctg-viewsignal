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
DROPBOX_URL = "https://www.dropbox.com/s/abc123xyz/ctu-chb-database.zip?dl=1"  # Reemplazar con tu enlace real

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

# Descarga si es necesario
download_and_extract_dropbox_zip(DROPBOX_URL)

# Entrada de usuario
record_name = st.text_input("Escribe el nombre del registro (por ejemplo, '2012')", value="2012")

if st.button("Mostrar gráficos"):
    with st.spinner("Procesando..."):
        try:
            # Preprocesamiento
            fhr, uc, fs, metadata_df = prep.preprocess_ctg_pipeline(
                record_name, DB_FOLDER, tolerance=1, interpolation_method='linear', plot=True
            )

            # Mostrar gráfico generado por matplotlib
            st.subheader("1️⃣ Señales FHR y UC")
            st.pyplot(plt.gcf())
            plt.clf()

            # Clasificación de desaceleraciones y eventos
            early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, dBaseline = dc.get_classified_decelerations(
                fhr, uc, fs, verbose=True
            )

            st.subheader("2️⃣ Información de eventos detectados")
            st.markdown(f"""
            - **Desaceleraciones tempranas:** {len(early_decs)}
            - **Desaceleraciones tardías:** {len(late_decs)}
            - **Desaceleraciones variables:** {len(variable_decs)}
            - **Total de contracciones:** {len(contractions)}
            - **Eventos emparejados (con desaceleración):** {len(paired_events)}
            """)

            # Animación de eventos emparejados
            st.subheader("3️⃣ Animación de eventos emparejados")
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                dc.animate_paired_events(fhr, uc, fs, dBaseline, decelerations, contractions, paired_events,
                                         out_path=tmpfile.name)
                st.image(tmpfile.name, caption="Animación")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
