import streamlit as st
from pathlib import Path
import os
import zipfile
import requests

from preprocessing import load_ctg_data, preprocess_ctg_pipeline, plot_ctg_signals
from preprocessing import plot_signal_with_gaps
from deceleration import get_classified_decelerations, animate_paired_events

st.title("Análisis interactivo de señales CTG")
st.markdown("""
Esta aplicación permite cargar registros de cardiotocografía (CTG), realizar un preprocesamiento 
automático de las señales de FHR (frecuencia cardiaca fetal) y UC (contracciones uterinas), 
y detectar deceleraciones fetales y contracciones uterinas para su análisis.
""")

# === CONFIGURACIÓN ===
DB_FOLDER = "ctu-chb-database"
DROPBOX_URL = "https://www.dropbox.com/scl/fi/q23kakkcrpail8ujxlrqs/ctu-chb-database.zip?rlkey=5s09onkezm3ysauccfvs70zi7&st=n3emf5f6&dl=1" 

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

# Descargar los datos si no están
download_and_extract_dropbox_zip(DROPBOX_URL)

# Sidebar inputs for record selection and parameters
st.sidebar.header("Seleccionar registro CTG")
record_name = st.sidebar.text_input("Nombre del registro (ID):", value="")

if st.sidebar.button("Procesar registro"):
    if not record_name:
        st.error("Por favor ingrese el ID del registro.")
    else:
        # Cargar datos originales
        try:
            original_fhr, original_uc, fs, metadata_df = load_ctg_data(record_name, DB_FOLDER)
        except Exception as e:
            st.error(f"Error al cargar el registro: {e}")
            st.stop()

        # Mostrar señales originales sin procesar
        st.subheader("Señales originales FHR y UC")
        fig_orig = plot_ctg_signals(original_fhr, original_uc, fs)
        st.pyplot(fig_orig)

        # Ejecutar pipeline de preprocesamiento (recorte e interpolación)
        result = preprocess_ctg_pipeline(record_name, DB_FOLDER, plot=True)
        if result[0] is None or result[1] is None:
            st.warning("La señal está prácticamente plana; se omite el análisis.")
            st.stop()
        
        fhr_clean, uc_clean, fs, _, figs_pipeline = result
        
        # Mostrar señales limpias y pasos intermedios del preprocesamiento
        st.subheader("Proceso de preprocesamiento y señales finales")
        for fig in figs_pipeline:
            st.pyplot(fig)

        # Detectar deceleraciones y contracciones, y clasificarlas
        early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, baseline_obj, figs_decel = \
            get_classified_decelerations(fhr_clean, uc_clean, fs, verbose=True)
        
        # Mostrar gráficos de detección y clasificación
        st.subheader("Eventos detectados en señales procesadas")
        for fig in figs_decel:
            st.pyplot(fig)

        # Mostrar resumen de deceleraciones clasificadas
        st.subheader("Resumen de deceleraciones detectadas")
        st.write(f"**Deceleraciones Early:** {len(early_decs)}")
        st.write(f"**Deceleraciones Late:** {len(late_decs)}")
        st.write(f"**Deceleraciones Variables:** {len(variable_decs)}")
        st.write(f"**Total deceleraciones detectadas:** {len(decelerations)}")
        st.write(f"**Total contracciones detectadas:** {len(contractions)}")


        # Botón para mostrar animación de emparejamiento
        if paired_events:
            gif_path = animate_paired_events(fhr_clean, uc_clean, fs, baseline_obj, decelerations, contractions, paired_events)
            st.image(gif_path, caption="Animación de eventos emparejados")
