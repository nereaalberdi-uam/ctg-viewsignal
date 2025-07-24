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
Esta aplicación permite cargar y analizar registros de cardiotocografía (CTG)
pertenecientes a la base de datos CTU-CHB Intrapartum Cardiotocography Database.
Los registros se identifican mediante sus ID, siguiendo la nomenclatura original del repositorio.
Los nombres de registro permitidos son: 1001, 1002, ..., 1506, 2001, ..., 2046.

En primer lugar, se muestran las señales originales de frecuencia cardíaca fetal (FHR)
y contracciones uterinas (UC). A continuación, se presentan los pasos sucesivos del
preprocesamiento automático y el resultado final.

La aplicación detecta y visualiza automáticamente tanto las deceleraciones como las 
contracciones sobre las señales. Finalmente, las deceleraciones se clasifican según 
su relación temporal con las contracciones, mostrando un resumen del conteo final 
y una animación del emparejamiento.
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

st.sidebar.markdown(
    "**Valores válidos:**  \n"
    "`1001` a `1506` y `2001` a `2046`  \n"
    "(No existen registros entre `1507` y `2000`)"
)

VALID_IDS = list(map(str, list(range(1001, 1507)) + list(range(2001, 2047))))

if st.sidebar.button("Procesar registro"):
    if not record_name:
        st.error("Por favor ingrese el ID del registro.")
    elif record_name not in VALID_IDS:
        st.error("ID no válido. Debe estar entre 1001–1506 o 2001–2046.")
    else:
        # Cargar datos originales
        try:
            original_fhr, original_uc, fs, metadata_df = load_ctg_data(record_name, DB_FOLDER)
        except Exception as e:
            st.error(f"Error al cargar el registro: {e}")
            st.stop()

        # Mostrar señales originales sin procesar
        st.subheader("Señales Originales FHR y UC")
        fig_orig = plot_ctg_signals(original_fhr, original_uc, fs)
        st.pyplot(fig_orig)

        # Ejecutar pipeline de preprocesamiento (recorte e interpolación)
        result = preprocess_ctg_pipeline(record_name, DB_FOLDER, plot=True)
        # Mostrar señales limpias y pasos intermedios del preprocesamiento
        st.subheader("Preprocesamiento Paso a Paso")
        st.markdown("""
        El preprocesamiento detecta y elimina tramos planos, interpola valores inválidos 
        y asegura una duración mínima, manteniendo la alineación temporal entre FHR y UC.
        """)
        if result[0] is None or result[1] is None:
            st.warning("Una de las señales (FHR o UC) no presenta variación significativa. El registro se descarta por falta de información útil.")
            st.stop()
        
        fhr_clean, uc_clean, fs, _, figs_pipeline = result
        for fig in figs_pipeline:
            st.pyplot(fig)

        # Detectar deceleraciones y contracciones, y clasificarlas
        duration_sec, early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, baseline, figs_decel = \
            get_classified_decelerations(fhr_clean, uc_clean, fs, verbose=True)

        if duration_sec < 900:
            st.warning("El registro tiene una duración inferior a los 15 minutos requeridos. Se descarta del análisis.")
            st.stop()
        
        # Mostrar gráficos de detección y clasificación
        st.subheader("Detección de Eventos: deceleraciones y contracciones.")
        for fig in figs_decel:
            st.pyplot(fig)

        # Mostrar resumen
        duration_min = duration_sec / 60
        st.subheader("Resumen de Eventos Detectados")
        st.write(f"**Duración del registro en minutos:** {duration_min:.1f}")
        st.write(f"**Deceleraciones detectadas:** {len(decelerations)}")
        st.write(f"**Contracciones detectadas:** {len(contractions)}")
        st.write(f"**Deceleraciones Tempranas:** {len(early_decs)}")
        st.write(f"**Deceleraciones Tardías:** {len(late_decs)}")
        st.write(f"**Deceleraciones Variables:** {len(variable_decs)}")


        # Animación de emparejamiento
        if paired_events:
            gif_path = animate_paired_events(fhr_clean, uc_clean, fs, baseline, decelerations, contractions, paired_events)
            st.image(gif_path, caption="Animación de Eventos Emparejados")
