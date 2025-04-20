import streamlit as st
from preprocessing import load_ctg_data, preprocess_ctg_pipeline, plot_ctg_signals
from preprocessing import plot_signal_with_gaps
from deceleration import get_classified_decelerations, plot_decc_contr, animate_paired_events

st.title("Análisis interactivo de señales CTG")
st.markdown("""
Esta aplicación permite cargar registros de cardiotocografía (CTG), realizar un preprocesamiento 
automático de las señales de FHR (frecuencia cardiaca fetal) y UC (contracciones uterinas), 
y detectar deceleraciones fetales y contracciones uterinas para su análisis.
""")

# Sidebar inputs for record selection and parameters
st.sidebar.header("Seleccionar registro CTG")
record_name = st.sidebar.text_input("Nombre del registro (ID):", value="")
db_path = st.sidebar.text_input("Ruta de la base de datos CTG:", value="")

if st.sidebar.button("Procesar registro"):
    if not record_name or not db_path:
        st.error("Por favor ingrese tanto el ID del registro como la ruta de la base de datos.")
    else:
        # Cargar datos originales
        try:
            original_fhr, original_uc, fs, metadata_df = load_ctg_data(record_name, db_path)
        except Exception as e:
            st.error(f"Error al cargar el registro: {e}")
            st.stop()

        # Mostrar metadatos del registro
        st.subheader("Metadatos del registro")
        st.dataframe(metadata_df)

        # Mostrar señales originales sin procesar
        st.subheader("Señales originales FHR y UC")
        fig_orig = plot_ctg_signals(original_fhr, original_uc, fs)
        st.pyplot(fig_orig)

        # Ejecutar pipeline de preprocesamiento (recorte e interpolación)
        fhr_clean, uc_clean, fs, _ = preprocess_ctg_pipeline(record_name, db_path)
        if fhr_clean is None or uc_clean is None:
            st.warning("La señal está prácticamente plana; se omite el análisis.")
            st.stop()

        # Mostrar señales limpias después del preprocesamiento
        st.subheader("Señales preprocesadas FHR y UC")
        fig_clean = plot_ctg_signals(fhr_clean, uc_clean, fs)
        st.pyplot(fig_clean)

        # Opcional: mostrar tramos planos detectados en las señales originales
        fhr_gaps, fhr_perc = [], 0
        uc_gaps, uc_perc = [], 0
        try:
            fhr_gaps, fhr_perc = plot_signal_with_gaps.__wrapped__.__globals__['detect_long_zero_gaps'](original_fhr, secs=30, freq=fs, union=3, tolerance=1)
            uc_gaps, uc_perc = plot_signal_with_gaps.__wrapped__.__globals__['detect_long_zero_gaps'](original_uc, secs=30, freq=fs, union=3, tolerance=1)
        except Exception:
            pass
        if fhr_gaps and uc_gaps:
            fig_gaps_fhr = plot_signal_with_gaps(original_fhr, fs, fhr_gaps, fhr_perc)
            fig_gaps_uc = plot_signal_with_gaps(original_uc, fs, uc_gaps, uc_perc)
            st.subheader("Tramos planos detectados en la señal original")
            st.pyplot(fig_gaps_fhr)
            st.pyplot(fig_gaps_uc)

        # Detectar deceleraciones y contracciones, y clasificarlas
        early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, baseline_obj = \
            get_classified_decelerations(fhr_clean, uc_clean, fs)

        # Mostrar resumen de deceleraciones clasificadas
        st.subheader("Resumen de deceleraciones detectadas")
        st.write(f"**Deceleraciones Early:** {len(early_decs)}")
        st.write(f"**Deceleraciones Late:** {len(late_decs)}")
        st.write(f"**Deceleraciones Variables:** {len(variable_decs)}")
        st.write(f"**Total deceleraciones detectadas:** {len(decelerations)}")
        st.write(f"**Total contracciones detectadas:** {len(contractions)}")

        # Mostrar gráfica combinada de FHR y UC con eventos detectados
        st.subheader("Señales FHR y UC con eventos detectados")
        fig_events = plot_decc_contr(fhr_clean, uc_clean, fs, decelerations, baseline_obj, contractions)
        st.pyplot(fig_events)

        # Botón para mostrar animación de emparejamiento
        if paired_events:
            if st.button("Mostrar animación de emparejamiento"):
                gif_path = animate_paired_events(fhr_clean, uc_clean, fs, baseline_obj, decelerations, contractions, paired_events)
                if gif_path:
                    st.subheader("Animación de deceleraciones (rojo) emparejadas con contracciones (verde)")
                    gif_bytes = open(gif_path, "rb").read()
                    st.image(gif_bytes, format="gif")
                else:
                    st.info("No hay deceleraciones del tipo seleccionado para animar.")
