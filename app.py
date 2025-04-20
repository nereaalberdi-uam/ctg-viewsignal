if st.button("Mostrar gráficos"):
    with st.spinner("Procesando..."):
        try:
            # Preprocesamiento
            fhr, uc, fs, metadata_df = prep.preprocess_ctg_pipeline(
                record_name, DB_FOLDER, tolerance=1, interpolation_method='linear', plot=True
            )

            # 💡 Mostrar cualquier gráfico que se haya generado
            st.subheader("1️⃣ Señales FHR y UC")
            st.pyplot(plt.gcf())
            plt.clf()

            # Clasificación de desaceleraciones
            early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, dBaseline = dc.get_classified_decelerations(
                fhr, uc, fs, verbose=True
            )

            # 💡 Mostrar cualquier gráfico generado internamente
            st.subheader("2️⃣ Gráficos de clasificación")
            st.pyplot(plt.gcf())
            plt.clf()

            # Mostrar resumen textual
            st.subheader("3️⃣ Información de eventos detectados")
            st.markdown(f"""
            - **Desaceleraciones tempranas:** {len(early_decs)}
            - **Desaceleraciones tardías:** {len(late_decs)}
            - **Desaceleraciones variables:** {len(variable_decs)}
            - **Total de contracciones:** {len(contractions)}
            - **Eventos emparejados:** {len(paired_events)}
            """)

            # Animación
            st.subheader("4️⃣ Animación de eventos emparejados")
            with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
                dc.animate_paired_events(fhr, uc, fs, dBaseline, decelerations, contractions, paired_events,
                                         out_path=tmpfile.name)
                st.image(tmpfile.name, caption="Animación")

        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
