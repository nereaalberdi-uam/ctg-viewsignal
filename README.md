# ctg-viewsignal

Esta aplicación web permite cargar y analizar registros de cardiotocografía (CTG) de la base de datos CTU-CHB Intrapartum Cardiotocography Database, identificados por sus IDs originales.

Funcionalidades:

  1. Carga registros válidos con IDs entre 1001-1506 y 2001-2046.
  2. Visualiza señales originales y preprocesadas de frecuencia cardiaca fetal (FHR) y contracciones uterinas (UC).
  3. Detecta y clasifica deceleraciones fetales y contracciones uterinas.
  4. Descarta registros con señales planas o duración menor a 15 minutos.
  5. Muestra un resumen y animación del emparejamiento entre deceleraciones y contracciones.

Uso:
  1. Introducir el ID del registro en la barra lateral.
  2. Presionar "Procesar registro".
  3. Revisar los resultados y visualizaciones.
