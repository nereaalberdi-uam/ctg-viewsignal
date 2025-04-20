import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline

# ----------------------------- Carga de datos CTG ----------------------------- #

def load_ctg_data(record_name, db_path):
    """
    Carga datos de CTG (Cardiotocografía) desde un registro de PhysioNet.

    Parámetros:
    - record_name (str): Nombre del registro a cargar.
    - db_path (str): Ruta al directorio que contiene los archivos del registro.

    Retorna:
    - fhr (numpy.ndarray): Señal de Fetal Heart Rate (FHR).
    - uc (numpy.ndarray): Señal de Uterine Contractions (UC).
    - fs (float): Frecuencia de muestreo de las señales.
    - metadata_df (pandas.DataFrame): DataFrame con metadatos del archivo .hea.
    """
    # Leer el registro utilizando wfdb
    record = wfdb.rdrecord(f"{db_path}/{record_name}")
    signals = record.p_signal        # Matriz de señales (columnas por tipo de señal)
    fields = record.sig_name         # Nombres de las señales
    fs = record.fs                  # Frecuencia de muestreo

    # Identificar índices de las señales FHR y UC
    try:
        fhr_idx = fields.index('FHR')
        uc_idx = fields.index('UC')
    except ValueError:
        raise ValueError("No se encontraron señales FHR o UC en el registro.")

    # Extraer las señales FHR y UC
    fhr = pd.Series(signals[:, fhr_idx])
    uc = pd.Series(signals[:, uc_idx])

    # Cargar metadatos desde el archivo .hea del registro
    header_path = f"{db_path}/{record_name}.hea"
    metadata = []
    with open(header_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                parts = line[1:].strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    metadata.append((key, value))
    metadata_df = pd.DataFrame(metadata, columns=["Parameter", "Value"])

    return fhr.values, uc.values, fs, metadata_df

# ---------------------------- Preprocesamiento CTG ---------------------------- #

def detect_long_zero_gaps(signal, secs=15, freq=4, union=0, tolerance=0):
    """
    Detecta tramos largos donde la señal permanece constante (derivada ~ 0).

    Parámetros:
    - signal (array-like): Señal de entrada (1D).
    - secs (float): Duración mínima en segundos para considerar un tramo como largo.
    - freq (float): Frecuencia de muestreo en Hz.
    - union (int): Máximo número de muestras no constantes entre tramos para unirlos.
    - tolerance (float): Tolerancia alrededor de cero para considerar la derivada ~ 0.

    Retorna:
    - final_gaps (list): Lista de tuplas (start_time, end_time, duration, percentage) de cada tramo constante.
    - total_percentage (float): Porcentaje total del tiempo de señal ocupado por estos tramos constantes.
    """
    signal = np.array(signal)
    total_duration = len(signal) / freq

    # Derivada de la señal
    derivative = np.diff(signal, prepend=signal[0])
    zero_indices = np.where(np.abs(derivative) <= tolerance)[0]

    # Identificar tramos constantes largos
    long_gaps = []
    if zero_indices.size > 0:
        start = zero_indices[0]
        prev = start
        for i in range(1, len(zero_indices)):
            if zero_indices[i] != prev + 1:
                # Fin de un tramo constante
                end = prev
                length = end - start + 1
                duration = length / freq
                if duration >= secs:
                    long_gaps.append((start, end, duration))
                # Iniciar nuevo tramo
                start = zero_indices[i]
            prev = zero_indices[i]
        # Último tramo
        end = prev
        length = end - start + 1
        duration = length / freq
        if duration >= secs:
            long_gaps.append((start, end, duration))

    # Unir tramos cercanos (separados por menos de `union` muestras)
    merged_gaps = []
    if long_gaps:
        merged_start, merged_end, _ = long_gaps[0]
        for i in range(1, len(long_gaps)):
            start, end, duration = long_gaps[i]
            if start - merged_end <= union:
                # Unir con el tramo anterior
                merged_end = end
            else:
                # Registrar tramo anterior y comenzar uno nuevo
                merged_duration = (merged_end - merged_start + 1) / freq
                merged_gaps.append((merged_start, merged_end, merged_duration))
                merged_start, merged_end = start, end
        # Añadir el último tramo unido
        merged_duration = (merged_end - merged_start + 1) / freq
        merged_gaps.append((merged_start, merged_end, merged_duration))

    # Convertir a tiempos y calcular porcentajes
    final_gaps = []
    total_percentage = 0.0
    for start_idx, end_idx, duration in merged_gaps:
        start_time = start_idx / freq
        end_time = end_idx / freq
        percentage = (duration / total_duration) * 100.0
        total_percentage += percentage
        final_gaps.append((start_time, end_time, duration, percentage))

    return final_gaps, total_percentage

def trim_signals(fhr, uc, fs, fhr_gaps, uc_gaps):
    """
    Recorta las señales FHR y UC para sincronizarlas, removiendo inicios o finales planos.

    Se eliminan porciones iniciales o finales de las señales si corresponden a tramos constantes detectados.

    Parámetros:
    - fhr (array-like): Señal FHR.
    - uc (array-like): Señal UC.
    - fs (float): Frecuencia de muestreo en Hz.
    - fhr_gaps (list): Lista de tramos constantes en FHR (tuplas con tiempos de inicio/fin).
    - uc_gaps (list): Lista de tramos constantes en UC.

    Retorna:
    - trimmed_fhr (numpy.ndarray): Señal FHR recortada.
    - trimmed_uc (numpy.ndarray): Señal UC recortada.
    """
    fhr = np.array(fhr)
    uc = np.array(uc)
    # Convertir tiempos de gaps a índices de muestra
    fhr_gaps_idx = [(int(start_time * fs), int(end_time * fs)) for start_time, end_time, _, _ in fhr_gaps]
    uc_gaps_idx = [(int(start_time * fs), int(end_time * fs)) for start_time, end_time, _, _ in uc_gaps]

    # Calcular recorte inicial y final según gaps al inicio o al final de cada señal
    start_cut = max(
        fhr_gaps_idx[0][1] + 1 if fhr_gaps_idx and fhr_gaps_idx[0][0] == 0 else 0,
        uc_gaps_idx[0][1] + 1 if uc_gaps_idx and uc_gaps_idx[0][0] == 0 else 0
    )
    end_cut = min(
        fhr_gaps_idx[-1][0] if fhr_gaps_idx and fhr_gaps_idx[-1][1] == len(fhr) - 1 else len(fhr),
        uc_gaps_idx[-1][0] if uc_gaps_idx and uc_gaps_idx[-1][1] == len(uc) - 1 else len(uc)
    )
    # Aplicar recorte
    trimmed_fhr = fhr[start_cut:end_cut]
    trimmed_uc = uc[start_cut:end_cut]
    return trimmed_fhr, trimmed_uc

def interpolate_zero_values(signal, tolerance=0.1, method='pchip'):
    """
    Reemplaza valores cercanos a cero en la señal por interpolaciones.

    Valores dentro de ±tolerance respecto a cero se consideran faltantes (NaN) y se interpolan 
    utilizando el método especificado.

    Parámetros:
    - signal (array-like): Señal de entrada.
    - tolerance (float): Umbral para considerar un valor como cero (default 0.1).
    - method (str): Método de interpolación ('pchip', 'linear', 'cubic', 'nearest').

    Retorna:
    - interpolated_signal (numpy.ndarray): Señal con valores cero interpolados.
    """
    signal = np.array(signal, dtype=float)
    # Identificar índices con valores ~0 y marcarlos como NaN
    missing_indices = np.where(np.abs(signal) <= tolerance)[0]
    signal[missing_indices] = np.nan

    # Si muy pocos datos válidos, no se puede interpolar
    valid_indices = np.where(~np.isnan(signal))[0]
    if len(valid_indices) < 2:
        raise ValueError("No hay suficientes datos válidos para interpolación.")

    # Elegir interpolador según método
    if method == 'pchip':
        interpolator = PchipInterpolator(valid_indices, signal[valid_indices])
    elif method == 'linear':
        interpolator = interp1d(valid_indices, signal[valid_indices], kind='linear', fill_value='extrapolate')
    elif method == 'cubic':
        interpolator = CubicSpline(valid_indices, signal[valid_indices])
    elif method == 'nearest':
        interpolator = interp1d(valid_indices, signal[valid_indices], kind='nearest', fill_value='extrapolate')
    else:
        raise ValueError("Método de interpolación no soportado. Use 'pchip', 'linear', 'cubic' o 'nearest'.")

    # Interpolar en los puntos marcados como faltantes
    interpolated_signal = signal.copy()
    interpolated_signal[missing_indices] = interpolator(missing_indices)
    # Asegurar que los valores interpolados sean positivos (si alguno quedó <= tolerance, elevarlos ligeramente)
    interpolated_signal[missing_indices][interpolated_signal[missing_indices] <= tolerance] = \
        np.nanmin(signal[valid_indices]) * 0.1

    return interpolated_signal

# ---------------------------- Funciones de Gráfica ---------------------------- #

def plot_ctg_signals(fhr, uc, fs, start_time=0, end_time=None):
    """
    Genera una visualización de las señales FHR y UC en función del tiempo.

    Parámetros:
    - fhr (numpy.ndarray): Señal de Fetal Heart Rate.
    - uc (numpy.ndarray): Señal de Uterine Contractions.
    - fs (float): Frecuencia de muestreo en Hz.
    - start_time (float): Tiempo de inicio en segundos para la visualización.
    - end_time (float): Tiempo de fin en segundos para la visualización (si None, hasta el final).
    
    Retorna:
    - fig (matplotlib.figure.Figure): Figura con dos subgráficos (FHR arriba, UC abajo).
    """
    # Determinar rango de índices a mostrar
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs) if end_time is not None else len(fhr)
    start_idx = max(0, start_idx)
    end_idx = min(len(fhr), end_idx)
    # Segmento de señal a mostrar
    fhr_segment = fhr[start_idx:end_idx]
    uc_segment = uc[start_idx:end_idx]
    # Vector de tiempo correspondiente
    duration = (end_idx - start_idx) / fs
    time = np.linspace(start_time, start_time + duration, len(fhr_segment))

    # Crear la figura con dos subplots compartiendo el eje X
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    # Gráfico de FHR
    ax1.plot(time, fhr_segment, label='FHR', color='blue')
    ax1.set_ylabel('FHR (BPM)')
    ax1.set_title('Fetal Heart Rate')
    ax1.legend()
    ax1.grid(True)
    # Gráfico de UC
    ax2.plot(time, uc_segment, label='UC', color='green')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('UC (au)')
    ax2.set_title('Uterine Contractions')
    ax2.legend()
    ax2.grid(True)
    fig.tight_layout()
    return fig

def plot_signal_with_gaps(signal, fs, long_gaps, total_percentage):
    """
    Genera una gráfica de una señal con tramos constantes destacados en color.

    Parámetros:
    - signal (numpy.ndarray): Señal de entrada.
    - fs (float): Frecuencia de muestreo en Hz.
    - long_gaps (list): Lista de tramos constantes (start_time, end_time, dur, %).
    - total_percentage (float): Porcentaje total del tiempo en tramos constantes.

    Retorna:
    - fig (matplotlib.figure.Figure): Figura con la señal y los tramos constantes sombreados.
    """
    time = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, signal, label="Signal", color="blue")
    # Sombrar los tramos de señal constante detectados
    for i, (start_time, end_time, _, _) in enumerate(long_gaps):
        ax.axvspan(start_time, end_time, color="red", alpha=0.5, label="Long Gap" if i == 0 else "")
    ax.set_xlabel("Tiempo (s)")
    ax.set_ylabel("Valor de la señal")
    ax.set_title(f"Señal con tramos constantes destacados (Total: {total_percentage:.2f}% del tiempo)")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig

def preprocess_ctg_pipeline(record_name, db_path, tolerance=1, interpolation_method='cubic'):
    """
    Pipeline completo de preprocesamiento de señales CTG:
      - Carga del registro.
      - Detección iterativa y recorte de tramos constantes en FHR y UC.
      - Interpolación de los valores cercanos a cero restantes.

    Parámetros:
    - record_name (str): Nombre del registro a procesar.
    - db_path (str): Ruta al directorio de la base de datos.
    - tolerance (float): Tolerancia para detectar derivada ~0 en detect_long_zero_gaps.
    - interpolation_method (str): Método de interpolación para valores cero ('cubic', 'pchip', etc.).

    Retorna:
    - fhr_clean (numpy.ndarray): Señal FHR final preprocesada.
    - uc_clean (numpy.ndarray): Señal UC final preprocesada.
    - fs (float): Frecuencia de muestreo.
    - metadata_df (pandas.DataFrame): DataFrame de metadatos del registro.
    """
    # Cargar señales originales
    fhr, uc, fs, metadata_df = load_ctg_data(record_name, db_path)

    # Recorte iterativo de tramos planos al inicio/final de las señales
    for _ in range(10):  # hasta 10 iteraciones máximo
        # Detectar tramos planos largos en cada señal
        fhr_gaps, fhr_perc = detect_long_zero_gaps(fhr, secs=30, freq=fs, union=3, tolerance=tolerance)
        uc_gaps, uc_perc = detect_long_zero_gaps(uc, secs=30, freq=fs, union=3, tolerance=tolerance)
        # Si la mayor parte de la señal es plana, abortar (posible registro corrupto o sin datos)
        if fhr_perc > 98 or uc_perc > 98:
            return None, None, None, None
        # Determinar si es necesario recortar extremos (si hay gaps al inicio o fin)
        trim_needed = False
        if (fhr_gaps and fhr_gaps[0][0] == 0) or (fhr_gaps and fhr_gaps[-1][1] >= len(fhr) / fs - 1):
            trim_needed = True
        if (uc_gaps and uc_gaps[0][0] == 0) or (uc_gaps and uc_gaps[-1][1] >= len(uc) / fs - 1):
            trim_needed = True
        if not trim_needed:
            break  # no hay recorte adicional necesario
        # Recortar señales al intervalo común válido
        fhr, uc = trim_signals(fhr, uc, fs, fhr_gaps, uc_gaps)

    # Interpolar valores cero en las señales finales
    fhr_clean = interpolate_zero_values(fhr, method=interpolation_method)
    uc_clean = interpolate_zero_values(uc, method=interpolation_method)
    return fhr_clean, uc_clean, fs, metadata_df
