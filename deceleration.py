import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from matplotlib.animation import FuncAnimation
import tempfile

# ------------------------------- Baseline FHR --------------------------------- #

class Baseline:
    def __init__(self, center, window_size=5.0):
        """
        Inicializa un objeto Baseline con un valor central y un ancho de banda.
        - center: valor central de la línea de base (BPM).
        - window_size: rango total (BPM) alrededor del centro para considerar "en banda".
        """
        self.center = center
        self.window_size = window_size
        self.lower_bound = center - window_size / 2
        self.upper_bound = center + window_size / 2

    def under_baseline(self, signal):
        """Array booleano indicando dónde la señal está por debajo de la línea base."""
        return signal < self.lower_bound

    def above_baseline(self, signal):
        """Array booleano indicando dónde la señal está por encima de la línea base."""
        return signal > self.upper_bound

    def in_baseline_band(self, signal):
        """Array booleano indicando dónde la señal está dentro del rango de la línea base."""
        return (signal >= self.lower_bound) & (signal <= self.upper_bound)

# ------------------------ Cálculo de baseline (media) ------------------------ #

def get_mean(signal, fs, start_time=0, end_time=None):
    """
    Calcula el valor medio de un segmento de la señal (usado como línea de base aproximada).
    
    Parámetros:
    - signal (numpy.ndarray): Señal de entrada.
    - fs (float): Frecuencia de muestreo en Hz.
    - start_time (float): Segundo de inicio del segmento (default 0).
    - end_time (float): Segundo de fin del segmento (default None, hasta el final).
    
    Retorna:
    - mean_value (float): Valor medio del segmento especificado.
    """
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs) if end_time is not None else len(signal)
    start_idx = max(0, start_idx)
    end_idx = min(len(signal), end_idx)
    segment = signal[start_idx:end_idx]
    mean_value = float(np.mean(segment))
    return mean_value

# ---------------------------- Deceleración FHR ----------------------------- #

class Deceleration:
    def __init__(self, start_idx, nadir_idx, end_idx, fs, nadir_value):
        """
        Representa una deceleración de FHR.
        - start_idx: Índice donde inicia la deceleración.
        - nadir_idx: Índice del nadir (mínimo) de la deceleración.
        - end_idx: Índice donde termina la deceleración.
        - fs: Frecuencia de muestreo (Hz).
        - nadir_value: Valor de FHR en el nadir.
        """
        self.start_idx = start_idx
        self.nadir_idx = nadir_idx
        self.end_idx = end_idx
        self.fs = fs
        self.nadir_value = nadir_value
        # Calcular tiempos en segundos
        self.start_time = self.start_idx / self.fs
        self.nadir_time = self.nadir_idx / self.fs
        self.end_time = self.end_idx / self.fs
        # Duración de la deceleración
        self.duration = self.end_time - self.start_time
        # Tipo de deceleración (clasificación)
        self.tipo = None

    def amplitude(self, baseline):
        """Amplitud de la deceleración: diferencia entre la línea base (upper_bound) y el valor nadir."""
        return baseline.upper_bound - self.nadir_value

    def check(self, baseline):
        """
        Verifica si la deceleración cumple criterios mínimos:
        amplitud > 15 BPM y duración > 15 segundos.
        """
        return (self.amplitude(baseline) > 15.0) and (self.duration > 15.0)

def find_deceleration(fhr, baseline, fs, start_idx=0):
    """
    Detecta la siguiente deceleración en la señal FHR a partir de un índice dado.

    Retorna un objeto Deceleration con los detalles, o None si no se encuentra otra deceleración.
    """
    # 1. Encontrar primer descenso por debajo de la línea base
    trigger_idx = None
    for i in range(start_idx, len(fhr) - 1):
        if (baseline.in_baseline_band(fhr[i]) or baseline.above_baseline(fhr[i])) and baseline.under_baseline(fhr[i+1]):
            trigger_idx = i
            break
    if trigger_idx is None:
        return None
    # 2. Encontrar punto donde la señal vuelve a subir a la línea base (fin de deceleración)
    end_idx = None
    for j in range(trigger_idx + 1, len(fhr)):
        if baseline.in_baseline_band(fhr[j]) or baseline.above_baseline(fhr[j]):
            end_idx = j
            break
    if end_idx is None:
        return None
    # 3. Encontrar el nadir (valor mínimo) entre trigger_idx y end_idx
    nadir_idx = int(np.argmin(fhr[trigger_idx:end_idx])) + trigger_idx
    nadir_value = fhr[nadir_idx]
    # Crear objeto Deceleration
    decel = Deceleration(trigger_idx, nadir_idx, end_idx, fs, nadir_value)
    return decel

def find_all_decelerations(fhr, baseline_obj, fs):
    """
    Identifica todas las deceleraciones en la señal FHR dada, usando una línea base Baseline.

    Retorna una lista de objetos Deceleration detectados (que cumplen criterios).
    """
    decelerations = []
    start_idx = 0
    while start_idx < len(fhr):
        decel = find_deceleration(fhr, baseline_obj, fs, start_idx)
        if decel is None:
            break
        if decel.check(baseline_obj):
            decelerations.append(decel)
        # Continuar buscando a partir del final de la deceleración encontrada
        start_idx = decel.end_idx
    return decelerations

# ---------------------------- Contracción UC ----------------------------- #

class Contraction:
    def __init__(self, start_idx, acme_idx, end_idx, fs, acme_value):
        """
        Representa una contracción detectada en la señal UC.
        - start_idx: Índice donde inicia la contracción.
        - acme_idx: Índice del acmé (pico) de la contracción.
        - end_idx: Índice donde termina la contracción.
        - fs: Frecuencia de muestreo (Hz).
        - acme_value: Valor UC en el acmé.
        """
        self.start_idx = start_idx
        self.acme_idx = acme_idx
        self.end_idx = end_idx
        self.fs = fs
        self.acme_value = acme_value
        # Tiempos en segundos
        self.start_time = self.start_idx / self.fs
        self.acme_time = self.acme_idx / self.fs
        self.end_time = self.end_idx / self.fs
        # Duración de la contracción
        self.duration = self.end_time - self.start_time

    def check(self):
        """Verifica si la contracción dura entre 45 y 120 segundos."""
        return 45 < self.duration < 120

def find_all_contractions(uc, fs, mode='optionN', cutoff=0.1, min_derivative_threshold=0.00005,
                          min_interval_sec=45, max_time_diff_sec=180, max_amp_diff=20, pairing_mode='furthest'):
    """
    Detecta todas las contracciones en la señal UC usando criterios configurables.

    Retorna:
    - final_contractions: Lista de objetos Contraction detectados.
    - smoothed_uc: Señal UC suavizada utilizada para la detección.
    """
    # Configuración según modo
    if mode == 'optionN':
        cutoff = 0.05   # Suavizado más fuerte
        max_amp_diff = None  # No filtrar por amplitud
        pairing_mode = 'nearest'
    elif mode == 'optionF':
        cutoff = 0.1    # Suavizado más ligero
        max_amp_diff = 20
        pairing_mode = 'furthest'

    # Suavizar señal UC con filtro pasa-bajos
    def lowpass_filter(signal, cutoff_freq, fs_rate, order=4):
        nyq = 0.5 * fs_rate
        normal_cutoff = cutoff_freq / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)
    smoothed_uc = lowpass_filter(uc, cutoff_freq=cutoff, fs_rate=fs)

    # Calcular derivada y detectar cruces por cero ascendentes significativos
    derivative = np.diff(smoothed_uc, append=smoothed_uc[-1])
    zero_crossings = np.where((derivative[:-1] < 0) & (derivative[1:] > 0))[0]
    valid_crossings = [idx for idx in zero_crossings if derivative[idx+1] > min_derivative_threshold]

    # Unir candidatos de inicio cercanos (separación mínima)
    min_interval_samples = int(min_interval_sec * fs)
    merged_starts = []
    last_keep = -np.inf
    for idx in valid_crossings:
        if idx - last_keep > min_interval_samples:
            merged_starts.append(idx)
            last_keep = idx
    merged_starts = np.array(merged_starts)

    # Emparejar posibles inicios con sus finales (según nearest/furthest)
    max_time_diff = int(max_time_diff_sec * fs)
    pairs = []
    for i, idx_i in enumerate(merged_starts):
        amp_i = smoothed_uc[idx_i]
        best_j = None
        for j in range(i+1, len(merged_starts)):
            idx_j = merged_starts[j]
            if idx_j - idx_i > max_time_diff:
                break
            amp_j = smoothed_uc[idx_j]
            if max_amp_diff is None or abs(amp_j - amp_i) <= max_amp_diff:
                best_j = idx_j
                if pairing_mode == 'nearest':
                    break
        if best_j is not None:
            pairs.append((idx_i, best_j))

    # Crear objetos Contraction a partir de los pares válidos
    contractions = []
    for start_idx, end_idx in pairs:
        if start_idx > end_idx:
            start_idx, end_idx = end_idx, start_idx
        segment = smoothed_uc[start_idx:end_idx+1]
        acme_rel_idx = int(np.argmax(segment))
        acme_idx = start_idx + acme_rel_idx
        acme_value = smoothed_uc[acme_idx]
        start_value = smoothed_uc[start_idx]
        end_value = smoothed_uc[end_idx]
        # Aceptar solo contracciones con pico claro (diferencia >= 2 unidades)
        if abs(acme_value - start_value) >= 2 and abs(acme_value - end_value) >= 2:
            contraction = Contraction(start_idx, acme_idx, end_idx, fs, acme_value)
            contractions.append(contraction)

    # Filtrar contracciones superpuestas (mantener las de mayor duración)
    contractions.sort(key=lambda c: c.duration, reverse=True)
    final_contractions = []
    used_intervals = []
    for contr in contractions:
        if all(contr.end_idx <= u_start or contr.start_idx >= u_end for (u_start, u_end) in used_intervals):
            final_contractions.append(contr)
            used_intervals.append((contr.start_idx, contr.end_idx))

    return final_contractions, smoothed_uc

# ----------------------- Emparejar y clasificar ------------------------ #

def emparejar(decelerations, contractions, tolerance=2.0):
    """
    Empareja cada deceleración con la contracción más cercana que ocurre antes o simultáneamente.
    Retorna una lista de tuplas (Deceleration, Contraction) emparejadas.
    """
    paired_events = []
    for decel in decelerations:
        # Contracciones que comienzan antes de cierto límite posterior al inicio de la deceleración
        valid_contrs = [contr for contr in contractions if contr.start_time <= decel.start_time + tolerance]
        if valid_contrs:
            # Elegir la contracción cuya acmé esté más cerca en el tiempo del nadir de la deceleración
            nearest_contr = min(valid_contrs, key=lambda contr: abs(decel.nadir_time - contr.acme_time))
            paired_events.append((decel, nearest_contr))
    return paired_events

def clasificar_dec(paired_events, diff=1.0, form_criteria=False):
    """
    Clasifica las deceleraciones emparejadas en early, late o variable.
    Criterios:
      - Early: |nadir_time - acme_time| <= diff (por defecto 1s).
      - Late: diferencia de tiempo > 20s (o criterios de forma adicionales si form_criteria es True).
      - Variable: el resto de casos por defecto.
    
    Retorna tuplas de listas: (early_decs, late_decs, variable_decs).
    """
    early_decs = []
    late_decs = []
    variable_decs = []
    for decel, contr in paired_events:
        time_diff = decel.nadir_time - contr.acme_time
        start_diff = abs(decel.start_time - decel.nadir_time)
        end_diff = abs(decel.end_time - decel.nadir_time)
        if abs(time_diff) <= diff:
            decel.tipo = "Early"
            early_decs.append(decel)
        elif abs(time_diff) > 20:
            decel.tipo = "Late"
            late_decs.append(decel)
        elif form_criteria and (start_diff > 30 or end_diff > 30):
            decel.tipo = "Late"
            late_decs.append(decel)
        else:
            decel.tipo = "Variable"
            variable_decs.append(decel)
    return early_decs, late_decs, variable_decs

# --------------------------- Funciones de Gráfica --------------------------- #

def plot_fhr_with_decelerations(fhr, fs, baseline_obj, decelerations):
    """
    Genera una gráfica de la señal FHR con la línea de base y deceleraciones destacadas.
    
    Parámetros:
    - fhr (numpy.ndarray): Señal FHR.
    - fs (float): Frecuencia de muestreo en Hz.
    - baseline_obj (Baseline): Objeto Baseline con la línea base de FHR.
    - decelerations (list): Lista de objetos Deceleration detectados.
    
    Retorna:
    - fig (matplotlib.figure.Figure): Figura de la señal FHR con deceleraciones.
    """
    time = np.arange(len(fhr)) / fs
    fig, ax = plt.subplots(figsize=(12, 5))
    # Señal FHR
    ax.plot(time, fhr, label='FHR', color='blue')
    # Línea base y banda
    ax.axhline(y=baseline_obj.center, color='green', linestyle='--', label=f'Baseline {baseline_obj.center:.0f} BPM')
    ax.fill_between(time, baseline_obj.center - baseline_obj.window_size/2,
                    baseline_obj.center + baseline_obj.window_size/2,
                    color='green', alpha=0.2, label=f'Baseline ±{baseline_obj.window_size/2:.1f} BPM')
    # Deceleraciones sombreadas y puntos clave
    for decel in decelerations:
        ax.fill_between(time[decel.start_idx:decel.end_idx], fhr[decel.start_idx:decel.end_idx],
                        baseline_obj.center, color='red', alpha=0.3)
        ax.plot([decel.start_time, decel.nadir_time, decel.end_time],
                [baseline_obj.center, decel.nadir_value, baseline_obj.center], 'ro')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('FHR (BPM)')
    ax.set_title('FHR con deceleraciones')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_cu_with_contractions(cu, fs, contractions):
    """
    Genera una gráfica de la señal UC destacando las contracciones detectadas.

    Parámetros:
    - cu (numpy.ndarray): Señal UC.
    - fs (float): Frecuencia de muestreo en Hz.
    - contractions (list): Lista de objetos Contraction detectados.

    Retorna:
    - fig (matplotlib.figure.Figure): Figura de la señal UC con contracciones.
    """
    time = np.arange(len(cu)) / fs
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(time, cu, label='UC Signal', color='blue')
    # Sombrar regiones de contracción
    for i, contr in enumerate(contractions):
        ax.fill_between(time[contr.start_idx:contr.end_idx+1], cu[contr.start_idx:contr.end_idx+1],
                        color='red', alpha=0.3, label='Contracción' if i == 0 else "")
        ax.plot([contr.start_time, contr.acme_time, contr.end_time],
                [cu[contr.start_idx], cu[contr.acme_idx], cu[contr.end_idx]], 'ro')
    ax.set_xlabel('Tiempo (s)')
    ax.set_ylabel('Unidad UC')
    ax.set_title('UC con contracciones detectadas')
    ax.legend(loc='upper right')
    ax.grid(True)
    fig.tight_layout()
    return fig

def plot_decc_contr(fhr, cu, fs, decelerations, baseline_obj, contractions):
    """
    Genera una visualización combinada de FHR y UC mostrando deceleraciones y contracciones.

    Parámetros:
    - fhr (numpy.ndarray): Señal FHR.
    - cu (numpy.ndarray): Señal UC.
    - fs (float): Frecuencia de muestreo en Hz.
    - decelerations (list): Lista de Deceleration detectadas en FHR.
    - baseline_obj (Baseline): Objeto Baseline de FHR.
    - contractions (list): Lista de Contraction detectadas en UC.

    Retorna:
    - fig (matplotlib.figure.Figure): Figura con dos subplots (FHR arriba, UC abajo).
    """
    time_fhr = np.arange(len(fhr)) / fs
    time_cu = np.arange(len(cu)) / fs
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    # Subplot FHR con deceleraciones
    ax1.plot(time_fhr, fhr, label='FHR', color='blue', alpha=0.8)
    ax1.axhline(y=baseline_obj.center, color='green', linestyle='--', label='Baseline FHR')
    ax1.fill_between(time_fhr, baseline_obj.center - baseline_obj.window_size/2,
                     baseline_obj.center + baseline_obj.window_size/2,
                     color='green', alpha=0.2, label='Baseline band')
    for decel in decelerations:
        ax1.fill_between(time_fhr[decel.start_idx:decel.end_idx], fhr[decel.start_idx:decel.end_idx],
                         baseline_obj.center, color='red', alpha=0.3)
        ax1.plot([decel.start_time, decel.nadir_time, decel.end_time],
                 [baseline_obj.center, decel.nadir_value, baseline_obj.center], 'ro')
    ax1.set_ylabel('FHR (BPM)')
    ax1.set_title('FHR con deceleraciones')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    # Subplot UC con contracciones
    ax2.plot(time_cu, cu, label='UC', color='blue', alpha=0.8)
    for i, contr in enumerate(contractions):
        ax2.fill_between(time_cu[contr.start_idx:contr.end_idx+1], cu[contr.start_idx:contr.end_idx+1],
                         color='red', alpha=0.3, label='Contracción' if i == 0 else "")
        ax2.plot([contr.start_time, contr.acme_time, contr.end_time],
                 [cu[contr.start_idx], cu[contr.acme_idx], cu[contr.end_idx]], 'ro')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Unidad UC')
    ax2.set_title('UC con contracciones')
    ax2.legend(loc='upper right')
    ax2.grid(True)
    fig.tight_layout()
    return fig

def animate_paired_events(fhr, cu, fs, baseline_obj, decelerations, contractions, paired_events, dec_type="all", out_path=None):
    """
    Genera una animación de las señales FHR y UC mostrando secuencialmente cada par deceleración-contracción emparejado.
    Si no se especifica out_path, la animación se guarda en un archivo temporal y se retorna su ruta.
    
    Parámetros:
    - fhr, cu (numpy.ndarray): Señales FHR y UC.
    - fs (float): Frecuencia de muestreo.
    - baseline_obj (Baseline): Objeto Baseline para FHR.
    - decelerations (list): Lista de todas las Deceleration detectadas.
    - contractions (list): Lista de todas las Contraction detectadas.
    - paired_events (list): Lista de tuplas (Deceleration, Contraction) emparejadas.
    - dec_type (str): Tipo de deceleraciones a animar ("all", "early", "late", "variable").
    - out_path (str): Ruta de archivo para guardar la animación (GIF). Si None, usa archivo temporal.

    Retorna:
    - path (str): Ruta al archivo GIF generado con la animación.
    """
    # Filtrar eventos emparejados según tipo solicitado
    if dec_type.lower() != "all":
        paired_events = [(dec, contr) for dec, contr in paired_events if dec.tipo and dec.tipo.lower() == dec_type.lower()]
    if not paired_events:
        return None  # No hay eventos a animar para el tipo dado

    time_fhr = np.arange(len(fhr)) / fs
    time_cu = np.arange(len(cu)) / fs

    # Preparar figura con dos subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    # Fondo: señal completa con deceleraciones y contracciones marcadas sutilmente
    ax1.plot(time_fhr, fhr, color='blue', alpha=0.7)
    ax1.axhline(y=baseline_obj.center, color='green', linestyle='--', label='Baseline FHR')
    for decel in decelerations:
        ax1.fill_between(time_fhr[decel.start_idx:decel.end_idx], fhr[decel.start_idx:decel.end_idx],
                         baseline_obj.center, color='red', alpha=0.1)
    ax1.set_ylabel('FHR (BPM)')
    ax1.set_title('Animación de deceleraciones (rojo) y contracciones (verde)')
    ax1.legend(loc='upper right')
    ax1.grid(True)
    ax2.plot(time_cu, cu, color='blue', alpha=0.7)
    for contr in contractions:
        ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], cu[contr.start_idx:contr.end_idx],
                         color='red', alpha=0.1)
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('UC (au)')
    ax2.grid(True)

    # Elementos variables (parpadeantes) en la animación
    decel_patch = ax1.fill_between([], [], [], color='red', alpha=0.5)  # parche para deceleración actual
    decel_points, = ax1.plot([], [], 'ro')  # puntos de inicio, nadir, fin de deceleración
    contr_patch = ax2.fill_between([], [], [], color='red', alpha=0.5)  # parche para contracción actual
    contr_points, = ax2.plot([], [], 'ro')  # puntos de inicio, acmé, fin de contracción
    # Mantener referencias a parches de contracciones ya mostradas para recolorearlos
    contraction_patches = {}

    # Dibujar inicialmente todas las contracciones emparejadas como verde pálido (ya "procesadas")
    for contr in contractions:
        contraction_patches[contr] = ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], 
                                                     cu[contr.start_idx:contr.end_idx], 
                                                     color='palegreen', alpha=0.3)

    # Función de actualización por frame
    def update(frame):
        decel, contr = paired_events[frame]
        # Actualizar parche de deceleración actual (rojo opaco)
        ax1.collections.remove(decel_patch)  # remover parche previo
        new_patch = ax1.fill_between(time_fhr[decel.start_idx:decel.end_idx], 
                                     fhr[decel.start_idx:decel.end_idx], baseline_obj.center,
                                     color='red', alpha=0.5)
        # Actualizar puntos de deceleración (inicio, nadir, fin)
        decel_points.set_data([decel.start_time, decel.nadir_time, decel.end_time],
                               [baseline_obj.center, decel.nadir_value, baseline_obj.center])
        # Actualizar parche de contracción actual (rojo opaco)
        ax2.collections.remove(contr_patch)
        new_patch_contr = ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], 
                                          cu[contr.start_idx:contr.end_idx], color='red', alpha=0.5)
        # Actualizar puntos de contracción (inicio, acmé, fin)
        contr_points.set_data([contr.start_time, contr.acme_time, contr.end_time],
                               [cu[contr.start_idx], cu[contr.acme_idx], cu[contr.end_idx]])
        # Después de mostrar el par actual, marcar la contracción como procesada (verde)
        if contr in contraction_patches:
            ax2.collections.remove(contraction_patches[contr])
        contraction_patches[contr] = ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], 
                                                     cu[contr.start_idx:contr.end_idx], 
                                                     color='palegreen', alpha=0.5)
        # Devolver artistas actualizados
        return new_patch, decel_points, new_patch_contr, contr_points

    # Crear animación
    ani = FuncAnimation(fig, update, frames=len(paired_events), interval=1000, blit=False, repeat=True)
    plt.close(fig)  # Cerrar la figura para que no se muestre estática

    # Guardar animación como GIF
    if out_path is None:
        tmpfile = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
        ani.save(tmpfile.name, writer='pillow')
        return tmpfile.name
    else:
        ani.save(out_path, writer='pillow')
        return out_path

# ----------------------------- Funciones Pipeline ----------------------------- #

def get_decelerations(fhr, fs):
    """
    Obtiene todas las deceleraciones de FHR y la línea base calculada.

    Retorna:
    - decelerations: Lista de objetos Deceleration detectados.
    - baseline_obj: Objeto Baseline utilizado como línea base.
    """
    baseline_value = get_mean(fhr, fs)
    baseline_obj = Baseline(baseline_value)
    decelerations = find_all_decelerations(fhr, baseline_obj, fs)
    return decelerations, baseline_obj

def get_contractions(uc, fs):
    """
    Obtiene todas las contracciones detectadas en la señal UC.

    Retorna:
    - contractions: Lista de objetos Contraction detectados.
    """
    contractions, _ = find_all_contractions(uc, fs)
    return contractions

def get_classified_decelerations(fhr, uc, fs):
    """
    Detecta deceleraciones y contracciones, las empareja y clasifica las deceleraciones.

    Retorna:
    - early_decs: Lista de deceleraciones tempranas (early).
    - late_decs: Lista de deceleraciones tardías (late).
    - variable_decs: Lista de deceleraciones variables.
    - decelerations: Lista completa de deceleraciones detectadas.
    - contractions: Lista completa de contracciones detectadas.
    - paired_events: Lista de pares (Deceleration, Contraction) emparejados.
    - baseline_obj: Objeto Baseline utilizado para FHR.
    """
    decelerations, baseline_obj = get_decelerations(fhr, fs)
    contractions = get_contractions(uc, fs)
    paired_events = emparejar(decelerations, contractions, tolerance=3.0)
    early_decs, late_decs, variable_decs = clasificar_dec(paired_events, diff=5.0, form_criteria=True)
    return early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, baseline_obj
