import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import mean
from scipy.signal import butter, filtfilt
from scipy.ndimage import uniform_filter1d

from matplotlib.animation import FuncAnimation
import streamlit as st
import tempfile
import os

def get_mean(signal, fs, start_time=0, end_time=None, plot = False):
    """
    Returns mean value. Used for getting the baseline of FHR.

    Parameters:
    - signal (numpy.ndarray): The input signal.
    - fs (float): Sampling frequency in Hz.
    - start_time (float, optional): Start time for the plot in seconds (default: 0).
    - end_time (float, optional): End time for the plot in seconds (default: full signal length).
    - plot (bool): Plot the signal with its mean if True (default False).

    Returns:
    - mean_value (float): The computed mean of the plotted segment.

    """

    # Convert times to sample indices
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs) if end_time else len(signal)

    # Ensure indices are within valid range
    start_idx = max(0, start_idx)  # Prevent negative indices
    end_idx = min(len(signal), end_idx)  # Prevent exceeding signal length

    # Extract the segment of the signal to be plotted
    signal_segment = signal[start_idx:end_idx]
    time = np.linspace(start_time, start_time + (end_idx - start_idx) / fs, len(signal_segment))

    # Compute the mean value of the segment
    mean_value = np.mean(signal_segment)

    return mean_value
    

#----------------------------------------- DECELERATION ------------------------------------
class Deceleration:
    def __init__(self, start_idx, nadir_idx, end_idx, fs, nadir_value):
        """
        Initialize a Deceleration instance.

        Parameters:
        - start_idx: Index where the deceleration starts.
        - nadir_idx: Index where the nadir occurs.
        - end_idx: Index where the deceleration ends.
        - fs: Sampling frequency (samples per second).
        - nadir_value: FHR value at the nadir.
        """
        self.start_idx = start_idx
        self.nadir_idx = nadir_idx
        self.end_idx = end_idx
        self.fs = fs
        self.nadir_value = nadir_value
        
        # Calculate the times based on the indices and fs
        self.start_time = self.start_idx / self.fs
        self.nadir_time = self.nadir_idx / self.fs
        self.end_time = self.end_idx / self.fs
        
        # Save the duration 
        self.duration = self.end_time - self.start_time

        # For classification
        self.tipo = None

    def amplitude(self, baseline):
        """
        Calculate the amplitude of the deceleration.
        
        Amplitude is the difference between the baseline and the nadir value.

        Parameters:
        - baseline: The baseline value of the FHR.
        
        Returns:
        - Amplitude: The difference between the baseline and nadir value.
        """
        return baseline.upper_bound - self.nadir_value

    def check(self, baseline):
        """
        Check if the deceleration meets the criteria of amplitude > 15 bpm and duration > 15 seconds.

        Parameters:
        - baseline: The baseline value of the FHR.
        
        Returns:
        - True if amplitude > 15 and duration > 15, else False.
        """
        return (self.amplitude(baseline) > 15.0) and (self.duration > 15.0)

def find_all_decelerations(fhr, fs, verbose=False):
    window_size = int(10 * 60 * fs)  # 10 minutes in samples
    if window_size % 2 == 0:
        window_size += 1  # ensure odd window for centered smoothing
    
    # Compute moving average (baseline)
    baseline = uniform_filter1d(fhr, size=window_size, mode='nearest')

    below_baseline = fhr < baseline
    decelerations = []

    in_decel = False
    start_idx = None

    for i in range(len(fhr)):
        if below_baseline[i] and not in_decel:
            start_idx = i
            in_decel = True
        elif not below_baseline[i] and in_decel:
            end_idx = i - 1
            segment = fhr[start_idx:end_idx+1]
            nadir_idx_rel = np.argmin(segment)
            nadir_idx = start_idx + nadir_idx_rel
            nadir_value = fhr[nadir_idx]

            decel = Deceleration(start_idx, nadir_idx, end_idx, fs, nadir_value)

            if decel.check(baseline[nadir_idx]):
                decelerations.append(decel)
                if verbose:
                    print(f"Deceleration from {decel.start_time:.1f}s to {decel.end_time:.1f}s, amplitude: {decel.amplitude(baseline[nadir_idx]):.1f}")
            in_decel = False

    # Handle edge case: if signal ends in a deceleration
    if in_decel:
        end_idx = len(fhr) - 1
        segment = fhr[start_idx:end_idx+1]
        nadir_idx_rel = np.argmin(segment)
        nadir_idx = start_idx + nadir_idx_rel
        nadir_value = fhr[nadir_idx]

        decel = Deceleration(start_idx, nadir_idx, end_idx, fs, nadir_value)
        if decel.check(baseline[nadir_idx]):
            decelerations.append(decel)
            if verbose:
                print(f"Deceleration from {decel.start_time:.1f}s to {decel.end_time:.1f}s, amplitude: {decel.amplitude(baseline[nadir_idx]):.1f}")

    if verbose:
        # Print the details of each valid deceleration
        for i, decel in enumerate(decelerations):
            print(f"Deceleration {i+1}:")
            print(f"  Start time: {decel.start_time:.2f} seconds")
            print(f"  Nadir time: {decel.nadir_time:.2f} seconds")
            print(f"  End time: {decel.end_time:.2f} seconds")
            print(f"  Nadir value: {decel.nadir_value:.2f} BPM")
            print(f"  Amplitude: {decel.amplitude(baseline[decel.nadir_idx]):.2f} BPM")
            print(f"  Duration: {decel.duration:.2f} seconds\n")
            
    return decelerations    

#----------------------------------------- CONTRACTION -------------------------------------
class Contraction:
    def __init__(self, start_idx, acme_idx, end_idx, fs, acme_value):
        """
        Initialize a Contraction instance.

        Parameters:
        - start_idx: Index where the deceleration starts.
        - acme_idx: Index where the acme occurs.
        - end_idx: Index where the deceleration ends.
        - fs: Sampling frequency (samples per second).
        - acme_value: FHR value at the acme.
        """
        self.start_idx = start_idx
        self.acme_idx = acme_idx
        self.end_idx = end_idx
        self.fs = fs
        self.acme_value = acme_value
        
        # Calculate the times based on the indices and fs
        self.start_time = self.start_idx / self.fs
        self.acme_time = self.acme_idx / self.fs
        self.end_time = self.end_idx / self.fs
        
        # Save the duration based on the times
        self.duration = self.end_time - self.start_time

    def check(self):
        """
        Check if the contraction meets the criteria of duration > 45 seconds and duration < 120 seconds.

        Returns:
        - True if duration > 45 and duration < 120, else False.
        """
        return 45 < self.duration #< 120

def find_all_contractions(uc, fs, mode='optionN',  # 'optionN' or 'optionF'
                         cutoff=0.1, min_derivative_threshold=0.00005,
                         min_interval_sec=45, max_time_diff_sec=180,
                         max_amp_diff=20, pairing_mode='furthest'):
    """
    Detect uterine contractions from the UC signal using configurable pairing and smoothing strategies.

    Parameters:
    - uc: Raw UC signal (numpy array)
    - fs: Sampling frequency (Hz)
    - mode: Detection mode ('optionN' or 'optionF')
        * 'optionN' = nearest pairing, no amplitude filtering, stronger smoothing
        * 'optionF' = furthest pairing, amplitude filtering, lighter smoothing
    - cutoff: Low-pass filter cutoff frequency (Hz)
    - min_derivative_threshold: Minimum derivative to detect valid upward zero-crossings
    - min_interval_sec: Minimum interval between detected start points (seconds)
    - max_time_diff_sec: Max time difference allowed between paired points (seconds)
    - max_amp_diff: Max amplitude difference between paired start/end points
    - pairing_mode: Pairing logic ('nearest' or 'furthest') [overridden by `mode`]

    Returns:
    - final_contractions: List of Contraction objects
    - smoothed_uc: Filtered UC signal
    """

    # --- Mode-based configuration ---
    if mode == 'optionN':
        cutoff = 0.05  # Stronger smoothing
        max_amp_diff = None  # Disable amplitude filtering for pairing
        pairing_mode = 'nearest'
    elif mode == 'optionF':
        cutoff = 0.1  # Lighter smoothing
        max_amp_diff = 20  # Require amplitude similarity for pairing
        pairing_mode = 'furthest'

    # --- Apply low-pass filter to smooth the signal ---
    def lowpass_filter(signal, cutoff, fs, order=4):
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return filtfilt(b, a, signal)

    smoothed_uc = lowpass_filter(uc, cutoff=cutoff, fs=fs)

    # --- Compute signal derivative and find valid zero-crossings (upward slope) ---
    derivative = np.diff(smoothed_uc, append=smoothed_uc[-1])
    zero_crossings = np.where((derivative[:-1] < 0) & (derivative[1:] > 0))[0]
    valid_crossings = zero_crossings[derivative[zero_crossings + 1] > min_derivative_threshold]

    # --- Merge close start candidates based on min interval ---
    min_interval_samples = int(min_interval_sec * fs)
    merged_starts = []
    last_kept = -np.inf
    for idx in valid_crossings:
        if idx - last_kept > min_interval_samples:
            merged_starts.append(idx)
            last_kept = idx
    merged_starts = np.array(merged_starts)

    # --- Pair points using 'nearest' or 'furthest' logic within time/amplitude constraints ---
    max_time_diff = int(max_time_diff_sec * fs)
    all_pairs = []

    for i, idx_i in enumerate(merged_starts):
        amp_i = smoothed_uc[idx_i]
        best_j = None

        for j in range(i + 1, len(merged_starts)):
            idx_j = merged_starts[j]
            if idx_j - idx_i > max_time_diff:
                break
            amp_j = smoothed_uc[idx_j]

            # Pair if amplitude is within bounds or if not constrained
            if max_amp_diff is None or abs(amp_j - amp_i) <= max_amp_diff:
                best_j = idx_j
                if pairing_mode == 'nearest':
                    break  # For nearest mode, take the first match

        if best_j is not None:
            all_pairs.append((idx_i, best_j))

    # --- Build Contraction objects from valid pairs ---
    contractions = []
    for idx1, idx2 in all_pairs:
        start_idx = min(idx1, idx2)
        end_idx = max(idx1, idx2)
        segment = smoothed_uc[start_idx:end_idx + 1]
        acme_rel_idx = np.argmax(segment)
        acme_idx = start_idx + acme_rel_idx
        acme_value = smoothed_uc[acme_idx]
        start_value = smoothed_uc[start_idx]
        end_value = smoothed_uc[end_idx]

        # Only accept contractions with a clear amplitude peak (≥ 2 units)
        if abs(acme_value - start_value) >= 2 and abs(acme_value - end_value) >= 2:
            contraction = Contraction(start_idx, acme_idx, end_idx, fs, acme_value)
            contractions.append(contraction)

    # --- Filter out overlapping contractions (keep longest ones first) ---
    contractions_sorted = sorted(contractions, key=lambda c: c.duration, reverse=True)
    final_contractions = []
    used_intervals = []
    for c in contractions_sorted:
        if all(c.end_idx <= u_start or c.start_idx >= u_end for u_start, u_end in used_intervals):
            final_contractions.append(c)
            used_intervals.append((c.start_idx, c.end_idx))

    return final_contractions

#----------------------------------------- GET PAIRS AND CLASIFY ---------------------------------------------
def emparejar(decelerations, contractions, tolerance=2.0):
    """
    Pair each deceleration with the nearest contraction that occurs before or at the same time.

    Parameters:
    - decelerations: List of Deceleration objects.
    - contractions: List of Contraction objects.
    - tolerance: Maximum allowed time difference for pairing.

    Returns:
    - List of tuples (Deceleration, Contraction).
    """
    paired_events = []
    
    for decel in decelerations:
        valid_contractions = [contr for contr in contractions if contr.start_time <= decel.start_time + tolerance]
        if valid_contractions:
            nearest_contraction = min(valid_contractions, key=lambda contr: abs(decel.nadir_time - contr.acme_time))
            paired_events.append((decel, nearest_contraction))
    
    return paired_events


def clasificar_dec(paired_events, diff=1.0, form_criteria=False, verbose = False):
    """
    Classifies decelerations as early, late or variable based on their relationship with uterine contractions.

    Parameters:
    - paired_events: List of tuples (Deceleration, Contraction).
    - diff: Allowed time difference (in seconds) for early deceleration classification.
    - form_criteria: If True, use additional criteria to classify some Variable decelerations as Late.

    Returns:
    - early_decelerations: List of early decelerations.
    - late_decelerations: List of late decelerations.
    - variable_decelerations: List of variable decelerations.
    """
    
    early_decelerations = []
    late_decelerations = []
    variable_decelerations = []
    
    # Classify each deceleration based on timing
    for decel, contr in paired_events:
        time_difference = decel.nadir_time - contr.acme_time  # Time difference in seconds
        start_diff = abs(decel.start_time - decel.nadir_time)  # Time difference between start and nadir
        end_diff = abs(decel.end_time - decel.nadir_time)  # Time difference between end and nadir

        # Early Deceleration
        if abs(time_difference) <= diff:
            decel.tipo = "Early"
            early_decelerations.append(decel)
        
        # Late Deceleration (if nadir is more than 20 sec from contraction acme)
        elif abs(time_difference) > 20:
            decel.tipo = "Late"
            late_decelerations.append(decel)
        
        # Check additional criteria if form_criteria is True
        elif form_criteria and (start_diff > 30 or end_diff > 30):
            decel.tipo = "Late"
            late_decelerations.append(decel)
        
        # Variable Deceleration (default case)
        else:
            decel.tipo = "Variable"
            variable_decelerations.append(decel)
    
    if verbose:
        # Print summary
        print("\nDeceleration Classification Summary:")
        print(f"  - Early Decelerations: {len(early_decelerations)}")
        print(f"  - Late Decelerations: {len(late_decelerations)}")
        print(f"  - Variable Decelerations: {len(variable_decelerations)}")
    
    return early_decelerations, late_decelerations, variable_decelerations

#----------------------------------------- PLOT ---------------------------------------------


def plot_fhr_with_decelerations(fhr, fs, decelerations):
    """
    Plot the FHR signal and highlight the deceleration areas with a different color.

    Parameters:
    - fhr: The FHR signal (numpy array).
    - fs: The sampling frequency (int).
    - baseline: The baseline FHR value (Baseline object).
    - decelerations: A list of Deceleration objects.
    """
    time = np.arange(len(fhr)) / fs

    # Calculate baseline (10-min moving average)
    window_size = int(10 * 60 * fs)
    if window_size % 2 == 0:
        window_size += 1
    baseline = uniform_filter1d(fhr, size=window_size, mode='nearest')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, fhr, label='Fetal Heart Rate (FHR)', color='blue')
    ax.plot(time, baseline, label='Baseline (Moving Avg)', color='green', linestyle='--')
    
    for decel in decelerations:
        t_segment = time[decel.start_idx:decel.end_idx+1]
        fhr_segment = fhr[decel.start_idx:decel.end_idx+1]
        baseline_segment = baseline[decel.start_idx:decel.end_idx+1]
        
        ax.fill_between(t_segment, fhr_segment, baseline_segment, 
                         where=fhr_segment < baseline_segment,
                         color='red', alpha=0.3)

        ax.plot(decel.start_time, fhr[decel.start_idx], 'ro', label='Start' if decel == decelerations[0] else "")
        ax.plot(decel.nadir_time, decel.nadir_value, 'ko', label='Nadir' if decel == decelerations[0] else "")
        ax.plot(decel.end_time, fhr[decel.end_idx], 'ro', label='End' if decel == decelerations[0] else "")

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Fetal Heart Rate (BPM)')
    ax.set_title('FHR with Decelerations')
    ax.legend(loc='upper right')
    ax.grid(True)
    return fig

def plot_cu_with_contractions(cu, fs, contractions):
    """
    Plot the CU signal and highlight the contraction areas with a different color.
    
    Parameters:
    - cu: The CU signal (numpy array).
    - fs: The sampling frequency (int).
    - contractions: A list of Contraction objects.
    """
    
    time = np.arange(len(cu)) / fs
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(time, cu, label='CU Signal', color='blue')

    for contr in contractions:
        ax.fill_between(time[contr.start_idx:contr.end_idx + 1], cu[contr.start_idx:contr.end_idx + 1], color='red', alpha=0.3)
        ax.plot([contr.start_time, contr.acme_time, contr.end_time], [cu[contr.start_idx], cu[contr.acme_idx], cu[contr.end_idx]], 'ro')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('CU Amplitude')
    ax.set_title('CU Signal with Contractions')
    ax.legend(loc='upper right')
    ax.grid(True)
    return fig

def plot_decc_contr(fhr, cu, fs, decelerations, contractions):
    fig1 = plot_fhr_with_decelerations(fhr, fs, decelerations)
    fig2 = plot_cu_with_contractions(cu, fs, contractions)
    return [fig1, fig2]

def animate_paired_events(fhr, cu, fs, baseline_fhr, decelerations, contractions, paired_events, dec_type="all"):
    """
    Animate FHR and CU signals in separate subplots, highlighting one paired deceleration-contraction at a time.
    Unpaired contractions remain unchanged, while paired contractions turn pale green over time.
    
    Parameters:
    - fhr: The FHR signal (numpy array).
    - cu: The CU signal (numpy array).
    - fs: Sampling frequency (int).
    - baseline_fhr: mean value of FHR.
    - decelerations: List of all detected Deceleration objects.
    - contractions: List of all detected Contraction objects.
    - paired_events: List of tuples (Deceleration, Contraction) that were successfully paired.
    - dec_type: Type of decelerations to show ("all", "early", "late", "variable").

    Returns:
    - Animation
    """

    # Filter paired events based on selected type
    if dec_type != "all":
        paired_events = [(dec, contr) for dec, contr in paired_events if dec.tipo.lower() == dec_type.lower()]
    
    if not paired_events:
        print(f"No decelerations found for type: {dec_type}")
        return
    
    # Generate time arrays
    time_fhr = np.arange(len(fhr)) / fs
    time_cu = np.arange(len(cu)) / fs

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Plot FHR in the first subplot ---
    ax1.plot(time_fhr, fhr, label='Fetal Heart Rate (FHR)', color='blue', alpha=0.7)
    ax1.axhline(y=baseline_fhr, color='green', linestyle='--', label='FHR Baseline')

    # Highlight all decelerations (default faded red)
    for decel in decelerations:
        ax1.fill_between(time_fhr[decel.start_idx:decel.end_idx], 
                         fhr[decel.start_idx:decel.end_idx], baseline_fhr, 
                         color='red', alpha=0.2)

    ax1.set_ylabel('FHR (BPM)')
    ax1.legend(loc='upper right')
    ax1.set_title('Fetal Heart Rate (FHR) with Decelerations')

    # --- Plot CU in the second subplot ---
    ax2.plot(time_cu, cu, label='Uterine Contractions (CU)', color='blue', alpha=0.7)

    # Highlight all contractions (default faded red)
    contraction_patches = {}  # Store patches to update colors during animation
    for contr in contractions:
        patch = ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], 
                                 cu[contr.start_idx:contr.end_idx], 
                                 color='red', alpha=0.2)
        contraction_patches[contr] = patch  # Store patch reference

    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Contractions')
    ax2.legend(loc='upper right')
    ax2.set_title('Uterine Contractions with Highlights')

    # Elements for animation (initialized empty and updated dynamically)
    highlight_decel = ax1.fill_between([], [], [], color='red', alpha=0.5)  # Placeholder
    highlight_contr = ax2.fill_between([], [], [], color='red', alpha=0.5)  # Placeholder

    decel_points, = ax1.plot([], [], 'ro', markersize=8)  # Points for deceleration
    contr_points, = ax2.plot([], [], 'ro', markersize=8)  # Points for contraction (same color as FHR)

    # Keep track of contractions that have been paired (to change color to pale green)
    paired_contractions = set()

    # Update function for animation
    def update(frame):
        decel, contr = paired_events[frame]

        # Remove previous highlights if they exist
        if hasattr(update, "highlight_decel"):
            update.highlight_decel.remove()
        if hasattr(update, "highlight_contr"):
            update.highlight_contr.remove()

        # Highlight the current deceleration
        update.highlight_decel = ax1.fill_between(time_fhr[decel.start_idx:decel.end_idx], 
                                                  fhr[decel.start_idx:decel.end_idx], 
                                                  color='red', alpha=0.5)

        # Highlight the current contraction
        update.highlight_contr = ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], 
                                                  cu[contr.start_idx:contr.end_idx], 
                                                  color='red', alpha=0.5)

        # Update deceleration points
        decel_points.set_data(
            [decel.start_time, decel.nadir_time, decel.end_time], 
            [baseline_fhr, decel.nadir_value, baseline_fhr]
        )

        # Update contraction points
        contr_points.set_data(
            [contr.start_time, contr.acme_time, contr.end_time], 
            [cu[contr.start_idx], cu[contr.acme_idx], cu[contr.end_idx]]
        )

        # Change contraction color to pale green after being paired
        if contr not in paired_contractions:
            paired_contractions.add(contr)  # Mark as paired
            contraction_patches[contr].remove()  # Remove the previous red patch
            ax2.fill_between(time_cu[contr.start_idx:contr.end_idx], 
                             cu[contr.start_idx:contr.end_idx], 
                             color='palegreen', alpha=0.5)  # Repaint as pale green

        return update.highlight_decel, update.highlight_contr, decel_points, contr_points

    # Create the animation
    n_frames = len(paired_events)
    ani = FuncAnimation(fig, update, frames=n_frames, interval=1000, repeat=True)

    plt.close(fig)  # Prevent duplicate static display
    
    # Guardar animación como gif
    if out_path is None:
        with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmpfile:
            ani.save(tmpfile.name, writer='pillow')
            return tmpfile.name
    else:
        ani.save(out_path, writer='pillow')
        return out_path

#----------------------------------------------PIPELINE-----------------------------------------------------------

def get_decelerations(fhr, fs, verbose = False):
    mean_fhr = get_mean(fhr, fs, plot = verbose)
    decelerations = find_all_decelerations(fhr, fs, verbose = verbose)
    if verbose:
        plot_fhr_with_decelerations(fhr, fs, decelerations)


    return decelerations, mean_fhr

def get_contractions(uc, fs, window_size = 3, verbose = False):
    contractions = find_all_contractions(uc, fs)
    if verbose:
        plot_cu_with_contractions(uc, fs, contractions)

    return contractions

def get_classified_decelerations(fhr, uc, fs, verbose = False):
    decelerations, mean_fhr = get_decelerations(fhr, fs, verbose = verbose)
    contractions = get_contractions(uc, fs, verbose = verbose)
    if verbose:
        plot_decc_contr(fhr, uc, fs, decelerations, contractions)
    
    paired_events = emparejar(decelerations, contractions, tolerance = 3)
    early_decs, late_decs, variable_decs = clasificar_dec(paired_events, diff=5.0, form_criteria=True, verbose = verbose)

    figs = []
    if verbose:
        figs = plot_decc_contr(fhr, uc, fs, decelerations, dBaseline, contractions)
    
    return early_decs, late_decs, variable_decs, decelerations, contractions, paired_events, dBaseline, figs
