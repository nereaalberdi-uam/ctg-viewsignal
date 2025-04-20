import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

#---------------------------------------- LINEAR FEATURES -----------------------------------------


def get_LTV(signal):
    
    valid_length = (len(signal) // 10) * 10  # Find the largest multiple of 10
    if valid_length == 0:
        return np.nan, np.nan  # Not enough data
    
    sFHR = np.nanmean(np.reshape(signal[:valid_length], (-1, 10)), axis=1)
    n = len(sFHR)
    j = 0
    LTV = 0
    for i in range(n-1):
        if(np.isnan(sFHR[i+1]) == True or np.isnan(sFHR[i]) == True):
            continue
        else:
            # print(sFHR[i+1], sFHR[i], sFHR[i+1]+sFHR[i])
            LTV = LTV + np.sqrt(sFHR[i+1]+sFHR[i])
            j += 1

    #LTV = LTV/n
    LTV = LTV/j
    
    return LTV 

def get_delta(signal, fs=4):
    """
    Compute Delta and Delta Total from an FHR signal.
    
    Args:
        signal (numpy.ndarray): 1D array of FHR values.
        fs (int, optional): Sampling frequency (Hz). Defaults to 4Hz.
        
    Returns:
        tuple: (delta, delta_total)
    """
    segment_size = 60 * fs  # 60 seconds per segment
    num_segments = len(signal) // segment_size

    if num_segments == 0:
        return np.nan, np.nan  # Not enough data

    part = np.reshape(signal[:num_segments * segment_size], (num_segments, segment_size))
    t = 0
    j = 0

    for i in range(num_segments):
        max_i = np.nanmax(part[i, :])
        min_i = np.nanmin(part[i, :])

        if np.isnan(max_i) or np.isnan(min_i):
            continue

        t += (max_i - min_i)
        j += 1

    delta = t / j if j > 0 else np.nan  # Avoid division by zero
    delta_total = np.nanmax(signal) - np.nanmin(signal)
    
    return delta, delta_total

def get_STV_II(signal):
    """
    Compute Short-Term Variability (STV) and Interval Index (II) from an FHR signal.
    """
    valid_length = (len(signal) // 10) * 10  # Find the largest multiple of 10
    if valid_length == 0:
        return np.nan, np.nan  # Not enough data
    
    sFHR = np.nanmean(np.reshape(signal[:valid_length], (-1, 10)), axis=1)
    n = len(sFHR)
    STV = 0
    j = 0

    for i in range(n - 1):
        if np.isnan(sFHR[i]) or np.isnan(sFHR[i + 1]):
            continue
        STV += abs(sFHR[i + 1] - sFHR[i])
        j += 1

    STV = STV / j if j > 0 else np.nan  # Avoid division by zero
    II = STV / np.nanstd(sFHR) if np.nanstd(sFHR) > 0 else np.nan  # Avoid division by zero

    return STV, II

def get_linear_features(signals, fs=4):
    """
    Compute multiple linear features from an array of FHR signals.
    
    Args:
        signals (numpy.ndarray): 2D array where each row is an FHR signal.
        fs (int, optional): Sampling frequency (Hz). Defaults to 4Hz.
        
    Returns:
        numpy.ndarray: Array of computed features.
    """
    mean_np = np.nanmean(signals, axis=1)
    std_np = np.nanstd(signals, axis=1)
    LTV_np = np.array([get_LTV(row) for row in signals])
    delta_np = np.array([get_delta(row, fs)[0] for row in signals])
    STV_II = np.array([get_STV_II(row) for row in signals])
    STV_np = STV_II[:, 0]
    II_np = STV_II[:, 1]

    linear = np.vstack([mean_np, std_np, LTV_np, delta_np, STV_np, II_np])
    return linear

    
#---------------------------------------- MORPHOLOGICAL FEATURES -----------------------------------------


def get_baseline_fit(l, signal):
    """
    Fit a spline model to the baseline of the signal.
    """
    if len(l) == 0 or len(signal) == 0:
        return np.nan, np.nan

    fit_result = UnivariateSpline(l, signal, s=5e5)
    baseline = np.nanmean(fit_result(l))

    return fit_result, baseline

def get_baseline(signal):
    """
    Estimate the baseline fetal heart rate.
    """
    signal = np.array(signal)
    
    if np.isnan(signal).all():
        return np.nan

    t = np.arange(len(signal))
    n = len(signal)
    mu = np.nanmean(signal)
    sigma = np.nanstd(signal)
    se = sigma / np.sqrt(n)
    up_border = mu + sigma + se
    down_border = mu - sigma - se

    gps = np.where((signal <= up_border) & (signal >= down_border))[0]
    
    if len(gps) == 0:
        return np.nan

    fit = get_baseline_fit(gps, signal[gps])
    baseline = np.nanmean(signal[gps])
    ratio = len(gps) / len(signal)

    return np.array([baseline, up_border, down_border, gps, fit, t, n, ratio, se], dtype="object")

def get_number_decelerations(signal):
    """
    Detect deceleration patterns in the FHR signal.
    """
    time = 15
    toppeak = -15

    if signal.size == 0 or np.isnan(signal).all():
        return 0, np.array([0, 0, 0, 0]), 0, 0

    baseline_object = get_baseline(signal)
    fit, _ = baseline_object[4]
    n1 = baseline_object[6]

    signal_fit = fit(np.arange(n1))

    if np.isnan(signal_fit).all():
        return 0, np.array([0, 0, 0, 0]), 0, 0

    diff = np.sign(signal - signal_fit)
    sign = -1 if diff[0] < 0 else 1

    matrix = []
    start = 0

    for i in range(n1 - 1):
        if diff[i] != sign and not np.isnan(diff[i]):
            stop = i - 1
            sign *= -1
            ind = np.arange(start, stop + 1)
            diff_aux = signal[ind] - signal_fit[ind]

            if diff_aux.size == 0 or np.isnan(diff_aux).all():
                continue

            peak = np.nanmin(diff_aux)
            matrix.append([start, stop, (stop - start) / 4, peak])
            start = i

    matrix = np.array(matrix)
    total_transition = len(matrix)

    if matrix.shape[0] > 0:
        matrix = matrix[matrix[:, 2] >= time]
        matrix = matrix[matrix[:, 3] <= toppeak]

    dcc_count = len(matrix) if matrix.shape[0] > 0 else 0
    dcc_matrix = matrix if matrix.shape[0] > 0 else np.array([0, 0, 0, 0])
    dcc_bpm = dcc_count / (len(signal) / 4)

    return np.array([dcc_count, dcc_matrix, dcc_bpm, total_transition], dtype="object")

def get_number_acelerations(signal):
    """
    Detect acceleration patterns in the FHR signal.
    """
    time = 15
    toppeak = 15

    if signal.size == 0 or np.isnan(signal).all():
        return 0, np.array([0, 0, 0, 0]), 0, 0

    baseline_object = get_baseline(signal)
    fit, _ = baseline_object[4]
    n1 = baseline_object[6]

    signal_fit = fit(np.arange(n1))
    diff = np.sign(signal - signal_fit)
    sign = -1 if diff[0] < 0 else 1

    matrix = []
    start = 0

    for i in range(n1 - 1):
        if diff[i] != sign and not np.isnan(diff[i]):
            stop = i - 1
            sign *= -1
            ind = np.arange(start, stop + 1)
            diff_aux = signal[ind] - signal_fit[ind]

            if diff_aux.size == 0 or np.isnan(diff_aux).all():
                continue

            peak = np.nanmax(diff_aux)
            matrix.append([start, stop, (stop - start) / 4, peak])
            start = i

    matrix = np.array(matrix)
    total_transition = len(matrix)

    if matrix.shape[0] > 0:
        matrix = matrix[matrix[:, 2] >= time]
        matrix = matrix[matrix[:, 3] >= toppeak]

    acc_count = len(matrix) if matrix.shape[0] > 0 else 0
    acc_matrix = matrix if matrix.shape[0] > 0 else np.array([0, 0, 0, 0])
    acc_bpm = acc_count / (len(signal) / 4)

    return np.array([acc_count, acc_matrix, acc_bpm, total_transition], dtype="object")

def get_morphological_features(signals):
    """
    Extract morphological features from a set of signals.
    """
    baselines_np = np.array([get_baseline(row)[0] for row in signals])
    dcc_np = np.array([get_number_decelerations(row)[0] for row in signals])
    acc_np = np.array([get_number_acelerations(row)[0] for row in signals])

    return np.vstack([baselines_np, dcc_np, acc_np])
