import wfdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, interp1d, CubicSpline

#------------------------------------------- LOAD -----------------------------------------------------------

def load_ctg_data(record_name, db_path):
    """
    Load CTG (Cardiotocography) data from the PhysioNet database.

    Parameters:
    - record_name (str): Name of the record to be loaded.
    - db_path (str): Path to the database files.

    Returns:
    - fhr (numpy.ndarray): Fetal heart rate (FHR) signal, either processed or raw.
    - uc (numpy.ndarray): Uterine contraction (UC) signal, either processed or raw.
    - fs (float): Sampling frequency of the signals.
    - metadata_df (pandas.DataFrame): DataFrame containing metadata from the header file.

    Example Usage:
    --------------
    record_name = '1137'  # Other examples: '1023', '1137'
    db_path = '../ctu-chb-database'  

    fhr, uc, fs, metadata_df = load_ctg_data(record_name, db_path)
    """

    # Load the record from the database
    record = wfdb.rdrecord(f"{db_path}/{record_name}")

    # Extract signals and sampling frequency
    signals = record.p_signal  # Multi-channel signal array
    fields = record.sig_name  # Signal names
    fs = record.fs  # Sampling frequency

    # Identify the indices for FHR (Fetal Heart Rate) and UC (Uterine Contractions)
    try:
        fhr_idx = fields.index('FHR')  # Find the index of the FHR signal
        uc_idx = fields.index('UC')  # Find the index of the UC signal
    except ValueError:
        raise ValueError("FHR or UC signals were not found in the record.")

    # Extract the FHR and UC signals as Pandas Series for easy handling
    fhr = pd.Series(signals[:, fhr_idx])  # FHR signal data
    uc = pd.Series(signals[:, uc_idx])  # UC signal data

    # Load metadata from the header file
    header_path = f"{db_path}/{record_name}.hea"  # Path to the header file
    metadata = []
    
    with open(header_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith("#"):  # Metadata lines start with "#"
                parts = line[1:].strip().split(maxsplit=1)
                if len(parts) == 2:
                    key, value = parts
                    metadata.append((key, value))

    # Convert metadata into a DataFrame for structured representation
    metadata_df = pd.DataFrame(metadata, columns=["Parameter", "Value"])

    return fhr.values, uc.values, fs, metadata_df



#------------------------------------------- PREPROCESSING -----------------------------------------------------------

def detect_long_zero_gaps(signal, secs=15, freq=4, union=0, tolerance=0, verbose=False):
    """
    Detects long gaps where the signal is constant (based on its derivative).
    These are regions where the signal's derivative is within ±tolerance from zero.
    
    Parameters:
    - signal (array-like): The input 1D signal.
    - secs (float): Minimum duration (in seconds) for a gap to be considered long.
    - freq (float): Sampling frequency in Hz.
    - union (int): Max number of non-constant points between gaps to merge.
    - tolerance (float): Tolerance around zero for derivative to be considered zero.
    - verbose (bool): Whether to print detailed output.
    
    Returns:
    - final_gaps (list of tuples): Each tuple is (start_time, end_time, duration, percentage).
    - total_percentage (float): Total percentage of time occupied by all detected gaps.
    """
    signal = np.array(signal)
    total_duration = len(signal) / freq

    # Compute derivative of signal
    derivative = np.diff(signal, prepend=signal[0])  # Same length as signal
    zero_indices = np.where(np.abs(derivative) <= tolerance)[0]

    long_gaps = []
    if zero_indices.size > 0:
        start = zero_indices[0]
        prev = start

        for i in range(1, len(zero_indices)):
            if zero_indices[i] != prev + 1:
                end = prev
                length = end - start + 1
                duration = length / freq

                if duration >= secs:
                    long_gaps.append((start, end, duration))

                start = zero_indices[i]

            prev = zero_indices[i]

        # Handle final segment
        end = prev
        length = end - start + 1
        duration = length / freq
        if duration >= secs:
            long_gaps.append((start, end, duration))

    # Merge close gaps
    merged_gaps = []
    if long_gaps:
        merged_start, merged_end, merged_duration = long_gaps[0]

        for i in range(1, len(long_gaps)):
            start, end, duration = long_gaps[i]

            if start - merged_end <= union:
                merged_end = end
                merged_duration = (merged_end - merged_start + 1) / freq
            else:
                merged_gaps.append((merged_start, merged_end, merged_duration))
                merged_start, merged_end, merged_duration = start, end, duration

        merged_gaps.append((merged_start, merged_end, merged_duration))

    # Convert to time and calculate percentages
    final_gaps = []
    total_percentage = 0
    for start_idx, end_idx, duration in merged_gaps:
        start_time = start_idx / freq
        end_time = end_idx / freq
        percentage = (duration / total_duration) * 100
        total_percentage += percentage
        final_gaps.append((start_time, end_time, duration, percentage))

    if verbose:
        print(f"Detected {len(final_gaps)} long constant-value gaps.")
        print(f"Total percentage of signal occupied by gaps: {total_percentage:.2f}%")
        for idx, (st, et, d, p) in enumerate(final_gaps):
            print(f"Gap {idx+1}: Start {st:.2f}s, End {et:.2f}s, Duration {d:.2f}s, {p:.2f}% of total time")

    return final_gaps, total_percentage


def trim_signals(fhr, uc, fs, fhr_gaps, uc_gaps, verbose = False):
    """
    Trims two signals (FHR and UC) to the same synchronized interval by removing leading
    and trailing long gaps from either signal.

    The gaps are given in seconds, so we must convert them back to array indices using fs.

    Parameters:
    - fhr (array-like): Fetal Heart Rate (FHR) signal.
    - uc (array-like): Uterine Contractions (UC) signal.
    - fs (float): Sampling frequency in Hz.
    - fhr_gaps (list of tuples): List of (start_time, end_time, duration, percentage) tuples representing long gaps in FHR.
    - uc_gaps (list of tuples): List of (start_time, end_time, duration, percentage) tuples representing long gaps in UC.

    Returns:
    - trimmed_fhr (numpy.ndarray): FHR signal trimmed to the synchronized interval.
    - trimmed_uc (numpy.ndarray): UC signal trimmed to the synchronized interval.
    """

    # Convert to numpy arrays
    fhr = np.array(fhr)
    uc = np.array(uc)

    if verbose:
        print(f"Original FHR length: {len(fhr)} samples")
        print(f"Original UC length: {len(uc)} samples")
        print(f"Sampling frequency: {fs} Hz")
    
        # Print detected gaps for debugging
        print(f"FHR long gaps (in seconds): {fhr_gaps}")
        print(f"UC long gaps (in seconds): {uc_gaps}")

    # Convert time-based gaps back to index-based gaps
    fhr_gaps_indices = [(int(start_time * fs), int(end_time * fs)) for start_time, end_time, _, _ in fhr_gaps]
    uc_gaps_indices = [(int(start_time * fs), int(end_time * fs)) for start_time, end_time, _, _ in uc_gaps]

    if verbose:
        print(f"FHR long gaps (converted to indices): {fhr_gaps_indices}")
        print(f"UC long gaps (converted to indices): {uc_gaps_indices}")

    # Find start trimming point (convert back to indices)
    fhr_start_trim = fhr_gaps_indices[0][1] + 1 if fhr_gaps_indices and fhr_gaps_indices[0][0] == 0 else 0
    uc_start_trim = uc_gaps_indices[0][1] + 1 if uc_gaps_indices and uc_gaps_indices[0][0] == 0 else 0

    start_trim = int(max(fhr_start_trim, uc_start_trim))  # Sync both signals

    # Find end trimming point (convert back to indices)
    fhr_end_trim = fhr_gaps_indices[-1][0] if fhr_gaps_indices and fhr_gaps_indices[-1][1] >= len(fhr) - 1 else len(fhr)
    uc_end_trim = uc_gaps_indices[-1][0] if uc_gaps_indices and uc_gaps_indices[-1][1] >= len(uc) - 1 else len(uc)

    end_trim = int(min(fhr_end_trim, uc_end_trim))  # Sync both signals

    if verbose: 
        print(f"Start trim index: {start_trim} (time: {start_trim / fs:.2f}s)")
        print(f"End trim index: {end_trim} (time: {end_trim / fs:.2f}s)")

    # Trim both signals
    trimmed_fhr = fhr[start_trim:end_trim]
    trimmed_uc = uc[start_trim:end_trim]

    if verbose:
        print(f"Trimmed FHR length: {len(trimmed_fhr)} samples")
        print(f"Trimmed UC length: {len(trimmed_uc)} samples")
    
        print(f"Trimming from {start_trim / fs:.2f}s to {end_trim / fs:.2f}s (syncing both signals)")

    return trimmed_fhr, trimmed_uc


def interpolate_zero_values(signal, tolerance=0.1, method='pchip'):
    """
    Detects near-zero values in a signal, replaces them with NaN, and interpolates missing values 
    using the specified interpolation method.

    Parameters:
    - signal (array-like): 1D input signal.
    - tolerance (float, optional): Values within ±tolerance from zero are treated as missing (default: 0.1).
    - method (str, optional): Interpolation method, one of:
        - 'pchip' (Piecewise Cubic Hermite Interpolation) [default]
        - 'linear' (Linear interpolation)
        - 'cubic' (Cubic spline interpolation)
        - 'nearest' (Nearest neighbor interpolation)

    Returns:
    - interpolated_signal (numpy.ndarray): Signal with missing values filled.

    Example Usage:
    --------------
    # Interpolating using different methods
    interpolated_pchip = interpolate_zero_values(trimmed_fhr, method='pchip')
    interpolated_linear = interpolate_zero_values(trimmed_fhr, method='linear')
    interpolated_cubic = interpolate_zero_values(trimmed_fhr, method='cubic')
    interpolated_nearest = interpolate_zero_values(trimmed_fhr, method='nearest')

    # Applying to a uterine contraction signal
    interpolated_uc = interpolate_zero_values(trimmed_uc, method='linear')

    # Handling different tolerances
    interpolated_fhr_tolerance_05 = interpolate_zero_values(trimmed_fhr, tolerance=0.05, method='pchip')
    interpolated_fhr_tolerance_2 = interpolate_zero_values(trimmed_fhr, tolerance=2, method='linear')
    """

    # Convert input to a NumPy array of type float (for NaN handling)
    signal = np.array(signal, dtype=float)

    # Step 1: Detect bellow-zero values and replace them with NaN
    missing_indices = np.where(signal <= tolerance)[0]
    signal[missing_indices] = np.nan

    # Step 2: Identify valid data points (non-NaN values)
    valid_indices = np.where(~np.isnan(signal))[0]
    if len(valid_indices) < 2:
        raise ValueError("Not enough valid data points for interpolation.")

    # Step 3: Select interpolation method
    if method == 'pchip':
        interpolator = PchipInterpolator(valid_indices, signal[valid_indices])
    elif method == 'linear':
        interpolator = interp1d(valid_indices, signal[valid_indices], kind='linear', fill_value='extrapolate')
    elif method == 'cubic':
        interpolator = CubicSpline(valid_indices, signal[valid_indices])
    elif method == 'nearest':
        interpolator = interp1d(valid_indices, signal[valid_indices], kind='nearest', fill_value='extrapolate')
    else:
        raise ValueError("Unsupported interpolation method. Choose from 'pchip', 'linear', 'cubic', or 'nearest'.")

    # Step 4: Apply interpolation to missing values
    interpolated_signal = signal.copy()
    interpolated_values = interpolator(missing_indices)

    # Ensure interpolated values are strictly positive
    interpolated_values[interpolated_values <= tolerance] = np.nanmin(signal[valid_indices]) * 0.1 

    interpolated_signal[missing_indices] = interpolated_values


    return interpolated_signal

#------------------------------------------- PLOT -----------------------------------------------------
def plot_ctg_signals(fhr, uc, fs, start_time=0, end_time=None, verbose = False):
    """
    Visualize CTG (Cardiotocography) signals.

    Parameters:
    - fhr (numpy.ndarray): Fetal Heart Rate (FHR) signal.
    - uc (numpy.ndarray): Uterine Contractions (UC) signal.
    - fs (float): Sampling frequency in Hz.
    - start_time (float, optional): Start time for the plot in seconds (default: 0).
    - end_time (float, optional): End time for the plot in seconds (default: full signal length).

    Example Usage:
    --------------
    plot_ctg_signals(fhr, uc, fs, start_time=0, end_time=60)
    """

    # Convert start and end times to sample indices
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs) if end_time else len(fhr)

    # Ensure indices are within valid range
    start_idx = max(0, start_idx)  # Ensure the start index is not negative
    end_idx = min(len(fhr), end_idx)  # Ensure the end index does not exceed signal length

    # Extract the desired segment of the signals
    fhr_segment = fhr[start_idx:end_idx]
    uc_segment = uc[start_idx:end_idx]

    # Generate corresponding time values for the x-axis
    time = np.linspace(start_time, start_time + (end_idx - start_idx) / fs, len(fhr_segment))

    # Create figure and subplots
    plt.figure(figsize=(12, 6))

    # Plot Fetal Heart Rate (FHR)
    plt.subplot(2, 1, 1)
    plt.plot(time, fhr_segment, label='Fetal Heart Rate (FHR)', color='blue')
    plt.xlabel('Time (s)')
    plt.ylabel('BPM')
    plt.title('Fetal Heart Rate')
    plt.legend()
    plt.grid(True)

    # Plot Uterine Contractions (UC)
    plt.subplot(2, 1, 2)
    plt.plot(time, uc_segment, label='Uterine Contractions (UC)', color='green')
    plt.xlabel('Time (s)')
    plt.ylabel('Intensity')
    plt.title('Uterine Contractions')
    plt.legend()
    plt.grid(True)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    if verbose:
        print("Basic Characteristics:")
        print(f"- Total signal duration: {len(fhr) / fs:.2f} seconds")
        print(f"- Sampling interval: {1 / fs:.4f} seconds")
        print(f"- Number of samples in the plotted segment: {len(fhr_segment)}")

def plot_signal_with_gaps(signal, fs, long_gaps, total_percentage):
    """
    Plots a signal and highlights detected long zero gaps.

    Parameters:
    - signal (numpy.ndarray): The input signal.
    - fs (float): Sampling frequency in Hz.
    - long_gaps (list of tuples): List of detected gaps, each defined as (start_time, end_time, duration, percentage).
    - total_percentage (float): Total percentage of time occupied by detected gaps.

    Example Usage:
    --------------
    plot_signal_with_gaps(fhr, fs, fhr_long_gaps, total_percentage)
    plot_signal_with_gaps(uc, fs, uc_long_gaps, total_percentage)
    """
    # Convert sample indices to time in seconds
    time = np.arange(len(signal)) / fs  

    # Create figure
    plt.figure(figsize=(12, 5))
    plt.plot(time, signal, label="Signal", color="blue")

    # Highlight detected long zero gaps
    for i, (start, end, _, _) in enumerate(long_gaps):
        plt.axvspan(start, end, color="red", alpha=0.5, label="Long Gap" if i == 0 else "")

    # Configure plot labels and title
    plt.xlabel("Time (seconds)")
    plt.ylabel("Signal Value")
    plt.title(f"Signal with Highlighted Long Zero Gaps (Total Gap: {total_percentage:.2f}%)")
    plt.legend()
    plt.grid()
    
    # Show the plot
    plt.show()

#----------------------------------------------PIPELINE-----------------------------------------------------------

def preprocess_ctg_pipeline(record_name, db_path, tolerance=1, interpolation_method='cubic', plot=False):
    # Load raw signals
    fhr, uc, fs, metadata_df = load_ctg_data(record_name, db_path)

    if plot:
        plot_ctg_signals(fhr, uc, fs)

    # --- Iterative trimming loop ---
    for _ in range(10):  # max 10 iterations
        # Detect long zero gaps again
        fhr_gaps, fhr_perc = detect_long_zero_gaps(fhr, secs=30, freq=fs, union=3, tolerance=tolerance)
        uc_gaps, uc_perc = detect_long_zero_gaps(uc, secs=30, freq=fs, union=3, tolerance=tolerance)

   #     if plot:
   #         plot_signal_with_gaps(fhr, fs, fhr_gaps, fhr_perc)
   #         plot_signal_with_gaps(uc, fs, uc_gaps, uc_perc)
    
        if fhr_perc > 98 or uc_perc > 98:
            if plot:
                print("Signal is completely flat. Skipping.")
            return None, None, None, None

        # Determine if start/end trimming is needed
        trim_needed = False
        if fhr_gaps and fhr_gaps[0][0] == 0 or fhr_gaps and fhr_gaps[-1][1] >= len(fhr) / fs - 1:
            trim_needed = True
        if uc_gaps and uc_gaps[0][0] == 0 or uc_gaps and uc_gaps[-1][1] >= len(uc) / fs - 1:
            trim_needed = True

        # Exit loop if no trimming needed
        if not trim_needed:
            break

        # Perform trimming
        fhr, uc = trim_signals(fhr, uc, fs, fhr_gaps, uc_gaps)

        if plot:
            print("Signals were trimmed again.")
            plot_ctg_signals(fhr, uc, fs)

    # Interpolate
    interpolated_fhr = interpolate_zero_values(fhr, method=interpolation_method)
    interpolated_uc = interpolate_zero_values(uc, method=interpolation_method)

    if plot:
        print("Final interpolated signals:")
        plot_ctg_signals(interpolated_fhr, interpolated_uc, fs)

    return interpolated_fhr, interpolated_uc, fs, metadata_df
