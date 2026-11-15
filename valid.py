import os
import shutil
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FOLDER = 'TAP_DATA'
OUTPUT_FOLDER = 'TAP_DATA_VALID'

# Constants derived from your CSV header example
NUM_ACC_ZONES = 6  # Indices 0 to 5
NUM_EMG_ZONES = 5  # Indices 0 to 4 based on your CSV string example

# Axis Labels
ACC_AXES = ['X', 'Y', 'Z']
EMG_AXES = ['CH1', 'CH2', 'CH3']

# Literal strings used in CSV headers for history indices
ACC_TAIL_IDX_STR = "ACC_PSD_HISTORY_SIZE - 1"
EMG_TAIL_IDX_STR = "EMG_PSD_HISTORY_SIZE - 1"

# Data Types Limits
# acc/emg/wl are uint16 packed (0 to 65534 due to MAX-1 logic)
MIN_UINT16 = 0
MAX_UINT16_PACKED = 65534 

# acc_rate is int8_t (Signed 8-bit: -128 to 127)
MIN_INT8 = -128
MAX_INT8 = 127

# Metadata columns that must exist before the features
METADATA_COLS = [
    "User", "Timestamp", "Metronome", "PointerState", 
    "TriggerPressure", "InputX", "InputY"
]

# ==========================================
# 1. GENERATE EXPECTED FEATURE ORDER
# ==========================================
def get_fw_feature_order():
    """
    Reconstructs the exact list of feature column names based on the 
    specific formatting provided (X/Y/Z, CH1/CH2/CH3, literal indices).
    """
    columns = []
    
    # --- 1. ACC DATA ---
    # ACC HEAD: p_core->acc_psd[0][AX][ZONE]
    for ax in ACC_AXES:
        for zone in range(NUM_ACC_ZONES):
            columns.append(f"p_core->acc_psd[0][{ax}][{zone}]")
            
    # ACC TAIL: p_core->acc_psd[ACC_PSD_HISTORY_SIZE - 1][AX][ZONE]
    for ax in ACC_AXES:
        for zone in range(NUM_ACC_ZONES):
            columns.append(f"p_core->acc_psd[{ACC_TAIL_IDX_STR}][{ax}][{zone}]")

    # ACC MEAN: p_core->acc_psd_mean[AX][ZONE]
    for ax in ACC_AXES:
        for zone in range(NUM_ACC_ZONES):
            columns.append(f"p_core->acc_psd_mean[{ax}][{zone}]")

    # --- 2. ACC RATE ---
    # p_core->acc_rate[AX][ZONE]
    for ax in ACC_AXES:
        for zone in range(NUM_ACC_ZONES):
            columns.append(f"p_core->acc_rate[{ax}][{zone}]")

    # --- 3. EMG DATA ---
    # EMG HEAD: p_core->emg_psd[0][CH][ZONE]
    for ax in EMG_AXES:
        for zone in range(NUM_EMG_ZONES):
            columns.append(f"p_core->emg_psd[0][{ax}][{zone}]")
            
    # EMG TAIL: p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1][CH][ZONE]
    for ax in EMG_AXES:
        for zone in range(NUM_EMG_ZONES):
            columns.append(f"p_core->emg_psd[{EMG_TAIL_IDX_STR}][{ax}][{zone}]")

    # EMG MEAN: p_core->emg_psd_mean[CH][ZONE]
    for ax in EMG_AXES:
        for zone in range(NUM_EMG_ZONES):
            columns.append(f"p_core->emg_psd_mean[{ax}][{zone}]")

    # --- 4. WL DATA ---
    # p_core->wl[CH], p_core->wl_head[CH], p_core->wl_tail[CH]
    for ax in EMG_AXES:
        columns.append(f"p_core->wl[{ax}]")
        columns.append(f"p_core->wl_head[{ax}]")
        columns.append(f"p_core->wl_tail[{ax}]")
        
    return columns

# ==========================================
# 2. VALIDATION LOGIC
# ==========================================
def validate_csv(filepath, expected_features):
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"[ERROR] Could not read CSV {filepath}: {e}")
        return False

    # --- Rule: Check Metadata presence ---
    # Specifically "PointerState" as requested, but we check all basic headers
    if "PointerState" not in df.columns:
        print(f"[FAIL] {filepath} missing 'PointerState' column.")
        return False

    # --- Rule: Check Feature Count & Order ---
    # We combine Metadata + Features to form the expected FULL header
    # or we just check that the features appear in the correct sequence after metadata.
    
    # Let's extract just the feature columns from the DF
    # We assume any column starting with "p_core->" is a feature
    df_feature_cols = [c for c in df.columns if "p_core->" in c]
    
    if len(df_feature_cols) != len(expected_features):
        print(f"[FAIL] {filepath} Feature count mismatch. Found {len(df_feature_cols)}, expected {len(expected_features)}.")
        return False
        
    if df_feature_cols != expected_features:
        print(f"[FAIL] {filepath} Feature order mismatch.")
        # Optional: Print first difference for debugging
        for i, (f_found, f_exp) in enumerate(zip(df_feature_cols, expected_features)):
            if f_found != f_exp:
                print(f"   Mismatch at index {i}: Found '{f_found}', Expected '{f_exp}'")
                break
        return False

    # --- Rule: Test Ranges ---
    # Define groups
    cols_acc = [c for c in expected_features if "acc_psd" in c] # Includes head, tail, mean
    cols_emg = [c for c in expected_features if "emg_psd" in c] # Includes head, tail, mean
    cols_wl  = [c for c in expected_features if "wl" in c]
    cols_rate = [c for c in expected_features if "acc_rate" in c]

    # Helper for range check
    def check_range(columns, min_val, max_val, name):
        if not columns: return True
        # Get min/max of the entire block
        block_max = df[columns].max().max()
        block_min = df[columns].min().min()
        
        if block_max > max_val:
            print(f"[FAIL] {filepath} {name} exceeds MAX {max_val} (Found {block_max})")
            return False
        if block_min < min_val:
            print(f"[FAIL] {filepath} {name} below MIN {min_val} (Found {block_min})")
            return False
        return True

    # Check UInt16 packed types (0 to 65534)
    if not check_range(cols_acc, MIN_UINT16, MAX_UINT16_PACKED, "ACC_PSD"): return False
    if not check_range(cols_emg, MIN_UINT16, MAX_UINT16_PACKED, "EMG_PSD"): return False
    if not check_range(cols_wl,  MIN_UINT16, MAX_UINT16_PACKED, "WL"):      return False
    
    # Check Int8 type (-128 to 127)
    if not check_range(cols_rate, MIN_INT8, MAX_INT8, "ACC_RATE"): return False

    # --- Rule: Test Variance ---
    # Grouped logic: "It cant be that whole feature group have variance zero"
    groups = {
        "ACC_PSD": cols_acc,
        "EMG_PSD": cols_emg,
        "WL": cols_wl,
        "ACC_RATE": cols_rate
    }

    for group_name, cols in groups.items():
        if not cols: continue
        # Calculate variance
        variances = df[cols].var()
        # Sum of variances == 0 means ALL columns in that group are constant
        if variances.sum() == 0:
            print(f"[FAIL] {filepath} Group {group_name} has ZERO variance (Flatline).")
            return False

    return True

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
def main():
    if not os.path.exists(INPUT_FOLDER):
        os.makedirs(INPUT_FOLDER)
        print(f"Created {INPUT_FOLDER}. Please put CSV files there.")
        return

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    expected_features = get_fw_feature_order()

    print(
        f"Configuration: ACC_ZONES={NUM_ACC_ZONES}, "
        f"EMG_ZONES={NUM_EMG_ZONES}"
    )
    print(f"Expected Features: {len(expected_features)} columns.")

    # ========================================================
    # FIND ALL CSV FILES RECURSIVELY
    # ========================================================

    files = []

    for root, dirs, filenames in os.walk(INPUT_FOLDER):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                files.append(
                    os.path.join(root, filename)
                )

    print(f"Found {len(files)} CSV files to process...")

    passed_count = 0
    failed_count = 0

    # ========================================================
    # HELPER: PREVENT SAME-FILENAME COLLISIONS
    # ========================================================

    def get_unique_destination(filename):
        """
        Everything goes directly into TAP_DATA_VALID.

        If two subfolders contain the same filename:
            test.csv
            test.csv

        they become:
            test.csv
            test_1.csv
        """

        destination = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        if not os.path.exists(destination):
            return destination

        name, extension = os.path.splitext(filename)

        counter = 1

        while True:
            new_filename = f"{name}_{counter}{extension}"

            destination = os.path.join(
                OUTPUT_FOLDER,
                new_filename
            )

            if not os.path.exists(destination):
                return destination

            counter += 1

    # ========================================================
    # PROCESS ALL FILES
    # ========================================================

    for filepath in files:

        filename = os.path.basename(filepath)

        # Only for logging, so we can see where it came from
        relative_path = os.path.relpath(
            filepath,
            INPUT_FOLDER
        )

        if validate_csv(filepath, expected_features):

            destination = get_unique_destination(
                filename
            )

            print(
                f"[PASS] {relative_path} "
                f"-> {os.path.basename(destination)}"
            )

            # ALL valid files go directly into TAP_DATA_VALID
            shutil.move(
                filepath,
                destination
            )

            passed_count += 1

        else:
            print(f"[REJECTED] {relative_path}")
            failed_count += 1

    print("---------------------------------------------------------")
    print("Processing Complete.")
    print(f"Total:  {len(files)}")
    print(f"Passed: {passed_count}")
    print(f"Failed: {failed_count}")
    print(f"Valid files moved directly to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()