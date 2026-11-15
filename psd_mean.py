import pandas as pd
import numpy as np
import os
import glob
import shutil

from concurrent.futures import ProcessPoolExecutor
from itertools import repeat


# =============================================================================
# CONFIGURATION
# =============================================================================

# --- 1:1 with FW Macros ---
# ACC_QUEUE_SIZE 32 -> Max Bin 15.
# EMG_QUEUE_SIZE 256 -> Max Bin 127.

# Counts = (EndBin - StartBin) + 1
EMG_BIN_COUNTS = [8, 10, 12, 46, 51]  # Zones 0-4
ACC_BIN_COUNTS = [2, 1, 1, 1, 2, 6]   # Zones 0-5


# -----------------------------------------------------------------------------
# PERFORMANCE SETTINGS
# -----------------------------------------------------------------------------

# Start with 8.
#
# NVMe / fast SSD:
#   8-16 is normally good.
#
# SATA SSD:
#   4-8 may be better.
#
# HDD:
#   2-4 is usually better.
#
# IMPORTANT:
# Changing this DOES NOT change calculation/results.
# It only changes how many files are processed simultaneously.
MAX_WORKERS = min(8, os.cpu_count() - 5 or 1)

# Printing thousands of lines also costs time.
# Print progress once every N completed files.
PRINT_EVERY = 25

# Marker written only after PSD values have been divided and rounded.
# Any recording with this flag set to 1 is already processed.
PSD_MEAN_FLAG = "psd_mean_applied"


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

ACC_AXES = ('X', 'Y', 'Z')
EMG_CHS = ('CH1', 'CH2', 'CH3')


ACC_GROUPS = (
    "p_core->acc_psd[0]",
    "p_core->acc_psd[ACC_PSD_HISTORY_SIZE - 1]",
    "p_core->acc_psd_mean",
)

EMG_GROUPS = (
    "p_core->emg_psd[0]",
    "p_core->emg_psd[EMG_PSD_HISTORY_SIZE - 1]",
    "p_core->emg_psd_mean",
)


# Pre-build exact column -> divisor mappings ONCE instead of rebuilding
# strings for every CSV.
#
# This does NOT change any calculation.
ACC_OPERATIONS = tuple(
    (
        f"{group}[{axis}][{z_idx}]",
        ACC_BIN_COUNTS[z_idx]
    )
    for group in ACC_GROUPS
    for axis in ACC_AXES
    for z_idx in range(len(ACC_BIN_COUNTS))
)


EMG_OPERATIONS = tuple(
    (
        f"{group}[{ch}][{z_idx}]",
        EMG_BIN_COUNTS[z_idx]
    )
    for group in EMG_GROUPS
    for ch in EMG_CHS
    for z_idx in range(len(EMG_BIN_COUNTS))
)


# =============================================================================
# PROCESS ONE CSV
# =============================================================================

def process_single_csv(file_path, input_dir, output_dir):
    """
    Process ONE CSV.

    IMPORTANT:
    Mathematical operations intentionally remain identical to the original
    implementation.
    """

    # -------------------------------------------------------------------------
    # READ
    # -------------------------------------------------------------------------

    # Intentionally keeping the exact same pandas read behavior.
    df = pd.read_csv(file_path)

    # Do not apply PSD bin division twice.
    if PSD_MEAN_FLAG in df.columns:
        if not df[PSD_MEAN_FLAG].eq(1).all():
            raise ValueError(
                f"{file_path}: '{PSD_MEAN_FLAG}' exists but not every value is 1."
            )
        # If the output is another directory, still copy the already processed
        # CSV there; do not recalculate its PSD columns.
        if os.path.abspath(input_dir) != os.path.abspath(output_dir):
            rel_path = os.path.relpath(file_path, input_dir)
            destination = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            shutil.copy2(file_path, destination)
        return file_path, "skipped"

    # Faster membership lookup than repeatedly searching df.columns.
    #
    # Does not affect column order or dataframe data.
    existing_columns = set(df.columns)


    # -------------------------------------------------------------------------
    # ACC FEATURES
    # -------------------------------------------------------------------------

    for col_name, bin_count in ACC_OPERATIONS:

        if col_name in existing_columns:

            # ================================================================
            # DO NOT CHANGE
            #
            # EXACT SAME MATH AS ORIGINAL SCRIPT
            # ================================================================

            df[col_name] = (
                df[col_name] / bin_count
            ).round()


    # -------------------------------------------------------------------------
    # EMG FEATURES
    # -------------------------------------------------------------------------

    for col_name, bin_count in EMG_OPERATIONS:

        if col_name in existing_columns:

            # ================================================================
            # DO NOT CHANGE
            #
            # EXACT SAME MATH AS ORIGINAL SCRIPT
            # ================================================================

            df[col_name] = (
                df[col_name] / bin_count
            ).round()


    # -------------------------------------------------------------------------
    # FINAL POLISH
    # -------------------------------------------------------------------------

    # EXACT SAME selection rule as original:
    #
    #     if "psd" in c
    #
    all_psd_cols = [
        c for c in df.columns
        if "psd" in c
    ]

    if all_psd_cols:

        # ================================================================
        # DO NOT CHANGE
        #
        # EXACT SAME clip + uint16 conversion
        # ================================================================

        df[all_psd_cols] = (
            df[all_psd_cols]
            .clip(0, 65535)
            .astype(np.uint16)
        )


    # Mark this recording as converted, after all PSD calculations.
    df[PSD_MEAN_FLAG] = 1

    # -------------------------------------------------------------------------
    # OUTPUT PATH
    # -------------------------------------------------------------------------

    rel_path = os.path.relpath(
        file_path,
        input_dir
    )

    output_file = os.path.join(
        output_dir,
        rel_path
    )

    output_file_dir = os.path.dirname(
        output_file
    )


    # exist_ok=True is important with multiprocessing.
    #
    # Multiple workers may attempt to create the same directory at the
    # same time.
    os.makedirs(
        output_file_dir,
        exist_ok=True
    )


    # -------------------------------------------------------------------------
    # WRITE
    # -------------------------------------------------------------------------

    # Exact same pandas CSV output call.
    df.to_csv(
        output_file,
        index=False
    )

    return file_path, "converted"


# =============================================================================
# PROCESS DIRECTORY
# =============================================================================

def process_offline_csvs(input_dir, output_dir):
    """
    Reads CSVs recursively from input_dir, applies the Mean-Binning correction,
    and saves new CSVs to output_dir while maintaining subfolder structure.

    Files are processed in parallel.

    Mathematical behavior inside every individual CSV remains unchanged.
    """

    search_pattern = os.path.join(
        input_dir,
        "**",
        "*.csv"
    )

    csv_files = glob.glob(
        search_pattern,
        recursive=True
    )

    total_files = len(csv_files)

    if total_files == 0:
        print(
            f"No CSV files found in "
            f"{input_dir} or its subfolders"
        )
        return


    print("=" * 100)
    print(f"Input      : {input_dir}")
    print(f"Output     : {output_dir}")
    print(f"CSV files  : {total_files:,}")
    print(f"Workers    : {MAX_WORKERS}")
    print("=" * 100)


    # -------------------------------------------------------------------------
    # PARALLEL FILE PROCESSING
    # -------------------------------------------------------------------------

    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS
    ) as executor:

        results = executor.map(
            process_single_csv,
            csv_files,
            repeat(input_dir),
            repeat(output_dir),

            # Keep each CSV as an individual task.
            # This gives good load balancing because CSV sizes vary.
            chunksize=1,
        )


        # Consume results so exceptions from workers are propagated.
        skipped_count = 0
        for completed, (_, status) in enumerate(
            results,
            start=1
        ):
            if status == "skipped":
                skipped_count += 1

            if (
                completed % PRINT_EVERY == 0
                or completed == total_files
            ):
                print(
                    f"[{completed:,}/{total_files:,}] "
                    f"{completed / total_files * 100:6.2f}%"
                )


    print("=" * 100)
    print(
        f"DONE: {total_files:,} CSV files checked; "
        f"{total_files - skipped_count:,} converted, "
        f"{skipped_count:,} already converted (skipped)."
    )
    print("=" * 100)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    FOLDER_A = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/TAP'
    )

    FOLDER_B = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/SWIPE'
    )

    FOLDER_C = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/FINGER'
    )

    FOLDER_D = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/ARM'
    )

    FOLDER_E = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/BAND'
    )

    FOLDER_F = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/WRIST'
    )

    FOLDER_X = (
        '/root/tree_run_script/tree_run_script/'
        'DATA/DATA_TAP/RFE'
    )

    FOLDER_Z = (
        '/root/tree_run_script/tree_run_script/DATA'
    )


    # folders = {
    #     FOLDER_A,
    #     FOLDER_B,
    #     FOLDER_C,
    #     FOLDER_D,
    #     FOLDER_E,
    #     FOLDER_F,
    # }

    folders = {
        FOLDER_Z,
    }


    for folder in folders:

        # Same behavior as your current script:
        #
        # input == output
        #
        # Therefore files are overwritten in-place.
        process_offline_csvs(
            folder,
            folder
        )
