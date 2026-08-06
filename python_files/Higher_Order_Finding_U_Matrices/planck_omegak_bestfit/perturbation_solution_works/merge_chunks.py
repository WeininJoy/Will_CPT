"""
Merge chunked data files from data_highK_integerK and data_highK_integerK_timeseries
into single combined files.
"""
import numpy as np
import os
import glob

def merge_chunks(base_folder, output_folder, file_pattern, merge_axis=0):
    """
    Merge numpy array files from multiple chunk folders.

    Parameters:
    -----------
    base_folder : str
        Base folder containing chunk_XX subdirectories
    output_folder : str
        Folder to save merged files
    file_pattern : str
        Pattern for files to merge (e.g., 'L70_*.npy')
    merge_axis : int
        Axis along which to concatenate arrays (default: 0)
    """
    # Find all chunk folders
    chunk_folders = sorted(glob.glob(os.path.join(base_folder, 'chunk_*')))

    if not chunk_folders:
        print(f"No chunk folders found in {base_folder}")
        return

    print(f"Found {len(chunk_folders)} chunk folders")

    # Get list of files from first chunk
    first_chunk = chunk_folders[0]
    file_list = glob.glob(os.path.join(first_chunk, file_pattern))
    file_basenames = [os.path.basename(f) for f in file_list]

    print(f"Found {len(file_basenames)} files to merge: {file_basenames}")

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Merge each file
    for filename in file_basenames:
        print(f"\nMerging {filename}...")

        arrays = []
        for chunk_folder in chunk_folders:
            filepath = os.path.join(chunk_folder, filename)
            if os.path.exists(filepath):
                arr = np.load(filepath)
                arrays.append(arr)
                print(f"  Loaded {os.path.basename(chunk_folder)}: shape {arr.shape}")
            else:
                print(f"  WARNING: {filepath} not found, skipping")

        if arrays:
            # Concatenate arrays
            merged = np.concatenate(arrays, axis=merge_axis)

            # Save merged file
            output_path = os.path.join(output_folder, filename)
            np.save(output_path, merged)
            print(f"  Saved merged file: {output_path}")
            print(f"  Final shape: {merged.shape}, Size: {merged.nbytes / 1e9:.2f} GB")
        else:
            print(f"  No arrays to merge for {filename}")

def main():
    base_dir = './data/'

    print("="*70)
    print("Merging U matrices and Xrec data")
    print("="*70)
    merge_chunks(
        base_folder=base_dir + 'data_highK_integerK/',
        output_folder=base_dir + 'data_highK_integerK_merged/',
        file_pattern='L70_*.npy',
        merge_axis=0
    )

    print("\n" + "="*70)
    print("Merging timeseries data")
    print("="*70)
    merge_chunks(
        base_folder=base_dir + 'data_highK_integerK_timeseries/',
        output_folder=base_dir + 'data_highK_integerK_timeseries_merged/',
        file_pattern='L70_*.npy',
        merge_axis=0
    )

    print("\n" + "="*70)
    print("Merging completed!")
    print(f"Merged data saved in:")
    print(f"  - {base_dir}data_highK_integerK_merged/")
    print(f"  - {base_dir}data_highK_integerK_timeseries_merged/")
    print("="*70)

if __name__ == '__main__':
    main()
