import os 
import shutil
# import numpy as np

def empty_folder(folder_path):
    """
    Deletes all files and subfolders in the given folder, but not the folder itself.
    """
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)         # Delete file or symlink
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)     # Delete subdirectory
        except Exception as e:
            print(f"Failed to delete {file_path}: {e}")

# def directional_round(x, decimals=0):
#     factor = 10 ** decimals
#     x_scaled = x * factor
#     result = np.where(x_scaled >= 0, np.ceil(x_scaled), np.floor(x_scaled))
#     return result / factor