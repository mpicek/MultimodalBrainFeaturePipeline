import os
import re
from datetime import datetime

def find_corresponding_wisci(mp4_filename, wisci_path):
    # Extract datetime from mp4 filename
    mp4_datetime = re.search(r'(\d{2}_\d{2}_\d{4}_\d{4}_\d{2})', mp4_filename).group(0)
    mp4_datetime = datetime.strptime(mp4_datetime, '%d_%m_%Y_%H%M_%S')
    
    # Find .mat files in the directory and subdirectories
    mat_files = []
    for root, dirs, files in os.walk(wisci_path):
        for file in files:
            if file.endswith('.mat'):
                mat_files.append(os.path.join(root, file))
    
    # Function to extract datetime from .mat filename
    def extract_datetime(mat_filename):
        mat_datetime = re.search(r'(\d{4}_\d{2}_\d{2}_\d{2}_\d{2})', mat_filename).group(0)
        return datetime.strptime(mat_datetime, '%Y_%m_%d_%H_%M')
    
    # Filter .mat files based on datetime
    valid_mat_files = [(mat_file, extract_datetime(mat_file)) for mat_file in mat_files]

    valid_mat_files.sort(key=lambda x: x[1])  # Sort by datetime

    # Find the closest lower time .mat file
    closest_mat_file = None
    for mat_file, mat_datetime in valid_mat_files:
        if mat_datetime <= mp4_datetime:
            closest_mat_file = mat_file
        else:
            break
    
    return closest_mat_file


if __name__ == "__main__":
    # Example usage:
    mp4_filename = "cam0_916512060805_record_04_10_2023_1341_07.mp4"
    wisci_path = "/media/vita-w11/T71/UP2001/WISCI/"
    corresponding_wisci = find_corresponding_wisci(mp4_filename, wisci_path)
    print("Corresponding .mat file:", corresponding_wisci)