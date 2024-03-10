import os
from openpyxl import Workbook

def count_directories_and_files(root_dir):
    # Create a new workbook
    wb = Workbook()
    # Select the active worksheet
    ws = wb.active
    # Set headers for the Excel sheet
    ws.append(['Directory', 'Number of Directories', 'Number of Files'])
    
    # Function to count directories and files recursively
    def count_items(directory):
        num_dirs = 0
        num_files = 0
        
        # Count directories and files
        for item in os.listdir(directory):
            full_path = os.path.join(directory, item)
            if os.path.isdir(full_path):
                num_dirs += 1
                sub_num_dirs, sub_num_files = count_items(full_path)
                num_dirs += sub_num_dirs
                num_files += sub_num_files
            else:
                num_files += 1
        
        return num_dirs, num_files
    
    # Function to count directories and files for each subdirectory
    def count_subdirectories(root_dir):
        for dir_name in os.listdir(root_dir):
            dir_path = os.path.join(root_dir, dir_name)
            if os.path.isdir(dir_path):
                num_dirs, num_files = count_items(dir_path)
                ws.append([dir_path, num_dirs, num_files])
    
    # Count directories and files for each subdirectory of the root directory
    count_subdirectories(root_dir)
    
    # Save the workbook
    wb.save('directory_statistics.xlsx')

# Provide the root directory here
root_directory = 'students of class'
count_directories_and_files(root_directory)
