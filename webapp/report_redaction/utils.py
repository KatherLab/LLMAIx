import os
import pandas as pd

def find_llm_output_csv(directory: str) -> pd.DataFrame | None:
    # List all files in the directory
    files = os.listdir(directory)

    # Iterate over the files to find the first one starting with 'llm-output'
    for file in files:
        if file.startswith('llm-output') and file.endswith('.csv'):
            # Construct the full path to the CSV file
            csv_file_path = os.path.join(directory, file)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
            
            # Return the DataFrame
            return df

    # If no file is found, return None
    return None