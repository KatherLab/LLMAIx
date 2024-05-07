import pandas as pd


def read_and_save_csv(input_file, output_file_path):
    """
    Reads a CSV file, treating each line as a single data entry, and saves it to a new file.

    Parameters:
    - input_file: werkzeug.datastructures.file_storage.FileStorage, the input CSV file.
    - output_file_path: str, the path where the reformatted CSV file will be saved.
    """
    # Initialize a list to hold the data
    data = []
    print(type(input_file))
    # Get the path of the input file

    # Read the file line by line directly from the FileStorage object
    input_file.stream.seek(0)  # Ensure we're reading from the beginning
    for line in input_file.stream:
        # Decode the binary data and strip whitespace/newlines
        data.append([line.decode('utf-8').strip()])
    # Create a DataFrame from the list
    # The column name is 'report'
    # delete the first row
    df = pd.DataFrame(data[1:], columns=['report'])
    print(df.head())

    # Save the DataFrame to a new CSV file
    df.to_csv(output_file_path, index=False, encoding='utf-8')
    print(f'File saved successfully to {output_file_path}')
