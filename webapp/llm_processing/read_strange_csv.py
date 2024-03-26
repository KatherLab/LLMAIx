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
    input_file_path = input_file

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

if __name__ == "__main__":
    input = "/home/jeff/Downloads/data3.csv"
    # read the file
    try:
        df = pd.read_csv(input)
        # print the first 5 rows of the dataframe

        print(df.head())
        # pandas.errors.ParserError: Error tokenizing data. C error: Expected 6 fields in line 3, saw 11
    except pd.errors.ParserError as e:
        print(e)
        print("The error message indicates that the number of fields in line 3 of the CSV file is not as expected. This means that the CSV file is not properly formatted and needs to be fixed.")
        # fix the file
        output = "/home/jeff/Downloads/data3_fixed.csv"
        read_and_save_csv(input, output)
        # read the fixed file
        df = pd.read_csv(output)
        # print the first 5 rows of the dataframe
        print(df.head())


