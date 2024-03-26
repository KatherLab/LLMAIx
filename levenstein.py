from thefuzz import process

def replace_personal_info(text, personal_info_list):
    masked_text = text
    for info in personal_info_list:
        # Get a list of best matches for the current personal information from the text
        best_matches = process.extract(info, text.split())
        best_score = best_matches[0][1]
        for match, score in best_matches:
            if score == best_score:
                # Replace best matches with asterisks (*) of the same length as the personal information
                masked_text = masked_text.replace(match, '*' * len(match))
    return masked_text

# Example usage
text = "John Smith lives in New York and was born on 01/01/1990."
personal_info_list = ["John", "Smith", "New York", "01/01/1990"]


# Replace personal information in text
masked_text = replace_personal_info(text, personal_info_list)
print(masked_text)
