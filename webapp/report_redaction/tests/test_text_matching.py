import difflib

def compare_text_and_count_removals(text_with_extra:str, clean_text:str, position=None):
    """
    Compare two texts and mark the differences to remove from text1 to match text2.
    Optionally, return the count of removed characters up to a specified position in text2.
    """
    differ = difflib.Differ()
    diff = list(differ.compare(text_with_extra, clean_text))
    
    cleaned_text = []
    removed_count = 0
    for idx, item in enumerate(diff):
        if position is not None and idx >= position:
            break
        if item.startswith('-'):
            removed_count += 1
        elif not item.startswith('?'):
            cleaned_text.append(item[-1])
    
    return ''.join(cleaned_text), removed_count

def measure_removed_until_position(clean_text, long_text, position):
    """
    Measure how many letters are removed in the long text until a certain position.
    """
    cleaned_text, removed_count = compare_text_and_count_removals(long_text, clean_text, position)
    return removed_count

# Example usage
clean_text = "The quick brown fox jumps over the lazy dog."
long_text = "The quick brown \n\nfox jumps over the lazy dog."

# Measure removed letters until a certain position
position = 18
removed_until_position = measure_removed_until_position(clean_text, long_text, position)
print(f"Removed letters until position {position}: {removed_until_position}")
