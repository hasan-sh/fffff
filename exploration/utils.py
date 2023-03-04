import string

def replace_punctuation_with_space(text):
    # Create a translation table with punctuation replaced by spaces
    translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation))

    # Use the translation table to replace punctuation with spaces
    return text.translate(translator)