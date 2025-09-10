#!/usr/bin/env python3

import re
import json
import keyword

try:
    import unidecode
    UNIDECODE_AVAILABLE = True
except ImportError:
    UNIDECODE_AVAILABLE = False

from nimbro_utils.utility.misc import assert_type_value

# normalization

def normalize_string(string, remove_underscores=False, remove_punctuation=False, remove_common_specials=False, reduce_whitespaces=False, remove_whitespaces=False, lowercase=False):
    """
    Normalize a given string by removing unwanted characters and applying various transformations.

    Args:
        string (str): The input string to process.
        remove_underscores (bool, optional): Whether to remove underscores. Defaults to False.
        remove_punctuation (bool, optional): Whether to remove punctuation marks (.,:!?). Defaults to False.
        remove_common_specials (bool, optional): Whether to remove common special characters ([]()/$€&+-=*'<>;%). Defaults to False.
        reduce_whitespaces (bool, optional): If True, all (consecutive) whitespaces (' ', '\n', '\t') are reduces to a single whitespace (' '). Defaults to False.
        remove_whitespaces (bool, optional): If True, `reduce_whitespaces` is ignored and all whitespaces (' ', '\n', '\t') are removed. Defaults to False.
        lowercase (bool, optional): Whether to convert the string to lowercase. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str: The normalized string.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=remove_underscores, type_or_value=bool, name="argument 'remove_underscores'")
    assert_type_value(obj=remove_punctuation, type_or_value=bool, name="argument 'remove_punctuation'")
    assert_type_value(obj=remove_common_specials, type_or_value=bool, name="argument 'remove_common_specials'")
    assert_type_value(obj=remove_whitespaces, type_or_value=bool, name="argument 'remove_whitespaces'")
    assert_type_value(obj=lowercase, type_or_value=bool, name="argument 'lowercase'")

    # normalize
    normalized = string
    if UNIDECODE_AVAILABLE:
        normalized = remove_unicode(normalized)
    normalized = remove_ansi_escape(normalized)
    normalized = remove_non_alpha_numeric(
        string=normalized,
        remove_whitespaces=False,
        remove_underscores=remove_underscores,
        remove_punctuation=remove_punctuation,
        remove_common_specials=remove_common_specials,
        replace_by_space=True
    )
    if reduce_whitespaces or remove_whitespaces:
        normalized = remove_whitespace(
            string=normalized,
            reduce_to_single_space=not remove_whitespaces
        )
    if lowercase:
        normalized = normalized.lower()

    return normalized

def remove_unicode(string):
    """
    Remove some Unicode characters (like emojis or flags), and normalize others (like accented characters).

    Args:
        string (str): The input string to process.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str: The string with Unicode characters removed or replaced.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # normalize
    normalized = unidecode.unidecode(string).strip()

    return normalized

def remove_whitespace(string, reduce_to_single_space=False):
    """
    Reduce (consecutive) whitespace characters (' ', '\t', '\n') to a single space (' ') or remove them entirely.

    Args:
        string (str): The input string to process.
        reduce_to_single_space (bool, optional): If True, all consecutive whitespaces are reduced to a single whitespace instead of removing all. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str: The string with reduced whitespace.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=reduce_to_single_space, type_or_value=bool, name="argument 'reduce_to_single_space'")

    # normalize
    normalized = re.sub(r'\s+', " " if reduce_to_single_space else "", string).strip()

    return normalized

def remove_non_alpha_numeric(string, remove_whitespaces=True, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, replace_by_space=False):
    """
    Remove non-alphanumeric characters from a string, with customizable options to remove specific characters or replace them with spaces.

    Args:
        string (str): The input string to process.
        remove_whitespaces (bool, optional): Whether to remove whitespaces (' ', '\t', '\n'). Defaults to True.
        remove_underscores (bool, optional): Whether to remove underscores. Defaults to True.
        remove_punctuation (bool, optional): Whether to remove punctuation marks (.,:!?). Defaults to True.
        remove_common_specials (bool, optional): Whether to remove common special characters ([]()/$€&+-=*'<>;%). Defaults to True.
        replace_by_space (bool, optional): If True, replaces non-alphanumeric characters with spaces. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str: The string with non-alphanumeric characters removed or replaced.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=remove_whitespaces, type_or_value=bool, name="argument 'remove_whitespaces'")
    assert_type_value(obj=remove_underscores, type_or_value=bool, name="argument 'remove_underscores'")
    assert_type_value(obj=remove_punctuation, type_or_value=bool, name="argument 'remove_punctuation'")
    assert_type_value(obj=remove_common_specials, type_or_value=bool, name="argument 'remove_common_specials'")
    assert_type_value(obj=replace_by_space, type_or_value=bool, name="argument 'replace_by_space'")

    # normalize
    pattern = r"^a-zA-Z0-9"
    if not remove_whitespaces:
        pattern += " \n\t"
    if not remove_underscores:
        pattern += "_"
    if not remove_punctuation:
        pattern += ".,:!?"
    if not remove_common_specials:
        pattern += "[]()/$€&+-=*'<>;%"
    else:
        string = string.replace("'", "")
    if replace_by_space:
        normalized = re.sub(f"[{pattern}]", " ", string).strip()
    else:
        normalized = re.sub(f"[{pattern}]", "", string).strip()

    return normalized

def remove_emoji(string):
    """
    Remove all emojis from a string.

    Args:
        string (str): The input string to process.

    Raises:
        AssertionError: If input arguments are invalid.
        ImportError: If `emoji` is not available.

    Returns:
        str: The string with emojis removed.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # normalize
    import emoji
    normalized = emoji.replace_emoji(string, "").strip()

    return normalized

def remove_ansi_escape(string):
    """
    Remove ANSI escape sequences from a string (commonly used for text formatting in the terminal).

    Args:
        string (str): The input string to process.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str: The string with ANSI escape sequences removed.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # normalize
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    normalized = ansi_escape.sub("", string)

    return normalized

# analysis

def levenshtein(string_a, string_b, normalization=False):
    """
    Calculate the Levenshtein distance between two strings, optionally normalizing them first.

    Args:
        string_a (str): The first string.
        string_b (str): The second string.
        normalization (bool, optional): Whether to normalize the strings before calculating the distance. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        int: The Levenshtein distance between the two strings.
    """
    # parse arguments
    assert_type_value(obj=string_a, type_or_value=str, name="argument 'string_a'")
    assert_type_value(obj=string_b, type_or_value=str, name="argument 'string_b'")
    assert_type_value(obj=normalization, type_or_value=bool, name="argument 'normalization'")

    # normalize
    if normalization:
        string_a = normalize_string(string=string_a, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, remove_whitespaces=True, lowercase=True)
        string_b = normalize_string(string=string_b, remove_underscores=True, remove_punctuation=True, remove_common_specials=True, remove_whitespaces=True, lowercase=True)

    # compute Levenshtein distance

    def _levenshtein_raw(a, b):
        if len(a) < len(b):
            return _levenshtein_raw(b, a)
        if not a:
            return len(b)

        previous_row = range(len(b) + 1)
        for i, c1 in enumerate(a):
            current_row = [i + 1]
            for j, c2 in enumerate(b):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    levenshtein_distance = _levenshtein_raw(a=string_a, b=string_b)

    return levenshtein_distance

def levenshtein_match(word, labels, threshold=0, normalization=False):
    """
    Find the closest matching label to a given word based on Levenshtein distance, with an optional threshold.

    Args:
        word (str): The word to match.
        labels (list): A list of labels to compare against.
        threshold (int | float, optional): The maximum allowable Levenshtein distance for a match.
            Use float in [0.0, 1.0] to use fractional threshold based the number of characters of `word`. Defaults to 0.
        normalization (bool, optional): Whether to normalize the word and labels before matching. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        str or None: The closest matching label, or None if no match is found within the threshold.
    """
    # parse arguments
    assert_type_value(obj=word, type_or_value=str, name="argument 'word'")
    assert_type_value(obj=labels, type_or_value=list, name="argument 'labels'")
    assert_type_value(obj=threshold, type_or_value=[int, float], name="argument 'threshold'")
    assert_type_value(obj=normalization, type_or_value=bool, name="argument 'normalization'")

    # shortcut
    if len(labels) == 0:
        return None

    # find minimum Levenshtein distance
    distances = [levenshtein(string_a=word, string_b=label, normalization=normalization) for label in labels]
    min_i = distances.index(min(distances))

    # fractional threshold
    if 0.0 < threshold < 1.0:
        threshold = int(len(word) * threshold) + 1

    # match
    if distances[min_i] <= threshold:
        match = labels[min_i]
    else:
        match = None

    return match

def is_url(string):
    """
    Check if a provided string is a valid URL.

    Args:
        string (str): The input string to process.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        bool: True, if `string` is a valid URL.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # identify URL
    # pattern = r'^(http|https):\/\/([\w.-]+)(\.[\w.-]+)+([\/\w\.-]*)*\/?$'
    pattern = r'^(https?):\/\/\S+$'
    # pattern = r'^(https?):\/\/[^\s\/$.?#].[^\s]*$'
    valid = bool(re.fullmatch(pattern, string))

    return valid

def is_attribute_name(string):
    """
    Check if a string is a valid Python class attribute name.

    A valid attribute name must be a valid identifier and must not
    be a reserved Python keyword. This ensures it can be safely used
    as an attribute name in classes or objects.

    Args:
        string (str): The string to validate.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        bool: True if the string is a valid attribute name, False otherwise.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # check validity
    valid = string.isidentifier() and not keyword.iskeyword(string)

    return valid

def count_tokens(string, encoding_name):
    """
    Count tokens in a string using a specified `tiktoken` encoding.

    Args:
        string (str): The input string to process.
        encoding_name (str): Name of the `tiktoken` encoding to use.

    Raises:
        AssertionError: If input arguments are invalid.
        ImportError: If `tiktoken` is not available.

    Returns:
        int: Number of tokens in the encoded string.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=encoding_name, type_or_value=str, name="argument 'encoding_name'")

    # tokenize
    import tiktoken
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))

    return num_tokens

def split_sentences(string):
    """
    Split a string into individual sentences based on punctuation marks.

    Args:
        string (str): The input string to process.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        list of str: A list of sentences.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")

    # define marks
    marks = [".", "!", "?"]

    # identify indices
    marks_idx = []
    for i in range(1, len(string) - 1):
        if string[i] in marks:
            if not string[i - 1].isdecimal() and string[i + 1].isspace(): # if string[i-1].isalpha() and string[i+1].isspace():
                marks_idx.append(i)

    # split sentences
    if len(marks_idx) < 1:
        sentences = [string]
    else:
        sentences = []
        i = 0
        for j in marks_idx:
            sentences.append(string[i:j + 1].strip())
            i = j + 2
        sentences.append(string[j + 2:].strip())

    return sentences

def extract_json(string, first_over_longest=False):
    """
    Extract the first or longest valid JSON object from a string.

    Args:
        string (str): The input string to process.
        first_over_longest (bool, optional): If True, the first encountered JSON object is returned instead of the longest. Defaults to False.

    Raises:
        AssertionError: If input arguments are invalid.

    Returns:
        any | None: The extracted JSON object, or None if no valid JSON object is found.
    """
    # parse arguments
    assert_type_value(obj=string, type_or_value=str, name="argument 'string'")
    assert_type_value(obj=first_over_longest, type_or_value=bool, name="argument 'first_over_longest'")

    # find JSON
    opening_indices = []
    opening_indices.extend([m.start() for m in re.finditer(r'[{\[]', string)])
    if len(opening_indices) > 0:
        closing_indices = []
        closing_indices.extend([m.start() for m in re.finditer(r'[}\]]', string)])
        if len(closing_indices) > 0:
            options = []
            for i in opening_indices:
                for j in closing_indices:
                    if j < i:
                        continue
                    else:
                        options.append(string[i:j + 1])
            if not first_over_longest:
                options.sort(key=len, reverse=True)
            for option in options:
                try:
                    return json.loads(option)
                except json.JSONDecodeError:
                    pass
