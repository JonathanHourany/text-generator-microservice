"""Functions used to split text for unsupervised learning"""

from typing import *


def split_input_target(text_chunk: str, split_index: int = 1) -> Tuple[str, str]:
    """Splits text into two chunks representing the input to be fed into the NN, and it's target label.

    Example
    -------
    >>> split_input_target("Python")
    "Pytho", "ython"
    """
    input_text = text_chunk[:-split_index]
    target_text = text_chunk[split_index:]

    return input_text, target_text
