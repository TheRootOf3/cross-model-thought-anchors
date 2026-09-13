"""Character ranges of each sentence inside a chain of thought.

get_chunk_ranges is COPIED VERBATIM from:

    external/thought-branches/faithfulness/utils.py, lines 96-195
    (submodule external/thought-branches @ 9e2bba4)

It is the function C_run_faith_transplantation.py:36-43 uses to build its
transplant prompts: cut k = question + reasoning_text[:chunk_ranges[k][0]].
Copied rather than reimplemented because it is quirky - a normalised-whitespace
fallback, a word-by-word rescue search, and a silent `continue` when a chunk is
not found (which shifts every later index) - and any reimplementation would
produce different cut points, so the comparison with their published curve
would no longer be like-for-like.

Its input `chunks` comes from split_solution_into_chunks. Ours lives in
src/common/split.py (copied from thought-anchors); it reproduces
thought-branches' utils.py:405 splitter on all 71 records of the Professor set,
so the two repos agree on where sentences begin.
"""

import re
from typing import List, Tuple

def get_chunk_ranges(
    full_text: str, chunks: List[str], pb_lazy_clean=True, window_size=100
) -> List[Tuple[int, int]]:
    # Get character ranges for each chunk in the full text
    chunk_ranges = []
    current_pos = 0

    for chunk in chunks:
        # Normalize the chunk for comparison (preserve length but standardize whitespace)
        normalized_chunk = re.sub(r"\s+", " ", chunk).strip()
        # normalized_chunk = ""

        # Try to find the chunk in the full text
        chunk_start = -1

        # First try exact match from current position
        exact_match_pos = full_text.find(chunk, current_pos)
        if exact_match_pos != -1:
            chunk_start = exact_match_pos
        else:
            # If exact match fails, try with normalized text
            chunk_words = normalized_chunk.split()

            # Search for the sequence of words, allowing for different whitespace
            for i in range(current_pos, len(full_text) - len(normalized_chunk)):
                # Check if this could be the start of our chunk
                text_window = full_text[i : i + len(normalized_chunk) + 20]  # Add some buffer
                normalized_window = re.sub(r"\s+", " ", text_window).strip()

                if normalized_window.startswith(normalized_chunk):
                    chunk_start = i
                    break

                # If not found with window, try word by word matching
                if (
                    i == current_pos + window_size
                ):  # Limit detailed search to avoid performance issues
                    for j in range(current_pos, len(full_text) - 10):
                        # Try to match first word
                        if re.match(
                            r"\b" + re.escape(chunk_words[0]) + r"\b",
                            full_text[j : j + len(chunk_words[0]) + 5],
                        ):
                            # Check if subsequent words match
                            match_text = full_text[j : j + len(normalized_chunk) + 30]
                            normalized_match = re.sub(r"\s+", " ", match_text).strip()
                            if normalized_match.startswith(normalized_chunk):
                                chunk_start = j
                                break
                    break

        if chunk_start == -1:
            print(f"Warning: Chunk not found in full text: {chunk[:50]}...")
            continue

        # For the end position, find where the content of the chunk ends in the full text
        chunk_content = re.sub(r"\s+", "", chunk)  # Remove all whitespace
        full_text_from_start = full_text[chunk_start:]
        full_text_content = re.sub(
            r"\s+", "", full_text_from_start[: len(chunk) + 50]
        )  # Remove all whitespace

        # Find how many characters of content match
        content_match_len = 0
        for i in range(min(len(chunk_content), len(full_text_content))):
            if chunk_content[i] == full_text_content[i]:
                content_match_len += 1
            else:
                break
        # print(f"Content match length: {content_match_len}")

        # Map content length back to original text with whitespace
        chunk_end = chunk_start
        content_chars_matched = 0
        for i in range(len(full_text_from_start)):
            if chunk_end + i >= len(full_text):
                break
            if not full_text[chunk_start + i].isspace():
                content_chars_matched += 1
            if content_chars_matched > content_match_len:
                break
            chunk_end = chunk_start + i

        chunk_end += 1  # Include the last character
        current_pos = chunk_end

        chunk_ranges.append((chunk_start, chunk_end))

    if pb_lazy_clean:
        chunk_ranges_ = []
        for i in range(len(chunk_ranges)):
            if i < len(chunk_ranges) - 1:
                chunk_range_new = (chunk_ranges[i][0], chunk_ranges[i + 1][0])
                chunk_ranges_.append(chunk_range_new)
            else:
                chunk_ranges_.append((chunk_ranges[i][0], len(full_text)))
        chunk_ranges = chunk_ranges_

    return chunk_ranges

