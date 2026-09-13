"""Sentence splitter.

split_solution_into_chunks is COPIED from:

    external/thought-anchors/utils.py, lines 360-449
    (submodule external/thought-anchors @ b53ed8c), with the "add the last
    chunk" block restored as it was BEFORE commit ff60857 (2025-08-07) - the
    version that produced the released dataset (2026-09-09; see the comment
    in the function). Verified: re-splitting every stored solution_text
    reproduces the released chunks in 40/40 traces with this block, 0/40
    without it.

The dataset's `chunks` were produced by this exact function, so reader
continuations must be split by it too - otherwise "first sentence of B's
continuation" would not mean the same thing as A's sentence i.
"""

from typing import List

def split_solution_into_chunks(solution_text: str) -> List[str]:
    """
    Split a solution into chunks for rollout generation.

    Args:
        solution_text: The full solution text

    Returns:
        List of chunks
    """
    # First, remove the prompt part if present
    if "<think>" in solution_text:
        solution_text = solution_text.split("<think>")[1].strip()

    # Remove the closing tag if present
    if "</think>" in solution_text:
        solution_text = solution_text.split("</think>")[0].strip()

    # Define patterns for chunk boundaries
    sentence_ending_tokens = [".", "?", "!"]
    paragraph_ending_patterns = ["\n\n", "\r\n\r\n"]

    # Split the text into chunks
    chunks = []
    current_chunk = ""

    # Process the text character by character
    i = 0
    while i < len(solution_text):
        current_chunk += solution_text[i]

        # Check for paragraph endings
        is_paragraph_end = False
        for pattern in paragraph_ending_patterns:
            if (
                i + len(pattern) <= len(solution_text)
                and solution_text[i : i + len(pattern)] == pattern
            ):
                is_paragraph_end = True
                break

        # Check for sentence endings followed by space or newline
        is_sentence_end = False
        if i < len(solution_text) - 1 and solution_text[i] in sentence_ending_tokens:
            next_char = solution_text[i + 1]
            if next_char == " " or next_char == "\n":
                is_sentence_end = True

        # If we found a boundary, add the chunk and reset
        if is_paragraph_end or is_sentence_end:
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
                current_chunk = ""

        i += 1

    # Add the last chunk if not empty. This block is the DATASET's version
    # (thought-anchors utils.py before commit ff60857 of 2025-08-07, which
    # commented it out after the data was generated): with it, re-splitting
    # every stored solution_text reproduces the released chunks in 40/40
    # traces; without it, 0/40 (each one chunk short). Restored 2026-09-09.
    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    # Merge small chunks (less than 10 characters)
    i = 0
    while i < len(chunks):
        if len(chunks[i]) < 10:
            # If this is the last chunk, merge with previous chunk if possible
            if i == len(chunks) - 1:
                if i > 0:
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    chunks.pop(i)
            # Otherwise merge with the next chunk
            else:
                chunks[i + 1] = chunks[i] + " " + chunks[i + 1]
                chunks.pop(i)
                # Don't increment i since we need to check the new merged chunk
            # If we're at the beginning and there's only one chunk, just keep it
            if i == 0 and len(chunks) == 1:
                break
        else:
            i += 1

    # chunk_boundaries = [(chunk_idxs[i], chunk_idxs[i + 1]) for i in range(len(chunk_idxs) - 1)]
    # chunk_boundaries.append((chunk_idxs[-1], len(solution_text)))

    # if get_idxs:
    # return chunks, chunk_boundaries
    # else:
    return chunks



# ---------------------------- COPIED BLOCK ENDS ----------------------------


def first_sentence(text: str) -> str | None:
    """First chunk of a continuation, or None if it splits to nothing.
    Used for the replaced pile: B's replacement for A's sentence i."""
    chunks = split_solution_into_chunks(text)
    return chunks[0] if chunks else None
