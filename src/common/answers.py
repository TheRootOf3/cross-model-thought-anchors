"""MATH answer extraction and equivalence.

The four functions below (extract_boxed_answers, normalize_answer, check_answer,
get_latex_equivalent, prepare_latex_for_sympy, normalize_latex) are COPIED
VERBATIM from:

    external/thought-anchors/utils.py, lines 111-328
    (submodule external/thought-anchors @ b53ed8c)

They are the author's own extraction and equivalence, so Phase 1 reader answers
are judged the same way the released A-rollout `is_correct` flags were.
Do not "improve" them: any change breaks comparability with the dataset.
Anything added by this project goes below the COPIED BLOCK ENDS marker.
"""

import re
from typing import List


def extract_boxed_answers(text: str) -> List[str]:
    """
    Extract answers enclosed in \boxed{} from the text with improved handling
    of nested braces and complex LaTeX expressions.

    Args:
        text: The text to extract boxed answers from

    Returns:
        List of extracted boxed answers
    """
    # Find all occurrences of \boxed{
    boxed_starts = [m.start() for m in re.finditer(r"\\boxed\{", text)]

    if not boxed_starts:
        return [""]

    answers = []

    for start_idx in boxed_starts:
        # Start after \boxed{
        idx = start_idx + 7
        brace_count = 1  # We've already opened one brace
        answer = ""

        # Parse until we find the matching closing brace
        while idx < len(text) and brace_count > 0:
            char = text[idx]

            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1

                # Skip the closing brace of \boxed{}
                if brace_count == 0:
                    break

            if brace_count > 0:  # Only add if we're still inside the boxed content
                answer += char

            idx += 1

        if answer:
            answers.append(answer)

    return answers if answers else [""]


def normalize_answer(answer: str, use_sympy: bool = False) -> str:
    """
    Get the final normalized and cleaned version of an answer.
    This function combines all normalization steps used in check_answer.

    Args:
        answer: The answer string to normalize
        use_sympy: Whether to use sympy to normalize the answer

    Returns:
        The normalized answer string
    """
    # First apply basic LaTeX normalization
    normalized = normalize_latex(answer)

    # Also prepare the answer for sympy if applicable
    if use_sympy:
        try:
            sympy_ready = prepare_latex_for_sympy(answer)
            if sympy_ready != normalized and len(sympy_ready) > 0:
                return sympy_ready
        except Exception:
            pass

    return normalized


def check_answer(answer: str, gt_answer: str) -> bool:
    """
    Check if the generated answer matches the ground truth answer
    after normalizing LaTeX formatting.

    Args:
        answer: The generated answer to check
        gt_answer: The ground truth answer to compare against

    Returns:
        True if the answers match after normalization, False otherwise
    """
    # Normalize both answers
    normalized_answer = normalize_latex(answer)
    normalized_gt_answer = normalize_latex(gt_answer)

    # First check if normalized strings match
    if normalized_answer == normalized_gt_answer:
        return True

    # If string comparison fails, try mathematical equivalence
    try:
        return get_latex_equivalent(answer, gt_answer)
    except Exception as e:
        # If SymPy parsing fails, fall back to string comparison result
        return False


def get_latex_equivalent(answer0, answer1):
    """
    Check if two LaTeX expressions are mathematically equivalent using SymPy.

    Args:
        answer0: First LaTeX expression
        answer1: Second LaTeX expression

    Returns:
        True if expressions are mathematically equivalent, False otherwise
    """
    try:
        from sympy.parsing.latex import parse_latex
        import sympy

        # Clean up the LaTeX expressions for parsing
        answer0 = prepare_latex_for_sympy(answer0)
        answer1 = prepare_latex_for_sympy(answer1)

        # Parse the LaTeX expressions
        expr1 = parse_latex(answer0)
        expr2 = parse_latex(answer1)

        # Check if they are mathematically identical
        equals = expr1.equals(expr2)
        # print(f"First: {answer0}, Second: {answer1}: equals={equals}")
        return equals
    except Exception as e:
        # print(f"Error comparing expressions: {e}")
        return False


def prepare_latex_for_sympy(latex_str):
    """
    Prepare a LaTeX string for SymPy parsing by removing unsupported commands
    and simplifying the expression.
    """
    if not isinstance(latex_str, str):
        return str(latex_str)

    # Remove \boxed{} command
    latex_str = re.sub(r"\\boxed\{(.*?)\}", r"\1", latex_str)

    # Replace common LaTeX commands that SymPy doesn't support
    replacements = {
        r"\\dfrac": r"\\frac",
        r"\\tfrac": r"\\frac",
        r"\\cdot": r"*",
        r"\\times": r"*",
        r"\\div": r"/",
        r"\\left": r"",
        r"\\right": r"",
        r"\\textbf": r"",
        r"\\text": r"",
        r"\\mathrm": r"",
        r"\\!": r"",
        r",": r"",
    }

    for old, new in replacements.items():
        latex_str = re.sub(old, new, latex_str)

    return latex_str


def normalize_latex(latex_str: str) -> str:
    """
    Normalize LaTeX string by applying various transformations.

    Args:
        latex_str: The LaTeX string to normalize

    Returns:
        Normalized LaTeX string
    """
    normalized = latex_str.strip().lower()

    # Replace different fraction notations
    normalized = normalized.replace("dfrac", "frac")
    normalized = normalized.replace("tfrac", "frac")

    # Normalize spaces
    normalized = re.sub(r"\s+", "", normalized)

    # Normalize percentages
    normalized = normalized.replace("\\%", "")

    # Normalize funny commas
    normalized = normalized.replace("{,}", "")

    # Normalize common mathematical notations
    normalized = normalized.replace("\\times", "*")
    normalized = normalized.replace("\\cdot", "*")

    # Normalize decimal representation
    normalized = re.sub(r"(\d+)[\.,](\d+)", r"\1.\2", normalized)

    # Remove unnecessary braces in simple expressions
    normalized = re.sub(r"{([^{}]+)}", r"\1", normalized)

    # Normalize common constants
    normalized = normalized.replace("\\pi", "pi")

    # Remove LaTeX text commands
    normalized = re.sub(r"\\text\{([^{}]+)\}", r"\1", normalized)
    normalized = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", normalized)

    # Normalize date formats (e.g., "October 30" vs "October\\ 30")
    normalized = re.sub(r"([a-z]+)\\+\s*(\d+)", r"\1\2", normalized)
    normalized = normalized.replace("\\text", "")

    return normalized



# ---------------------------- COPIED BLOCK ENDS ----------------------------


def extract_answer(text: str) -> str | None:
    """FIRST \\boxed{} in a completion, or None - the generator's convention
    (generate_rollouts.py:458, `extracted_answers[0]`), so reader answers are
    judged the same way the dataset's stored rollouts were."""
    # extract_boxed_answers returns [''] (not []) when there is no box. Readers
    # also quote the instruction's literal \\boxed{...} inside their reasoning
    # (A never does), so a placeholder box is skipped, then FIRST of the rest.
    # Parity check: none of 6000 sampled A rollouts has a placeholder first.
    # decided 2026-09-08.
    real = [b for b in extract_boxed_answers(text) if b.strip() not in ("", "...", "\u2026")]
    return real[0] if real else None


def is_correct(answer: str | None, gt_answer: str) -> bool:
    """Always a bool. The generator short-circuits on a falsy answer
    (generate_rollouts.py: `if gt_answer and answer`); sympy's .equals() can
    return None when undecided, which check_answer passes through."""
    if not answer:
        return False
    # LaTeX spacing commands (\\, \\! \\; \\: "\\ ") are typography, not content:
    # gpt-oss writes \\boxed{88\\,572} for 88572 and normalize_latex leaves the
    # \\, in. 0/2400 sampled stored A answers contain one, so this is
    # parity-neutral on the dataset (decided 2026-09-08).
    answer = _strip_latex_spacing(answer)
    try:
        return bool(check_answer(answer, _strip_latex_spacing(gt_answer)))
    except Exception:
        return False


def _strip_latex_spacing(s: str) -> str:
    return re.sub(r"\\[,!;: ]", "", s)


def p_correct(rows: list[dict]) -> dict:
    """P(correct) the way the dataset computes it: rollouts with no answer
    (cap hit, no box) are dropped from numerator AND denominator - the same
    rule in every accuracy thought-anchors computes (analyze_rollouts.py
    :745-770 dissimilar/next, :1502-1514 per-chunk, :1463-1479 forced).

    `p_correct` is None when nothing was answered: "no data" must not become
    "0% correct", which would enter an importance difference as a maximal
    value. Report n_no_answer beside every p_correct - with heavy capping the
    estimate rests on few samples. decided 2026-09-08.

    `p_answered` is the nuisance statistic: if it differs between the kept and
    replaced piles of a sentence, that pile's importance partly reflects
    differential termination rather than the sentence."""
    answered = [r for r in rows if r.get("answer")]
    n = len(rows)
    return {"p_correct": (sum(bool(r["correct"]) for r in answered) / len(answered)) if answered else None,
            "n_answered": len(answered), "n_total": n, "n_no_answer": n - len(answered),
            "p_answered": (len(answered) / n) if n else None}
