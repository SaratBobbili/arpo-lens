import sys
import os
import re
import math
sys.path.append(os.getcwd())

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string

def _fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def _unwrap_text_braces(string):
    # Strip bare \text{X} wrappers produced by MATH500 ground truths like
    # "\\text{Evelyn}", "\\text{(C)}", so they match plain predictions "Evelyn" / "(C)".
    # Only removes the wrapper; anything left inside the braces is preserved verbatim.
    pattern = re.compile(r"\\text\{([^{}]*)\}")
    prev = None
    while prev != string:
        prev = string
        string = pattern.sub(r"\1", string)
    return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    string = string.replace("$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # Unwrap plain \text{...} wrappers (non-unit) so e.g. \text{east} == east.
    string = _unwrap_text_braces(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # strip digit-grouping commas (1,000 -> 1000) without touching tuple/list separators
    string = re.sub(r"(?<=\d),(?=\d{3}(?:\D|$))", "", string)

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    # Canonicalize plain decimal forms so numerically equivalent strings match.
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", string):
        string = string.rstrip("0").rstrip(".") if "." in string else string

    return string

def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = _strip_string(str1)
        ss2 = _strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except:
        return str1 == str2


def _to_number(s):
    """Parse a scalar answer (decimal, a/b, or \\frac{a}{b}) to float, else None."""
    if s is None:
        return None
    t = _strip_string(str(s))
    m = re.fullmatch(r"\\frac\{(-?\d+(?:\.\d+)?)\}\{(-?\d+(?:\.\d+)?)\}", t)
    if not m:
        m = re.fullmatch(r"(-?\d+(?:\.\d+)?)/(-?\d+(?:\.\d+)?)", t)
    if m:
        num, den = float(m.group(1)), float(m.group(2))
        return num / den if den != 0 else None
    try:
        return float(t)
    except ValueError:
        return None


def numeric_equal(str1, str2, rel_tol=1e-9, abs_tol=1e-9):
    """True when both parse to numbers that are close (0.5 == \\frac12 == 1/2)."""
    n1, n2 = _to_number(str1), _to_number(str2)
    if n1 is None or n2 is None:
        return False
    return math.isclose(n1, n2, rel_tol=rel_tol, abs_tol=abs_tol)


_SAFE_EXPR_RE = re.compile(r"^[\sA-Za-z0-9_+\-*/^().,{}\\]+$")
_UNSAFE_TOKENS = ("__", "import", "lambda", "eval", "exec")


def _looks_safe_expr(s):
    if not s or len(s) > 200:
        return False
    if not _SAFE_EXPR_RE.match(s):
        return False
    return not any(tok in s for tok in _UNSAFE_TOKENS)


def _prep_sympy(s):
    """Convert common LaTeX forms to a sympify-parseable expression string."""
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\pi", "pi")
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"((\1)/(\2))", s)
    s = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", s)
    s = s.replace("\\", "")
    s = s.replace("^", "**")
    s = s.replace("{", "(").replace("}", ")")
    return s


def symbolic_equal(str1, str2):
    """True when both parse to symbolic expressions whose difference simplifies to 0.
    Guarded to safe expression strings only (e.g. x + 11 == 11 + x)."""
    if str1 is None or str2 is None:
        return False
    e1, e2 = _prep_sympy(str1.strip()), _prep_sympy(str2.strip())
    if not (_looks_safe_expr(e1) and _looks_safe_expr(e2)):
        return False
    try:
        from sympy import simplify, sympify
        a, b = sympify(e1), sympify(e2)
        return bool(simplify(a - b) == 0)
    except Exception:
        return False


def math_answers_equal(prediction, reference):
    """Trustworthy semantic equality: string-normalized OR numeric OR symbolic."""
    if prediction is None or reference is None:
        return prediction is None and reference is None
    if is_equiv(prediction, reference):
        return True
    if numeric_equal(prediction, reference):
        return True
    if symbolic_equal(prediction, reference):
        return True
    return False