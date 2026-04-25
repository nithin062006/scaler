"""Humanizing functions for numbers."""

import math
import re
from fractions import Fraction

powers = [10**x for x in (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 100)]
human_powers = (
    "thousand", "million", "billion", "trillion", "quadrillion",
    "quintillion", "sextillion", "septillion", "octillion",
    "nonillion", "decillion", "googol",
)


def ordinal(value):
    """Convert an integer to its ordinal string (1 → '1st', 2 → '2nd', etc.).

    Examples:
        >>> ordinal(1)
        '1st'
        >>> ordinal(12)
        '12th'
        >>> ordinal(103)
        '103rd'
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    t = ("th", "st", "nd", "rd", "th", "th", "th", "th", "th", "th")
    if value % 100 in (11, 12, 13):
        return f"{value}th"
    return f"{value}{t[value % 10]}"


def intcomma(value, ndigits=None):
    """Convert an integer to a string with commas every three digits.

    Examples:
        >>> intcomma(1000000)
        '1,000,000'
        >>> intcomma(1234567.25)
        '1,234,567.25'
    """
    try:
        if isinstance(value, str):
            float(value.replace(",", ""))
        else:
            float(value)
    except (TypeError, ValueError):
        return value

    if ndigits:
        orig = "{0:.{1}f}".format(value, ndigits)
    else:
        orig = str(value)

    new = re.sub(r"^(-?\d+)(\d{3})", r"\g<1>,\g<2>", orig)
    if orig == new:
        return new
    return intcomma(new)


def intword(value, format="%.1f"):
    """Convert a large integer to a friendly text representation.

    Examples:
        >>> intword(1000000)
        '1.0 million'
        >>> intword(1200000000)
        '1.2 billion'
    """
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if value < powers[0]:
        return str(value)
    for ordinal_idx, power in enumerate(powers[1:], 1):
        if value < power:
            chopped = value / float(powers[ordinal_idx - 1])
            count = math.ceil(chopped)
            label = human_powers[ordinal_idx - 1]
            plural = label + "s" if count != 1 else label
            if float(format % chopped) == float(10**3):
                chopped = value / float(powers[ordinal_idx])
                count = math.ceil(chopped)
                label = human_powers[ordinal_idx]
                plural = label + "s" if count != 1 else label
                return (format + " %s") % (chopped, plural)
            return (format + " %s") % (chopped, plural)
    return str(value)


def apnumber(value):
    """Convert integers 0–9 to their AP-style word equivalents.

    Examples:
        >>> apnumber(5)
        'five'
        >>> apnumber(10)
        '10'
    """
    words = ("zero", "one", "two", "three", "four",
             "five", "six", "seven", "eight", "nine")
    try:
        value = int(value)
    except (TypeError, ValueError):
        return value
    if not 0 <= value < 10:
        return str(value)
    return words[value]


def fractional(value):
    """Convert a float to a human-readable fractional string.

    Examples:
        >>> fractional(0.3)
        '3/10'
        >>> fractional(1.3)
        '1 3/10'
        >>> fractional(1)
        '1'
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    whole = int(number)
    frac = Fraction(number - whole).limit_denominator(1000)
    n, d = frac.numerator, frac.denominator
    if whole and not n and d == 1:
        return f"{whole:.0f}"
    elif not whole:
        return f"{n:.0f}/{d:.0f}"
    return f"{whole:.0f} {n:.0f}/{d:.0f}"


def scientific(value, precision=2):
    """Return a number in scientific notation (e.g. 5.00 x 10²).

    Examples:
        >>> scientific(500)
        '5.00 x 10²'
        >>> scientific(0.3)
        '3.00 x 10⁻¹'
    """
    exponents = {
        "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴",
        "5": "⁵", "6": "⁶", "7": "⁷", "8": "⁸", "9": "⁹",
        "+": "⁺", "-": "⁻",
    }
    negative = False
    try:
        if "-" in str(value):
            value = str(value).replace("-", "")
            negative = True
        if isinstance(value, str):
            value = float(value)
        fmt = "{:.%se}" % str(int(precision))
        n = fmt.format(value)
    except (ValueError, TypeError):
        return value
    part1, part2 = n.split("e")
    part2 = part2.replace("-0", "-").replace("+0", "")
    new_part2 = []
    if negative:
        new_part2.append(exponents["-"])
    for char in part2:
        new_part2.append(exponents[char])
    return part1 + " x 10" + "".join(new_part2)


def clamp(value, format="{:}", floor=None, ceil=None, floor_token="<", ceil_token=">"):
    """Return a number formatted and clamped between floor and ceil.

    Examples:
        >>> clamp(123.456)
        '123.456'
        >>> clamp(0.001, floor=0.01)
        '<0.01'
        >>> clamp(999, ceil=100)
        '>100'
    """
    if value is None:
        return None
    if floor is not None and value < floor:
        value, token = floor, floor_token
    elif ceil is not None and value > ceil:
        value, token = ceil, ceil_token
    else:
        token = ""
    if isinstance(format, str):
        return token + format.format(value)
    elif callable(format):
        return token + format(value)
    raise ValueError("format must be a string or callable")
