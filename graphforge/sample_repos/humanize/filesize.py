"""Bits and bytes related humanization."""

suffixes = {
    "decimal": ("kB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"),
    "binary": ("KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"),
    "gnu": "KMGTPEZY",
}


def naturalsize(value, binary=False, gnu=False, format="%.1f"):
    """Format a number of bytes as a human-readable file size (e.g. 10 kB).

    By default, decimal suffixes (kB, MB) are used.

    Examples:
        >>> naturalsize(3000000)
        '3.0 MB'
        >>> naturalsize(300, False, True)
        '300B'
        >>> naturalsize(3000, True)
        '2.9 KiB'
    """
    if gnu:
        suffix = suffixes["gnu"]
    elif binary:
        suffix = suffixes["binary"]
    else:
        suffix = suffixes["decimal"]

    base = 1024 if (gnu or binary) else 1000
    bytes_ = float(value)
    abs_bytes = abs(bytes_)

    if abs_bytes == 1 and not gnu:
        return "%d Byte" % bytes_
    elif abs_bytes < base and not gnu:
        return "%d Bytes" % bytes_
    elif abs_bytes < base and gnu:
        return "%dB" % bytes_

    for i, s in enumerate(suffix):
        unit = base ** (i + 2)
        if abs_bytes < unit and not gnu:
            return (format + " %s") % ((base * bytes_ / unit), s)
        elif abs_bytes < unit and gnu:
            return (format + "%s") % ((base * bytes_ / unit), s)
    if gnu:
        return (format + "%s") % ((base * bytes_ / unit), s)
    return (format + " %s") % ((base * bytes_ / unit), s)
