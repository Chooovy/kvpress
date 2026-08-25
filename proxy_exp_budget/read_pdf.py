# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Minimal PDF text extractor, for reading a paper on a box with no ``pypdf`` and no ``poppler``.

Not a general PDF library. It handles exactly what a LaTeX-produced paper needs: locate the content
streams, inflate the FlateDecode ones, and pull the strings out of the text-showing operators
(``Tj``, ``TJ``, ``'``, ``"``). Font encodings are not resolved, so a paper using a subsetted font
with a non-identity ``ToUnicode`` map can come out garbled -- which is checked for and reported
rather than silently returned, since a garbled extraction that *looks* like prose is worse than none.

Usage::

    python read_pdf.py /path/to/paper.pdf > paper.txt
"""

from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path


def inflate_streams(raw: bytes) -> list[bytes]:
    """Every FlateDecode stream in the file, inflated. Corrupt ones are skipped."""
    out = []
    for match in re.finditer(rb"stream\r?\n", raw):
        start = match.end()
        end = raw.find(b"endstream", start)
        if end < 0:
            continue
        blob = raw[start:end].rstrip(b"\r\n")
        try:
            out.append(zlib.decompress(blob))
        except zlib.error:
            try:  # some writers leave a raw deflate stream with no zlib header
                out.append(zlib.decompressobj(-15).decompress(blob))
            except zlib.error:
                continue
    return out


#: Matches a PDF literal string, honouring backslash escapes so `\)` does not end it early.
_LITERAL = re.compile(rb"\((?:\\.|[^\\()])*\)", re.S)


def unescape(chunk: bytes) -> str:
    """Decode one PDF literal string's bytes to text."""
    body = chunk[1:-1]
    body = re.sub(rb"\\([nrtbf])", lambda m: {b"n": b"\n", b"r": b"\r", b"t": b"\t",
                                             b"b": b"", b"f": b""}[m.group(1)], body)
    body = re.sub(rb"\\([0-7]{1,3})", lambda m: bytes([int(m.group(1), 8) & 0xFF]), body)
    body = re.sub(rb"\\(.)", rb"\1", body)
    return body.decode("latin-1")


def extract_text(stream: bytes) -> str:
    """Text from one inflated content stream, in operator order."""
    pieces: list[str] = []
    # Split on text-showing operators. TJ takes an array of strings and kerns; Tj/'/" take one.
    for match in re.finditer(rb"\[((?:\\.|[^\[\]\\]|\\\])*)\]\s*TJ|(\((?:\\.|[^\\()])*\))\s*(?:Tj|'|\")",
                             stream, re.S):
        if match.group(1) is not None:
            pieces.append("".join(unescape(s) for s in _LITERAL.findall(match.group(1))))
        else:
            pieces.append(unescape(match.group(2)))
        pieces.append(" ")
    # Newlines: Td/TD/T*/TL moves and ET are the paragraph-ish boundaries. Cheap approximation --
    # insert a break wherever the stream repositioned the text cursor between shows.
    text = "".join(pieces)
    return text


def main() -> None:
    path = Path(sys.argv[1])
    raw = path.read_bytes()
    streams = inflate_streams(raw)
    chunks = []
    for s in streams:
        if b"Tj" in s or b"TJ" in s:
            t = extract_text(s)
            if t.strip():
                chunks.append(t)
    text = "\n\n".join(chunks)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # Sanity gate: a subsetted font with a custom encoding yields high-entropy junk that still looks
    # like "text". Require that common English words survive, and say so loudly if they do not.
    lowered = text.lower()
    hits = sum(lowered.count(w) for w in (" the ", " of ", " and ", " that ", " is ", " to "))
    sys.stderr.write(
        f"[read_pdf] {len(streams)} streams, {len(chunks)} with text, {len(text)} chars, "
        f"{hits} common-word hits\n"
    )
    if hits < 20:
        sys.stderr.write(
            "[read_pdf] WARNING: almost no common English words found. The fonts are probably "
            "subsetted with a non-identity encoding, so this extraction is NOT trustworthy.\n"
        )
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
