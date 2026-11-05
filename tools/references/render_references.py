#!/usr/bin/env python3
"""Render plain-text reference lines from references.bib.

The master bibliography at the repository root is the single source of the
scientific references. This script renders one plain-text line per entry
(authors, title, venue, volume, pages, year, DOI) and wraps it as a comment
block for the C++ headers, the R roxygen blocks or the Python docstrings, so
that the three languages carry identical text.

Usage:
    render_references.py --list
    render_references.py [--format plain|cpp|r|py|pylit|md] [--width N] KEY [KEY ...]
    render_references.py --all

No third-party dependency: the parser covers the subset of BibTeX used in
references.bib (braced values, nested braces, "and"-separated authors).
"""

import argparse
import re
import sys
import textwrap
from pathlib import Path

BIB_PATH = Path(__file__).resolve().parents[2] / "references.bib"

# LaTeX accent commands to Unicode, applied before braces are stripped.
_ACCENTS = {
    "'": {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "y": "ý", "A": "Á", "E": "É", "O": "Ó"},
    "`": {"a": "à", "e": "è", "i": "ì", "o": "ò", "u": "ù", "A": "À", "E": "È"},
    '"': {"a": "ä", "e": "ë", "i": "ï", "o": "ö", "u": "ü", "A": "Ä", "O": "Ö", "U": "Ü"},
    "^": {"a": "â", "e": "ê", "i": "î", "o": "ô", "u": "û"},
    "~": {"n": "ñ", "a": "ã", "o": "õ"},
    "c": {"c": "ç", "C": "Ç"},
}


def latex_to_text(value):
    """Convert the LaTeX subset used in the bibliography to plain Unicode text."""
    text = value

    def accent(match):
        cmd, letter = match.group(1), match.group(2)
        return _ACCENTS.get(cmd, {}).get(letter, letter)

    # {\'e}, \'e, {\"o}, \"o, {\^i}, {\c c}
    text = re.sub(r"\{\\(['`\"^~c])\s?\{?([A-Za-z])\}?\}", accent, text)
    text = re.sub(r"\\(['`\"^~c])\s?\{?([A-Za-z])\}?", accent, text)
    text = text.replace(r"\&", "&")
    text = re.sub(r"\\emph\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\url\{([^{}]*)\}", r"\1", text)
    text = text.replace("{", "").replace("}", "")
    text = text.replace("---", "-").replace("--", "-")
    text = text.replace("``", '"').replace("''", '"')
    return " ".join(text.split())


def parse_bib(text):
    """Parse the bibliography into {key: {"type": ..., fields...}}."""
    entries = {}
    for match in re.finditer(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,", text, re.M):
        kind = match.group(1).lower()
        if kind in ("comment", "string", "preamble"):
            continue
        key = match.group(2)
        start = text.index("{", match.start())
        depth, pos = 0, start
        while True:
            char = text[pos]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    break
            pos += 1
        body = text[match.end():pos]
        fields = {"type": kind}
        for field in _split_top_level(body):
            if "=" not in field:
                continue
            name, value = field.split("=", 1)
            value = value.strip()
            if value.startswith("{") and value.endswith("}"):
                value = value[1:-1]
            elif value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            fields[name.strip().lower()] = " ".join(value.split())
        entries[key] = fields
    return entries


def _split_top_level(body):
    parts, depth, current = [], 0, []
    for char in body:
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        if char == "," and depth == 0:
            parts.append("".join(current))
            current = []
        else:
            current.append(char)
    parts.append("".join(current))
    return [p.strip() for p in parts if p.strip()]


def format_author(name):
    """'Surname, Given M.' or 'Given M. Surname' -> 'G. M. Surname'."""
    name = latex_to_text(name)
    if name == "others":
        return "et al."
    if "," in name:
        surname, given = [part.strip() for part in name.split(",", 1)]
    else:
        tokens = name.split()
        surname, given = tokens[-1], " ".join(tokens[:-1])
    initials = []
    for token in given.split():
        # keep hyphenated initials such as J.-P.
        pieces = token.split("-")
        initials.append("-".join(p[0] + "." for p in pieces if p))
    return (" ".join(initials) + " " + surname).strip()


def format_authors(field):
    names = [format_author(n) for n in re.split(r"\s+and\s+", field)]
    if names and names[-1] == "et al.":
        names = names[:-1]
        head = ", ".join(names)
        return head + " et al."
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return names[0] + " and " + names[1]
    return ", ".join(names[:-1]) + ", and " + names[-1]


def render(entry):
    """One plain-text line in the IEEE-like order used by the T-Rex papers."""
    kind = entry["type"]
    get = lambda name: latex_to_text(entry.get(name, ""))
    authors = format_authors(entry["author"]) if "author" in entry else ""
    title = get("title")
    year = get("year")
    doi = get("doi")
    pages = get("pages")
    tail = []

    if kind == "article":
        parts = [f'{authors}, "{title}," {get("journal")}']
        if get("volume"):
            parts.append(f"vol. {get('volume')}")
        if get("number"):
            parts.append(f"no. {get('number')}")
        if pages:
            parts.append(f"pp. {pages}")
        if get("eid"):
            parts.append(f"Art. no. {get('eid')}")
        parts.append(year)
        line = ", ".join(parts)
    elif kind == "inproceedings":
        parts = [f'{authors}, "{title}," in {get("booktitle")}']
        if get("series"):
            parts.append(f"{get('series')} vol. {get('volume')}" if get("volume") else get("series"))
        parts.append(year)
        if pages:
            parts.append(f"pp. {pages}")
        line = ", ".join(parts)
    elif kind == "book":
        parts = [f"{authors}, {title}"]
        if get("series"):
            parts.append(f"{get('series')}, no. {get('number')}" if get("number") else get("series"))
        line = ". ".join(parts) + f". {get('publisher')}"
        if get("address"):
            line += f", {get('address')}"
        line += f", {year}"
    elif kind == "unpublished":
        note = _lower_first(get("note")) or "unpublished"
        line = f'{authors}, "{title}," {note}'
        if not get("note"):
            line += f", {year}"
    elif kind == "online":
        line = f'{authors}, "{title}," arXiv:{get("eprint")}, {year}'
        if get("note"):
            tail.append(get("note"))
    else:  # misc and anything else
        parts = [f"{authors}, {title}" if authors else title]
        if get("url"):
            parts.append(get("url"))
        parts.append(year)
        line = ", ".join(p for p in parts if p)

    if kind in ("article", "book") and get("note"):
        tail.append(get("note"))
    if doi:
        line += f", doi: {doi}"
    if tail:
        line += " (" + "; ".join(tail) + ")"
    return line + "."


def _lower_first(text):
    return text[:1].lower() + text[1:] if text else text


# Line prefix per output format. "cpp" is the Doxygen block style of the headers
# (" * " lines, references indented by two more spaces, continuation by two
# more), "r" the roxygen style, "pylit" C++ string literals for a pybind11
# docstring, "py" an indented plain block, "md" one Markdown bullet per entry
# with the DOI (or arXiv id) as a link, unwrapped (the README lists).
_PREFIXES = {"plain": "", "cpp": " *   ", "r": "#' ", "py": "    ", "pylit": "        ", "md": "* "}

_DOI_RE = re.compile(r"doi: (\S+?)(\.?)$")
_ARXIV_RE = re.compile(r"arXiv:(\d{4}\.\d{4,5}(?:v\d+)?)")


def _markdown(line):
    """Link the DOI or the arXiv id of a rendered line."""
    line = _DOI_RE.sub(lambda m: f"[doi: {m.group(1)}](https://doi.org/{m.group(1)}){m.group(2)}", line)
    return _ARXIV_RE.sub(lambda m: f"[arXiv:{m.group(1)}](https://arxiv.org/abs/{m.group(1)})", line)

# Tokens that must stay on one line with the word that follows them.
_GLUE = ("vol.", "no.", "pp.", "Art.", "doi:")


def _wrap(text, width):
    for token in _GLUE:
        text = text.replace(token + " ", token + "\x00")
    wrapped = textwrap.wrap(text, width=width, subsequent_indent="  ",
                            break_long_words=False, break_on_hyphens=False)
    return [w.replace("\x00", " ") for w in wrapped]


def _literal(text):
    """One line of a pybind11 docstring as a C++ string literal."""
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '\\n"'


def render_block(entries, keys, fmt, width, prefix=None):
    prefix = _PREFIXES[fmt] if prefix is None else prefix
    blocks = []
    for key in keys:
        if key not in entries:
            sys.exit(f"unknown key: {key}")
        if fmt == "md":
            blocks.append(prefix + _markdown(render(entries[key])))
            continue
        wrapped = _wrap(render(entries[key]), width - len(prefix) - (4 if fmt == "pylit" else 0))
        if fmt == "pylit":
            blocks.append("\n".join(prefix + _literal(w) for w in wrapped))
        else:
            blocks.append("\n".join(prefix + w for w in wrapped))
    if fmt == "r":
        # roxygen: a blank "#'" line between entries keeps them separate paragraphs
        return "\n#'\n".join(blocks)
    return "\n".join(blocks)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("keys", nargs="*", help="bibliography keys to render")
    parser.add_argument("--format", choices=sorted(_PREFIXES), default="plain")
    parser.add_argument("--width", type=int, default=110)
    parser.add_argument("--prefix", help="override the line prefix of the format")
    parser.add_argument("--list", action="store_true", help="list the keys")
    parser.add_argument("--all", action="store_true", help="render every entry")
    parser.add_argument("--bib", type=Path, default=BIB_PATH)
    args = parser.parse_args()

    entries = parse_bib(args.bib.read_text(encoding="utf-8"))
    if args.list:
        print("\n".join(entries))
        return
    keys = list(entries) if args.all else args.keys
    if not keys:
        parser.error("give keys, --all or --list")
    print(render_block(entries, keys, args.format, args.width, args.prefix))


if __name__ == "__main__":
    main()
