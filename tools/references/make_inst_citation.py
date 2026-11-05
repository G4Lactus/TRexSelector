#!/usr/bin/env python3
"""Write R/TRexSelector/inst/CITATION from references.bib.

The package entry (author "TRexSelector Team", version from DESCRIPTION) is
fixed here; the method papers are taken from the master bibliography so
that their text versions match the class documentation of all three
languages (see render_references.py).

Usage:
    make_inst_citation.py            # writes the file
    make_inst_citation.py --print    # prints it instead
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render_references import BIB_PATH, latex_to_text, parse_bib, render  # noqa: E402

ROOT = BIB_PATH.parent
PKG = ROOT / "R" / "TRexSelector"
OUT = PKG / "inst" / "CITATION"

# Method papers of the T-Rex framework, in citation order.
METHOD_KEYS = [
    "machkour2025trex",
    "machkour2025datrex",
    "machkour2022gvs",
    "machkour2023ien",
    "machkour2023screen",
    "machkour2024spca",
    "scheidt2023bigtrex",
    "scheidt2026greedy",
    "koka2026vdtrex",
    "machkour2025indextracking",
    "machkour2024ar1",
]

_R_TYPES = {"article": "Article", "inproceedings": "InProceedings", "online": "Misc"}


def r_string(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def r_person(name):
    name = latex_to_text(name)
    if "," in name:
        family, given = [p.strip() for p in name.split(",", 1)]
        return f'as.person("{given} {family}")'
    return f'as.person("{name}")'


def r_authors(field):
    persons = [r_person(n) for n in re.split(r"\s+and\s+", field)]
    if len(persons) == 1:
        return persons[0]
    return "c(" + ",\n                   ".join(persons) + ")"


def bibentry(entry):
    kind = _R_TYPES[entry["type"]]
    fields = [("title", r_string(latex_to_text(entry["title"]))),
              ("author", r_authors(entry["author"]))]
    if kind == "Article":
        fields.append(("journal", r_string(latex_to_text(entry["journal"]))))
        if "volume" in entry:
            fields.append(("volume", r_string(entry["volume"])))
        if "number" in entry:
            fields.append(("number", r_string(entry["number"])))
        if "pages" in entry:
            fields.append(("pages", r_string(entry["pages"].replace("--", "-"))))
        elif "eid" in entry:
            fields.append(("pages", r_string(entry["eid"])))
    elif kind == "InProceedings":
        fields.append(("booktitle", r_string(latex_to_text(entry["booktitle"]))))
        if "pages" in entry:
            fields.append(("pages", r_string(entry["pages"].replace("--", "-"))))
    else:  # arXiv preprint
        fields.append(("note", r_string("arXiv:" + entry["eprint"])))
    fields.append(("year", r_string(entry["year"])))
    if "doi" in entry:
        fields.append(("doi", r_string(entry["doi"])))
    fields.append(("textVersion", r_string(render(entry))))
    width = max(len(name) for name, _ in fields)
    body = ",\n".join(f"  {name:<{width}} = {value}" for name, value in fields)
    return f'bibentry("{kind}",\n{body}\n)'


def package_version():
    text = (PKG / "DESCRIPTION").read_text(encoding="utf-8")
    return re.search(r"^Version:\s*(\S+)", text, re.M).group(1)


def build():
    entries = parse_bib(BIB_PATH.read_text(encoding="utf-8"))
    version = package_version()
    pkg_text = ("TRexSelector Team, \\\"TRexSelector: T-Rex Selector: High-Dimensional Variable "
                f"Selection & FDR Control,\\\" 2026, R package version {version}. [Online]. "
                "Available: https://github.com/trexselector/TRexSelector")
    head = f'''citHeader("To cite 'TRexSelector' in publications, please use the package entry and the
          paper of the method you used:")

bibentry("Manual",
  title       = "{{TRexSelector: T-Rex Selector: High-Dimensional Variable Selection & FDR Control}}",
  author      = person(given = "TRexSelector Team"),
  note        = "R package version {version}",
  year        = "2026",
  url         = "https://github.com/trexselector/TRexSelector",
  textVersion = "{pkg_text}"
)
'''
    blocks = [bibentry(entries[key]) for key in METHOD_KEYS]
    tail = '''
citFooter("The library builds on Eigen: G. Guennebaud, B. Jacob et al., Eigen v3, https://libeigen.gitlab.io/, 2010.")
'''
    return head + "\n" + "\n\n".join(blocks) + "\n" + tail


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--print", action="store_true", help="print instead of writing")
    args = parser.parse_args()
    text = build()
    if args.print:
        print(text, end="")
    else:
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(METHOD_KEYS)} method papers)")


if __name__ == "__main__":
    main()
