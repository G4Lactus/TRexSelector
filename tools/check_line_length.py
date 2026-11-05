#!/usr/bin/env python3
"""Report tracked text files with lines over the project ruler (120 characters).

Usage: python3 tools/check_line_length.py [--max 120] [--summary] [PATH ...]

Exempt: binary and data files (images, CSV, RDS, RDA, BIN, PDF), generated files
(the pybind11 stubs, the roxygen man pages, RcppExports, inst/CITATION, the
vendored C++ copy of the R package, the Makevars written by bundle_for_cran.sh),
third-party style files (CSL), Markdown table rows (a row cannot break), and
lines that carry a URL or a DOI. Exit status 1 when a line is reported, so the
script serves as a CI step.
"""
import argparse
import re
import subprocess
import sys

EXEMPT_SUFFIXES = (".png", ".pdf", ".jpg", ".jpeg", ".svg", ".ico", ".bin", ".rds", ".rda",
                   ".csv", ".tsv", ".pyi", ".Rd", ".csl", ".lock")
EXEMPT_PATTERNS = (
    re.compile(r"(^|/)RcppExports\.(R|cpp)$"),
    re.compile(r"^R/TRexSelector/src/(ml_methods|trex_selector_methods|tsolvers|utils)/"),
    re.compile(r"^R/TRexSelector/src/Makevars(\.in|\.win)?$"),
    re.compile(r"^R/TRexSelector/configure$"),
    re.compile(r"^R/TRexSelector/inst/CITATION$"),
    re.compile(r"^\.clang-tidy$"),
)
URL_RE = re.compile(r"https?://|doi\.org/|doi: 10\.")
TABLE_ROW_RE = re.compile(r"^\s*\|")


def tracked_files(paths):
    out = subprocess.run(["git", "ls-files", "--", *paths], capture_output=True, text=True, check=True)
    return [f for f in out.stdout.splitlines() if f]


def exempt(path):
    if path.endswith(EXEMPT_SUFFIXES):
        return True
    return any(p.search(path) for p in EXEMPT_PATTERNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="*", default=["."])
    parser.add_argument("--max", type=int, default=120)
    parser.add_argument("--summary", action="store_true", help="counts per top-level folder only")
    args = parser.parse_args()

    hits = []
    for path in tracked_files(args.paths):
        if exempt(path):
            continue
        try:
            with open(path, encoding="utf-8") as fh:
                for number, line in enumerate(fh, 1):
                    text = line.rstrip("\n")
                    if len(text) > args.max and not URL_RE.search(text) and not TABLE_ROW_RE.match(text):
                        hits.append((path, number, len(text)))
        except (UnicodeDecodeError, OSError):
            continue

    if args.summary:
        per_folder = {}
        files = {}
        for path, _, _ in hits:
            top = path.split("/", 1)[0] if "/" in path else "(root)"
            per_folder[top] = per_folder.get(top, 0) + 1
            files.setdefault(top, set()).add(path)
        for top in sorted(per_folder):
            print(f"{top:28s} {per_folder[top]:5d} lines in {len(files[top]):3d} files")
        print(f"{'total':28s} {len(hits):5d} lines in {len({h[0] for h in hits}):3d} files")
    else:
        for path, number, length in hits:
            print(f"{path}:{number}: {length}")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
