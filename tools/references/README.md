# Scientific references of the class documentation

`references.bib` at the repository root is the master list of every paper the library credits. The class
documentation of C++, R and Python carries the same references as plain text, rendered from that file, so the
three languages never drift apart and no citation machinery (Doxygen `\cite`, Rdpack, Sphinx roles) is needed.

## Files

| File | Purpose |
| --- | --- |
| `../../references.bib` | Master bibliography. Comment lines above an entry record where its fields were read from. |
| `references_map.json` | Which keys each documented class cites, in citation order (C++ class names). |
| `render_references.py` | Renders keys as one-line plain-text references, wrapped for each language. |
| `make_inst_citation.py` | Writes `R/TRexSelector/inst/CITATION` (package entry plus the T-Rex method papers). |

Render a block:

```bash
python3 tools/references/render_references.py --format cpp efron2004lars machkour2025trex
python3 tools/references/render_references.py --format r --width 110 efron2004lars
python3 tools/references/render_references.py --format pylit --width 100 efron2004lars
python3 tools/references/render_references.py --format md machkour2025trex   # a Markdown bullet, DOI linked
python3 tools/references/render_references.py --list        # every key
```

The reference list of the root README is rendered with `--format md` (one unwrapped bullet per entry, the DOI
or the arXiv id as a link); the command stands in an HTML comment above the list. The two application papers
of the README's opening story live in the bibliography's "Application references" section.

One reference line reads: initials and surnames, "title," venue, volume, number, pages, year, DOI. The entries
of a class follow the order of `references_map.json`: the origin papers of the method first, then the T-Rex paper
for the terminating form (the greedy-pursuits paper for the greedy solvers).

## Block format per language

The reference block is the last paragraph of the documentation block that belongs to the class (the Doxygen
block directly above the C++ class, the roxygen block above the R6 class or wrapper function, the pybind11 class
docstring). Lines stay within 110 characters.

C++ (Doxygen block, ` * ` line prefix; headers that use `* ` without the leading space use `*  References:`
and the renderer's `--prefix "*   "`):

```text
 *
 *  References:
 *  -----------
 *   B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
 *     no. 2, pp. 407-499, 2004.
 *   J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
 *     High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
 *     Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
 */
```

R (roxygen, placed before `@examples`; a blank `#'` line between entries keeps them separate paragraphs):

```text
#' @references
#' B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, "Least Angle Regression," Ann. Statist., vol. 32,
#'   no. 2, pp. 407-499, 2004.
#'
#' J. Machkour, M. Muma, and D. P. Palomar, "The Terminating-Random Experiments Selector: Fast
#'   High-Dimensional Variable Selection with False Discovery Rate Control," Signal Process., vol. 231,
#'   Art. no. 109894, 2025, doi: 10.1016/j.sigpro.2025.109894.
```

Python (pybind11 class docstring, a numpy-style section appended to the existing text; the `pylit` format
emits the lines as adjacent C++ string literals):

```text
        "...existing docstring...\n"
        "\n"
        "References\n"
        "----------\n"
        "B. Efron, T. Hastie, I. Johnstone, and R. Tibshirani, \"Least Angle Regression,\" Ann.\n"
        "  Statist., vol. 32, no. 2, pp. 407-499, 2004.\n"
```

After docstring edits the `.pyi` stubs under `Python/trexselector/src/trexselector/_core/` are
regenerated with pybind11-stubgen, and after roxygen edits `man/` is regenerated with
`roxygen2::roxygenise("R/TRexSelector", load_code = roxygen2::load_installed)`.

## Rules

- Fields are never typed by hand in the sources: change `references.bib`, re-render, paste.
- Existing hand-written citation lists in the class blocks are replaced by the rendered block; prose mentions
  such as "Section III-A of Blumensath and Davies" stay.
- Comments state the technical content only: no names of people from the project, no work packages, no dates.
- The five-line limit for code comments does not apply to these documentation blocks (decided for the
  documentation refresh).
