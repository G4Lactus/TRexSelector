# ---- .Rprofile  (project local) ---------------------------------
if (requireNamespace("lintr", quietly = TRUE)) {
  options(
    lintr.linters = lintr::linters_with_defaults(
      object_name_linter = NULL, # disable snake_case enforcement
      object_usage_linter = NULL, # disable unused variable squiggles
      line_length_linter = lintr::line_length_linter(120L),
      commented_code_linter = NULL,
      return_linter = NULL       # disable explicit return squiggles
    )
  )
}
# -----------------------------------------------------------------
