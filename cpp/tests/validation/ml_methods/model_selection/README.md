# Model-selection reference: coordinate-descent EN vs R `glmnet`

## What this is

`demo_en_glmnet_compare.R` is the R **reference generator** for the C++ coordinate-descent
elastic-net solver
(`trex::ml_methods::model_selection::elastic_net_gaussian` / `elastic_net_cv_gaussian`)
against R `glmnet` / `cv.glmnet`, on the same sparse factor-model design used for the SPCA
and t-solver evaluations. Because `glmnet` **is** cyclic coordinate descent, the C++ engine
matches its coefficient *path* to solver tolerance (not merely within CV noise).

`rdump_en/` holds the frozen reference (committed, ~300 KB): centered design `Xn`/`y`, and
per-`alpha ∈ {0.0, 0.5, 1.0}` the `glmnet` lambda grid, coefficient path, intercepts, and
`cv.glmnet` `lambda.min`/`lambda.1se`.

## Consumer

Unlike the other validations here, there is **no migrated C++ gate/diagnostic** for this
reference: its C++ counterpart is the demo
`TRexSelector_Examples/cpp/ml_methods/model_selection/demo_mlm_ms_02_enet_cv_ccd`. The
generator and its dumps are kept in the library so the cross-language EN reference lives
with the code it validates and can be reproduced independently.

## Regenerate

```bash
Rscript demo_en_glmnet_compare.R   # needs CRAN glmnet; writes rdump_en/ next to the script
```
