# Audit of the SLSN–ORACLE comparison

Reviewed 2026-11-06. Scope: the supplied thread, every code cell in `Glue.ipynb`, `classification_64.csv`, Christina's three CSVs, the two generated numerical CSVs, and the repository's SLSN label and trigger-time definitions. Cell numbers below are zero-based. This is a review; the notebook and its existing figures have not been edited.

**Conclusion:** the notebook executes and its main binned count calculation is internally consistent. Its quoted times are exploratory ratio-separation forecasts, not validated 5σ detections. Several statements in the referenced thread need correction.

## Confirmed checks

- Executed all code cells from a fresh namespace in the VT environment, in a temporary directory. No computation errors; the only warnings concerned displaying figures through the non-interactive Agg backend.
- Both regenerated CSVs match the existing `confusion_by_z.csv` and `glue_by_z.csv`.
- Independently recomputed all nine confusion entries in every bin with a separate groupby calculation.
- Checked every aggregation of Christina's 0.1 bins into 0.2 bins against their original lower and upper edges. Totals are naive 281,141, O 466,939, Fe 917,999; the former Fe/O swap is fixed.
- All 463,528 classification rows have distinct SNIDs, finite redshifts in the configured range, and SLSN truth labels consistent with the split at z = 1.
- The binned forward model correctly keeps SLSN leakage proportional to each model's SLSN population and holds the non-SLSN term fixed.
- The Poisson count errors and first-order ratio errors are algebraically correct conditional on fixed calibration and independent Poisson category counts. Thin count bands are expected at large counts.

## 1. The final plot's crossing labels use a different model from its curves

Cell 19 draws significance curves from `results`, which includes redshift-dependent completeness and both SLSN leakage terms. Its legend calls `days_to_5sigma(**oracle)`, which uses the simpler cell-14 model instead.

Under the notebook's existing ratio-separation formula, the curves cross at survey day 67.719 (Fe/naive), 77.215 (O/naive), and 81.886 (Fe/O), including the assumed 64-day delay. The legend instead evaluates 67.193, 75.655, and 76.520. At the last displayed crossing time the Fe/O curve is only 4.183 in the plotted separation units.

These are corrected crossings of the existing curves, **not scientifically validated detection times**. A local consistency fix is to calculate each label directly from the same `results` entries used for its curve. This alone would leave the issues below unresolved.

## 2. The precision/recall scan does not preserve the ORACLE leakage model

Cells 14–16 use `FP = recall * N_naive * (1 - precision) / precision` and hold this entire quantity fixed. This is a valid definition for a deliberately simplified detector with only missed detections and external background. It does not reconstruct a three-class detector from its diagonal recall and precision.

The effective ORACLE precision in cell 15 has SLSN cross-class leakage in its denominator. Consequently, the scan moves that leakage into the fixed FP term, although it should increase with the SLSN rate model. It also replaces the redshift-dependent response with a naive-weighted average and uses all redshift bins rather than the binned calculation's support mask.

For example, the binned Fe calculation contains about 26,376 true high-z SLSNe predicted low-z; this contribution is model dependent. The two precision/recall numbers per class do not determine the off-diagonal SLSN rates. A realistic scan needs an explicit response matrix or a stated prescription for how leakage changes with recall, alongside separate external-background parameters.

The scan's two ORACLE markers are only references: because the grid forces equal precision and recall for both classes, neither point represents the actual unequal-class ORACLE operating configuration.

## 3. The current separation formula does not establish a 5σ rejection

Cells 12 and 14 use

`abs(R_a - R_b) / sqrt(var_a + var_b)`.

This describes separation of two independently fluctuating ratio estimates. In the intended experiment there is one observed catalogue and competing model predictions. Adding two hypothetical data variances is not, by itself, a hypothesis test or a p-value calibration. One must specify which model is true, which is being rejected, and what normalization and calibration parameters are fitted.

A defensible next calculation is a likelihood for observed categories, with

`lambda_j(model, theta, time) = exposure(time) * (sum_i response_ji(theta) * N_i(model) + background_j(theta))`,

plus constraints from calibration data. Fit nuisance parameters under each hypothesis, calculate expected likelihood-ratio separation, then validate its null distribution and power with simulations. Non-nested rate models require care when converting a likelihood ratio to Gaussian significance; a square root alone is not a universal conversion. Use observed categories, not unobservable true-redshift bins, in the data likelihood.

Cowan et al. provide the standard likelihood, nuisance-parameter, and expected-sensitivity framework: https://arxiv.org/abs/1007.1727 .

## 4. The systematic “cancellation” is too narrow to support the thread's claim

Cell 14 calculates `fp_sys**2 * (d_a - d_b)**2`. That propagates one fully correlated fractional shift through the *difference of two predicted ratios*. It is not a general treatment of the uncertainty in distinguishing a measured ratio from a null prediction.

Low-z and high-z contamination can have separate uncertainties; rates of individual contaminating populations can vary differently with redshift; response probabilities have calibration errors and covariance. Shared physical nuisance parameters should be represented consistently in one likelihood, but their allowed fitted values must be considered under each competing hypothesis. Shared calibration does not imply its effect disappears from the inference.

As a diagnostic, conditional first-order propagation of SLSN multinomial response errors and Poisson contaminant numerators gives individual 10-year ratio calibration errors of about 0.023, 0.036, and 0.054 for naive, O, and Fe. Their Poisson-only errors are about 0.010, 0.014, and 0.016. Even using common calibration draws, the Fe–O predicted-ratio contrast retains about 0.022 uncertainty in this restricted calculation. This is not a complete uncertainty budget: it conditions on denominator sample sizes and omits contaminant-population uncertainty and simulation selection effects. It demonstrates why the assertion that calibration uncertainty can simply be left out is unsupported. It does not establish how much the early-time crossings move.

## 5. The separation-panel ±1 band is not derived from the calculation

Cell 19 adds and subtracts 1 from a deterministic forecast curve. No sampling distribution or uncertainty propagation produces this band. With the existing sum-of-two-variances normalization, even a simple measured-ratio-minus-fixed-prediction statistic need not have unit scatter. A real sampling band needs simulations or a derived distribution for the chosen test statistic.

The earlier explanation that a 67-day expected crossing corresponds to “roughly 55–80 days” is particularly inconsistent with this notebook's hard 64-day delay: no detections can contribute before day 64 under that assumption. A distribution of first crossing times also differs from a pointwise uncertainty band and requires a sequential-analysis definition.

## 6. Contaminant normalization remains unvalidated

`c_oth_as_* = n_oth_as_* / n_true_slsn` retains the classification sample's class proportions. Anchoring to Christina's naive SLSN count makes the background equal across rate hypotheses, but does not convert it into a calibrated sky background. `elasticc_norm = 1` remains an explicit unverified assumption, and one global normalization cannot repair incorrect relative contaminant abundances.

A realistic background requires population counts after the alert selection for each contaminant class, multiplied by class-conditional misclassification probabilities. The present source has a class column suitable for estimating those probabilities, but not validated full-survey population weights. The provided Christina CSVs contain SLSN counts, not these background populations. The high-z contaminant sample is dominated by SNII and PISN, so their rate assumptions deserve specific treatment. The thread's earlier “32%” PISN statement also needs a denominator correction: in this classification CSV PISNe are 31.65% of non-SLSN contaminants predicted high-z SLSN, but only 4.34% of all high-z SLSN predictions. These are sample proportions, not sky-weighted forecasts.

## 7. Timing, support, and uncertainty limits

- Linear scaling of full-survey yields to the first few days assumes a constant detection rate, established templates, and no seasonal or commissioning effects. It is a toy exposure model, not a realistic early-survey forecast.
- The repository defines the classification phase relative to the first detection/trigger. The 64-day shift is therefore a useful assumed classification age, not a separately verified complete survey-latency model.
- The last supported confusion bin is 3.8–4.0, although classification data stop around 3.86. Applying its rate throughout the bin already extrapolates over its upper tail. Only bins beginning at 4.0 are explicitly dropped. Dropped totals (177, 1,248, 2,517) are reproduced.
- The scan uses all Christina bins, whereas the binned model truncates unsupported bins. Align support before comparing the two.
- Christina's files have efficiency-error columns, but the notebook does not propagate simulation/efficiency uncertainty. Their physical yield normalization, cadence compatibility, and 10-year interpretation cannot be independently established from these CSV column values alone.
- The scan does not validate probability inputs. Recall or precision above 1 is accepted; negative recall yields NaN; precision 0 silently produces an infinite time. Require finite `0 < precision <= 1`, `0 < recall <= 1` for a finite-response API, or explicitly implement the zero-recall boundary, and validate systematic and significance inputs too.
- Changing precision and recall in the scan does not update the final count-versus-time figure, which always uses the binned `glue` calculation.

## Interpretation of “ORACLE beats perfect”

The raw high/low ratio is a lossy summary. Adding known background can change how this particular summary responds to differences between rate models; it can therefore appear to improve this proxy relative to a different lossy summary of ideal data. This is not evidence that contamination adds information, nor proof by itself of an arithmetic error. Compare both experiments with the same calibrated inference procedure. Subtracting background and correcting response is another possible summary, but it needs the leakage matrix and propagated background/calibration covariance.

## Reproduction

Run from any directory:

```sh
/Users/vedshah/anaconda3/envs/VT/bin/python /Users/vedshah/Documents/Research/NU-Miller/Projects/Hierarchical-VT/Glue/audit_glue.py
```

`audit_glue.py` executes the notebook with generated files confined to a temporary directory and checks the original CSVs. `audit_results.txt` records the numerical diagnostics from this review. No original notebook, CSV, or figure was overwritten.
