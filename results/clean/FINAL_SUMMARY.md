# Final results: which AutoML pipeline component drives SHAP explanation variation

## Phase 1 — re-measurement of the original pipelines (items 1-5)
- Classifier dominance as originally reported (0.18 vs 0.0075) does not survive matched measurement.
- On credit-g, spambase, Phishing and bank-marketing the original KernelExplainer values are at the noise floor (D_bar/delta_0 = 1.00-1.05); on diabetes (exact Kernel SHAP) one-way R^2: classifier 0.29, rescaling 0.18, preprocessor 0.13.
- The original code mislabels features on credit-g and bank-marketing (verified by recomputation; corrected).

## Phase 2 — balanced factorial (items 6-9), TV, all pipelines
| dataset | classifier share | preprocessor share | balancing share | interaction (added R^2) | P(preprocessor > classifier), bootstrap |
|---|---|---|---|---|---|
| diabetes | 11% | 89% | 0.3% | 0.059 | 1.00 |
| credit-g | 22% | 78% | 0.1% | 0.090 | 1.00 |
| spambase | 27% | 73% | 0.2% | 0.092 | 1.00 |
| PhishingWebsites | 29% | 71% | 0.0% | 0.097 | 1.00 |
| bank-marketing | 37% | 62% | 1.2% | 0.082 | 1.00 |

Paired effects (changing only one component), delta_0 units: diabetes: classifier 3.3, preprocessor 5.9, balancing 1.6; credit-g: classifier 3.0, preprocessor 4.0, balancing 1.4; spambase: classifier 2.7, preprocessor 3.6, balancing 1.1; PhishingWebsites: classifier 4.9, preprocessor 6.6, balancing 1.1; bank-marketing: classifier 4.3, preprocessor 4.9, balancing 1.8

Rashomon sets: diabetes and credit-g — preprocessor largest; classifier share rises with the accuracy constraint (diabetes: 11% -> 29% -> 34%; credit-g: 22% -> 28% -> 36%). spambase, Phishing, bank-marketing — near-optimal explanations agree within noise (D_bar/delta_0 = 0.79-1.01).

Whole components (hyperparameters on both sides), preprocessor share: diabetes 83%, credit-g 71%, spambase 72%, PhishingWebsites 70%, bank-marketing 61%

## Phase 3 — mechanism (items 10-11)
- Item 10: hyperparameters carry credit-g R_1se 31%, credit-g R_2se 38%, credit-g all pipelines 45%, diabetes R_1se 42%, diabetes R_2se 20%, diabetes all pipelines 59%, spambase all pipelines 31%, bank-marketing all pipelines 41%, PhishingWebsites all pipelines 38% of the classifier part. Within-family dispersion is larger for flexible than for low-capacity learners on datasets credit-g, diabetes, spambase, bank-marketing, PhishingWebsites.
- Item 11 (corrected rules): non-identity predicted-invariant cells, median distance diabetes 0.00, credit-g 0.00, spambase 0.00, PhishingWebsites 0.00, bank-marketing 0.00 delta_0 vs predicted-changing diabetes 5.45, credit-g 3.52, spambase 3.46, PhishingWebsites 5.46, bank-marketing 4.35 delta_0. The pre-registered rules are also significant (p <= 1e-59).

## Robustness
- Bootstrap (200x): P(preprocessor > classifier) = diabetes 1.00, credit-g 1.00, spambase 1.00, PhishingWebsites 1.00, bank-marketing 1.00.
- Leave-one-preprocessor-out: the preprocessor is the largest component in 63 of 63 removals.
- Seed check: shares with seed 0 vs 3 seeds differ by at most 0.015; same largest component on every dataset: True.
- Distance choice: cases where the classifier is the largest component: diabetes R_1se (jac3); diabetes R_1se (jac5); credit-g R_2se (jac3); credit-g R_2se (jac5); credit-g R_1se (jac3); credit-g R_1se (jac5); credit-g R_1se (rbo); spambase all pipelines (jac3); spambase all pipelines (jac5); spambase R_2se (tv); spambase R_2se (jac3); spambase R_2se (jac5); spambase R_2se (rbo); bank-marketing all pipelines (jac3); bank-marketing R_2se (tv); bank-marketing R_2se (jac3); bank-marketing R_2se (jac5); bank-marketing R_2se (rbo); bank-marketing R_1se (tv); bank-marketing R_1se (jac3); bank-marketing R_1se (jac5); bank-marketing R_1se (rbo).

## Deviations, choices and corrections
- Phase 1 reuses the original KernelExplainer SHAP (option C); deviations from item 1 in item1_deviations_reused_shap.csv.
- Feature-label bug of the original code corrected for credit-g and bank-marketing.
- LDA/QDA predict_proba kept as auto-sklearn outputs it (sigma(2P-1)); recorded, not corrected.
- Item 5 not available on diabetes (no matching RF pipeline in the filtered set).
- Phase-1 seed refits use the full training set (auto-sklearn run models used an internal subset).
- Rescaling pinned to standardize (plus mean imputation, one-hot, no coalescence); item 11's affine and monotone columns therefore untested.
- k = 5 configurations per cell, paired draws across cells; E = 200 held-out rows, B = 100 stratified training rows.
- Full factorial explained at seed 0; seeds as replicates only for Rashomon sets and delta_0 (seed check above).
- One explainer run (r = 0) for every pipeline; extra runs only on a 50-configuration subset (lever 3), restricted to configurations explained in <= 1 h.
- CV: seed 0, SE of L(p*) from per-row out-of-fold losses; 1 h per CV fold, 6 h per explanation, 3/6 GB memory limits.
- PERMANOVA implemented in Python (validated: SS identity, MANOVA identity, null p ~ 0.5); vegan not used.
- RBO_ext with p = 0.9; resolvability threshold D_bar/delta_0 >= 1.2 (our choice).
- Item 10 nested terms use exact configuration ids (correction of a first version that used draw indices).
- Item 11 rule corrections (fast_ica whiten=False = rotation; LDA/QDA rotation-invariant) were data-triggered; both pre-registered and corrected results are reported.
- auto-sklearn PCA always truncates (keep_variance <= 0.9999), so rotation is tested via fast_ica only.
- Items 12-15 out of scope.

## Extension — other explanation methods (PFI, LIME) on the same factorial pipelines
Same ~7,750 pipelines (all 7,656 seed-0 refits identical to the SHAP-explained models), same raw features, same decomposition. PFI = log-loss increase (5 repeats, <=2,000 held-out rows, shared permutations, negatives clipped); LIME = LimeTabularExplainer defaults on the same 200 rows E, global = mean |weight|.

| dataset | method | classifier share | preprocessor share | balancing share | P(preprocessor > classifier) |
|---|---|---|---|---|---|
| diabetes | SHAP | 11% | 89% | 0.3% | 1.00 |
| diabetes | PFI | 31% | 68% | 0.1% | 1.00 |
| diabetes | LIME | 15% | 85% | 0.5% | 1.00 |
| credit-g | SHAP | 22% | 78% | 0.1% | 1.00 |
| credit-g | PFI | 39% | 61% | 0.1% | 1.00 |
| credit-g | LIME | 28% | 72% | 0.2% | 1.00 |
| spambase | SHAP | 27% | 73% | 0.2% | 1.00 |
| spambase | PFI | 33% | 67% | 0.2% | 1.00 |
| spambase | LIME | 32% | 68% | 0.2% | 1.00 |
| PhishingWebsites | SHAP | 29% | 71% | 0.0% | 1.00 |
| PhishingWebsites | PFI | 43% | 57% | 0.0% | 0.96 |
| PhishingWebsites | LIME | 32% | 68% | 0.0% | 1.00 |
| bank-marketing | SHAP | 37% | 62% | 1.2% | 1.00 |
| bank-marketing | PFI | 53% | 46% | 1.2% | 0.08 |
| bank-marketing | LIME | 45% | 54% | 1.4% | 0.97 |

- LIME: preprocessor largest on 5 of 5 datasets (preprocessor 54-85%).
- PFI: preprocessor largest on 4 of 5 datasets; classifier share higher than under SHAP (31-53%), consistent with loss-based importance being sensitive to calibration, which the classifier controls.
- Invariance test holds for both methods: non-identity predicted-invariant cells median PFI 0.00-0.00, LIME 0.00-0.00 delta_0 vs predicted-changing PFI 2.5-5.4, LIME 3.5-4.9 delta_0.
- Agreement on which pipelines differ (Spearman correlation of TV distance matrices): SHAP-LIME 0.84-0.95, SHAP-PFI 0.45-0.90, PFI-LIME 0.45-0.81.

## Manipulation analysis (item 15; X-hacking)
Alternatives differ from a base pipeline in one component and keep CV accuracy >= base - 0.01; noise baseline = retraining the same configuration with another seed.
- All pipelines, share of single switches pushing a top-3 feature out of the top 3 (all methods): preprocessor 0.15-0.57, classifier 0.11-0.63, balancing 0.04-0.23, seed only 0.04-0.20.
- Top feature replaced by some allowed switch: preprocessor 0.17-0.91, classifier 0.26-0.94.
- Worst-case rank drop of a top-3 feature (positions): preprocessor 2.0-22.9, classifier 1.6-22.0.
- Near-optimal pipelines (R_2se, SHAP): the classifier is the stronger per-switch lever on 4 of 5 datasets.
- Summary: preprocessing dominates how much the overall explanation varies; the classifier more often decides which feature is on top, especially among near-optimal pipelines; balancing is about as strong as seed noise.

## Additional choices (extension and manipulation analysis)
- LIME 0.2.0.1 installed with --no-deps (scikit-image not needed for tabular LIME; environment unchanged).
- PFI on log-loss with 5 repeats on <=2,000 held-out rows; negative importances clipped at 0; pipelines with all-zero importance excluded (PFI: 4-10 per dataset).
- Manipulation analysis uses an accuracy tolerance of 0.01 (the original study's filter); per-switch rates correct for different numbers of alternatives; near-optimal sets on spambase, Phishing and bank-marketing are small.
