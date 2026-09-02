# Figure production specification

## Global policy

- English visible text only.
- ASCII hyphens only.
- No watermark, banner, status marker, diagonal text, or opaque text over data.
- Panel letters, legends, titles, metrics, and color bars must use reserved margins or dedicated axes.
- Use fixed probability limits 0 to 1.
- Use one recorded RGB stretch for T1 and T2 within a case.
- Use recorded fixed SAR display limits.
- Preserve the exact same crop and support for all compared methods.
- Produce vector PDF, editable SVG, and PNG at 300 dpi or higher.
- Figure numbers are assigned by the manuscript, not embedded in the image.

## Dimensions

| ID | Figure | Width | Height | PNG minimum |
|---|---|---:|---:|---:|
| M1 | Stage I architecture | 190 mm | 90 mm | 2245 x 1063 px |
| M2 | Stage II architecture | 190 mm | 86 mm | 2245 x 1016 px |
| QF1 | Severe-cloud comparison | 190 mm | 95 mm | 2245 x 1122 px |
| QF2 | Observability and unresolved positives | 190 mm | 88 mm | 2245 x 1039 px |
| QF3 | Past-only retrieval | 190 mm | 90 mm | 2245 x 1063 px |
| QF4 | Gate mechanism | 190 mm | 86 mm | 2245 x 1016 px |
| QF5 | Proposal and annotation audit | 190 mm | 88 mm | 2245 x 1039 px |
| QF6 | Failure taxonomy | 190 mm | 90 mm | 2245 x 1063 px |
| QF7 | Inter-city contact sheet | 190 mm | 86 mm per page | 2245 x 1016 px |
| Q0 | Quantitative plots | 88 mm | 65 mm | 1039 x 768 px |

## M1 and M2

Method figures are vector schematics derived from the implementation and scientific contracts. They must not contain fabricated satellite thumbnails, model outputs, or measured values.

M1 shows the corrected flow: tokenization, Cloud-Mix, separate optical and SAR encoders, projection 192 to 384, asymmetric gated SAR-to-optical cross-attention, past-only targets, safety controls to reconstruction loss, structural fallback, three-term objective, and parallel encoder transfer.

M2 follows the approved 190 by 86 mm layout in `layout_references/method_figures/`. Native probability and binary outputs are created before V12 composition. V12 is independent of the decoder. Binary prediction and V12 meet only at the three-state compositor.

## QF1 severe-cloud comparison

Selection: mean cloud burden above 0.70, coverage above 0.05, then median positive OA-MAE gain, upper-quartile positive gain, and largest regression. Freeze selection before viewing.

Columns:

1. observed Sentinel-2 T1;
2. observed Sentinel-2 T2;
3. observed Sentinel-1 change;
4. mean cloud probability;
5. reference and V12;
6. strongest optical-only baseline probability;
7. CROMA probability;
8. OA-MAE probability;
9. error maps.

Every map must be from the same observed sample and grid. Include a regression case.

## QF2 observability diagnostics

Show observed T2 optical, bright-surface or land-cover context, cloud probability, V12 at 0.75, 0.85, and 0.95, reference change, positive support, unresolved positives, and OA-MAE errors. Include at least one bright-surface cloud-product failure.

## QF3 past-only retrieval

Show four states: three valid candidates, two candidates, one candidate, and fallback. Panels must use observed historical acquisitions and archived retrieval records. Show product dates, ages, selected candidate order, target median, fallback mask, and safety weight. Do not create these panels from formulas alone.

## QF4 gate mechanism

Show clear optical evidence, severe-cloud positive gain, and severe-cloud regression. Use observed optical, observed SAR change, archived cloud gate, archived SAR reliability gate, archived effective gate, CROMA probability, OA-MAE probability, and OA-MAE errors.

## QF5 proposal and annotation audit

Show regular fabric, small-roof dense fabric, complex texture, and bright surface. Use observed source views, proposal geometry, independent reference changes, missed changes, false proposals, annotator A, annotator B, and adjudication. The annotation gate must pass.

## QF6 failure taxonomy

Include OA-MAE regression, small-object false negative, support-boundary false positive, high-gate false positive, unresolved positive, and a case where both methods fail. Distinguish model error, support limitation, resolution limitation, and reference ambiguity.

## QF7 inter-city contact sheet

For each city select unique minimum-cloud, median-cloud, maximum-cloud, and lowest-gain cases by deterministic rules. Show optical T2, CROMA, and OA-MAE. Use two pages when needed to preserve readable panels.

## Quantitative plots

Produce label-noise robustness, abstention and coverage trade-off, cloud-burden gain, target-window sensitivity, cloud-mask sensitivity, compute comparison, and per-city primary effects. Values must be read from the active result registry.
