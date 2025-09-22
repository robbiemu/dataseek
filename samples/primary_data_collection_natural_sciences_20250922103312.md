# Data Prospecting Report
**Target Characteristic**: `Primary Data Collection` **Search Domain**: `Natural Sciences`
**Source URL**: `"https://www.nature.com/articles/s41467-020-19870-y"` **Source Title**: `"Nutrients cause grassland biomass to outpace herbivory | Nature Communications"`
----
## Justification for Selection 
* **Alignment with `Primary Data Collection`**: This is a multisite, manipulative field experiment that generated original primary data. The study applied factorial treatments (nutrient addition and large-vertebrate herbivore exclusion) to 5 × 5 m plots at 58 grassland sites across six continents, with pretreatment data and 2–10 years of post-treatment measurements. Methods describe plot-level experimental setup (treatments, plot size, fence design), treatment dosages and schedules (N, P, K, micronutrients), replication and blocking, standardized biomass sampling (clipping, drying, weighing), soil sampling and assays, and site metadata collection (soil, climate, herbivore indices). The study reports raw data deposits (Environmental Data Initiative) and provides detailed statistical modeling steps, confirming primary, empirical data collection rather than secondary analysis.
* **Potential for High Yield**: The paper contains many explicit, extractable procedural elements that exemplify primary-data collection in ecology: global coordinated experimental protocol, exact treatment compositions and application rates, fence construction specifications, plot dimensions and sampling units, biomass clipping and processing details (strip sizes, drying temp/time), soil core dimensions and pre-treatment sampling, herbivore index methodology, durations and replication counts, and the statistical framework for analyzing experimental time-series. These repeated, concrete, quantitative details across field implementation, sample processing, and analysis make it a rich source for extracting exemplars of primary-data collection workflows.

----
## Retrieved Content (Markdown) `[CACHE_REFERENCE: call_url_to_markdown]` or a curated excerpt in Markdown

- Experimental scope and sites
  - Multisite distributed experiment replicated at 58 grassland sites spanning six continents (sites listed in Supplementary Table 1).
  - Most sites had three replicate blocks; all sites collected 1 year of pretreatment data and 2–10 consecutive years of post-treatment data.

- Experimental design and treatments
  - Full factorial design: nutrient addition (“all nutrients”) vs control and vertebrate herbivore exclusion via fencing (“fenced”) vs control.
  - Plot size: 5 × 5 m experimental plots.
  - Nutrient treatment composition and schedule:
    - N: 10 g N m−2 yr−1 as time-release urea [(NH2)2CO].
    - P: 10 g P m−2 yr−1 as triple-super phosphate [Ca(H2PO4)2].
    - K: 10 g K m−2 yr−1 as potassium sulfate [K2SO4].
    - Micronutrient mix: 100 g m−2 at experiment initiation (Fe 15%, S 14%, Mg 1.5%, Mn 2.5%, Cu 1%, Zn 1%, B 0.2%, Mo 0.05%).
    - Macronutrients applied annually; micronutrients applied once at start (year 1).
  - Fence/exclosure specifications:
    - 230 cm tall fences.
    - Lower 90 cm: 1 cm woven wire mesh with 30 cm outward-facing ground flange stapled to exclude diggers (e.g., rabbits, voles).
    - Upper 90 cm: three evenly spaced barbless wires to restrict larger vertebrates (e.g., bison, elk).
    - Some sites deviated from the exact fence design (documented in Supplementary Table 2).

- Vegetation and biomass sampling
  - Annual peak-season aboveground live biomass measured by clipping all plants rooted within two 0.1 m2 (10 × 100 cm) strips per plot.
  - Clipped vegetation separated into live and dead, dried at 60 °C for 48 h, weighed to nearest 0.01 g.
  - Percent cover estimated to nearest 1% within a permanently marked 1 × 1 m subplot for species richness and composition.

- Soil sampling and assays
  - Prior to treatments (year 0), two soil cores (2.5 cm diameter × 10 cm depth) taken per plot, composited, homogenized through 2 mm sieve, air-dried for laboratory assays.
  - Assays included %N and %C (dry combustion GC), soil P, K, micronutrients, pH, organic matter, and texture (external analytical lab).
  - Site-level climate characterized via WorldClim and other gridded datasets; atmospheric N deposition estimated from global chemistry-transport model outputs.

- Herbivore quantification
  - Two site-level metrics:
    - Empirical herbivore index: PI documented all herbivore species (>2 kg) and assigned importance values (1–5) for impact/frequency; index = sum of importance values.
    - Modeled wild grazer biomass extracted from published global dataset using site coordinates.

- Replication, timing, and scope
  - Pretreatment baseline data collected (year 0) at all sites.
  - Post-treatment monitoring lasted 2–10 years depending on site; analyses performed on subsets (e.g., ≥5 years, ≥8 years) to assess duration effects.
  - Enables within-site repeated measures and cross-site comparisons across broad climatic and edaphic gradients.

- Statistical analysis (method summary)
  - Mixed-effects models (lmer, lme4 in R) with site and treatment year nested within site as random intercepts; log10-transform of live biomass.
  - Computation of treatment effects as log differences between treatment and control (e.g., fenced vs unfenced).
  - Models included interactions with site-level covariates (soil N, P, pH, climate, species richness, herbivore index).
  - Model diagnostics and validation (residual checks, cross-validation subsets) described; supplementary tables report model coefficients and tests.

- Data availability and provenance
  - Source data (plant, herbivore, soil nitrogen) provided with paper and archived in Environmental Data Initiative (EDI) with DOI (provided in paper).
  - Supplementary information includes site-level metadata, treatment implementation details, and additional analyses.

(Excerpt curated from Methods, Results, and Supplementary Materials of: Borer E.T., Harpole W.S., Adler P.B., et al. Nutrients cause grassland biomass to outpace herbivory. Nat Commun 11, 6036 (2020). https://doi.org/10.1038/s41467-020-19870-y)