# Physics-Guided-PSInSAR-LSM

**Physics-guided PS-InSAR-derived indices and machine learning for landslide susceptibility mapping**

![First-page screenshot of the published article](Screenshot%202026-11-23%20161056.png)

*Published-article first-page preview. The article is copyright-protected; reuse of this image should be checked against the publisher's terms. See the DOI for the version of record.*

## Published research article

**Title:** Integrating physics-guided Persistent Scatterer Interferometry (PS-InSAR) derived indices and machine learning for landslide susceptibility mapping  
**Authors:** Diwakar Khadka, Ananta Man Singh Pradhan, Jie Zhang, Atma Sharma, and Meng Lu  
**Journal:** *Advances in Space Research* **78** (2026), 8772–8800  
**Published online:** 22 August 2026  
**DOI:** [10.1016/j.asr.2026.08.058](https://doi.org/10.1016/j.asr.2026.08.058)

## Research overview and novelty

Conventional landslide susceptibility maps use predominantly static terrain and environmental predictors. Directly adding raw PS-InSAR deformation can be misleading where radar viewing geometry, phase coherence, or persistent-scatterer availability limits measurement reliability. This study develops **observation-aware, physics-guided input indices** that interpret deformation in its radar, slope, and hydrological context before it is used by machine-learning models.

The contribution is **physics-guided feature construction**, rather than a new physics solver or a claim that these indices measure the probability of slope failure directly.

## Study area and main inputs

- **Region:** Nyalam–Tatopani–Dolalghat section of the China–Nepal Friendship Highway, approximately 101 km, with a 5 km study buffer.
- **Land- and terrain-related inputs:** ALOS PALSAR RTC DEM (12.5 m), terrain derivatives, roads/rivers from OpenStreetMap, Sentinel-2 NDVI, ESA WorldCover, CHIRPS precipitation, and ERA5-Land soil moisture.
- **InSAR:** 107 Sentinel-1A ascending scenes from January 2019 to July 2022, processed using SNAP–StaMPS; 22,129 persistent scatterers were reported.
- **Inventory:** 232 mapped landslide polygons, of which 165 met the common-support requirements for the controlled modelling experiment; 165 matched pseudo-absence sites formed a 330-record modelling dataset.
- **Spatial reference:** UTM Zone 45N (EPSG:32645); susceptibility modelling uses a common 12.5 m analysis grid.

## Research methodology

1. **Data harmonization:** Align conditioning-factor rasters to a common grid, determine valid spatial support, and derive slope, aspect, curvature, topographic position, stream power and topographic wetness proxies.
2. **PS-InSAR and geometric interpretation:** Use processed LOS velocities and radar/terrain geometry to estimate an apparent downslope-constrained velocity, with a stabilization parameter for weakly observable geometry; restrict unsupported deformation rather than treating missing signal as slope stability.
3. **InSAR Reliability Score (IRS):** Combine normalized coherence, normalized local PS density and geometric observability. IRS acts as an internal weight, **not** as a standalone ML predictor.
4. **Hydro-Kinematic Fusion Index (HKFI):** Combine reliability-weighted kinematic information, Topographic Wetness Index (TWI), and coarse-support ERA5-Land soil moisture. The reference formulation uses equal nominal component weights with robust index normalization.
5. **Slope-Normalized Deformation Ratio (SNDR):** Form a reliability-weighted downslope deformation measure normalized relative to slope geometry, then normalize the supported index. SNDR is an empirical conditioning indicator, **not** a factor of safety or mechanical stress ratio.
6. **Machine learning:** Compare six controlled predictor configurations across XGBoost (XGB), Random Forest (RF), and Logistic Regression (LR), using nested spatial cross-validation and identical sampling/fold definitions.
7. **Evaluation and sensitivity:** Compare pooled out-of-fold ROC-AUC, PR-AUC, balanced accuracy, F1-score and Brier score; evaluate paired spatial-block bootstrap differences, pseudo-absence buffers (100/500/1000 m), HKFI weighting, and map-area variability.

### Predictor configurations

| Configuration | Predictors |
|---|---|
| A | Twelve conventional static conditioning factors (baseline) |
| B | A + downslope-positive velocity (DV+) |
| C | A + HKFI |
| D | A + SNDR |
| E | A + HKFI + SNDR |
| F | A + DV+ + TWI + soil moisture, entered separately |

## Principal results

**Reported pooled out-of-fold ROC-AUC** (from Table 3 of the published paper):

| Configuration | XGB | RF | LR |
|---|---:|---:|---:|
| A: Static baseline | 0.873 | 0.875 | 0.859 |
| B: Baseline + DV+ | 0.869 | 0.872 | 0.859 |
| C: Baseline + HKFI | 0.885 | 0.884 | 0.858 |
| D: Baseline + SNDR | **0.890** | 0.887 | 0.857 |
| E: Baseline + HKFI + SNDR | 0.889 | **0.888** | 0.856 |
| F: Untransformed source variables | 0.867 | 0.876 | 0.856 |

- Raw downslope-positive velocity did **not** improve the static baseline.
- SNDR alone achieved the best XGB ROC-AUC (**0.890**); the combined HKFI–SNDR configuration achieved the best RF ROC-AUC (**0.888**).
- Physics-guided configurations produced supported improvements for nonlinear XGB and RF models, but not LR.
- Combining HKFI and SNDR did **not** demonstrate a statistically supported additional gain beyond each individual index in the reported paired E–C/E–D comparisons.
- The study emphasizes the role of observation reliability and geometric interpretation, while noting limitations from a single ascending radar geometry, incomplete spatial support, and within-corridor validation.

## Research code

The repository provides an annotated Jupyter notebook covering terrain and hydro-meteorological preparation, PS-InSAR-derived inputs, IRS/HKFI/SNDR construction, model preparation, XGB/RF/LR analysis, evaluation and sensitivity checks.

**Notebook:** [`notebooks/Physics_Guided_PSInSAR_LSM_Public.ipynb`](notebooks/Physics_Guided_PSInSAR_LSM_Public.ipynb)

The public version of the original notebook retains **225 cells** (117 code, 108 markdown) and retains the scientific calculation/ML steps. For safe public sharing, saved cell outputs were cleared, the original machine-specific working directory was replaced with the `LSM_DATA_DIR` environment variable (default `./data`), and Earth Engine credential settings were replaced with environment-based configuration. Credential diagnostics were replaced by a generic setup example. **This is a publication-prepared notebook, not a verified end-to-end rerun.**

### Software requirements

The analysis uses Python with Jupyter, NumPy, Pandas, SciPy, rasterio, GeoPandas, Fiona, Shapely, PyProj, WhiteboxTools, Google Earth Engine API, scikit-learn, XGBoost, matplotlib, and associated spatial libraries. An indicative dependency list is provided in [`requirements.txt`](requirements.txt). The original notebook metadata recorded **Python 3.8.19**; the exact original full environment and package versions were not provided.

```bash
python -m pip install -r requirements.txt
jupyter lab
```

Set your project paths and Earth Engine configuration before executing the notebook. For example, in a Windows PowerShell session:

```powershell
$env:LSM_DATA_DIR = "D:\your-research-data"
$env:EE_PROJECT = "your-earth-engine-enabled-project"
# Optional, only if using a service-account JSON stored OUTSIDE the repository:
$env:EE_SERVICE_ACCOUNT_KEY = "D:\private-credentials\service-account.json"
```

**Important limitations of execution:** Raw input rasters, landslide inventory, derived products, fitted models and generated results are **not** included in this repository. Certain code sections expect previous outputs and may reference local supporting Python modules or intermediate paths. The upstream Sentinel-1 SNAP–StaMPS processing workflow is described in the paper but is not recreated end-to-end by this notebook. Users must provide authorized datasets and adapt environment paths/settings to their installation. Package version compatibility and full reproduction of reported results have **not** been verified in a clean environment.

## Data availability

The paper states: *“The datasets generated during and/or analyzed during the current study are available from the corresponding author on reasonable request.”* See the DOI-linked paper for details. Dataset use and redistribution may be governed by their original providers' terms. No private data or credential files should be committed to this repository.

## How to cite

Khadka, D., Pradhan, A. M. S., Zhang, J., Sharma, A., & Lu, M. (2026). Integrating physics-guided Persistent Scatterer Interferometry (PS-InSAR) derived indices and machine learning for landslide susceptibility mapping. *Advances in Space Research, 78*, 8772–8800. https://doi.org/10.1016/j.asr.2026.08.058

See also [`CITATION.cff`](CITATION.cff).

## License and reuse

No open-source license has yet been assigned to this repository. Public access permits viewing and downloading through GitHub, but does not automatically grant broad rights to reuse, redistribute or modify the code. The published article and its screenshot remain subject to applicable publisher rights. Contact the authors/rights holders for permission as appropriate.
