# Extending CyAN CI to NASA PACE OCI

**Advancing CyanoHAB monitoring with hyperspectral data from NASA PACE: First results and validation**

[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.jag.2025.105032-blue)](https://doi.org/10.1016/j.jag.2025.105032)
[![Python 3.10](https://img.shields.io/badge/python-3.10-green.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the source code, processing workflows, and shareable data required to reproduce the analysis and figures presented in the paper **"Advancing CyanoHAB monitoring with hyperspectral data from NASA PACE: First results and validation"**, published in the *International Journal of Applied Earth Observation and Geoinformation (JAG)*.

## 📄 Abstract
This study presents the first assessment of NASA's Phytoplankton, Aerosol, Cloud, ocean Ecosystem (PACE) mission's hyperspectral Ocean Color Imager (OCI) for cyanobacterial harmful algal blooms (cyanoHABs) monitoring. We conducted a direct comparison of PACE OCI with Sentinel-3's multispectral Ocean and Land Color Instrument (OLCI) and the operational CyAN product using imagery from summer 2024 blooms in Lake Erie, Green Bay, and Clear Lake. The code in this repository reproduces the validation of the Cyanobacteria Index ($CI_{Cyano}$), Cell Density (CCD), and Chlorophyll-a biomass retrievals against in-situ measurements.

## 🚀 Getting Started

### Prerequisites
The analysis was performed using Python 3.10. You can install the required dependencies using the provided environment file:

```
conda env create -f environment.yml
conda activate pace_cyan
```

### Data Availability
- Satellite Data: Raw L1B data for PACE OCI and Sentinel-3 OLCI are available from [NASA Earthdata](https://search.earthdata.nasa.gov/) and the [Copernicus Browser](https://browser.dataspace.copernicus.eu/), respectively.

- In-situ Data: The processed match-up dataset (radiometry and Chl-a pigment concentrations) used for validation is included in data/processed. Full raw in-situ datasets are available upon request from either [chintanmaniyar@uga.edu](mailto:chintanmaniyar@uga.edu) or [ak68883@uga.edu](ak68883@uga.edu).

## 🔗 Citation
If you use this code or data in your research, please cite this paper as follows:
```
@article{Kumar2026_PACE_CyAN,
  title = {Advancing CyanoHAB monitoring with hyperspectral data from NASA PACE: First results and validation},
  author = {Kumar, Abhishek and Maniyar, Chintan B. and Tesfayi, Nathan and Grunert, Brice K. and Fiorentino, Isabella R. and Herweck, Kendra and Hyland, Emily and Liu, Bingqing and Bartelme, Bradley and Mishra, Deepak R.},
  journal = {International Journal of Applied Earth Observation and Geoinformation},
  volume = {146},
  pages = {105032},
  year = {2026},
  doi = {10.1016/j.jag.2025.105032},
  publisher = {Elsevier}
}
```
---

*Center for Geospatial Research, Department of Geography, University of Georgia*
