# Geoscientific Data Sources & Acquisition Audit
**Project:** LandslideSense — ML-Based Landslide Prediction, Risk Assessment, and Early Warning System  
**Institution:** Galgotias University, Greater Noida  
**Milestone:** 7th Semester Mid-Term Evaluation (MTE) / Interim Project Report-I (IPR-I)

---

## 1. Overview and Scientific Integrity Notice

In developing machine learning models for natural hazard prediction, scientific rigor requires explicit documentation of data origins, acquisition protocols, spatial-temporal resolution, and licensing constraints. 

A critical finding in geoscientific computing is that **public landslide portals generally publish spatial inventory atlases or GIS polygons rather than pre-assembled, tabular, machine-learning-ready CSV files**. Furthermore, rainfall observations are maintained separately by meteorological satellites or surface raingauges. 

This document provides a transparent audit of the four official data repositories evaluated for LandslideSense, detailing their actual formats, accessibility, and the steps required to transition from spatial data to tabular ML predictors.

---

## 2. Official Geoscientific Data Repositories

### A. Geological Survey of India (GSI) — Landslide Hazard Portal
* **Official URL:** [https://gsi.gov.in/landslide-hazard/](https://gsi.gov.in/landslide-hazard/)
* **Publisher:** Geological Survey of India, Ministry of Mines, Government of India
* **Geographical Coverage:** India (Indian Himalayas, Western Ghats, Nilgiris, Eastern Ghats, and North-Eastern Hill Ranges)
* **Data Types & Delivery:**
  * National Landslide Susceptibility Mapping (NLSM) at 1:50,000 scale.
  * Historical Landslide Inventory polygons/points recorded post-monsoon.
  * Spatial WebGIS layers hosted on the GSI Bhukosh portal.
* **Access Conditions & Constraints:**
  * Web map layers are viewable by the public; however, bulk programmatic export of raw GIS shapefiles or geodatabases requires registered researcher access on the GSI Bhukosh portal.
  * Data is **spatial (GIS vector polygons)**, not a time-series tabular CSV.
* **Role in LandslideSense:**
  * Serves as the primary spatial ground-truth source for validated landslide occurrences in the Indian subcontinent.

---

### B. ISRO / NRSC — Landslide Atlas of India
* **Official URL:** [https://www.isro.gov.in/ISRO_EN/Landslide_Atlas_India.html](https://www.isro.gov.in/ISRO_EN/Landslide_Atlas_India.html)
* **Publisher:** National Remote Sensing Centre (NRSC), Indian Space Research Organisation (ISRO)
* **Publication Period / Coverage:** 1998–2022 inventory spanning 17 States and 2 Union Territories of India.
* **Data Types & Delivery:**
  * Comprehensive geospatial atlas document (PDF) detailing landslide hazard exposure and district-level vulnerability rankings (e.g., Rudraprayag, Tehri Garhwal, Wayanad).
  * Interactive raster/vector hazard layers visualizable on the ISRO Bhuvan Geo-platform ([https://bhuvan.nrsc.gov.in](https://bhuvan.nrsc.gov.in)).
* **Access Conditions & Constraints:**
  * The PDF atlas is freely downloadable for academic reference.
  * Vector geometries require web-GIS sampling; no direct CSV download endpoint exists for raw historical events.
* **Role in LandslideSense:**
  * Provides benchmark district-level susceptibility indices across India and historical frequency distributions used to calibrate regional risk weighting.

---

### C. NASA Earth Data — Global Precipitation Measurement (GPM)
* **Official URL:** [https://gpm.nasa.gov/data/directory](https://gpm.nasa.gov/data/directory)
* **Publisher:** National Aeronautics and Space Administration (NASA) / JAXA
* **Coverage & Resolution:** Global, 0.1° × 0.1° (~10 km) spatial grid, 30-minute to daily temporal resolution.
* **Data Products:**
  * Integrated Multi-satellite Retrievals for GPM (IMERG) Early, Late, and Final Run products.
  * Formats: NetCDF-4, HDF5, GeoTIFF, and OpenDAP/HTTP query streams.
* **Access Conditions & Constraints:**
  * Freely accessible to the global research community under NASA Open Data Policy (requires free NASA EarthData user login).
  * Requires spatial subsetting and temporal aggregation (e.g., aggregating 30-min feeds into 24-hour, 3-day, and 7-day cumulative precipitation sums).
* **Role in LandslideSense:**
  * Primary source for dynamic hydrological triggering inputs (`rainfall_24h`, `rainfall_3d`, `rainfall_7d`).

---

### D. USGS Landslide Inventories across the United States (Ver 3.0)
* **Official URL:** [https://www.usgs.gov/data/landslide-inventories-across-united-states-ver-30-february-2025](https://www.usgs.gov/data/landslide-inventories-across-united-states-ver-30-february-2025)
* **Publisher:** United States Geological Survey (USGS)
* **Release Date:** February 2025 (Version 3.0)
* **Coverage:** United States ONLY (Contiguous US, Alaska, Hawaii, Puerto Rico)
* **Data Format:** Esri File Geodatabase, GeoPackage, Shapefile, and point/polygon geometries.
* **Access Conditions:** Public domain (U.S. Government Open Data).
* **CRITICAL GEOGRAPHICAL NOTICE:**
  * This dataset represents **United States terrain exclusively** and does **NOT** represent India.
  * If utilized during model development, it is strictly as a methodological benchmark for pipeline verification and must never be misrepresented as Indian topography.

---

## 3. Ground-Truth Data Processing Pipeline for Capstone Milestone

To transform official spatial records into a verified training dataset:

```
[ GSI / NRSC Landslide Polygons ]       [ SRTM 30m / Cartosat DEM ]       [ NASA GPM IMERG Grids ]
             │                                      │                                 │
             ▼                                      ▼                                 ▼
   Event Spatial Coordinates                Slope Angle & Elevation         Antecedent Rainfall (24h/3d/7d)
   (Latitude, Longitude, Date)              Extraction at Location           Temporal Cumulative Extraction
             │                                      │                                 │
             └──────────────────────────────────────┴─────────────────────────────────┘
                                                    │
                                                    ▼
                                     [ Spatial Intersection / GIS Join ]
                                                    │
                                                    ├─ Positive Samples (Landslide = 1)
                                                    └─ Non-Landslide Sampling (Landslide = 0)
                                                       (Sampled across stable slopes during non-monsoon/dry spells)
                                                    │
                                                    ▼
                                    `data/templates/landslide_training_template.csv`
```

### Class Balance & Negative Sampling Protocol
Inventory catalogs exclusively record *positive occurrences* (places where landslides were observed). An inventory omission does **NOT** inherently signify confirmed stability. To build an unbiased classifier:
1. Negative observations (Class `0`) must be systematically sampled from slopes verified stable via multi-temporal satellite imagery.
2. The negative samples must be paired with historical non-triggering rainfall records to prevent artificial bias in model decision boundaries.
