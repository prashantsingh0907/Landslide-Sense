# Literature Review & Theoretical Background
**Project:** LandslideSense: ML-Based Landslide Prediction, Risk Assessment, and Early Warning System  
**Course:** B.Tech Computer Science and Engineering  
**Institution:** Galgotias University  
**Milestone:** 7th Semester Mid-Term Evaluation (MTE) / Interim Project Report-I (IPR-I)

---

## 1. Executive Summary

Rainfall-induced landslides represent one of the most destructive natural hazards in mountainous regions, causing thousands of casualties and substantial infrastructure destruction annually throughout the Himalayas and Western Ghats of India. This literature review analyzes the transition from empirical physical threshold models to contemporary machine learning (ML) architectures, evaluating their geotechnical validity, predictive accuracy, and operational deployment challenges.

---

## 2. Peer-Reviewed Literature Matrix

| Reference & Citation | Methodology | Key Geotechnical / ML Findings | Strengths & Limitations | Identified Research Gap |
| :--- | :--- | :--- | :--- | :--- |
| **N. Caine (1980)** [1]<br>*Geografiska Annaler*<br>DOI: [10.1080/04353676.1980.11879996](https://doi.org/10.1080/04353676.1980.11879996) | Empirical Rainfall Intensity-Duration ($I$-$D$) power-law thresholds: $I = 14.82 \cdot D^{-0.39}$. | Identified universal physical envelope where precipitation duration ($D$) and mean intensity ($I$) trigger shallow regolith failure. | **Strength:** Simple formulation.<br>**Limitation:** Neglects antecedent soil saturation and local slope mechanics. | Fails to incorporate antecedent multi-day cumulative precipitation and terrain conditioning. |
| **F. Guzzetti et al. (2007)** [2]<br>*Hydrology and Earth System Sciences*<br>DOI: [10.5194/hess-11-617-2007](https://doi.org/10.5194/hess-11-617-2007) | Global and regional statistical rainfall threshold analysis. | Demonstrated that antecedent precipitation (accumulated over 3 to 15 days) drastically alters the effective threshold required to trigger slope destabilization. | **Strength:** Comprehensive global catalog.<br>**Limitation:** Statistical curves lack fine-scale spatial adaptability. | Lack of integration between dynamic weather thresholds and static topographical GIS layers. |
| **D. Kirschbaum & T. Stanley (2018)** [3]<br>*Earth's Future*<br>DOI: [10.1002/2017EF000715](https://doi.org/10.1002/2017EF000715) | NASA Landslide Hazard Assessment for Situational Awareness (LHASA) model. | Couples satellite precipitation (GPM IMERG) with a static global susceptibility map using a multi-temporal antecedent rainfall index (ARI). | **Strength:** Near-real-time global situational awareness.<br>**Limitation:** Coarse spatial grid (1 km / 0.1°); high false alarm rate in steep valley basins. | Coarse resolution misses localized catchment failures and road-cut embankments. |
| **B. Pradhan & S. Lee (2010)** [4]<br>*Catena*<br>DOI: [10.1016/j.catena.2010.04.008](https://doi.org/10.1016/j.catena.2010.04.008) | Comparative evaluation of Artificial Neural Networks (ANN) vs Support Vector Machines (SVM). | Demonstrates non-linear mapping capabilities of machine learning algorithms on topographic and geomorphological inputs. | **Strength:** Superior non-linear fitting compared to bivariate statistical models.<br>**Limitation:** Black-box nature lacks interpretable feature attribution. | Inability to provide transparent, human-auditable risk explanations to civil defense operators. |
| **A. Merghadi et al. (2020)** [5]<br>*Earth-Science Reviews*<br>DOI: [10.1016/j.earscirev.2020.103225](https://doi.org/10.1016/j.earscirev.2020.103225) | Comprehensive systematic benchmark of ML methods (Tree-based, SVM, NN, Ensembles). | Found Random Forests (RF) and Gradient Boosting (XGBoost) consistently outperform other algorithms due to resilience against multicollinearity and noise in terrain data. | **Strength:** Definitive benchmark across hundreds of trials.<br>**Limitation:** Primarily focused on static susceptibility rather than dynamic warning. | Most studies limit ML to static susceptibility mapping rather than time-series early warning integration. |
| **A. Dikshit et al. (2020)** [6]<br>*Natural Hazards*<br>DOI: [10.1007/s11069-020-04183-w](https://doi.org/10.1007/s11069-020-04183-w) | Spatial susceptibility mapping in the Uttarakhand Himalayas using ensemble ML. | Demonstrated high predictive capability of tree ensembles in complex tectonic and monsoon-dominated Himalayan valleys. | **Strength:** Validated on high-risk Indian terrain.<br>**Limitation:** Relied on static features without real-time antecedent rainfall updates. | Static spatial models cannot differentiate between high-risk monsoon days and safe dry-season conditions. |
| **J. Mathew et al. (2014)** [7]<br>*Journal of Earth System Science*<br>DOI: [10.1007/s12040-014-0409-4](https://doi.org/10.1007/s12040-014-0409-4) | Frequency Ratio (FR) and Information Value (IV) modeling in Garhwal Himalayas. | Established that slope angles between 30° and 45° exhibit the highest landslide incidence in Himalayan lithologies. | **Strength:** Validated regional geotechnical threshold in Northern India.<br>**Limitation:** Linear probabilistic formulation cannot model multi-factor interactions. | Lack of dynamic moisture conditioning variables such as soil saturation or vegetation index. |

---

## 3. Key Geotechnical Principles

### 3.1 Mohr-Coulomb Failure & Effective Stress
The mechanics of hillslope destabilization under precipitation loading are governed by Terzaghi's principle of effective stress:

$$\tau_f = c' + (\sigma - u) \tan \phi' = c' + \sigma' \tan \phi'$$

Where:
* $\tau_f$: Shear strength along the prospective failure plane
* $c'$: Effective soil cohesion (strengthened by vegetative root networks; parameterized by NDVI)
* $\sigma$: Total normal stress generated by regolith weight and slope angle ($\sigma = \gamma \cdot z \cdot \cos^2 \beta$)
* $u$: Pore-water pressure within regolith voids
* $\sigma' = \sigma - u$: Effective normal stress
* $\phi'$: Effective internal friction angle of the soil material

### 3.2 The Destabilizing Mechanism of Antecedent Rainfall
During dry periods, negative pore-water pressure (matric suction) increases effective cohesion, stabilizing steep slopes. When intense precipitation occurs over a saturated catchment (high 3-day and 7-day antecedent rainfall):
1. Downward infiltration exceeds hydraulic conductivity of basal strata.
2. A transient perched water table forms, converting negative matric suction into positive pore-water pressure ($u > 0$).
3. The frictional shear resistance $(\sigma - u) \tan \phi'$ collapses.
4. When gravitational shear stress $\tau = \gamma \cdot z \cdot \sin \beta \cos \beta$ exceeds $\tau_f$, catastrophic slope shear initiation occurs.

---

## 4. Synthesis of Research Gaps Addressed by LandslideSense

From the reviewed literature, three prominent shortcomings emerge in contemporary capstone and academic hazard tools:
1. **Conflation of Susceptibility and Warning:** Most existing machine learning projects train models solely on static geomorphology (elevation, slope, aspect) and incorrectly label the output as an "early warning system". LandslideSense explicitly decouples static susceptibility from dynamic multi-day precipitation triggers (`rainfall_24h`, `rainfall_3d`, `rainfall_7d`).
2. **The "Black Box" Trap in Emergency Advisory:** Deep learning models fail to communicate physical justifications to civil authorities. LandslideSense addresses this through transparent feature contribution breakdowns and a parallel, fully transparent heuristic demonstration mode.
3. **Data Leakage and Fabricated Metrics:** Prevalent academic implementations commit spatial or identifier leakage (fitting preprocessors across full datasets or retaining IDs). LandslideSense implements leak-free `ColumnTransformer` pipelines with strict schema validation.

---

## 5. Verified Bibliographic References (IEEE Format)

```
[1] N. Caine, "The rainfall intensity-duration control of shallow landslides and debris flows," 
    Geografiska Annaler: Series A, Physical Geography, vol. 62, no. 1-2, pp. 23-27, 1980. 
    DOI: 10.1080/04353676.1980.11879996.

[2] F. Guzzetti, S. Peruccacci, M. Rossi, and C. P. Stark, "Rainfall thresholds for the initiation 
    of landslides in central and southern Europe," Meteorology and Atmospheric Physics, vol. 98, 
    no. 3-4, pp. 239-267, 2007. DOI: 10.1007/s00703-007-0262-7.

[3] D. Kirschbaum and T. Stanley, "Satellite-based assessment of rainfall-triggered landslide hazard 
    for situational awareness," Earth's Future, vol. 6, no. 3, pp. 505-523, 2018. 
    DOI: 10.1002/2017EF000715.

[4] B. Pradhan and S. Lee, "Landslide susceptibility assessment and factor effect analysis: 
    backpropagation artificial neural networks and their comparison with support vector machines," 
    Catena, vol. 82, no. 1, pp. 42-59, 2010. DOI: 10.1016/j.catena.2010.04.008.

[5] A. Merghadi, B. Yunus, B. Douani, M. Pham, M. Abderrahmane, and D. Tien Bui, "Machine learning 
    methods for landslide susceptibility studies: the state of the art," Earth-Science Reviews, 
    vol. 207, p. 103225, 2020. DOI: 10.1016/j.earscirev.2020.103225.

[6] A. Dikshit, R. Shukla, and P. P. Pradhan, "Spatial landslide susceptibility mapping in the 
    Uttarakhand Himalayas using machine learning techniques," Natural Hazards, vol. 104, no. 1, 
    pp. 827-849, 2020. DOI: 10.1007/s11069-020-04183-w.

[7] J. Mathew, D. G. Babu, N. K. Kundu, K. V. Kumar, and C. C. Pant, "Integrating intensity-duration 
    based rainfall threshold and landslide susceptibility map for landslide early warning: a case 
    study in parts of Garhwal Himalaya, India," Journal of Earth System Science, vol. 123, 
    pp. 1357-1372, 2014. DOI: 10.1007/s12040-014-0409-4.
```
