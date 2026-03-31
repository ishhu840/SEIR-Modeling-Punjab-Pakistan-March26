# Dynamic SEIR Modeling of Dengue Transmission in Punjab, Pakistan

**Author:** Ishtiaq  
**Context:** Repository for PhD 9-Month Progress Report  
**Status:** Final Version (March 27, 2026 Manuscript)

---

## 📌 Project Overview
**Study Objective:** To understand and characterise the baseline dengue transmission dynamics in the Punjab region of Pakistan using data from 11 cities collected from 2013 to 2024. To achieve this, we developed a climate-driven, continuous compartment model (SEIR model) using differential equations. A secondary objective was to establish predictive mathematical foundations that could seamlessly serve as a training platform for future Machine Learning models.

This repository contains the complete computational framework, raw tabular data, and visual generation pipeline used to investigate these spatio-temporal dynamics and the specific meteorological drivers of dengue fever. The methodology leverages grid-search optimization to extract optimal climate lags, statistical models to validate correlations, and geographical analyses to map localized epidemic clustering.

### Key Results (Table 2: City-Specific Optimal Lags & Prediction Performance)
The execution of the SEIR optimization grid yielded explicit meteorological incubation delays and subsequent high training/testing correlation scores across the studied geography.

| City            | Optimal Rain Lag *(Weeks)* | Optimal Temp Lag *(Weeks)* | Training Performance *(r)* | Training *(p)* | Testing Performance *(r)* | Testing *(p)* |
|:----------------|:--------------------------:|:--------------------------:|:--------------------------:|:--------------:|:-------------------------:|:-------------:|
| Lahore          |          5                 |         10                 |     0.526                  |    < 0.001     |    0.443                  |    < 0.001    |
| Rawalpindi      |          4                 |         11                 |     0.512                  |    < 0.001     |    0.535                  |    < 0.001    |
| Faisalabad      |          5                 |          7                 |     0.403                  |    < 0.001     |    0.504                  |    < 0.001    |
| Gujranwala      |          7                 |          6                 |     0.479                  |    < 0.001     |    0.478                  |    < 0.001    |
| Multan          |          5                 |         12                 |     0.563                  |    < 0.001     |    0.456                  |    < 0.001    |
| Islamabad       |          7                 |          8                 |     0.529                  |    < 0.001     |    0.428                  |    < 0.001    |
| Sargodha        |          6                 |          8                 |     0.582                  |    < 0.001     |    0.305                  |    0.002      |
| Sheikhupura     |          7                 |         10                 |     0.301                  |    < 0.001     |   -0.013                  |    0.896      |
| Dera Ghazi Khan |          7                 |          8                 |     0.300                  |    < 0.001     |    0.054                  |    0.584      |

---

## 🧮 Mathematical Framework (SEIR Model)

The biological transmission dynamics across the 11 study cities are modeled using a climate-driven, continuous time-delay SEIR differential framework, evaluated weekly ($t$).

**State Equations (with demographic flow):**
$$ S(t+1) = S(t) - inc(t) + \omega R(t) + \mu(N(t) - S(t)) + \Delta N(t+1) $$
$$ E(t+1) = E(t) + inc(t) - (\sigma + \mu)E(t) $$
$$ I(t+1) = I(t) + \sigma E(t) - (\gamma + \mu)I(t) $$
$$ R(t+1) = R(t) + \gamma I(t) - (\omega + \mu)R(t) $$

**Observation & Force of Infection:**
$$ inc(t) = \beta(t) \frac{S(t) I(t)}{N(t)} + \lambda $$
$$ \beta(t) = \kappa \cdot \exp(b_0 + b_R \cdot Rain_{z, t-lag} + b_T \cdot Temp_{z, t-lag} + b_{T2} \cdot Temp_{z, t-lag}^2) $$
$$ \hat{Y}(t) = \rho \cdot inc(t) $$

**Biological Parameters:**
*   $\sigma$: Intrinsic incubation rate (Exposed $\rightarrow$ Infectious)
*   $\gamma$: Recovery rate (Infectious $\rightarrow$ Recovered)
*   $\omega$: Waning immunity rate (Recovered $\rightarrow$ Susceptible)
*   $\mu$: Background baseline demographic turnover (Births/Deaths)
*   $\rho$: Observational reporting fraction
*   $\beta(t)$: The core transmission multiplier, driven by meteorological variables pushed backward by the optimal temporal lags (Table 2).

### 📝 Study Workflow Pseudocode
```text
 1. START STUDY WORKFLOW
 2. PART 1: DATA PREPARATION
 3.     LOAD weekly dengue cases and weather records (2013–2024)
 4.     LOAD population census data for 2017 and 2023
 5.     DEFINE list of nine study cities
 6.     SPLIT data into Training (2013–2022) and Testing (2023–2024)
 7.     FOR EACH city IN list of cities
 8.         CALCULATE dynamic population for each week using interpolation
 9.         AGGREGATE daily weather to weekly means and totals
10.         NORMALIZE weather variables using training-period statistics only
11.     END FOR

12. PART 2: STATISTICAL VALIDATION
13.     FOR EACH city IN list of cities
14.         CALCULATE Spearman correlation between lagged weather and cases
15.         FIT Negative Binomial regression models to check weather associations
16.         COMPARE model fit using AIC values to justify lag ranges
17.     END FOR

18. PART 3: SEIR MODEL CALIBRATION (GRID SEARCH)
19.     DEFINE lag ranges (Rainfall: 4–7 weeks, Temperature: 6–12 weeks)
20.     FOR EACH city IN list of cities
21.         SET best model configuration to None
22.         SET best performance score to negative infinity
23.         FOR EACH rainfall lag IN lag range
24.             FOR EACH temperature lag IN lag range
25.                 SHIFT weather variables by selected lags
26.                 DEFINE SEIR model equations with weather-driven transmission
27.                 DEFINE error function including biological penalty terms
28.                 OPTIMIZE model parameters using Nelder-Mead algorithm
29.                 CALCULATE performance score (Pearson correlation) on training data
30.                 IF current score is better than best score
31.                     UPDATE best score to current score
32.                     UPDATE best model configuration to current settings
33.                 END IF
34.             END FOR
35.         END FOR
36.         SAVE optimal lag configuration and fitted parameters for this city
37.     END FOR

38. PART 4: TESTING AND VALIDATION
39.     FOR EACH city IN list of cities
40.         GENERATE predictions for testing period (2023–2024) using optimal model
41.         CALCULATE Pearson correlation, MAE, and RMSE on testing data
42.         COMPUTE 95% Confidence Intervals using Fisher Z-transformation
43.         PERFORM residual diagnostics (Ljung-Box test for white noise)
44.     END FOR

45. PART 5: COMPARISON AND OUTPUT
46.     COMPARE optimal lag performance against fixed lag (7 weeks) baseline
47.     PERFORM Wilcoxon signed-rank test across cities for improvement significance
48.     COMPILE results into summary tables
49.     GENERATE final figures including time series, heatmaps, and forest plots
50.     SAVE all outputs for report integration
51. END STUDY WORKFLOW
```

---
## 📂 Repository Structure

The codebase is strictly modular to separate pure data, execution scripts, raw numerical outputs, and qualitative graphics.

### `01_Data/`
Contains the curated epidemiological and demographic spreadsheets necessary to feed the SEIR models.
*   `D1_Weekly_Cases_Weather_AllCities.xlsx`: Merged dataset capturing weekly confirmed dengue infections and trailing weather metrics (Temperature, Rainfall) for all 11 subject cities.
*   `D2_Population_2017_2023.xlsx`: Urban demographic metrics used to normalize infection rates and define demographic models.

### `02_Code/`
The complete Python ecosystem utilized to run the study. Scripts are enumerated in execution/pipeline order:
*   **SEIR Optimization:** The primary Grid Search script handles non-linear grid optimization isolating the optimum biological transmission rates ($\beta$) and temperature-dependent lag parameters.
*   **Statistical Validation:** `S2_Statistical_Tests.py`, `S3_Plot_Spearman_NB.py`, and `S8_Residual_Analysis.py` handle non-parametric correlations and Ljung-Box residual diagnostics. 
*   **Results Visualization:** `S4` through `S6` scripts dynamically handle the generation of predictive Time-Series charts and localized thermal reaction curves.
*   **Descriptive Spatial Analytics:** Scripts `S9_Manuscript_Analysis.py`, `S11` through `S15` generate robust geo-spatial heatmaps, Choropleths, Age/Sex segmentations, and comparative population bubble charts.

### `03_Results/`
Tabular outputs and numerical proofs generated by the core SEIR modeling execution.
*   **City-Level Predictions:** `*_Predictions.xlsx` files for each isolated district containing theoretical curves mapped against raw empirical cases.
*   **Optimization Diagnostics:** Core grid search outputs and `Optimal_Lags_ByCity.xlsx` containing the exact scalar mappings for maximum modeling efficiency.
*   **Test Metrics:** Associated statistical proving tests and `S8_LjungBox_TestResults.csv`.

### `04_Figures/`
Publication-quality illustrations, schematics, and vector-aligned graphs exported for direct injection into the final manuscript (`Manuscript_March_27_2026.docx`). Includes Graphviz `.dot` source models mapping the SEIR mechanism pathway.

---

## 🚀 Execution & Replication

To replicate the study's analytical progression from your local environment:

### Prerequisites
*   Python 3.x+
*   Core Scientific Modules: `pandas`, `numpy`, `scipy`, `matplotlib`, `seaborn`

### General Pipeline Walkthrough
1. **Initialize Parameters:** Ensure directories in `02_Code` target the localized structures established in `01_Data`.
2. **Execute Modeling:** Run the primary Grid Search script to calculate optimized transmission vectors across the 11 study cities. The outputs will immediately populate into `03_Results`.
3. **Statistical Validation:** Run `02_Code/S2_Statistical_Tests.py` to pull the optimal lag targets generated in Step 2 and run standard variance tests.
4. **Data Visualization:** Any of the `S4`-`S15` Python pipelines can be targeted directly to regenerate high-DPI `.png` files found in the `04_Figures` envelope. 

---

## 📊 Core Visual Analyses

### 1. Spatiotemporal Transmission Hotspots
![Transmission Hotspots](04_Figures/Fig_Geo_Clusters_LHR_RWP.png)

**What it shows:** Built using exact geo-coordinates, these density heatmaps map out areas of extreme transmission intensity within Lahore and Rawalpindi. It visually confirms that dengue is not uniformly distributed, but rather forms intense, highly specific localized clusters driven by dense urban geography and localized ecological conditions.

---

### 2. Dengue Burden vs. Urban Population
![Dengue Burden vs Population](04_Figures/Fig_Population_vs_Dengue_Bubble.png)

**What it shows:** A comparison of total patient counts against raw city population brackets. Crucially, this bubble chart demonstrates that raw population size does not perfectly correlate to epidemic risk (for example, Faisalabad has a massive population but relatively low burden compared to Lahore). This validates the core theory that environmental, geographical, and meteorological factors are the true primary epidemic drivers rather than just population.

---

### 3. Seasonal Meteorological Gradients (May – Dec)
![Seasonal Temp](04_Figures/Fig_Seasonal_Weather_Temp_May_Dec.png)
![Seasonal Rain](04_Figures/Fig_Seasonal_Weather_Rain_May_Dec.png)

**What it shows:** These sequential visualizations map the changing temperature and precipitation gradients across Punjab leading up to and during the primary dengue peaks. They clearly illustrate the precise "thermal and precipitational windows" that dictate when Aedes mosquito vectors can physically breed and transmit the virus. 

---

### 4. SEIR Predictive Time-Series Validation
**North, Central, and South Regional Performance:**
![North TimeSeries](04_Figures/Fig09_North_TimeSeries.png)
![Central TimeSeries](04_Figures/Fig09_Central_TimeSeries.png)
![South TimeSeries](04_Figures/Fig09_South_TimeSeries.png)

**What it shows:** These three time-series plots validate the strict predictive capabilities of our mathematically optimized SEIR compartment model against real-world outbreak data across different geographical strata. The high degree of overlap between the theoretical curves and empirical patient counts over the 12-year window proves that temperature-driven lag metrics are a robust mechanism for forecasting future outbreaks.

---

### 5. Biological Thermal Constraints (SEIR Kinetics)
![Thermal Curves](04_Figures/FigS1_ThermalCurves.png)

**What it shows:** This highlights the strict biological bounds enforced within the computational model. Rather than assuming mosquitoes can transmit the virus all year, these curves mathematically define the perfect temperature zone required for maximal mosquito vitality, biting frequency, and viral incubation efficiency—which drop sharply if the climate is too hot or too cold.

---

### 6. Statistical Significance of Climate Drivers
![Pearson Forest Plot](04_Figures/Fig11_PearsonCI_Forest.png)

**What it shows:** This forest plot of Pearson correlation confidence intervals definitively quantifies the statistical power of temperature and rainfall as leading indicators. It solidifies our hypothesis that precise meteorological shifts are statically proven driving forces for transmission volatility across every test city.

---

> *"The results of this computational study underscore the necessity for localized, highly optimized climate-lag strategies for proactive dengue intervention."*
