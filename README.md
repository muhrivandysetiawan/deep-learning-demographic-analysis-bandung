# Deep Learning for Demographic Big Data Analysis: Towards Efficient and Accurate Government Services (Bandung City Case Study) </br>

## Project Overview </br>

This project implements Deep Learning methodologies, specifically **Autoencoder** and **Artificial Neural Network (ANN)**, to analyze demographic **big data** for Bandung City spanning 2017–2024. The primary objectives are to detect anomalies and inconsistencies across institutional datasets (Statistics Indonesia [BPS] and Population and Civil Registration Office [Disdukcapil]) and to predict e-KTP ownership gaps at the district/subdistrict level. The initiative aims to enhance population governance efficiency, accuracy, and responsiveness while delivering targeted policy recommendations for local government stakeholders. </br>

## Research and Engineering Challenges Addressed </br>

### Research Challenges </br>

- **Population Data Quality** : The associated journal article identifies critical issues in demographic data quality, including inter-agency inconsistencies between BPS and Disdukcapil datasets, duplicates, missing values, and delayed updates that impede effective public policy utilization.</br>
- **Anomaly Detection** : Absence of robust methods to identify data anomalies (e.g., unexplained population surges without documented migration, inconsistent inputs) that lead to misdirected policy interventions.</br>
- **e-KTP Ownership Disparities** : Spatial gaps in e-KTP possession across Bandung's districts and subdistricts requiring focused policy interventions.</br>
- **Targeted Policy Prediction** : Need for predictive tools to prioritize intervention areas for e-KTP coverage improvement based on demographic and spatial variables.</br>

### Engineering Challenges </br>

- **Large-Scale Data Management and Cleaning** : Handling voluminous, heterogeneous population data from multiple sources with robust pipelines for deduplication, missing value treatment, and transformation. Addressed via GeneralDataLoader.py , DuplicationCleanerMerger.py , and FinalDemographicCleaner.py . </br>
- **Deep Learning Model Implementation** : Translating Autoencoder and ANN theoretical frameworks into optimized, functional code ( UltimateDemographicAI.py ) using TensorFlow/Keras. </br>
- **Complex Data Preprocessing** : Systematic procedures for duplicate removal, missing value imputation, categorical-to-numeric transformation, and feature scaling for deep learning readiness. </br>
- **Model Evaluation and Validation** : Comprehensive metrics (MSE, RMSE, MAE, R²) and validation techniques (train-test split, k-fold cross-validation) ensuring model stability and generalizability. </br>
- **Component Integration** : Cohesive workflow orchestration across data loading, cleaning, EDA, training, and prediction modules via main.py . </br>

### Project Structure  </br>

deep-learning-demographic-analysis-bandung/   </br>
├── README.md                            </br>
├── data/                                </br>
│   ├── total_e_ktp_ownership_coverage_bandung.csv  </br>
│   ├── population_by_religion_bandung.csv  </br>
│   ├── population_by_blood_type_bandung.csv  </br>
│   ├── population_by_gender_bandung.csv  </br>
│   ├── population_by_occupation_bandung.csv  </br>
│   ├── population_by_education_level_bandung.csv  </br>
│   ├── population_by_district_bandung.csv  </br>
│   ├── population_by_age_group_bandung.csv  </br>
│   ├── population_by_subdistrict_bandung.csv  </br>
│   ├── population_by_marital_status_bandung.csv  </br>
│   ├── total_population_bandung.csv  </br>
│   └── population_eligible_for_id_cards_bandung.csv  </br>
├── src/                                </br>
│   ├── GeneralDataLoader.py  </br>
│   ├── DuplicationCleanerMerger.py  </br>
│   ├── FinalDemographicCleaner.py  </br>
│   ├── ProfessionalDemographicEDA.py  </br>
│   ├── UltimateDemographicAI.py  </br>
│   └── main.py  </br>
├── docs/                               </br>
│   └── jurnal_registratie_5572.pdf      </br>
├── requirements.txt                     </br>
└── LICENSE                             </br>

## Datasets Utilized  </br>

The datasets focus on Bandung City's population data across multiple demographic dimensions:  </br>
e-KTP Ownership Coverage in Bandung City  </br>
- Bandung Population by Religion  </br>
- Bandung Population by Blood Type  </br>
- Bandung Population by Gender  </br>
- Bandung Population by Occupation Type  </br>
- Bandung Population by Education Level  </br>
- Bandung Population by District  </br>
- Bandung Population by Age Group  </br>
- Bandung Population by Subdistrict  </br>
- Bandung Population by Marital Status  </br>
- Total Bandung Population  </br>
- Bandung KTP-Eligible Population  </br>

Associated Journal Article </br>
This research is grounded in the following peer-reviewed publication: </br>
Title: APPLICATION OF DEEP LEARNING FOR POPULATION BIG DATA ANALYSIS: TOWARDS EFFICIENCY AND ACCURACY IN GOVERNMENT SERVICES </br>
Author: Muh Rivandy Setiawan </br>
Journal: Jurnal Registratie 7(2), August 2025: 111-127
Link: 
- https://ejournal.ipdn.ac.id/index.php/jurnalregistratie/issue/view/405 </br>
- https://ejournal.ipdn.ac.id/index.php/jurnalregistratie/article/view/5572 </br>
Full PDF available in docs/ folder. </br>

## Installation and Usage </br>
### Prerequisites </br>
*   Python 3.8+ </br> 
*   pip  </br>
### Setup Instructions </br>
1.  **Clone the repository:** </br>
    ```bash </br>
    git clone https://github.com/your-username/deep-learning-demographic-analysis-bandung.git 
    cd deep-learning-demographic-analysis-bandung
    ``` 
2.  **Create virtual environment (recommended):** </br>
    ```bash </br>
    python -m venv venv 
    source venv/bin/activate  # Pada Windows gunakan `venv\Scripts\activate` 
    ```
3.  **Install dependencies:** </br>
    ```bash </br>
    pip install -r requirements.txt 
    ``` 
4.  **Run the project:** </br>
    ```bash </br>
    python src/main.py 
    ``` 
    *(Additional parameter/mode instructions can be added here as needed.)* </br>
 
##  Contributing </br>
Please refer to CONTRIBUTING.md for contribution guidelines. </br>
 ## License </br>
This project is licensed under the MIT LICENSE. See the LICENSE file for details.</br>
##  Contact </br>
For questions or suggestions, please contact muhrivandysetiawan@gmail.com. </br>

