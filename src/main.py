----------------------------------------------------------------------
GeneralDataLoader
----------------------------------------------------------------------

loader = GeneralDataLoader(data_dir='./data/') 
print(f"\nTotal datasets loaded: {len(loader.dataframes)}")

for df_name, df in loader.dataframes.items():
    print(f"\n--- Preview of: {df_name} ---")
    print(df.head())

----------------------------------------------------------------------
DuplicationCleanerMerger
----------------------------------------------------------------------

cleaner = DuplicationCleanerMerger(
    dataframes=loader.dataframes,
    output_dir="final_cleaned_output"
)
cleaner.clean_and_merge()

----------------------------------------------------------------------
DuplicationCleanerMerger
----------------------------------------------------------------------

cleaner = FinalDemographicCleaner(
    dataframes=loader.dataframes,
    output_dir="ultimate_clean_output"
)

cleaner.clean_and_export()

----------------------------------------------------------------------
ProfessionalDemographicEDA
----------------------------------------------------------------------

eda_engine = ProfessionalDemographicEDA()
eda_engine.generate_professional_eda()

----------------------------------------------------------------------
UltimateDemographicAI
----------------------------------------------------------------------

ai = UltimateDemographicAI()
ai.run_pipeline()
