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
