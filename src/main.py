# Initialize the loader
loader = GeneralDataLoader(data_dir='./data/') # Change to your path

# Display summary of loaded data
print(f"\nTotal datasets loaded: {len(loader.dataframes)}")

for df_name, df in loader.dataframes.items():
    print(f"\n--- Preview of: {df_name} ---")
    print(df.head())
