class GeneralDataLoader:
    """
    A utility class to automatically load all CSV files from a directory 
    into a dictionary of Pandas DataFrames.
    """
    
    def __init__(self, data_dir: str = '/content/'):
        """
        Initializes the loader and triggers the data loading process.
        
        Args:
            data_dir (str): The path to the directory containing CSV files.
        """
        self.data_dir = data_dir
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self._load_data()

    def _sanitize_name(self, filename: str) -> str:
        """Standardizes filenames into clean dictionary keys."""
        return (
            filename.replace('.csv', '')
            .replace(' ', '_')
            .replace('-', '_')
            .lower()
        )

    def _load_data(self) -> None:
        """Iterates through the directory and loads CSV files into the dictionary."""
        if not os.path.exists(self.data_dir):
            print(f"Error: Directory '{self.data_dir}' not found.")
            return

        files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        
        if not files:
            print(f"Warning: No CSV files found in {self.data_dir}")
            return

        for filename in files:
            df_name = self._sanitize_name(filename)
            filepath = os.path.join(self.data_dir, filename)
            
            try:
                self.dataframes[df_name] = pd.read_csv(filepath)
                print(f"Successfully loaded: {filename} -> key: '{df_name}'")
            except Exception as e:
                print(f"Failed to load {filename}: {e}")

    def get_df(self, name: str) -> Optional[pd.DataFrame]:
        """Safely retrieve a dataframe by its key."""
        return self.dataframes.get(name)
        
