import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union
import glob

class DataLoader:
    def __init__(self, data_path: Optional[str] = None, kaggle_folder: Optional[str] = None):
        self.data_path = data_path or Path("data/NEVO2023_8.0.csv")
        self.kaggle_folder = kaggle_folder and Path(kaggle_folder)
        self.df: Optional[pd.DataFrame] = None
        self.kaggle_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load both NEVO and Kaggle datasets into separate DataFrames."""
        # Load NEVO dataset
        try:
            self.df = pd.read_csv(self.data_path, sep='|', on_bad_lines='skip')  # Adjust sep if needed
            self.df.columns = self.df.columns.str.strip()  # Strip whitespace from column names
            print("Loaded NEVO dataset columns:", self.df.columns.tolist())  # Debugging output
            self.df['data_source'] = 'nevo'
        except Exception as e:
            raise Exception(f"Error loading NEVO dataset: {str(e)}")

        # Load Kaggle dataset if folder provided
        if self.kaggle_folder:
            try:
                # Find all group files
                group_files = sorted(glob.glob(str(self.kaggle_folder / "FOOD-DATA-GROUP*.csv")))
                
                if group_files:
                    # Read and combine all group files into a separate DataFrame
                    dfs = []
                    for file in group_files:
                        print(f"Loading Kaggle group file: {file}")
                        df = pd.read_csv(file, sep='|', on_bad_lines='skip')  # Adjust sep if needed
                        df.columns = df.columns.str.strip()  # Strip whitespace from column names
                        print("Loaded Kaggle dataset columns:", df.columns.tolist())  # Debugging output
                        dfs.append(df)
                    
                    if dfs:
                        # Combine all group dataframes into a single Kaggle DataFrame
                        self.kaggle_df = pd.concat(dfs, ignore_index=True)
                        self.kaggle_df['data_source'] = 'kaggle'
                        print("Loaded Kaggle dataset columns:", self.kaggle_df.columns.tolist())  # Debugging output
            except Exception as e:
                print(f"Warning: Error loading Kaggle dataset: {str(e)}")

        return self.df  # Return the NEVO dataset as default

    def get_nevo_data(self) -> pd.DataFrame:
        """Return the NEVO dataset."""
        return self.df

    def get_kaggle_data(self) -> pd.DataFrame:
        """Return the Kaggle dataset."""
        return self.kaggle_df 

    def get_data(self, source: str = "combined") -> pd.DataFrame:
        """Get the dataset based on the specified source.
        
        Args:
            source: Which dataset to return ("nevo", "kaggle", or "combined"). Defaults to "combined".
            
        Returns:
            pd.DataFrame: The requested dataset
        """
        if self.df is None:
            self.load_data()
            
        if source == "nevo":
            return self.get_nevo_data()
        elif source == "kaggle":
            return self.get_kaggle_data()
        elif source == "combined":
            # Create combined dataset if both sources are available
            if self.kaggle_df is not None:
                # Identify common columns
                common_cols = set(self.df.columns) & set(self.kaggle_df.columns)
                print(f"Common columns between datasets: {common_cols}")
                
                # Combine datasets using only common columns
                return pd.concat([
                    self.df,
                    self.kaggle_df[list(common_cols)]
                ], ignore_index=True)
            else:
                return self.df
        else:
            raise ValueError(f"Unknown data source: {source}") 