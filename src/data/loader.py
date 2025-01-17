import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Union
import glob
import re

class DataLoader:
    def __init__(self, data_path: Optional[str] = None, kaggle_folder: Optional[str] = None):
        self.data_path = data_path or Path("data/NEVO2023_8.0.csv")
        self.kaggle_folder = Path("data/FINAL FOOD DATASET") if kaggle_folder is None else Path(kaggle_folder)
        self.df: Optional[pd.DataFrame] = None
        self.kaggle_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Load both NEVO and Kaggle datasets into separate DataFrames."""
        # Load NEVO dataset
        try:
            self.df = pd.read_csv(self.data_path, sep='|', on_bad_lines='skip')
            self.df.columns = self.df.columns.str.strip()
            print("Loaded NEVO dataset columns:", self.df.columns.tolist())
            self.df['data_source'] = 'nevo'
        except Exception as e:
            print(f"Warning: Error loading NEVO dataset: {str(e)}")
            self.df = pd.DataFrame()

        # Load Kaggle dataset from group files
        try:
            # Find all group files
            group_files = sorted(glob.glob(str(self.kaggle_folder / "FOOD-DATA-GROUP*.csv")))
            
            if group_files:
                print(f"Found {len(group_files)} group files")
                dfs = []
                
                for file in group_files:
                    print(f"Loading group file: {file}")
                    group_df = pd.read_csv(file)
                    
                    # Extract group number from filename
                    group_num = re.search(r'GROUP(\d+)', file).group(1)
                    group_df['Group'] = int(group_num)
                    
                    # Add descriptive name based on group
                    group_df['Food_Name'] = f"Food Group {group_num}"
                    
                    dfs.append(group_df)
                
                # Combine all group dataframes
                self.kaggle_df = pd.concat(dfs, ignore_index=True)
                
                # Map column names to match NEVO format
                column_mapping = {
                    'Caloric Value': 'ENERCC (kcal)',
                    'Fat': 'FAT (g)',
                    'Carbohydrates': 'CHO (g)',
                    'Protein': 'PROT (g)',
                    'Dietary Fiber': 'FIBT (g)',
                }
                
                # Rename columns
                self.kaggle_df = self.kaggle_df.rename(columns=column_mapping)
                
                # Add required columns for compatibility
                self.kaggle_df['Voedingsmiddelnaam/Dutch food name'] = self.kaggle_df.apply(
                    lambda row: f"Food Group {row['Group']} - {row.get('Food_Name', '')}", axis=1
                )
                self.kaggle_df['Engelse naam/Food name'] = self.kaggle_df['Voedingsmiddelnaam/Dutch food name']
                self.kaggle_df['data_source'] = 'kaggle'
                
                print("Loaded Kaggle dataset columns:", self.kaggle_df.columns.tolist())
                print(f"Total Kaggle dataset entries: {len(self.kaggle_df)}")
            else:
                print("Warning: No Kaggle group files found")
                self.kaggle_df = pd.DataFrame()
                
        except Exception as e:
            print(f"Warning: Error loading Kaggle dataset: {str(e)}")
            self.kaggle_df = pd.DataFrame()

        # Return combined dataset if both are available, otherwise return whichever is available
        if not self.df.empty and not self.kaggle_df.empty:
            # Ensure all required columns exist in both dataframes
            required_columns = [
                'Voedingsmiddelnaam/Dutch food name',
                'ENERCC (kcal)',
                'FAT (g)',
                'CHO (g)',
                'PROT (g)',
                'FIBT (g)',
                'data_source'
            ]
            
            # Add missing columns with NaN values
            for col in required_columns:
                if col not in self.df.columns:
                    self.df[col] = pd.NA
                if col not in self.kaggle_df.columns:
                    self.kaggle_df[col] = pd.NA
            
            # Combine datasets
            combined_df = pd.concat([self.df, self.kaggle_df], ignore_index=True)
            print(f"Combined dataset sources: {combined_df['data_source'].unique()}")
            print(f"Total combined dataset entries: {len(combined_df)}")
            return combined_df
        elif not self.df.empty:
            return self.df
        elif not self.kaggle_df.empty:
            return self.kaggle_df
        else:
            raise ValueError("Failed to load both NEVO and Kaggle datasets")

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