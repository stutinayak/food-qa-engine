import pandas as pd
from typing import List, Dict, Any

class DataProcessor:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        # Debug info
        print("\nAvailable columns:", self.df.columns.tolist())
        print("\nSample row:", self.df.iloc[0])
        self.clean_data()

    def clean_data(self) -> None:
        """Clean and prepare the dataset."""
        # Replace missing values with NaN
        self.df = self.df.replace(['', 'NA', 'N/A'], pd.NA)
        
        # Convert European number format (comma to dot)
        numeric_columns = self.df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                # Replace comma with dot and convert to float
                self.df[col] = self.df[col].str.replace(',', '.').astype(float)
            except:
                continue

    def get_food_by_name(self, name: str) -> Dict[str, Any]:
        """Search for a food item by name."""
        result = self.df[self.df['Voedingsmiddelnaam/Dutch food name'].str.contains(name, case=False, na=False)]
        # Debug info
        print(f"\nSearching for: {name}")
        print(f"Found {len(result)} matches")
        if len(result) > 0:
            print("First match columns:", result.iloc[0].to_dict())
        if len(result) == 0:
            return {}
        return result.iloc[0].to_dict()

    def get_vegetables(self) -> pd.DataFrame:
        """Get all vegetables from the dataset."""
        # Debug info
        print("\nFood groups available:", self.df['Food group'].unique())
        vegetables = self.df[self.df['Food group'] == 'Vegetables']
        print(f"\nFound {len(vegetables)} vegetables")
        return vegetables

    def search_foods(self, query: str) -> pd.DataFrame:
        """Search for food items containing the query string."""
        return self.df[self.df['Voedingsmiddelnaam/Dutch food name'].str.contains(query, case=False, na=False)]

    def get_column_stats(self, column: str) -> Dict[str, float]:
        """Get basic statistics for a numeric column."""
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in dataset")
        
        return {
            'mean': self.df[column].mean(),
            'median': self.df[column].median(),
            'min': self.df[column].min(),
            'max': self.df[column].max()
        } 