import pandas as pd
import json
from typing import List, Dict, Literal

class BaseDataLoader:
    """
    BaseDataLoader loads question-answer data from supported format.

    Parameters:
    --------------------
    file_path : str
        Path to the input data file.

    file_type : str, optional
        Extension of data file. Must be one of: "csv", "json", "parquet", "xml"
        Default is "csv".

    """
    SUPPORTED_FILE_EXTENSIONS = ["csv", "json", "parquet", "xml"]

    def __init__(self, file_path: str, file_type: Literal["csv", "json", "parquet", "xml"] = "csv"):
        self.file_path = file_path
        self.file_type = file_type.lower()
        if self.file_type not in self.SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(f"Unsupported file type '{self.file_type}'. Supported file types are: {self.SUPPORTED_FILE_EXTENSIONS}")

    def load(self) -> List[Dict[str, str]]:
        if self.file_type == "csv":
            df = pd.read_csv(self.file_path)
        elif self.file_type == "json":
            with open(self.file_path, 'r', encoding = 'utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif self.file_type == "parquet":
            df = pd.read_parquet(self.file_path)
        elif self.file_type == "xml":
            df = pd.read_xml(self.file_path)
        
        return df
    
    @staticmethod
    def help():
        """
        Prints help about how to use BaseDataLoader.
        """
        print(BaseDataLoader.__doc__)
        
    
     
