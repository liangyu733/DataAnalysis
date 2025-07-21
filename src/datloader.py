import pandas as pd

# def dat_load(filepath: str) -> pd.DataFrame:
#         """
#         Data loading
#         """
#         df = pd.read_csv(filepath)
#         print(f"Import successfully; nrow = {df.shape[0]}, ncol = {df.shape[1]}.")
#         return df

class Data(pd.DataFrame):
    def __init__(self, filepath: str):
        # Data loading
        df = pd.read_csv(filepath)
        df = self._fix_dtypes(df)
        super().__init__(df)
        print(f"Import successfully; nrow = {self.shape[0]}, ncol = {self.shape[1]}.")

    @classmethod
    def from_DataFrame(cls, df: pd.DataFrame):
        df = cls._fix_dtypes(df)
        obj = cls.__new__(cls)
        pd.DataFrame.__init__(obj, df)
        return obj
    
    @staticmethod
    def _fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            dtype = df[col].dtype

        # 1. int64 → Int64
            if dtype == "int64":
                df[col] = df[col].astype("Int64")

        # 2. float64(by missing) → Int64 
            elif dtype == "float64":
                series = df[col].dropna()
                if (series == series.astype(int)).all():
                    df[col] = df[col].astype("Int64")

        # 3. bool / boolean → Int64
            elif dtype in ("bool", pd.BooleanDtype()):
                df[col] = df[col].astype("boolean").astype("Int64")

        return df

    def summary(self):
        """
        Summary of the dataset
        """
        print("\n Summary:")
        print(self.info())
        print("\n NA:")
        print(self.isnull().sum())

    def impute(self):
        df = self.copy()
        for rv in df.columns:
            if df[rv].isnull().any():
                if pd.api.types.is_float_dtype(df[rv]):
                    imp_val = df[rv].median()
                
                else:
                    imp_val = df[rv].mode()[0]
                df[rv] = df[rv].fillna(imp_val)
        
        return self.from_DataFrame(df)

    # def explain(self, response: str):
    #     if response not in self.columns:
    #         print(f"[Error] '{response}' not in DataFrame columns.")
    #         return
        
    #     df = self.dropna(subset=[response]).copy()
    #     y = df[response]
    #     if df[response].dtype in ["object", "bool", "category"]:

     
 