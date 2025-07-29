import pandas as pd
import numpy as np
import statsmodels.api as sm

class Data(pd.DataFrame):
    def __init__(self, filepath: str, as_category: list = None, impute: bool = False):
        # Data loading
        df = pd.read_csv(filepath)
        if impute:
            df = self._impute(df)
        else:
            ini_nrow = df.shape[0]
            df = df.dropna()
            if df.shape[0] != ini_nrow:
                print(f"[Warning] Dropped {ini_nrow - df.shape[0]} rows with missing values.")
        df = self._fix_dtypes(df, as_category)
        super().__init__(df)
        print(f"Import successfully; nrow = {self.shape[0]}, ncol = {self.shape[1]}.")

    @classmethod
    def from_DataFrame(cls, df: pd.DataFrame):
        df = cls._fix_dtypes(df)
        obj = cls.__new__(cls)
        pd.DataFrame.__init__(obj, df)
        return obj
    
    @staticmethod
    def _impute(df: pd.DataFrame) -> pd.DataFrame:
        for rv in df.columns:
            if df[rv].isnull().any():
                if pd.api.types.is_float_dtype(df[rv]):
                    imp_val = df[rv].median()
                
                else:
                    imp_val = df[rv].mode()[0]
                df[rv] = df[rv].fillna(imp_val)
        
        return df
    
    @staticmethod
    def _fix_dtypes(df: pd.DataFrame, as_category: dict = None) -> pd.DataFrame:
        if isinstance(df, Data):
            df = pd.DataFrame(df)
        
        as_category = as_category or {}

        for col in df.columns:
            dtype = df[col].dtype

            # 0. as_category → category
            if col in as_category:
                baseline = as_category[col]
                labels = df[col].unique().tolist()
                if baseline in labels:
                    labels.remove(baseline)
                    labels.insert(0, baseline)
                else: 
                    print(f"[Warning] Baseline '{baseline}' not found in column '{col}' categories: '{labels}'.")
                df[col] = pd.Categorical(df[col], categories = labels)
            
            # 1. object → category
            elif dtype == "object":
                df[col] = pd.Categorical(df[col])

            # 2. int64
            elif dtype == "int64":
                pass

            # 3. float64(by missing) → int64 
            elif dtype == "float64":
                series = df[col].dropna()
                if (series == series.astype(int)).all():
                    df[col] = df[col].astype("int64")

            # 4. bool / boolean → int64
            elif dtype in ("bool", pd.BooleanDtype()):
                df[col] = df[col].astype("boolean").astype("int64")

        return df

    def summary(self):
        """
        Summary of the dataset
        """
        print("\n Summary:")
        print(self.info())
        print("\n NA:")
        print(self.isnull().sum())

    def _to_design(self, response: str) -> np.ndarray:
        df = self.drop(response, axis = 1)
        df = pd.get_dummies(df, drop_first = True)
        df = sm.add_constant(df)
        df = Data._fix_dtypes(df)
        return df

    def explain(self, response: str, resp_type: str):  
        if response not in self.columns:
            print(f"[Error] '{response}' not in DataFrame columns.")
            return
        
        X = self._to_design(response)
        y = self[response]
        if resp_type == 'numerical':
            y = y.astype(float)
            model = sm.OLS(y, X).fit()

        elif resp_type == 'categorical':
            y = y.astype("category")
            cate_num = len(y.unique())
            y = y.cat.codes
            if cate_num == 2:
                model = sm.Logit(y, X).fit()
            elif cate_num > 2:
                model = sm.MNLogit(y, X).fit()
            else:
                print(f"[Error] Categorical response variable have less than 2 classes.")
        
        elif resp_type == 'count':
            y = y.astype(int)
            if (y == 0).mean()/len(y) > 0.4:
                if y.var() < y.mean() * 2:
                    model = sm.ZeroInflatedNegativeBinomialP(y, X).fit()
                else:
                    model = sm.ZeroInflatedGeneralizedPoisson(y, X).fit()
            else:
                if y.var() < y.mean() * 2:
                    model = sm.NegativeBinomial(y, X).fit()
                else:
                    model = sm.Poisson(y, X).fit()

        else:
            print(f"[Error] 'Invalid {resp_type}'; only 'numerical', 'count', and 'categorical' are allowed.")
        print(model.summary())
        return model