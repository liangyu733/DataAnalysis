import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats

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

    def summary(self, col:list = None):
        """
        Summary of the dataset
        """
        
        if col is None:
            col = list(range(min(6, self.shape[1])))
        
        nrows = (len(col) + 2) // 3
        fig, axes = plt.subplots(nrows = nrows, ncols = 3, figsize = (15, 5*nrows))
        if len(col) != 1:
            axes = axes.flatten() 
        else: 
            axes = [axes]

        for i, idx in enumerate(col):
            y = self.iloc[:, idx]
            ax = axes[i]
            nlvl = len(y.unique())

            if y.dtype == "category":
                cnt = y.value_counts()
                colors = ['#A6B0BE', '#B2B8A3', '#C0C9CC']
                if nlvl > 3:
                    others = y.value_counts()[2:].sum() 
                    cnt = cnt[:2]
                    cnt.loc['Others'] = others
                else:
                    colors = colors[:nlvl]
                ax.pie(cnt, labels = cnt.index, autopct = '%1.1f%%', colors = colors)
                ax.set_ylabel('')
            
            else: 
                if nlvl < 10 and all(np.diff(np.sort(y.unique())) == 1):
                    bins = nlvl
                else:
                    bins = 'auto'
                ax.hist(y, bins = bins, color = '#A6B0BE', edgecolor = '#C0C9CC')
            ax.set_title(f"{y.name}")
            
        for i in range(len(col), len(axes)):
            fig.delaxes(axes[i])
        
        plt.tight_layout()
        plt.show()



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
            ylab = y.cat.categories[0]
            y = y.cat.codes
            if cate_num >= 2:
                if cate_num > 2:
                    y = (y == 0).astype(int)
                    print(f"[Warning] Variable '{response}' has more than 2 classes; transform to binary values\n 0: {ylab} and 1: not {ylab}.")
                model = sm.GLM(y, X, family = sm.families.Binomial()).fit()
            else:
                print(f"[Error] Categorical response variable have less than 2 classes.")
        
        elif resp_type == 'count':
            y = y.astype(int)
            # if (y == 0).mean()/len(y) > 0.4:
            #     if y.var() > y.mean() * 2:
            #         model = sm.ZeroInflatedNegativeBinomialP(y, X).fit()
            #     else:
            #         model = sm.ZeroInflatedGeneralizedPoisson(y, X).fit()
            # else:
            if y.var() > y.mean() * 2:
                model = sm.GLM(y, X, family = sm.families.NegativeBinomial()).fit()
            else:
                model = sm.GLM(y, X, family = sm.families.Poisson()).fit()

        else:
            print(f"[Error] 'Invalid {resp_type}'; only 'numerical', 'count', and 'categorical' are allowed.")
        print(model.summary())
        if hasattr(model, "deviance"):
            print(f"Null Deviance: {model.null_deviance:.4f}, Degrees of Freedom: {int(model.nobs-1)}")
            print(f"Residual Deviance: {model.deviance:.4f}, Degrees of Freedom: {int(model.df_resid)}")
        return _ModelInfo(model, X, y, response, resp_type)

class _ModelInfo:
    def __init__(self, model, X, y, response: str, resp_type: str):
        self.model = model
        self.X = X
        self.y = y
        self.response = response
        self.resp_type = resp_type
        self.leverage = model.get_influence().hat_matrix_diag

    def __getattr__(self, name):
        return getattr(self.model, name)
    
    def _mkax(self, ax = None, figsize = (8, 6)):
        if ax is None:
            fig, ax = plt.subplots(figsize = figsize)
            return fig, ax
        else:
            return None, ax

    def resid_vs_fit(self, ax = None):
        fig, ax = self._mkax(ax)
        # ax.set_aspect('equal', adjustable='box')
        # Diagnostic plot: Residuals vs Fitted Values
        if isinstance(self.model.model, sm.GLM):
            fitted = self.model.predict(which = 'linear')
            resid = self.resid_pearson
            resid_type = "Pearson Residuals"
        else: 
            fitted = self.fittedvalues
            resid = self.resid
            resid_type = "Residuals"
        lowess = sm.nonparametric.lowess(resid, fitted, frac = 2/3)
        ax.plot(lowess[:, 0], lowess[:, 1], color = "#980417", linestyle = '-', linewidth = 2)
        ax.scatter(fitted, resid, alpha = 0.7, color = '#B2B8A3')
        ax.axhline(0, color = "gray", linestyle = "--")
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel(f"{resid_type}")
        ax.set_title(f"{resid_type} vs Fitted Values")
        ax.grid(True, linestyle = '--', alpha = 0.5)
        if fig: plt.show()
    
    def qqplot(self, ax = None):
        fig, ax = self._mkax(ax)
        # ax.set_aspect('equal', adjustable='box')
        # Diagnostic plot: Q-Q plot for standard residuals
        if isinstance(self.model.model, sm.GLM):
            resid = np.abs(self.resid_deviance / np.sqrt(self.scale * (1 - self.leverage)))
            resid_type = "Abs. Std. Devaince Residuals"
        else: 
            resid = np.abs(self.resid / np.sqrt(self.scale * (1 - self.leverage)))
            resid_type = "Abs. Std. Residuals"
        
        q_samp = np.sort(resid)
        n = len(q_samp)
        probs = (np.arange(1, n + 1) - 0.5) / n
        q_theo = stats.norm.ppf(0.5 + 0.5 * probs)
        ax.scatter(q_theo, q_samp, alpha = 0.7, color = '#B2B8A3')
        q1 = (np.percentile(q_theo, 25), np.percentile(q_samp, 25))
        q3 = (np.percentile(q_theo, 75), np.percentile(q_samp, 75))
        ax.axline(q1, q3, color = "#C0C9CC", linestyle = '--', linewidth = 2)
        ax.set_title(f"Normal Q-Q Plot of {resid_type}")
        ax.set_xlabel("Theoretical Quantiles")
        ax.set_ylabel(f"{resid_type}")
        ax.grid(True, linestyle = '--', alpha = 0.5)
        if fig: plt.show()
    
    def scale_location(self, ax = None):
        fig, ax = self._mkax(ax)
        # ax.set_aspect('equal', adjustable='box')
        # Diagnostic plot: Scale-Location plot for sqrt of standardized residuals
        if isinstance(self.model.model, sm.GLM):
            fitted = self.model.predict(which = 'linear')
            resid = np.sqrt(np.abs((self.resid_pearson / np.sqrt(self.scale * (1 - self.leverage)))))
            resid_type = "Sqrt. Abs. Std. Pearson Residuals"
        else:
            fitted = self.fittedvalues
            resid = np.sqrt(np.abs((self.resid / np.sqrt(self.scale * (1 - self.leverage)))))
            resid_type = "Sqrt. Abs. Std. Residuals"

        lowess = sm.nonparametric.lowess(resid, fitted, frac = 2/3)
        ax.plot(lowess[:, 0], lowess[:, 1], color = "#980417", linestyle = '-', linewidth = 2)
        ax.scatter(fitted, resid, alpha = 0.7, color = '#B2B8A3')
        ax.set_xlabel("Fitted Values")
        ax.set_ylabel(f"{resid_type}")
        ax.set_title(f"Scale-Location Plot of {resid_type}")
        ax.grid(True, linestyle = '--', alpha = 0.5)
        if fig: plt.show()

    def resid_vs_leverage(self, ax = None):
        fig, ax = self._mkax(ax)
        # ax.set_aspect('equal', adjustable='box')
        # Diagnostic plot: Residuals vs Leverage
        leverage = self.get_influence().hat_matrix_diag
        # cooks_d = self.get_influence().cooks_distance[0]
        if isinstance(self.model.model, sm.GLM):
            resid = self.resid_pearson / np.sqrt(self.scale * (1 - self.leverage))
            resid_type = "Std. Pearson Residuals"
        else: 
            resid = self.resid / np.sqrt(self.scale * (1 - self.leverage))
            resid_type = "Std. Residuals"
        lowess = sm.nonparametric.lowess(resid, leverage, frac = 2/3)
        ax.plot(lowess[:, 0], lowess[:, 1], color = "#980417", linestyle = '-', linewidth = 2)
        ax.scatter(leverage, resid, alpha = 0.7, color = '#B2B8A3')
        ax.axhline(0, color = "#C0C9CC", linestyle = "--")
        ax.axvline(0, color = "#C0C9CC", linestyle = "--")
        ax.set_title(f"{resid_type} vs Leverage")
        ax.set_xlabel("Leverage")
        ax.set_ylabel(f"{resid_type}")
        ax.grid(True, linestyle = '--', alpha = 0.5)
        if fig: plt.show()

    def diagnose(self):
        fig, axes = plt.subplots(nrows = 2, ncols = 2, figsize = (16, 15))
        
        self.resid_vs_fit(axes[0, 0])
        self.qqplot(axes[0, 1])
        self.scale_location(axes[1, 0])
        self.resid_vs_leverage(axes[1, 1])
        plt.tight_layout()
        plt.show()
    
    def out(self, path = "model_result.csv"):
        df = self.model.summary2().tables[1]
        
        if self.model.summary2().tables[0].iloc[0, 1] == 'GLM':
            if self.model.summary2().tables[0].iloc[1, 1] == 'Logit':
                ylab = 'Odds Ratio'
            if self.model.summary2().tables[0].iloc[1, 1] == 'Log':
                ylab = 'Rate Ratio'
            df[ylab] = np.exp(df['Coef.'])
            df['CI Lower'] = np.exp(df['[0.025'])
            df['CI Upper'] = np.exp(df['0.975]'])
        else:
            ylab = 'Coefficient'
            df[ylab] = df['Coef.']
            df['CI Lower'] = df['[0.025']
            df['CI Upper'] = df['0.975]']
        df = df.drop(columns = ['[0.025', '0.975]'])
        
        print(df)
        return df.to_csv(path, index = True)
    
