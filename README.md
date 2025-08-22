content = """# Data Analysis Workflow Module

## Overview  
This project provides a **Python module** designed for a simple data analysis workflow, applicable to different data types.  
The workflow includes:  
1. **Data preprocessing** – handling missing values (imputation or drop) and automatic data type conversion  
2. **Statistical modeling** – explaining the effect of each variable on the target outcome  
3. **Model diagnostics and export** – residual checking and exporting results  

---

## Features  
- Load and preprocess CSV data into a custom `Data` object (a subclass of `pandas.DataFrame`)  
- Automatically handle missing values and data type transformation  
- Fit different statistical models depending on the response type:  
  - OLS regression for numerical response  
  - Logistic regression for binary categorical response  
  - Poisson / Negative Binomial GLM for count data  
- Diagnostic plots for model evaluation  
- Export model results (coefficients, odds ratios/rate ratios with confidence intervals) as `.csv`  

---

## Example Usage  

```python
import datloader as dl

# Load dataset
df = dl.Data('../data/CustomerTravel.csv', impute=False)

# Summarize variables
df.summary()

# Fit logistic regression model
mod1 = df.explain(response="Target", resp_type="categorical")

# Export results
mod1.out("model_result.csv")
