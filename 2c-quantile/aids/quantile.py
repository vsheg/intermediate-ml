# %%
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import QuantileRegressor
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector
import numpy as np
from pathlib import Path

# %%
df = pd.read_csv("aids.csv")

y = df.pop("time")
censored = df.pop("censored")

X = df.copy()

# %%
model_median = QuantileRegressor(quantile=0.5, alpha=0.0)
sfs = SequentialFeatureSelector(
    model_median,
    direction="backward",
    scoring="neg_mean_absolute_error",
    cv=5,
)
sfs.fit(X, y)

mask_features = sfs.get_support()
X = X.loc[:, mask_features]

# %%
coeffs_dict = {}
quantiles = np.linspace(0.05, 0.95, 20)

for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0.0)
    model.fit(X, y)
    coeffs_dict[q] = model.coef_

df_coeffs = pd.DataFrame(coeffs_dict, index=X.columns).T
df_coeffs.index.name = "quantile"

# %%
path = Path(__file__).parent / "coeffs.csv"
df_coeffs.to_csv(path)

# %%
