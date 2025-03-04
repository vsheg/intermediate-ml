# %%
import numpy as np
import pandas as pd
import seaborn as sns
from sksurv.datasets import load_aids
from sklearn.linear_model import QuantileRegressor
import scienceplots
import matplotlib.pyplot as plt

# %%
X, y_frame = load_aids()
X = X.astype(float)

# %%
y_data = pd.DataFrame(y_frame)
_, y = y_data.values.T
y = y.astype(float)

# %%
quantiles = np.linspace(0.1, 0.9, 20)
coeffs = {}

for q in quantiles:
    model = QuantileRegressor(quantile=q, alpha=0.01)
    model.fit(X, y)
    coeffs[q] = model.coef_

# %%
df_coeffs = pd.DataFrame(coeffs, index=X.columns)
features_important = df_coeffs.abs().sum(axis=1).sort_values(ascending=False).head(4)
df_coeffs = df_coeffs.loc[features_important.index].reset_index(names="feature")
df_coeffs = df_coeffs.melt("feature", var_name="quantile", value_name="coefficient")

# %%
plt.style.use("default")
plt.style.use("no-latex")
plot = sns.relplot(
    data=df_coeffs,
    x="quantile",
    y="coefficient",
    col="feature",
    col_wrap=2,
    kind="line",
    height=2,
    aspect=4 / 3,
)

# %%
plot.savefig("aids.svg")
