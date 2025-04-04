# %%
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, QuantileRegressor

# %%
x = np.linspace(-5, 5, 128)
y = x
y_lognormal = y + np.random.lognormal(0, 4, 128)
y_laplace = y + np.random.laplace(0, 4, 128)

# %%
bimodal_noise = np.concatenate([np.random.normal(-2, 1, 64), np.random.normal(+2, 3, 64)])
np.random.shuffle(bimodal_noise)
y_bimodal = x + bimodal_noise


# %%
df = pd.DataFrame(
    {
        "x": x,
        "y": y,
        "y_lognormal": y_lognormal,
        "y_laplace": y_laplace,
        "y_bimodal": y_bimodal,
    }
)

# %%
for col in df.columns[2:]:
    X = x.reshape(-1, 1)

    model_ls = LinearRegression()
    model_ls.fit(X, df[col])
    df[f"{col}_pred_ls"] = model_ls.predict(X)

    model_qr = QuantileRegressor(quantile=0.5, alpha=0)
    model_qr.fit(X, df[col])
    df[f"{col}_pred_qr"] = model_qr.predict(X)


# %%
parent = Path(__file__).parent
df.to_csv(parent / "out.csv", index=False)
