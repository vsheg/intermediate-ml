# %%
import numpy as np
import seaborn as sns
from sklearn.linear_model import QuantileRegressor, LinearRegression
import matplotlib.pyplot as plt

# %%
plt.style.use("../assets/plot.mplstyle")

# %% Generate Pareto data
np.random.seed(1)
x = np.linspace(0, 1, 100)
y_paretto = np.random.pareto(3, 100) + x

# %% Generate heteroscedastic data
y_heteroscedastic = np.random.normal(0, 1 + x, 100) + x


# %%
def plot_model(model, label=None):
    y_pred = model.predict(x.reshape(-1, 1))
    plt.plot(x, y_pred, label=label)


# %%
def plot_data(model, x, y, label=None):
    model.fit(x.reshape(-1, 1), y)
    plot_model(model, label="mean")

    quantiles = np.array([0.9, 0.75, 0.5, 0.25, 0.1])

    for q in quantiles:
        model_quantile = QuantileRegressor(quantile=q, alpha=0.0)
        model_quantile.fit(x.reshape(-1, 1), y)
        plot_model(model_quantile, label=f"{q:.2f}")

    sns.scatterplot(x=x, y=y)
    plt.legend()
    plt.savefig("pareto.svg", transparent=True)


# %%
plot_data(LinearRegression(), x, y_paretto)
plt.show()

# %%
plot_data(LinearRegression(), x, y_heteroscedastic)
plt.show()
