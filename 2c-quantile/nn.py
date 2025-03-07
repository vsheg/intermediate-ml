# %%
import pandas as pd
import seaborn as sns
import lightning as L
import torch as T
import matplotlib.pyplot as plt

# %% Generate data
x = T.linspace(-5, 5, 1024)
X = x.reshape(-1, 1)

y = x + 5 * x.cos() + T.normal(0, 1, size=x.shape)
y = y.reshape(-1, 1)


# %%
class QuantileLoss(L.LightningModule):
    def __init__(self, q: float):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        return T.where(
            (epsilon := y_true - y_pred) >= 0,
            self.q * epsilon,
            (self.q - 1) * epsilon,
        ).mean()


# %% Define simple model
class Model(L.LightningModule):
    def __init__(self, q: float | None = None):
        super().__init__()

        self.model = T.nn.Sequential(
            T.nn.LazyLinear(64),
            T.nn.GELU(),
            T.nn.LazyLinear(64),
            T.nn.GELU(),
            T.nn.LazyLinear(1),
        )

        self.loss = QuantileLoss(q) if q else T.nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = self.loss(y_pred, y)
        return loss

    def train_dataloader(self):
        dataset = T.utils.data.TensorDataset(X, y)
        return T.utils.data.DataLoader(
            dataset, batch_size=128, shuffle=True, pin_memory=True
        )

    def configure_optimizers(self):
        return T.optim.NAdam(self.parameters(), lr=1e-2)


# %%
quantiles = [0.025, 0.5, 0.975]
df = pd.DataFrame({"x": X.reshape(-1), "y": y.reshape(-1)})

# %% Predict mean
model = Model()
trainer = L.Trainer(max_epochs=100)
trainer.fit(model)
df["mean"] = model(X).detach()

# %% Predict quantiles
for q in quantiles:
    model = Model(q=q)
    trainer = L.Trainer(max_epochs=100)
    trainer.fit(model)
    df[q] = model(X).detach()

# %%
df = df.melt(
    id_vars=["x", "y"],
    value_vars=["mean"] + quantiles,
    var_name="model",
    value_name="y_pred",
)

# %% Plot
with plt.style.context("../assets/plot.mplstyle"):
    sns.scatterplot(data=df, x="x", y="y", s=5, alpha=0.2)
    sns.lineplot(data=df, x="x", y="y_pred", hue="regression")
    plt.xlabel("$x$")
    plt.ylabel("$y$")

# %%
