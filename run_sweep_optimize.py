# run_sweep_with_tuning.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor

from data_utils import CoilData
from baselines import LookUpRBF, PolyInverseRegressor, KRRInverseRegressor, NNInverseRegressor

# --- safe MAE that ignores NaNs ---
def _safe_mae(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    return np.nan if mask.sum()==0 else mean_absolute_error(y_true[mask], y_pred[mask])

# --- RBF + Brent eval (no tuning) ---
def _eval_rbf(data, train_df, test_df):
    data.fit_scalers(train_F=train_df, train_G=train_df)
    Xtr, ytr = data.get_xy(train_df, forward=True,  scaled=False)
    Xte, yte = data.get_xy(test_df,  forward=False, scaled=False)
    model = LookUpRBF(bounds=(0,5)).fit(Xtr, ytr)
    ypred = np.array([model._inverse_one(B,x,y) for B,x,y in Xte])
    return _safe_mae(yte, ypred)

# --- Polynomial Regression with degree tuning ---
def _eval_poly(data, train_df, test_df):
    data.fit_scalers(train_F=train_df, train_G=train_df)
    # prepare raw and scaled
    Xtr, ytr = data.get_xy(train_df, forward=False, scaled=True)
    Xte, _   = data.get_xy(test_df,  forward=False, scaled=True)
    ytrue    = data.get_xy(test_df, forward=False, scaled=False)[1]

    # inner grid search over degree
    pipe = make_pipeline(PolynomialFeatures(), 
                         1.)  # placeholder
    param_grid = {'polynomialfeatures__degree': [3],
                  'linearregression__fit_intercept': [True]}
    # build pipeline with real estimator
    from sklearn.linear_model import LinearRegression
    pipe = make_pipeline(PolynomialFeatures(), LinearRegression())
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)

    ypred = best.predict(Xte)
    # inverse scale
    ypred = data.sc_out_G.inverse_transform(ypred.reshape(-1,1)).ravel()
    return _safe_mae(ytrue, ypred)

# --- KRR–RBF with alpha/gamma tuning ---
def _eval_krr(data, train_df, test_df):
    data.fit_scalers(train_F=train_df, train_G=train_df)
    Xtr, ytr = data.get_xy(train_df, forward=False, scaled=True)
    Xte, _   = data.get_xy(test_df,  forward=False, scaled=True)
    ytrue    = data.get_xy(test_df, forward=False, scaled=False)[1]

    # grid search
    kr = KernelRidge(kernel='rbf')
    param_grid = {'alpha': [1e-4, 1e-3, 1e-2],
                  'gamma': [0.1, 1.0, 10.0]}
    grid = GridSearchCV(kr, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_

    ypred_s = best.predict(Xte)
    ypred   = data.sc_out_G.inverse_transform(ypred_s.reshape(-1,1)).ravel()
    return _safe_mae(ytrue, ypred)

# --- NN-MLP with simple tuning of alpha & hidden layers ---
def _eval_nn(data, train_df, test_df):
    data.fit_scalers(train_F=train_df, train_G=train_df)
    Xtr, ytr = data.get_xy(train_df, forward=False, scaled=True)
    Xte, _   = data.get_xy(test_df,  forward=False, scaled=True)
    ytrue    = data.get_xy(test_df, forward=False, scaled=False)[1]

    mlp = MLPRegressor(max_iter=2000, early_stopping=True)
    param_grid = {
        'hidden_layer_sizes': [(32,32),(64,32),(128,64)],
        'alpha': [1e-6, 1e-5,1e-4,1e-3],
        'learning_rate_init': [1e-2,1e-3, 1e-4]
    }
    grid = GridSearchCV(mlp, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1)
    grid.fit(Xtr, ytr)
    best = grid.best_estimator_
    print("Best params:", grid.best_params_)
    ypred_s = best.predict(Xte)
    ypred   = data.sc_out_G.inverse_transform(ypred_s.reshape(-1,1)).ravel()
    return _safe_mae(ytrue, ypred)

METHODS = {
    "RBF-Brent": _eval_rbf,
    "PolyReg":   _eval_poly,
    "KRR-RBF":   _eval_krr,
    "NN-MLP":    _eval_nn,
}

def sweep_random(data, ratios):
    res = {name:[] for name in METHODS}
    for r in ratios:
        tr, vd, te = data.random_split(train_ratio=r, valid_ratio=(1-r)/2)
        for name, fn in METHODS.items():
            mae = fn(data, tr, te)
            res[name].append(mae)
        print(f"Random split {r:.2f} done")
    return res

def sweep_block(data, xs):
    res = {name:[] for name in METHODS}
    for x in xs:
        tr, vd, te = data.block_split(test_x=x, valid_ratio=0.2)
        for name, fn in METHODS.items():
            mae = fn(data, tr, te)
            res[name].append(mae)
        print(f"Block split x={x} done")
    return res

def plot_sweep(xvals, results, xlabel, title):
    plt.figure(figsize=(8,5))
    for name, maes in results.items():
        plt.plot(xvals, maes, marker='o', label=name)
    plt.xlabel(xlabel); plt.ylabel("MAE (A)")
    plt.title(title); plt.legend(); plt.grid(True)
    plt.tight_layout()

if __name__ == "__main__":
    data = CoilData('Real_data.csv', random_state=42)

    ratios = np.linspace(0.1, 0.7, 7)
    rr = sweep_random(data, ratios)
    plot_sweep(ratios, rr, "Train ratio", "Random‐Split MAE")

    xs = [10,20,30,40,50]
    br = sweep_block(data, xs)
    plot_sweep(xs, br, "Test slice x (mm)", "Block‐Split MAE")

    plt.show()
