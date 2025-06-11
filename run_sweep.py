# run_sweep.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from data_utils import CoilData
from baselines import LookUpRBF,PolyInverseRegressor,KRRInverseRegressor,NNInverseRegressor
from joblib import dump
METHODS = {
    "RBF-Brent":    lambda data, tr, te: _eval_rbf(data, tr, te),
    "PolyReg(3)":   lambda data, tr, te: _eval_poly(data, tr, te),
    "KRR-RBF":      lambda data, tr, te: _eval_krr(data, tr, te),
    "NN-MLP":       lambda data, tr, te: _eval_nn(data, tr, te),
}

def _safe_mae(y_true, y_pred):
    mask = ~np.isnan(y_pred)
    if mask.sum() == 0:
        return np.nan
    return mean_absolute_error(y_true[mask], y_pred[mask])

def _eval_rbf(data, train_df, test_df):
    Xtr, ytr = data.get_xy(train_df, forward=True,  scaled=False)
    Xte, yte = data.get_xy(test_df,  forward=False, scaled=False)
    model    = LookUpRBF(bounds=(0,5)).fit(Xtr, ytr)
    # 并行预测
    ypred = np.array([model._inverse_one(B, x, y) for B, x, y in Xte])
    return _safe_mae(yte, ypred)

def _eval_poly(data, train_df, test_df):
    Xtr, ytr = data.get_xy(train_df, forward=False, scaled=True)
    Xte, yte = data.get_xy(test_df,  forward=False, scaled=True)
    model    = PolyInverseRegressor(degree=3).fit(Xtr, ytr)
    dump(model, 'poly_model_degree3.joblib')
    print("Saved Poly model to poly_model_degree3.joblib")
    ypred_s  = model.predict(Xte)
    ypred    = data.sc_out_G.inverse_transform(ypred_s.reshape(-1,1)).ravel()
    y_true   = data.get_xy(test_df, forward=False, scaled=False)[1]
    return _safe_mae(y_true, ypred)

def _eval_krr(data, train_df, test_df):
    Xtr, ytr = data.get_xy(train_df, forward=False, scaled=True)
    Xte, yte = data.get_xy(test_df,  forward=False, scaled=True)
    model    = KRRInverseRegressor(kernel='rbf', alpha=1e-3, gamma=1).fit(Xtr, ytr)
    ypred_s  = model.predict(Xte)
    ypred    = data.sc_out_G.inverse_transform(ypred_s.reshape(-1,1)).ravel()
    y_true   = data.get_xy(test_df, forward=False, scaled=False)[1]
    return _safe_mae(y_true, ypred)

def _eval_nn(data, train_df, test_df):
    Xtr, ytr = data.get_xy(train_df, forward=False, scaled=True)
    Xte, yte = data.get_xy(test_df,  forward=False, scaled=True)
    model    = NNInverseRegressor().fit(Xtr, ytr)
    ypred_s  = model.predict(Xte)
    ypred    = data.sc_out_G.inverse_transform(ypred_s.reshape(-1,1)).ravel()
    y_true   = data.get_xy(test_df, forward=False, scaled=False)[1]
    return _safe_mae(y_true, ypred)

def sweep_random(data, ratios):
    """随机切分：train_ratio in ratios"""
    results = {m: [] for m in METHODS}
    for r in ratios:
        train_df, valid_df, test_df = data.random_split(train_ratio=r, valid_ratio=(1-r)/2)
        # ！！重新 fit scalers ！！  
        data.fit_scalers(train_F=train_df, train_G=train_df)

        for name, fn in METHODS.items():
            mae = fn(data, train_df, test_df)
            results[name].append(mae)
        print(f"Random split {r:.2f} done")
    return results

def sweep_block(data, xs):
    """Block split：test剖面 x in xs"""
    results = {m: [] for m in METHODS}
    for x in xs:
        train_df, valid_df, test_df = data.block_split(test_x=x, valid_ratio=0.2)
        # ！！重新 fit scalers ！！  
        data.fit_scalers(train_F=train_df, train_G=train_df)

        for name, fn in METHODS.items():
            mae = fn(data, train_df, test_df)
            results[name].append(mae)
        print(f"Block split x={x} done")
    return results

def plot_sweep(xvals, results, xlabel, title):
    plt.figure(figsize=(8,5))
    for name, maes in results.items():
        plt.plot(xvals, maes, marker='o', label=name)
    plt.xlabel(xlabel)
    plt.ylabel("MAE (A)")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

if __name__=="__main__":
    data = CoilData('Real_data.csv', random_state=42)

    # 1) 随机分割扫描
    ratios = np.linspace(0.1, 0.7, 7)
    res_rand = sweep_random(data, ratios)
    plot_sweep(ratios, res_rand, xlabel="Train ratio", title="Random Split Sweep")

    # 2) Block 剖面扫描
    xs = [10,20,30,40,50]
    res_block = sweep_block(data, xs)
    plot_sweep(xs, res_block, xlabel="Test slice x (mm)", title="Block Split Sweep")

    plt.show()
