# run_bench.py
import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from data_utils import CoilData
from baselines import LookUpRBF,PolyInverseRegressor,KRRInverseRegressor,NNInverseRegressor

import utils



def evaluate_model(name, model, X_test, y_test, scaler_out=None):
    """统一评测并画图"""
    tic = time.perf_counter()
    y_pred = model.predict(X_test)
    t_ms = (time.perf_counter() - tic) / len(y_test) * 1e3

    # 如果给了 scaler_out，就反标准化
    if scaler_out is not None:
        y_pred = scaler_out.inverse_transform(y_pred.reshape(-1, 1)).ravel()

    mask = ~np.isnan(y_pred)
    mae = mean_absolute_error(y_test[mask], y_pred[mask])
    r2 = r2_score     (y_test[mask], y_pred[mask])


    print(f"[{name}] MAE={mae:.4f} A,  R²={r2:.4f},  {t_ms:.2f} ms/sample")
    utils.plot_pred_vs_true(
        y_true=y_test[mask],
        y_pred=y_pred[mask],
        title=f"{name} ({args.split.upper()} split)"
    )


if __name__ == "__main__":
    # ---------------- parse args ----------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['block', 'random'], default='block')
    args = parser.parse_args()

    # --------------- prepare data ---------------
    data = CoilData('Real_data.csv', random_state=42)
    if args.split == 'block':
        train_df, valid_df, test_df = data.block_split(test_x=30., valid_ratio=0.2)
    else:
        train_df, valid_df, test_df = data.random_split(train_ratio=0.7, valid_ratio=0.15)

    # fit scalers on train
    data.fit_scalers(train_F=train_df, train_G=train_df)

    X_train_f, y_train_f = data.get_xy(train_df, forward=True,  scaled=False)
    X_train_f_s, y_train_f_s = data.get_xy(train_df, forward=True,  scaled=True)
    X_test_inv, y_test_inv = data.get_xy(test_df, forward=False, scaled=False)
    X_train_inv, y_train_inv = data.get_xy(train_df, forward=False, scaled=False)
    X_train_inv_s, y_train_inv_s = data.get_xy(train_df, forward=False, scaled=True)
    X_test_inv_s,  y_test_inv_s  = data.get_xy(test_df,  forward=False, scaled=True)

    # ================= Baseline ①: RBF + Brent =================
 
    rbf_model = LookUpRBF(bounds=(0., 5.)).fit(X_train_f, y_train_f)

    
    evaluate_model("RBF-Brent", rbf_model, X_test_inv, y_test_inv)

    # ================ Baseline ②a: Poly =======================
   
    poly = PolyInverseRegressor(degree=3).fit(X_train_inv_s, y_train_inv_s)
    evaluate_model("PolyReg(deg=3)", poly, X_test_inv_s, y_test_inv, scaler_out=data.sc_out_G)

    # ================ Baseline ②b: KRR ========================
    # 缩放特征后训练


    krr = KRRInverseRegressor(kernel='rbf', alpha=1e-3, gamma=1)
    krr.fit(X_train_inv_s, y_train_inv_s)

    evaluate_model(
        "KRR-RBF",
        krr,
        X_test_inv_s,
        y_test_inv,
        scaler_out=data.sc_out_G
    )

# ================ Baseline ②c: NN ========================
# NN 同样在标准化后的空间训练，再反标准化输出
X_train_inv_s, y_train_inv_s = data.get_xy(train_df, forward=False, scaled=True)
X_test_inv_s,  y_test_inv_s  = data.get_xy(test_df,  forward=False, scaled=True)

nn = NNInverseRegressor(
    hidden_layer_sizes=(64,64,32),
    activation='relu',
    alpha=1e-4,
    max_iter=2000
).fit(X_train_inv_s, y_train_inv_s)

evaluate_model(
    "NN-MLP",
    nn,
    X_test_inv_s,
    y_test_inv,
    scaler_out=data.sc_out_G
)



# import torch
# from pinn_forward import ForwardPINN

# # 设备设置
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # 1. 准备正向训练数据 (scaled)
# Xf_s, yf_s = data.get_xy(train_df, forward=True, scaled=True)
# I_tr = torch.tensor(Xf_s[:, 0], requires_grad=True,  dtype=torch.float32, device=device)
# x_tr = torch.tensor(Xf_s[:, 1], requires_grad=True,  dtype=torch.float32, device=device)
# y_tr = torch.tensor(Xf_s[:, 2], requires_grad=True,  dtype=torch.float32, device=device)
# B_tr = torch.tensor(yf_s,       requires_grad=True, dtype=torch.float32, device=device)

# # 2. 初始化 PINN，并训练
# pinn = ForwardPINN(layers=[3,64,64,64,1]).to(device)
# # 假设我们有一个无限循环的数据迭代器 loader，再简单用全量一次 batch:
# def data_loader():
#     while True:
#         yield I_tr, x_tr, y_tr, B_tr

# loader = data_loader()
# pinn.train_pinn(loader,
#                 epochs=2000,
#                 lr=1e-5,
#                 weight_phys=1e-5,
#                 use_lbfgs=True)

# # 3. 在测试集上前向预测 B
# X_test_f_s, y_test_f_s = data.get_xy(test_df, forward=True, scaled=True)
# I_ts = torch.tensor(X_test_f_s[:, 0], requires_grad=True, dtype=torch.float32, device=device)
# x_ts = torch.tensor(X_test_f_s[:, 1], requires_grad=True, dtype=torch.float32, device=device)
# y_ts = torch.tensor(X_test_f_s[:, 2], requires_grad=True, dtype=torch.float32, device=device)

# # 用 compute_B 计算 B_pred_s (scaled)

# B_pred_s = pinn.compute_B(I_ts, x_ts, y_ts).cpu().numpy()

# # 4. 反标准化回物理单位
# B_pred = data.sc_out_F.inverse_transform(B_pred_s.reshape(-1,1)).ravel()
# import matplotlib.pyplot as plt
# plt.plot(B_pred)
# plt.show()
# # 5. 可视化对比
# utils.plot_pred_vs_true(
#     y_true=test_df['B'].values,
#     y_pred=B_pred,
#     title=f"PINN Forward ({args.split})"
# )
