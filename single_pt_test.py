# test_poly_inference.py

import numpy as np
import joblib
from data_utils import CoilData
from baselines import PolyInverseRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression

def train_and_save():
    """
    1) 读取标定数据
    2) 训练 Polynomial Regression 模型
    3) 保存模型和 Scaler
    """
    # 1) 读取并切分
    data = CoilData('Real_data.csv', random_state=42)
    train_df, valid_df, test_df = data.random_split(train_ratio=0.7, valid_ratio=0.15)
    
    # 2) fit scalers on train
    data.fit_scalers(train_F=train_df, train_G=train_df)
    
    # 3) 准备数据 (逆向：B,x,y -> I), scaled=True
    X_train_s, y_train_s = data.get_xy(train_df, forward=False, scaled=True)    

    # 4) 构建 Pipeline 并训练
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=3, include_bias=False)),
        ('sc', StandardScaler()),        # 标准化回归输出
        ('lr', LinearRegression())
    ])
    # 先把 y_train_s 转成二维
    y_train_s_2d = y_train_s.reshape(-1,1)
    pipe.fit(X_train_s, y_train_s_2d)

    # 5) 保存 Pipeline 以及输入/输出 Scaler
    joblib.dump(pipe, 'poly_pipeline.joblib')
    joblib.dump(data.sc_in_G, 'sc_in_G.joblib')
    joblib.dump(data.sc_out_G, 'sc_out_G.joblib')
    print("Training complete. Saved: poly_pipeline.joblib, sc_in_G.joblib, sc_out_G.joblib")


def load_and_predict(B_new, x_new, y_new):
    """
    1) 加载模型与 Scaler
    2) 对单个样本(B_new,x_new,y_new)做预测
    """
    # 加载
    pipe    = joblib.load('poly_pipeline.joblib')
    sc_in_G = joblib.load('sc_in_G.joblib')
    sc_out_G= joblib.load('sc_out_G.joblib')

    # 构造单样本并标准化输入
    X_new = np.array([[B_new, x_new, y_new]])
    X_new_s = sc_in_G.transform(X_new)

    # Pipeline 中 poly->scaler->LinearRegression
    y_pred_s_2d = pipe.predict(X_new_s)     # shape (1,1)
    y_pred_s = y_pred_s_2d.ravel()          # shape (1,)

    # 反标准化得到物理电流
    I_pred = sc_out_G.inverse_transform(y_pred_s.reshape(-1,1)).ravel()[0]
    print(f"Input (B, x, y) = ({B_new}, {x_new}, {y_new})")
    print(f"Predicted current I = {I_pred:.4f} A")
    return I_pred


if __name__ == "__main__":
    # 第一次运行：训练并保存
    # train_and_save()

    # 测试单样本预测
    # 替换成你想预测的 B (T), x (mm), y (mm)
    B_new, x_new, y_new = 1.68,30, 20
    load_and_predict(B_new, x_new, y_new)
