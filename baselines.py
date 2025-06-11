
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor

class LookUpRBF:
    """
    正向插值  +  Brent 反求 I
    --------------------------------------------------
    fit:   训练正向 RBF  (I,x,y) -> B
    predict: 给 (B*,x,y) 返回 I
    """
    def __init__(self,
                 kernel: str = 'thin_plate_spline',
                 bounds: tuple = (-10., 10.)):
        self.kernel = kernel
        self.bounds = bounds
        self.rbf    = None

    # ----------------- 训练正向插值器 ------------------------------
    def fit(self, X_fwd, y_fwd):
        """
        X_fwd shape = (n,3)  columns = [I,x,y]
        y_fwd shape = (n,)
        """
        self.rbf = RBFInterpolator(X_fwd, y_fwd,
                                   kernel=self.kernel)
        return self

    # ----------------- 单点求 I (Brent) ---------------------------
    def _inverse_one(self, B_star, x, y):
        f = lambda I: self.rbf([[I, x, y]])[0] - B_star
        try:
            return brentq(f, *self.bounds)
        except ValueError:
            return np.nan

    # ----------------- 批量预测 -----------------------------------
    def predict(self, X_inv):
        """
        X_inv shape = (m,3) columns = [B*,x,y]
        returns np.ndarray (m,) 电流预测
        """
        Bv, xv, yv = X_inv.T
        preds = [self._inverse_one(B,x,y) for B,x,y in zip(Bv,xv,yv)]
        return np.asarray(preds)
    

class PolyInverseRegressor:
    """
    使用多项式特征 + 线性回归，拟合 (B, x, y) -> I
    """
    def __init__(self, degree: int = 3):
        """
        degree: 多项式最高次数，常用 2~5
        """
        self.degree = degree
        self.model = make_pipeline(
            PolynomialFeatures(degree, include_bias=False),
            LinearRegression()
        )

    def fit(self, X, y):
        """
        X shape=(n,3) 列顺序 = [B, x, y]
        y shape=(n,)
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        X shape=(m,3)
        return shape=(m,)
        """
        return self.model.predict(X)

# baseline_krr.py
from sklearn.kernel_ridge import KernelRidge

class KRRInverseRegressor:
    """
    使用核岭回归拟合 (B, x, y) -> I，kernel 可选 'rbf'、'laplacian'、'polynomial' 等
    """
    def __init__(self, kernel: str = 'rbf', alpha: float = 1e-2, gamma: float = None, degree: int = 3):
        """
        kernel: 核类型
        alpha : 正则化强度
        gamma : RBF 核宽度 (若 None sklearn 会自动选择)
        degree: 仅在 kernel='polynomial' 时生效
        """
        self.model = KernelRidge(kernel=kernel,
                                 alpha=alpha,
                                 gamma=gamma,
                                 degree=degree)

    def fit(self, X, y):
        """
        X shape=(n,3): [B, x, y]
        y shape=(n,)
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        X shape=(m,3)
        """
        return self.model.predict(X)
    
class NNInverseRegressor:
    """
    普通神经网络 (MLP) 端到端逆向预测 (B, x, y) -> I
    """
    def __init__(self,
                 hidden_layer_sizes=(64,64,32),
                 activation='relu',
                 solver='adam',
                 alpha=1e-4,
                 learning_rate='adaptive',
                 max_iter=2000,
                 random_state=42):
        """
        hidden_layer_sizes: 隐藏层配置
        activation        : 激活函数
        solver            : 优化器
        alpha             : L2 正则化
        learning_rate     : 学习率策略
        max_iter          : 最大迭代次数
        """
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate=learning_rate,
            max_iter=max_iter,
            early_stopping=True,
            n_iter_no_change=50,
            random_state=random_state,
            verbose=False
        )

    def fit(self, X, y):
        """
        X shape = (n_samples, 3)  —— [B, x, y]
        y shape = (n_samples,)
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        X shape = (m_samples, 3)
        returns shape = (m_samples,)
        """
        return self.model.predict(X)