from read_data import DataProcessPD
from scipy.interpolate import RBFInterpolator
from scipy.optimize import brentq
import numpy as np
class ComparisonMethods():
    def __init__(self):
        data_path = 'Real_data.csv'
        pd_process = DataProcessPD(data_path, 42)
        long_df = pd_process._read_data()
        train_df, valid_df, test_df = pd_process._block_split_data(long_df, 40, 0.2)
        self.X_train_s, self.y_train, self.X_valid_s, self.y_valid, self.X_test_s, self.y_test = pd_process._scaler_data(train_df, valid_df, test_df, True)
        self.X_train_s_inv, self.y_train_inv, self.X_valid_s_inv, self.y_valid_inv, self.X_test_s_inv, self.y_test_inv = pd_process._scaler_data(train_df, valid_df, test_df, False)




    def _look_table_method(self):
        self.rbf   = RBFInterpolator(self.X_train_s, self.y_train, kernel='thin_plate_spline')

        # 假设 X_test_s_inv 是一个 numpy 数组 (n_samples, 3)
        # I_pred_baseline = [self.inverse_one(self.X_test_s_inv[i, 0], self.X_test_s_inv[i, 1], self.X_test_s_inv[i, 2]) for i in range(self.X_test_s_inv.shape[0])]
        I_pred_baseline = []

        # 遍历 X_test_s_inv 的每一行
        for i in range(self.X_test_s_inv.shape[0]):
            # 获取当前行的 B, x, y 值
            B = self.X_test_s_inv[i, 0]
            x = self.X_test_s_inv[i, 1]
            y = self.X_test_s_inv[i, 2]

            # 调用 inverse_one 函数
            I_sol = self.inverse_one(B, x, y)

            # 将结果添加到 I_pred_baseline 列表中
            I_pred_baseline.append(I_sol)

        baseline_mae = np.nanmean(np.abs(np.array(I_pred_baseline) - self.y_test_inv))
        print("Baseline  MAE(A):", baseline_mae)

    def inverse_one(self, B_star, x, y, I_bounds=(-10, 10)):
        func = lambda I: self.rbf([[I, x, y]])[0] - B_star
        try:
            I_sol = brentq(func, *I_bounds)
        except ValueError:          # brentq 找不到根
            I_sol = np.nan
        return I_sol
    



if __name__ == "__main__":
    cm = ComparisonMethods()
    cm._look_table_method()