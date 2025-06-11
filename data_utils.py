# data_utils.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class CoilData:
    """
    负责：读取 csv ➜ 宽表转长表 ➜ 数据切分 ➜ 缩放
    """
    def __init__(self, csv_path: str, random_state: int = 42):
        self.csv_path      = csv_path
        self.random_state  = random_state
        self.long_df: pd.DataFrame | None = None         # (I,x,y,B)
        # 四把 scaler (in/out × 正/逆)
        self.sc_in_F  = StandardScaler()
        self.sc_out_F = StandardScaler()
        self.sc_in_G  = StandardScaler()
        self.sc_out_G = StandardScaler()

    # ---------- 0. 读 CSV 并转成长表 ---------------------------------
    def load_long(self) -> pd.DataFrame:
        if self.long_df is not None:
            return self.long_df

        df = pd.read_csv(self.csv_path)
       
        df = df.rename(columns={df.columns[0]: 'I'})
       
        new_cols = {c: c.strip('()').replace(' ', '') for c in df.columns[1:]}
        df = df.rename(columns=new_cols)
        long_df = df.melt(id_vars='I', var_name='coord', value_name='B')
        long_df[['x','y']] = long_df['coord'].str.split(',', expand=True).astype(float)
        long_df = long_df[['I','x','y','B']].sort_values(['I','x','y']).reset_index(drop=True)

        self.long_df = long_df
        return long_df

    # ---------- 1. block split：留整块 x 剖面作 test -------------------
    def block_split(self,
                    test_x: List[float] | float,
                    valid_ratio: float = .2
                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        test_x: 单个 float 或列表，表示哪些 x-剖面留做测试
        """
        if isinstance(test_x, float) or isinstance(test_x, int):
            test_x = [float(test_x)]

        df      = self.load_long()
        test_df = df[df['x'].isin(test_x)].reset_index(drop=True)
        pool_df = df[~df['x'].isin(test_x)].reset_index(drop=True)

        train_df, valid_df = train_test_split(pool_df,
                                              test_size   = valid_ratio,
                                              random_state= self.random_state,
                                              shuffle     = True)
        return train_df, valid_df, test_df

    # ---------- 2. 训练四把 scaler (正向 F, 逆向 G)---------------------
    def fit_scalers(self, train_F: pd.DataFrame, train_G: pd.DataFrame):
        # 只用 ndarray，不带列名
        self.sc_in_F .fit(train_F[['I','x','y']].values)
        self.sc_out_F.fit(train_F[['B']].values.reshape(-1, 1))
        self.sc_in_G .fit(train_G[['B','x','y']].values)
        self.sc_out_G.fit(train_G[['I']].values.reshape(-1, 1))


    # ---------- 3. 取数据并做 transform（自动挑正/逆格式） -------------

    def get_xy(self,
                df: pd.DataFrame,
                forward: bool = True,
                scaled:  bool = True):
        """forward=True => (I,x,y)->B ; forward=False => (B,x,y)->I"""
        if forward:
            X_df, y_series = df[['I','x','y']], df['B']
            scaler_X, scaler_y = self.sc_in_F, self.sc_out_F
        else:
            X_df, y_series = df[['B','x','y']], df['I']
            scaler_X, scaler_y = self.sc_in_G, self.sc_out_G

        if scaled:

            X = scaler_X.transform(X_df.values)
            y = scaler_y.transform(y_series.to_numpy().reshape(-1,1)).ravel()
        else:
            # 不缩放时，保持 DataFrame 和 Series
            X = X_df.values   # 直接取 numpy array
            y = y_series.values

        return X, y


    
    def random_split(self,
                     train_ratio: float = .7,
                     valid_ratio: float = .15
                     ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        随机打乱后按比例切 train/valid/test
        train_ratio + valid_ratio 应 < 1.0
        """
        df = self.load_long()

        train_df, temp_df = train_test_split(
            df,
            test_size   = 1.0 - train_ratio,
            random_state= self.random_state,
            shuffle     = True,
        )
        valid_size = valid_ratio / (1.0 - train_ratio)
        valid_df,  test_df = train_test_split(
            temp_df,
            test_size   = 1.0 - valid_size,
            random_state= self.random_state,
            shuffle     = True,
        )
        return train_df.reset_index(drop=True), \
               valid_df.reset_index(drop=True), \
               test_df .reset_index(drop=True)
