import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
class DataProcessPD():
    def __init__(self, data_path, random_state):
        self.data_path = data_path
        self.random_state = random_state

    def _read_data(self):
        
        df = pd.read_csv(self.data_path)

        # --- 2.1 预处理列名 -------------------------------------------------
        # 把第一列重命名为 Current
        df = df.rename(columns={df.columns[0]: 'I'})

        # 其余列名形如 "(10,0)"，去掉括号
        new_cols = {}
        for c in df.columns[1:]:
            c_clean = c.strip('()').replace(' ', '')   # '10,0'  or '10,0.5'
            new_cols[c] = c_clean
        df = df.rename(columns=new_cols)

        # --- 2.2 宽表 -> 长表 ------------------------------------------------
        long_df = df.melt(id_vars='I', var_name='coord', value_name='B')

        # --- 2.3 拆分 coord = "10,0" -> x, y --------------------------------
        long_df[['x','y']] = long_df['coord'].str.split(',', expand=True).astype(float)
        long_df = long_df.drop(columns='coord')

        # 按需要排序 / 重置索引
        long_df = long_df[['I','x','y','B']].sort_values(['I','x','y']).reset_index(drop=True)

        print(long_df.head())

        return long_df
    
    def _random_split_data(self, df):

        train_df, temp_df = train_test_split(df,
                                     test_size=0.3,       # 30 % 留给 valid+test
                                     random_state=self.random_state,
                                     shuffle=True)

        valid_df, test_df = train_test_split(temp_df,
                                        test_size=0.5,       # 各占一半 → 15 % + 15 %
                                        random_state=self.random_state,
                                        shuffle=True)
        
        return train_df, valid_df, test_df
    
    def _block_split_data(self, df, test_x, valid_size):
        test_df  = df[df['x'] == test_x].reset_index(drop=True)
        pool_df  = df[df['x'] != test_x].reset_index(drop=True)

        train_df, valid_df = train_test_split(
        pool_df,
        test_size   = valid_size,         
        random_state= self.random_state,
        shuffle     = True,
        )

        return train_df, valid_df, test_df

    def _scaler_data(self, train, valid, test, iffw):

        if iffw:
            X_train = train[['I', 'x', 'y']].values
            y_train = train['B'].values
            X_valid = valid[['I', 'x', 'y']].values
            y_valid = valid['B'].values
            X_test  = test [['I', 'x', 'y']].values
            y_test  = test ['B'].values

        else:
            X_train = train[['B', 'x', 'y']].values
            y_train = train['I'].values
            X_valid = valid[['B', 'x', 'y']].values
            y_valid = valid['I'].values
            X_test  = test [['B', 'x', 'y']].values
            y_test  = test ['I'].values

        # scaler = StandardScaler().fit(X_train)
        # X_train_s = scaler.transform(X_train)
        # X_valid_s = scaler.transform(X_valid)
        # X_test_s  = scaler.transform(X_test)
        X_train_s = X_train
        X_valid_s = X_valid
        X_test_s  = X_test

        return X_train_s, y_train, X_valid_s, y_valid, X_test_s, y_test




    
if __name__ == "__main__":
    data_path = 'Real_data.csv'
    pd_process = DataProcessPD(data_path, 42)
    long_df = pd_process._read_data()
    train_df, valid_df, test_df = pd_process._random_split_data(long_df)

    train_df, valid_df, test_df = pd_process._block_split_data(long_df, 40, 0.2)
    # 3.2 结果
    print(len(train_df), len(valid_df), len(test_df))
    # 例如：70 % / 15 % / 15 %
