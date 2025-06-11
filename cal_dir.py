import numpy as np

def compute_finger_directions(theta: float, phi0: float = 0.0):
    """
    计算三根手指(指尖朝内)在末端坐标系下的方向向量, 三指均匀分布(相差120°)绕末端z轴.

    参数:
    --------
    theta : float
        指尖与末端 z 轴的夹角(弧度制). 0表示与z轴平行, 值越大表示越倾斜.
    phi0 : float, optional
        三指在水平方向的基准偏角(弧度). 缺省为0, 表示第一指相对x轴负方向.

    返回:
    --------
    directions : list of np.ndarray
        包含三个长度为3的numpy向量, 对应三指方向(单位向量).

    示例:
    --------
    >>> dirs = compute_finger_directions(theta=np.radians(60))
    >>> for i, d in enumerate(dirs):
    ...     print(f"Finger {i+1} direction: {d}")
    """
    directions = []
    # 每根手指水平方向相差120度(2*pi/3)
    for i in range(3):
        phi_i = phi0 + 2.0 * np.pi * i / 3.0
        # 如果希望指尖“朝内”, 则在水平分量上加负号:
        #  v_inward = [-sin(theta)*cos(phi), -sin(theta)*sin(phi), cos(theta)]
        x = -np.sin(theta) * np.cos(phi_i)
        y = -np.sin(theta) * np.sin(phi_i)
        z =  np.cos(theta)
        directions.append(np.array([x, y, z], dtype=float))

    return directions

if __name__ == "__main__":
    # 测试示例: 让theta=60度
    theta_deg = 60.0
    theta_rad = np.radians(theta_deg)
    finger_dirs = compute_finger_directions(theta=theta_rad, phi0=0.0)

    print(f"Given theta = {theta_deg} degrees ({theta_rad:.3f} rad), the finger directions are:")
    for idx, vec in enumerate(finger_dirs, start=1):
        print(f"  Finger {idx}: {vec}")
