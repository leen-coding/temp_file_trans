import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.linalg import expm

def normalize(vector):
    """归一化向量"""
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("零向量没有方向，请输入有效的旋转轴向量。")
    return vector / norm

def skew_symmetric_matrix(v):
    """生成一个向量的斜对称矩阵（用于指数映射）"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def rotation_matrix_from_axis_angle(n, theta):
    """
    使用指数映射从旋转轴 n 和角度 theta 生成旋转矩阵
    n: 旋转轴（单位向量）
    theta: 旋转角度（弧度）
    """
    n = normalize(n)
    skew_matrix = skew_symmetric_matrix(n)
    R = expm(skew_matrix * theta)
    return R

def generate_rotating_field(n, omega, t, B_initial):
    """
    生成旋转磁场 B_t，使用旋转轴 n 和角速度 omega 通过指数映射更新旋转。
    n: 旋转轴
    omega: 角速度（单位：弧度/秒）
    t: 当前时间（秒）
    B_initial: 初始磁场向量
    """
    theta = omega * t  # 根据时间计算旋转角度
    R = rotation_matrix_from_axis_angle(n, theta)  # 生成旋转矩阵
    B_t = R @ B_initial  # 更新磁场方向
    return B_t

# 设置旋转轴、角速度和初始磁场
n = np.array([0, 0, 1])  # 旋转轴（例如 Z 轴）
omega = 2 * np.pi  # 1 Hz 的角速度（2π 弧度/秒）
B_initial = np.array([1, 0, 0])  # 初始磁场方向（沿 X 轴）

# 初始化图形
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-1.5, 1.5])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Rotating Magnetic Field')

# 绘制旋转轴
axis_length = 1.5
ax.plot([0, n[0]*axis_length], [0, n[1]*axis_length], [0, n[2]*axis_length],
        color='gray', linestyle='dashed', linewidth=1, label='Rotation Axis (n)')

# 初始化磁场向量（Line3D 对象）
B_t_initial = generate_rotating_field(n, omega, 0, B_initial)
line, = ax.plot([0, B_t_initial[0]], [0, B_t_initial[1]], [0, B_t_initial[2]],
                color='r', linewidth=3, label='Magnetic Field B(t)')

# 添加图例
ax.legend()

# 更新动画帧
def update(frame):
    t = frame / 20  # 调整时间步长以控制旋转速度
    B_t = generate_rotating_field(n, omega, t, B_initial)
    # 更新 Line3D 数据
    line.set_data([0, B_t[0]], [0, B_t[1]])
    line.set_3d_properties([0, B_t[2]])
    return line,

# 创建动画
ani = FuncAnimation(fig, update, frames=200, interval=50, blit=True)

plt.show()
