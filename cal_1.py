import sympy as sp
import math
# ------------------------------
# 1. 定义符号与已知参数
# ------------------------------
theta = sp.symbols('theta', real=True)  # 要求解的角度（单位：弧度）

# 示例参数（可根据实际情况修改）
xA, yA = 144.019, 63     # 举例：A点坐标 (可根据实际修改)
d      = 8.736            # A到D的距离
yE_target     = 179.185         # 已知E点的纵坐标 (x=0, y=yE)

# ------------------------------
# 2. 构建方程
# ------------------------------
# 2.1 D点坐标：D = A + d*(sinθ, cosθ)
xD = xA + d * sp.sin(theta)
yD = yA + d * sp.cos(theta)

# 2.2 射线DE方向：与CA杆顺时针加60° => 方向角 = theta + 60° (60°转换为弧度：pi/3)
alpha = theta + sp.pi/3

# 2.3 射线方程：E = D + s*(sin(alpha), cos(alpha))
#     交点条件：E的x坐标 = 0
s = - xD / sp.sin(alpha)

# 2.4 E点的y坐标（作为theta的函数）
yE_expr = yD + s * sp.cos(alpha)

# 2.5 构造方程：要求 yE_expr == yE_target
eq = sp.Eq(yE_expr, yE_target)

# ------------------------------
# 3. 数值求解反求θ
# ------------------------------
# 给定初始猜测值，单位为弧度
initial_guess = 2  # 例如0.5弧度

solution = sp.nsolve(eq, theta, initial_guess)
theta_val = float(solution)
theta_deg =  theta_val * 180.0 / sp.pi

print("求得的θ (弧度) =", theta_val)
print("求得的θ (度)   =", 360-float(theta_deg))
x_B = 107.299
r_4 = 35.0  # 杆长
y = sp.Symbol('y', real=True)  # 目标变量
# 设定方程
theta_val = sp.N(sp.rad(float(theta_deg))) 
equation = (xA + r_4 * sp.sin(theta_val) - x_B)**2 + (yA + r_4 * sp.cos(theta_val) - y)**2 - 25**2

# 解方程，求 y
solution = sp.solve(equation, y)
print(solution)

theta = 360-float(theta_deg)

theta_x_de = 90 - (theta - 60)
d_y = yA + d * math.cos(theta/180*math.pi)
f_e = yE_target-d_y

de = f_e / math.sin(theta_x_de/180*math.pi)
x = de - 148.382
print(x)


