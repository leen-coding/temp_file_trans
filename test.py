import sympy as sp
import math
# =========== 1. 定义符号与已知参数 ============
theta = sp.Symbol('theta', real=True)  # 要求解的角度(弧度)
xA, yA = 144.019, 63     # 举例：A点坐标 (可根据实际修改)
d      = 8.736            # A到D的距离
yE     = 110          # 已知E点的纵坐标 (x=0, y=yE)

# =========== 2. 构建方程 ============

# 2.1 D点坐标 (在杆CA上，离A距离d)
#     注意：theta=0 时表示CA正好竖直向下，故D相对A为( d*sin(theta), d*cos(theta) )
xD = xA + d*sp.sin(theta)
yD = yA + d*sp.cos(theta)

# 2.2 DE射线与CA成60° => 射线方向角 alpha = theta + 60°
#     (将60°转换为弧度 pi/3)
alpha = theta + sp.pi/sp.Integer(3)  # 60° = pi/3

# 2.3 与 x=0 相交 => 解出参数 s
#     射线方程: X = xD + s*sin(alpha),  Y = yD + s*cos(alpha)
#     要 x=0 => s = - xD / sin(alpha)
s = - xD / sp.sin(alpha)

# 2.4 yE_expr = 射线在此 s 下的纵坐标
yE_expr = yD + s*sp.cos(alpha)

# 2.5 构造方程：yE_expr == yE
eq = sp.Eq(yE_expr, yE)

# =========== 3. 使用 nsolve 数值求解 ============
# 需要给一个初始猜测值 guess
guess =2.2 # 1.0 rad ~ 57.3°
solution = None

try:
    solution = sp.nsolve(eq, theta, guess)
    # 若成功求解，solution 即为弧度制的数值
    print("数值解(弧度) =", float(solution))
    print(4.03*180/math.pi)

    print("数值解(度)   =", float(solution) * 180.0 /math.pi) 
except Exception as e:
    print("求解失败:", e)
