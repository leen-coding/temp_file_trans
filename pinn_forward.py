# pinn_forward.py

import torch
import torch.nn as nn
import torch.autograd as autograd

class ForwardPINN(nn.Module):
    def __init__(self, layers=[3,64,64,64,1], mu0=4e-7*torch.pi):
        super().__init__()
        seq = []
        for i in range(len(layers)-2):
            seq += [nn.Linear(layers[i], layers[i+1]), nn.Tanh()]
        seq.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*seq)
        self.mu0 = mu0

    def forward(self, I, x, y):
        inp = torch.stack([I, x, y], dim=1)
        return self.net(inp).squeeze(-1)

    def compute_B(self, I, x, y):
        # 确保 x,y 需要梯度
        xg = x.clone().detach().requires_grad_(True)
        yg = y.clone().detach().requires_grad_(True)
        Ig = I.clone().detach()

        A = self.forward(Ig, xg, yg)
        # 一阶导，并保留计算图（create_graph=True）
        grads = autograd.grad(
            A, [xg, yg],
            grad_outputs=torch.ones_like(A),
            create_graph=True,  # <-- 这句保留图
            retain_graph=True
        )
        dA_dx, dA_dy = grads
        r = torch.sqrt(xg**2 + yg**2) + 1e-6
        B = (dA_dx * xg + dA_dy * yg) / r
        return B.detach()  # 推理时 detach 回 CPU

    def physics_residual(self, I, x, y):
        xg = x.clone().detach().requires_grad_(True)
        yg = y.clone().detach().requires_grad_(True)
        Ig = I.clone().detach().requires_grad_(True)

        A = self.forward(Ig, xg, yg)
        gradA = autograd.grad(
            A, [xg, yg],
            grad_outputs=torch.ones_like(A),
            create_graph=True
        )
        dA_dx, dA_dy = gradA
        d2A_dx2 = autograd.grad(
            dA_dx, xg,
            grad_outputs=torch.ones_like(dA_dx),
            create_graph=True
        )[0]
        d2A_dy2 = autograd.grad(
            dA_dy, yg,
            grad_outputs=torch.ones_like(dA_dy),
            create_graph=True
        )[0]
        lapA = d2A_dx2 + d2A_dy2
        R = lapA + self.mu0 * Ig
        return R

    def loss(self, I, x, y, B_true, weight_phys=1e-3):
        B_pred = self.compute_B(I, x, y)
        mse_data = nn.functional.mse_loss(B_pred, B_true)
        R = self.physics_residual(I, x, y)
        mse_phys = torch.mean(R**2)
        return mse_data + weight_phys * mse_phys, mse_data, mse_phys

    def train_pinn(self, data_loader, epochs=2000,
                   lr=1e-3, weight_phys=1e-3,
                   use_lbfgs=False):
        # choose optimizer
        if use_lbfgs:
            optimizer = torch.optim.LBFGS(self.parameters(), lr=1.0)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        batches = data_loader

        # phase 1: data-only
        for epoch in range(epochs // 2):
            I_b, x_b, y_b, B_b = next(batches)

            if use_lbfgs:
                # define closure capturing this batch
                def closure():
                    optimizer.zero_grad()
                    loss_data = nn.functional.mse_loss(self.compute_B(I_b, x_b, y_b), B_b)
                    loss_data.backward()
                    return loss_data

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss_data = nn.functional.mse_loss(self.compute_B(I_b, x_b, y_b), B_b)
                loss_data.backward()
                optimizer.step()

        # phase 2: data + physics
        for epoch in range(epochs // 2, epochs):
            I_b, x_b, y_b, B_b = next(batches)

            if use_lbfgs:
                def closure():
                    optimizer.zero_grad()
                    loss_all, _, _ = self.loss(I_b, x_b, y_b, B_b, weight_phys)
                    loss_all.backward()
                    return loss_all

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                loss_all, _, _ = self.loss(I_b, x_b, y_b, B_b, weight_phys)
                loss_all.backward()
                optimizer.step()

        return self
