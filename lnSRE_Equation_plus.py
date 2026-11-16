import torch
import torch.nn as nn
import math

class lnSRE_Equation_plus(nn.Module):
    def __init__(self, params, device='cpu',
                 pgd_steps=300,           # Default: 300
                 pgd_lr_base=0.05,        # Default: 0.05
                 kkt_tol=1e-3,            # KKT early-stopping tolerance
                 max_pgd_steps=500):      # Maximum PGD steps safeguard
        super().__init__()
        self.device = device
        self.pgd_steps = int(pgd_steps)
        self.pgd_lr_base = float(pgd_lr_base)
        self.kkt_tol = float(kkt_tol)
        self.max_pgd_steps = int(max_pgd_steps)

        def to_t(val):
            return torch.tensor(val, device=self.device, dtype=torch.float64)

        self.r = to_t(params['r'])
        self.mu_base = to_t(params['mu_base'])
        self.mu_load = to_t(params['mu_load'])
        self.sigma_0 = to_t(params['sigma_0'])
        self.gamma_mat = to_t(params['gamma_mat'])

        if 'corr_matrix' not in params:
            raise ValueError("[Error] corr_matrix must be provided in params for intertemporal hedging.")
        C = to_t(params['corr_matrix'])

        self.N = self.mu_base.shape[0]
        self.M = self.sigma_0.shape[1]
        self.d = C.shape[0] - self.M

        self.r_vec = torch.full((self.N,), self.r.item(), dtype=torch.float64, device=self.device)
        self.rho_c = C[:self.d, self.d:self.d + self.M]                     # (d, M)

        # Precompute baseline inverse covariance matrix for PGD
        Sigma_0 = torch.matmul(self.sigma_0, self.sigma_0.t())
        self.inv_Sigma_0 = torch.linalg.pinv(Sigma_0)                       # (N, N)

    def forward(self, t, check_P_t, check_Lambda_t, V_t):
        B = V_t.shape[0]
        check_Lambda_t_active = check_Lambda_t[:, :self.d]                  # (B, d)

        # State-dependent excess return
        mu_V = self.mu_base + torch.matmul(V_t, self.mu_load.t())
        mu_tilde = mu_V - self.r_vec

        # State-dependent volatility and covariance
        lin_gamma_V = torch.clamp(1.0 + torch.matmul(V_t, self.gamma_mat.t()), min=1e-6)
        sigma_V = lin_gamma_V.unsqueeze(-1) * self.sigma_0.unsqueeze(0)
        Sigma_V = torch.bmm(sigma_V, sigma_V.transpose(1, 2))

        #  Adjoint vector theta
        Lambda_mapped = torch.matmul(check_Lambda_t_active, self.rho_c)
        cross_term = torch.bmm(sigma_V, Lambda_mapped.unsqueeze(-1)).squeeze(-1)
        theta_V = mu_tilde + cross_term

        # ==========  FISTA solver for constrained infimum ==========
        with torch.no_grad():
            Sigma_ng = Sigma_V.detach()
            theta_ng = theta_V.detach()

            # Warm-start initialization via clamped reciprocal
            inv_lin_ng = 1.0 / lin_gamma_V.detach()
            inv_Sigma_V_ng = inv_lin_ng.unsqueeze(-1) * self.inv_Sigma_0.unsqueeze(0) * inv_lin_ng.unsqueeze(1)

            pi_unc = -torch.bmm(inv_Sigma_V_ng, theta_ng.unsqueeze(-1)).squeeze(-1)
            pi_0 = torch.relu(pi_unc)

            trace_Sigma = torch.diagonal(Sigma_ng, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True)
            lr = self.pgd_lr_base / (trace_Sigma + 1e-8)

            # FISTA momentum initialization
            y = pi_0.clone()
            pi_curr = pi_0.clone()
            t_fista = 1.0

            actual_steps = 0
            for step in range(self.max_pgd_steps):
                grad = 2.0 * (torch.bmm(Sigma_ng, y.unsqueeze(-1)).squeeze(-1) + theta_ng)
                pi_next = torch.relu(y - lr * grad)
                actual_steps += 1

                # Early-stopping check 
                if step >= 10:
                    current_kkt = torch.sum((pi_next * grad) ** 2, dim=-1)
                    if torch.mean(current_kkt) < self.kkt_tol:
                        pi_curr = pi_next
                        break

                # Nesterov momentum update
                t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista)) / 2.0
                momentum = (t_fista - 1.0) / t_next
                y = pi_next + momentum * (pi_next - pi_curr)
                pi_curr = pi_next
                t_fista = t_next

            pi_star = pi_curr

            # Final KKT residual for external monitoring
            final_grad = 2.0 * (torch.bmm(Sigma_ng, pi_star.unsqueeze(-1)).squeeze(-1) + theta_ng)
            self._step_kkt_res = torch.sum((pi_star * final_grad) ** 2, dim=-1)

        # ==========  Symbolic drift evaluation  ==========
        # Inject detached optimizer pi_star back into graph-tracked Sigma_V and theta_V
        pi_Sigma_pi = torch.sum(pi_star * torch.bmm(Sigma_V, pi_star.unsqueeze(-1)).squeeze(-1), dim=-1)
        theta_pi = 2.0 * torch.sum(theta_V * pi_star, dim=-1)
        inf_val = pi_Sigma_pi + theta_pi

        lambda_sq = torch.sum(check_Lambda_t_active ** 2, dim=-1)
        bracket = 2.0 * self.r + inf_val + 0.5 * lambda_sq
        drift = -bracket

        return drift.unsqueeze(-1)