import torch
import math
import time

class ConstrainedLnSRE_LSQMC_P_Minus:
    """
    Truncated LSQMC initial value estimator for the log-transformed P- equation.
    Solves a regularized surrogate BSDE with norm-clamped quadratic growth via
    backward polynomial ridge regression, providing a fast scalar anchor for initial_p0.
    """
    def __init__(self, dt, model_params, device="cpu",
                 degree=2, rcond=None, pgd_steps=35, pgd_lr_base=0.08):
        self.dt = float(dt)
        self.device = torch.device(device)
        self.degree = int(degree)
        self.rcond = rcond

        self.pgd_steps = int(pgd_steps)
        self.pgd_lr = float(pgd_lr_base)

        self.r = torch.tensor(model_params["r"], dtype=torch.float64, device=self.device)
        self.mu_base = torch.tensor(model_params["mu_base"], dtype=torch.float64, device=self.device)
        self.mu_load = torch.tensor(model_params["mu_load"], dtype=torch.float64, device=self.device)
        self.sigma_0 = torch.tensor(model_params["sigma_0"], dtype=torch.float64, device=self.device)
        self.gamma_mat = torch.tensor(model_params["gamma_mat"], dtype=torch.float64, device=self.device)
        self.corr = torch.tensor(model_params["corr_matrix"], dtype=torch.float64, device=self.device)

        self.sigma_V = torch.tensor(model_params["sigma_V"], dtype=torch.float64, device=self.device)
        self.inv_sigma_V = torch.linalg.inv(self.sigma_V)

        self.N = self.mu_base.shape[0]
        self.d = self.mu_load.shape[1]
        self.M = self.sigma_0.shape[1]
        self.r_vec = self.r.repeat(self.N)
        self.rho_c = self.corr[:self.d, self.d:self.d + self.M]
        self._triu_idx = torch.triu_indices(self.d, self.d, device=self.device)

    def _build_basis_std(self, V, degree):
        B = V.shape[0]
        basis_list = [V]
        if degree >= 2:
            quad = V[:, self._triu_idx[0]] * V[:, self._triu_idx[1]]
            basis_list.append(quad)
        X_raw = torch.cat(basis_list, dim=1)
        X_mean = X_raw.mean(dim=0, keepdim=True)
        X_std = X_raw.std(dim=0, keepdim=True).clamp(min=1e-8)
        X_norm = (X_raw - X_mean) / X_std
        ones = torch.ones(B, 1, dtype=V.dtype, device=V.device)
        X_final = torch.cat([ones, X_norm], dim=1)
        return X_final, X_mean.squeeze(0), X_std.squeeze(0)

    def _ridge_regression(self, X, Y, alpha=1e-3):
        B, K = X.shape
        XTX = X.t() @ X
        XTY = X.t() @ Y
        penalty = torch.ones(K, dtype=X.dtype, device=X.device)
        penalty[0] = 0.0
        I_pen = torch.diag(penalty)
        beta = torch.linalg.solve(XTX + alpha * B * I_pen, XTY)
        return beta

    def _lambda_from_surface(self, V, beta_P, X_mean, X_std):
        d = self.d
        w = beta_P.reshape(-1)[1:] / X_std
        grad = w[:d].unsqueeze(0).expand(V.shape[0], -1).clone()
        if self.degree >= 2 and w.numel() > d:
            wq = w[d:]
            ii = self._triu_idx[0]
            jj = self._triu_idx[1]
            grad.index_add_(1, ii, wq.unsqueeze(0) * V[:, jj])
            grad.index_add_(1, jj, wq.unsqueeze(0) * V[:, ii])
        Lambda_V = grad @ self.sigma_V
        return Lambda_V, grad

    def _solve_constrained_pi(self, Sigma, theta):
        """
        Solve the constrained minimization for P-: min_{pi >= 0} (pi^T Sigma pi - 2 pi^T theta).
        Unconstrained stationary point is pi = Sigma^{-1} theta.
        """
        try:
            pi_unc = torch.linalg.solve(Sigma, theta.unsqueeze(-1)).squeeze(-1)
        except Exception:
            pi_unc = torch.linalg.lstsq(Sigma, theta.unsqueeze(-1)).solution.squeeze(-1)
        pi = torch.relu(pi_unc)

        trace = torch.diagonal(Sigma, dim1=-2, dim2=-1).sum(-1, keepdim=True)
        lr = self.pgd_lr / (trace + 1e-8)
        y = pi.clone()
        pi_curr = pi.clone()
        t_fista = 1.0

        for _ in range(self.pgd_steps):
            # Gradient with -theta for P-
            grad = 2.0 * (torch.bmm(Sigma, y.unsqueeze(-1)).squeeze(-1) - theta)
            pi_next = torch.relu(y - lr * grad)
            t_next = (1.0 + math.sqrt(1.0 + 4.0 * t_fista * t_fista)) / 2.0
            momentum = (t_fista - 1.0) / t_next
            y = pi_next + momentum * (pi_next - pi_curr)
            pi_curr = pi_next
            t_fista = t_next

        return pi_curr

    def _compute_generator(self, V, Lambda_active, pi=None):
        mu_V = self.mu_base + V @ self.mu_load.t()
        mu_tilde = mu_V - self.r_vec
        lin_gamma = torch.clamp(1.0 + V @ self.gamma_mat.t(), min=1e-6)
        sigma = lin_gamma.unsqueeze(-1) * self.sigma_0.unsqueeze(0)
        Sigma = torch.bmm(sigma, sigma.transpose(1, 2))

        Lambda_map = Lambda_active @ self.rho_c
        theta = mu_tilde + torch.bmm(sigma, Lambda_map.unsqueeze(-1)).squeeze(-1)

        if pi is None:
            pi = self._solve_constrained_pi(Sigma, theta)

        Sigma_pi = torch.bmm(Sigma, pi.unsqueeze(-1)).squeeze(-1)

        # Linear term has negative sign for P-
        H = torch.sum(pi * Sigma_pi, dim=-1) - 2.0 * torch.sum(theta * pi, dim=-1)

        lam_norm = torch.norm(Lambda_active, p=2, dim=-1)
        lam_sq = torch.clamp(lam_norm, max=6.0) ** 2
        f = 2.0 * self.r + H + 0.5 * lam_sq

        return f, Sigma, theta, pi

    @torch.no_grad()
    def estimate_p0(self, V_traj, dW_traj, Xi_traj=None, verbose=False):
        start = time.time()
        n_inc, n_paths, _ = dW_traj.shape
        V = V_traj.to(self.device, dtype=torch.float64)

        if Xi_traj is not None:
            Xi = Xi_traj.to(self.device, dtype=torch.float64)
        else:
            Xi = torch.zeros(n_inc, n_paths, self.d, dtype=torch.float64, device=self.device)

        P_next = torch.zeros(n_paths, 1, dtype=torch.float64, device=self.device)
        Lambda_active = torch.zeros(n_paths, self.d, dtype=torch.float64, device=self.device)

        if verbose:
            print(f">>> Constrained LSQMC (P-) Start | paths={n_paths}, steps={n_inc}, pgd_steps={self.pgd_steps}")

        for k in range(n_inc - 1, -1, -1):
            V_k = V[k]
            Xi_k = Xi[k]
            X_quad, mu_x, sd_x = self._build_basis_std(V_k, degree=self.degree)

            # Explicit Euler backward step using Lambda_active from step k+1
            f, _, _, pi_star = self._compute_generator(V_k, Lambda_active)
            grad_P = Lambda_active @ self.inv_sigma_V
            skorokhod_comp = torch.sum(grad_P * Xi_k, dim=-1)

            if k == 0:
                P_k = (P_next[:, 0] + f * self.dt - skorokhod_comp).mean()
                P_k = P_k * torch.ones(n_paths, dtype=torch.float64, device=self.device)
            else:
                Y_P = P_next[:, 0] + f * self.dt - skorokhod_comp
                beta_P = self._ridge_regression(X_quad, Y_P.unsqueeze(1), alpha=1e-4)
                P_k = (X_quad @ beta_P).squeeze(-1)

                # Update analytical Lambda from fitted polynomial surface for next backward step
                Lambda_active, _ = self._lambda_from_surface(V_k, beta_P, mu_x, sd_x)

            P_next = P_k.reshape(-1, 1)

            if verbose and (k % 50 == 0 or k == n_inc - 1 or k == 0):
                print(
                    f"k={k:4d} | mean check_P- = {P_k.mean().item():10.6f} "
                    f"| rms Lambda = {Lambda_active.pow(2).mean().sqrt().item():8.4f} "
                    f"| mean H-part f = {f.mean().item():10.6f}"
                )

        check_p0 = P_k.mean().item()
        p0 = math.exp(check_p0)

        if verbose:
            print(f">>> LSQMC (P-) Done in {time.time() - start:.2f}s")
            print(f">>> check_P-0 = {check_p0:.6f}, P-0 = {p0:.6f}")

        return check_p0, p0