import torch
import numpy as np
import pandas as pd
import time
import json
import os
import matplotlib.pyplot as plt

class StateDependentAssetModel_GPU:
    """
    Multi-asset dynamics GPU simulator with reflected factor process.
    """
    def __init__(self,
                 d_dim, N_dim, M_dim,
                 alpha, beta, sigma_V, radius,
                 mu_base, mu_load, sigma_0, gamma_mat, corr_matrix,
                 dt, T, num_paths, warmup_steps, device=None):

        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))

        self.d = int(d_dim)
        self.N = int(N_dim)
        self.M = int(M_dim)
        self.M_noise = self.d + self.M
        self.radius = float(radius)
        self.dt = float(dt)
        self.T = float(T)
        self.num_paths = int(num_paths)
        self.warmup_steps = int(warmup_steps)
        self.n_steps = int(round(self.T / self.dt))
        self.time_grid = torch.linspace(0, self.T, self.n_steps + 1, device=self.device)

        # 1. Factor process V tensors
        self.alpha = torch.tensor(alpha, dtype=torch.float32, device=self.device)
        self.beta = torch.tensor(beta, dtype=torch.float32, device=self.device)
        self.sigma_V = torch.tensor(sigma_V, dtype=torch.float32, device=self.device)

        # 2. Asset process S tensors
        self.mu_base = torch.tensor(mu_base, dtype=torch.float32, device=self.device)
        self.mu_load = torch.tensor(mu_load, dtype=torch.float32, device=self.device)
        self.sigma_0 = torch.tensor(sigma_0, dtype=torch.float32, device=self.device)
        self.gamma_mat = torch.tensor(gamma_mat, dtype=torch.float32, device=self.device)

        # 3. Joint correlation matrix and orthogonal decomposition
        C = torch.tensor(corr_matrix, dtype=torch.float32, device=self.device)
        self.chol_L = torch.linalg.cholesky(C)

        self.rho_VS = C[:self.d, self.d:self.d + self.M]

        I_M = torch.eye(self.M, dtype=torch.float32, device=self.device)
        C_perp = I_M - torch.matmul(self.rho_VS.t(), self.rho_VS)
        self.chol_perp = torch.linalg.cholesky(C_perp)

    def _apply_specular_reflection(self, V_curr, V_next):
        """Compute tangent-plane specular reflection on the spherical boundary."""
        norms = torch.norm(V_next, p=2, dim=1, keepdim=True)
        is_reflected_mask = norms > self.radius

        if not is_reflected_mask.any():
            # Return zero displacement vector when no reflection occurs
            return V_next, is_reflected_mask.squeeze(-1), torch.zeros_like(V_next)

        eps = 1e-8
        Delta_V = V_next - V_curr
        A = torch.sum(Delta_V ** 2, dim=1, keepdim=True)
        B_coef = 2.0 * torch.sum(V_curr * Delta_V, dim=1, keepdim=True)
        C_coef = torch.sum(V_curr ** 2, dim=1, keepdim=True) - self.radius**2

        discriminant = torch.clamp(B_coef**2 - 4.0 * A * C_coef, min=0.0)
        tau = (-B_coef + torch.sqrt(discriminant)) / (2.0 * A + eps)
        tau = torch.clamp(tau, 0.0, 1.0)

        # Collision point c and unit normal vector n
        c = V_curr + tau * Delta_V
        c_norm = torch.norm(c, p=2, dim=1, keepdim=True)
        n = c / (c_norm + eps)

        # Reverse normal component and keep tangential component
        remain_V = V_next - c
        inner_prod = torch.sum(remain_V * n, dim=1, keepdim=True)
        V_specular = c + remain_V - 2.0 * inner_prod * n

        # Secondary radial projection safeguard
        spec_norms = torch.norm(V_specular, p=2, dim=1, keepdim=True)
        V_specular = torch.where(spec_norms > self.radius, V_specular * (self.radius / (spec_norms + eps)), V_specular)

        V_next_corrected = torch.where(is_reflected_mask, V_specular, V_next)

        # Compute boundary reflection displacement (Skorokhod increment)
        Delta_xi = V_next_corrected - V_next

        return V_next_corrected, is_reflected_mask.squeeze(-1), Delta_xi

    @torch.no_grad()
    def simulate(self, v0, return_full_path=True, auto_analyze=False, verbose=False, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        if auto_analyze:
            return_full_path = True

        v0_tensor = torch.tensor(v0, dtype=torch.float32, device=self.device)
        if torch.norm(v0_tensor) > self.radius:
            raise ValueError(f"Initial point v0 exceeds boundary radius {self.radius}")

        V_curr = v0_tensor.repeat(self.num_paths, 1)
        sqrt_dt = torch.sqrt(torch.tensor(self.dt, device=self.device))

        if verbose: start_time = time.time()

        # --- Stage 1: Factor warmup period ---
        if self.warmup_steps > 0:
            for _ in range(self.warmup_steps):
                Z_V = torch.randn((self.num_paths, self.d), device=self.device)
                drift_V = torch.matmul(self.alpha - V_curr, self.beta.t()) * self.dt
                diff_V = torch.matmul(Z_V, self.sigma_V.t()) * sqrt_dt
                V_next_raw = V_curr + drift_V + diff_V
                # Discard Delta_xi during warmup
                V_curr, _, _ = self._apply_specular_reflection(V_curr, V_next_raw)

        # --- Stage 2: Joint simulation period ---
        S_curr = torch.ones((self.num_paths, self.N), dtype=torch.float32, device=self.device)
        reflection_counter = torch.zeros(self.num_paths, dtype=torch.float32, device=self.device)

        if return_full_path:
            V_traj = torch.empty((self.n_steps + 1, self.num_paths, self.d), device=self.device)
            S_traj = torch.empty((self.n_steps + 1, self.num_paths, self.N), device=self.device)
            dW_S_traj = torch.empty((self.n_steps, self.num_paths, self.M_noise), device=self.device)
            # Pre-allocate reflection displacement tensor
            Xi_traj = torch.zeros((self.n_steps, self.num_paths, self.d), device=self.device)

            V_traj[0] = V_curr
            S_traj[0] = S_curr

        for i in range(1, self.n_steps + 1):
            Z_V = torch.randn((self.num_paths, self.d), device=self.device)
            Z_indep_perp = torch.randn((self.num_paths, self.M), device=self.device)
            Z_perp = torch.matmul(Z_indep_perp, self.chol_perp.t())
            Z_S = torch.matmul(Z_V, self.rho_VS) + Z_perp

            # Factor evolution and specular reflection
            drift_V = torch.matmul(self.alpha - V_curr, self.beta.t()) * self.dt
            diff_V = torch.matmul(Z_V, self.sigma_V.t()) * sqrt_dt
            V_next_raw = V_curr + drift_V + diff_V

            # Apply reflection and record displacement
            V_next, is_reflected, Delta_xi = self._apply_specular_reflection(V_curr, V_next_raw)
            reflection_counter += is_reflected.float()

            # Asset evolution (linear-affine volatility)
            mu_V = self.mu_base + torch.matmul(V_curr, self.mu_load.t())
            lin_gamma_V = 1.0 + torch.matmul(V_curr, self.gamma_mat.t())

            diff_S = torch.einsum('bn, nm, bm -> bn', lin_gamma_V, self.sigma_0, Z_S) * sqrt_dt
            ito_correction = 0.5 * torch.einsum('bn, nm -> bn', lin_gamma_V**2, self.sigma_0**2)
            S_next = S_curr * torch.exp((mu_V - ito_correction) * self.dt + diff_S)

            if return_full_path:
                V_traj[i] = V_next
                S_traj[i] = S_next
                dW_S_traj[i-1] = torch.cat([Z_V, Z_perp], dim=1) * sqrt_dt
                # Record current step reflection displacement
                Xi_traj[i-1] = Delta_xi

            V_curr = V_next
            S_curr = S_next

        if verbose:
            print(f"-> GPU simulation completed! | Warmup: {self.warmup_steps} steps | Trading: {self.n_steps} steps | Elapsed: {time.time() - start_time:.4f} s")

        if return_full_path:
            # Compute and cache geometric fields d(v) and n(v)
            eps = 1e-8
            V_norms = torch.norm(V_traj, p=2, dim=2) # Shape: (n_steps+1, num_paths)
            self._cached_d = torch.clamp(self.radius - V_norms, min=0.0) # Shape: (n_steps+1, num_paths)
            self._cached_n = V_traj / (V_norms.unsqueeze(-1) + eps)      # Shape: (n_steps+1, num_paths, d)

            self._cached_V = V_traj
            self._cached_S = S_traj
            self._cached_dW_S = dW_S_traj
            self._cached_Xi = Xi_traj # Cache Xi trajectory
            self._cached_reflections = reflection_counter

            if auto_analyze:
                print("\n" + "="*70)
                print(">> [Auto Analyze] Syncing data from GPU to CPU memory...")
                self.analyze_factors()
                self.analyze_scenarios()
                print("="*70 + "\n")

            # Return trajectory tensors including Xi_traj
            return V_traj, S_traj, dW_S_traj, Xi_traj

        return V_curr, S_curr

    @torch.no_grad()
    def analyze_factors(self, print_limit=5, visualize_reflection=False):
        if not hasattr(self, '_cached_V') or not hasattr(self, '_cached_reflections'):
            raise RuntimeError("Missing cached trajectories. Run simulate with return_full_path=True.")

        print("-> [CPU Analysis] Diagnosing latent state factors V ...")
        V_np = self._cached_V.cpu().numpy()
        path_means = np.mean(V_np, axis=0)
        path_vars = np.var(V_np, axis=0)
        reflection_counts = self._cached_reflections.cpu().numpy()

        stats_dict = {}
        for i in range(self.d):
            stats_dict[f'Mean_V{i+1}'] = path_means[:, i]
            stats_dict[f'Var_V{i+1}'] = path_vars[:, i]
        stats_dict['Reflections'] = reflection_counts

        df_stats = pd.DataFrame(stats_dict, index=[f"Path_{i}" for i in range(self.num_paths)])
        df_t = df_stats.T
        df_t.index.name = 'Factor_Metrics'
        df_t['Global_Average'] = df_t.mean(axis=1)

        print("\n📊 Factor Diagnostics Report:")
        format_func = lambda x: f"{x:.4f}"
        if self.num_paths > print_limit:
            cols = list(df_t.columns[:print_limit]) + ['Global_Average']
            print(df_t[cols].to_string(float_format=format_func))
            print(f"... (omitted {self.num_paths - print_limit} intermediate paths)")
        else:
            print(df_t.to_string(float_format=format_func))

        print("\n" + "-"*60)
        print("💡 Spherical Boundary Reflection Statistics:")
        print(f" - Domain boundary radius (Radius)          : {self.radius}")
        print(f" - Max reflections per path                 : {int(reflection_counts.max())}")
        print(f" - Mean reflections per path                : {reflection_counts.mean():.2f}")
        print(f" - Percentage of paths with zero reflection : {(np.sum(reflection_counts == 0) / self.num_paths * 100):.2f}%")
        print("-" * 60 + "\n")

        if visualize_reflection:
            plt.figure(figsize=(9, 5))
            max_ref = int(reflection_counts.max())
            bins = range(max_ref + 2)
            plt.hist(reflection_counts, bins=bins, color='#1f77b4', edgecolor='black', alpha=0.75)
            plt.xlabel('Number of Boundary Reflections per Path', fontsize=12)
            plt.ylabel('Number of Paths', fontsize=12)
            plt.title('Distribution of Elastic Reflection Counts Across Paths', fontsize=14, fontweight='bold')
            plt.grid(True, linestyle=':', alpha=0.6)
            for count in range(max_ref + 1):
                num_paths_with_count = np.sum(reflection_counts == count)
                if num_paths_with_count > 0:
                    plt.text(count + 0.5, num_paths_with_count + self.num_paths * 0.01,
                             str(num_paths_with_count), ha='center', va='bottom', fontsize=10)
            plt.tight_layout()
            plt.show()

        return df_t

    @torch.no_grad()
    def analyze_scenarios(self, print_limit=5):
        if not hasattr(self, '_cached_S'):
            raise RuntimeError("Missing cached trajectories. Run simulate with return_full_path=True.")

        print("-> [CPU Analysis] Comparing multi-asset steady-state analytical values with Monte Carlo estimates...")

        full_corr = torch.matmul(self.chol_L, self.chol_L.t()).cpu().numpy()
        R_SS = full_corr[self.d:, self.d:]

        alpha_np = self.alpha.cpu().numpy()
        mu_base_np = self.mu_base.cpu().numpy()
        mu_load_np = self.mu_load.cpu().numpy()
        sigma_0_np = self.sigma_0.cpu().numpy()
        gamma_np = self.gamma_mat.cpu().numpy()

        # Steady-state linear-affine volatility
        sigma_alpha = np.diag(1.0 + gamma_np @ alpha_np) @ sigma_0_np
        theory_cov = sigma_alpha @ R_SS @ sigma_alpha.T
        theory_vol = np.sqrt(np.diag(theory_cov))
        theory_return = (mu_base_np + mu_load_np @ alpha_np) - 0.5 * np.diag(theory_cov)

        theory_dict = {}
        for i in range(self.N): theory_dict[f'Return_S{i+1}'] = theory_return[i]
        for i in range(self.N): theory_dict[f'Vol_S{i+1}'] = theory_vol[i]
        for i in range(self.N):
            for j in range(i + 1, self.N):
                theory_dict[f'Corr_S{i+1}_S{j+1}'] = theory_cov[i, j] / (theory_vol[i] * theory_vol[j])

        S_data = self._cached_S.cpu().numpy()
        log_returns = np.diff(np.log(S_data), axis=0)

        return_ann = np.log(S_data[-1] / S_data[0]) / self.T
        vol_daily = np.std(log_returns, axis=0, ddof=1)
        vol_ann = vol_daily * np.sqrt(1.0 / self.dt)

        stats_dict = {}
        for i in range(self.N): stats_dict[f'Return_S{i+1}'] = return_ann[:, i]
        for i in range(self.N): stats_dict[f'Vol_S{i+1}'] = vol_ann[:, i]

        if self.N > 1:
            T_steps = log_returns.shape[0]
            mu = np.mean(log_returns, axis=0)
            X_centered = log_returns - mu
            for i in range(self.N):
                for j in range(i + 1, self.N):
                    cov_ij = np.sum(X_centered[:, :, i] * X_centered[:, :, j], axis=0) / (T_steps - 1)
                    denom = vol_daily[:, i] * vol_daily[:, j]
                    stats_dict[f'Corr_S{i+1}_S{j+1}'] = np.divide(cov_ij, denom, out=np.zeros_like(cov_ij), where=(denom != 0))

        df_stats = pd.DataFrame(stats_dict, index=[f"Path_{i}" for i in range(self.num_paths)])
        df_t = df_stats.T
        df_t.index.name = 'Asset_Metrics'

        df_t['Global_Average'] = df_t.mean(axis=1)
        df_t.insert(0, 'Theoretical', pd.Series(theory_dict))
        df_t['Bias(Avg-Theory)'] = df_t['Global_Average'] - df_t['Theoretical']

        print("\n📈 Asset Performance Analysis Completed!")
        format_func = lambda x: f"{x:.4f}"
        if self.num_paths > print_limit:
            cols_to_print = ['Theoretical'] + list(df_t.columns[1:print_limit+1]) + ['Global_Average', 'Bias(Avg-Theory)']
            print(df_t[cols_to_print].to_string(float_format=format_func))
        else:
            print(df_t.to_string(float_format=format_func))
        print("\n" + "="*70 + "\n")

        return df_t

    def save_to_csv(self, filepath):
        if not hasattr(self, '_cached_S') or not hasattr(self, '_cached_dW_S'): raise RuntimeError("Run simulate with return_full_path=True first.")
        print(f"-> [CPU IO] Serializing data and geometric fields to {filepath} ...")

        metadata = {
            "d_dim": self.d, "N_dim": self.N, "M_dim": self.M,
            "alpha": self.alpha.cpu().tolist(), "beta": self.beta.cpu().tolist(),
            "sigma_V": self.sigma_V.cpu().tolist(), "radius": self.radius,
            "mu_base": self.mu_base.cpu().tolist(), "mu_load": self.mu_load.cpu().tolist(),
            "sigma_0": self.sigma_0.cpu().tolist(), "gamma_mat": self.gamma_mat.cpu().tolist(),
            "corr_matrix": torch.matmul(self.chol_L, self.chol_L.t()).cpu().tolist(),
            "dt": self.dt, "T": self.T, "num_paths": self.num_paths, "warmup_steps": self.warmup_steps
        }

        V_data = self._cached_V.cpu().numpy().reshape(-1, self.d)
        S_data = self._cached_S.cpu().numpy().reshape(-1, self.N)

        # Flatten cached geometric fields
        d_data = self._cached_d.cpu().numpy().flatten()
        n_data = self._cached_n.cpu().numpy().reshape(-1, self.d)

        dW_S_padded = np.zeros((self.n_steps + 1, self.num_paths, self.M_noise))
        dW_S_padded[:-1, :, :] = self._cached_dW_S.cpu().numpy()
        dW_S_data = dW_S_padded.reshape(-1, self.M_noise)

        # Pad and flatten Xi data
        Xi_padded = np.zeros((self.n_steps + 1, self.num_paths, self.d))
        Xi_padded[:-1, :, :] = self._cached_Xi.cpu().numpy()
        Xi_data = Xi_padded.reshape(-1, self.d)

        t_col = np.repeat(self.time_grid.cpu().numpy(), self.num_paths)
        p_col = np.tile(np.arange(self.num_paths), self.n_steps + 1)

        df_dict = {'Time': t_col, 'Path_ID': p_col, 'Distance_d': d_data}
        df_dict.update({f'Normal_n{i+1}': n_data[:, i] for i in range(self.d)})
        df_dict.update({f'V{i+1}': V_data[:, i] for i in range(self.d)})
        df_dict.update({f'Asset_S{i+1}': S_data[:, i] for i in range(self.N)})
        df_dict.update({f'dW_S{i+1}': dW_S_data[:, i] for i in range(self.M_noise)})
        # Add Xi columns
        df_dict.update({f'Xi_{i+1}': Xi_data[:, i] for i in range(self.d)})

        with open(filepath, 'w', encoding='utf-8') as f: f.write(f"# METADATA:{json.dumps(metadata)}\n")
        pd.DataFrame(df_dict).to_csv(filepath, mode='a', index=False)
        print("✅ Data, geometric fields (d, n), and metadata saved.")

    @classmethod
    def from_csv(cls, filepath, device=None):
        with open(filepath, 'r', encoding='utf-8') as f:
            meta = json.loads(f.readline().strip().replace("# METADATA:", ""))

        instance = cls(
            d_dim=meta["d_dim"], N_dim=meta["N_dim"], M_dim=meta["M_dim"],
            alpha=meta["alpha"], beta=meta["beta"], sigma_V=meta["sigma_V"], radius=meta["radius"],
            mu_base=meta["mu_base"], mu_load=meta["mu_load"], sigma_0=meta["sigma_0"],
            gamma_mat=meta["gamma_mat"], corr_matrix=meta["corr_matrix"],
            dt=meta["dt"], T=meta["T"], num_paths=meta["num_paths"], warmup_steps=meta["warmup_steps"],
            device=device
        )

        df = pd.read_csv(filepath, comment='#').sort_values(by=['Time', 'Path_ID'])
        v_data = df[[f'V{i+1}' for i in range(instance.d)]].values
        s_data = df[[f'Asset_S{i+1}' for i in range(instance.N)]].values
        dw_s_data = df[[f'dW_S{i+1}' for i in range(instance.M_noise)]].values
        # Load Xi data
        xi_data = df[[f'Xi_{i+1}' for i in range(instance.d)]].values

        # Reconstruct geometric field data
        d_data = df['Distance_d'].values
        n_data = df[[f'Normal_n{i+1}' for i in range(instance.d)]].values

        t_dim, p_dim = instance.n_steps + 1, instance.num_paths

        instance._cached_V = torch.tensor(v_data.reshape(t_dim, p_dim, instance.d), dtype=torch.float32, device=instance.device)
        instance._cached_S = torch.tensor(s_data.reshape(t_dim, p_dim, instance.N), dtype=torch.float32, device=instance.device)
        instance._cached_d = torch.tensor(d_data.reshape(t_dim, p_dim), dtype=torch.float32, device=instance.device)
        instance._cached_n = torch.tensor(n_data.reshape(t_dim, p_dim, instance.d), dtype=torch.float32, device=instance.device)

        dw_s_tensor_padded = torch.tensor(dw_s_data.reshape(t_dim, p_dim, instance.M_noise), dtype=torch.float32, device=instance.device)
        instance._cached_dW_S = dw_s_tensor_padded[:-1, :, :]

        # Reconstruct and attach Xi_traj
        xi_tensor_padded = torch.tensor(xi_data.reshape(t_dim, p_dim, instance.d), dtype=torch.float32, device=instance.device)
        instance._cached_Xi = xi_tensor_padded[:-1, :, :]

        instance._cached_reflections = torch.zeros(instance.num_paths, device=instance.device)

        print("✅ Instance, trajectory tensors, and geometric fields (d, n) reconstructed.")
        return instance