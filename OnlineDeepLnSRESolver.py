import torch
import torch.nn as nn
import torch.optim as optim
import os
import pandas as pd
import math
import copy

class OnlineDeepLnSRESolver(nn.Module):
    """
    Log-space online Deep BSDE solver for Extended Stochastic Riccati Equations.
    """

    def __init__(self, params, lambda_net, sre_equation, dt=1/252, T=1.0, device='cpu',
                 initial_p0=0.0, p_lower=1.0, p_upper=200.0,
                 lambda_wall=100.0, phi_threshold=None):
        super(OnlineDeepLnSRESolver, self).__init__()

        self.params = params
        self.dt = dt
        self.T = T
        self.N_steps = int(T / dt)
        self.device = device
        self.p_lower = p_lower
        self.p_upper = p_upper
        self.log_p_lower = math.log(p_lower) if p_lower > 0 else -float('inf')
        self.log_p_upper = math.log(p_upper)

        self.lambda_net = lambda_net.to(device)
        self.sre_equation = sre_equation.to(device)
        self.P0 = nn.Parameter(torch.tensor([initial_p0], dtype=torch.float64, device=device))

        self.sigma_V = torch.tensor(self.params['sigma_V'], dtype=torch.float64, device=self.device)

        # Exact inverse matrix for spatial gradient computation
        self.inv_sigma_V = torch.linalg.inv(self.sigma_V)
        self.inv_sigma_V_T = self.inv_sigma_V.t()

        self.radius = self.params['radius']
        self.d_dim = self.sigma_V.shape[0]

        # Base tensors for state-dependent physical prior rho(V_t)
        self.mu_base = torch.tensor(self.params['mu_base'], dtype=torch.float64, device=self.device)
        self.mu_load = torch.tensor(self.params['mu_load'], dtype=torch.float64, device=self.device)
        self.sigma_0 = torch.tensor(self.params['sigma_0'], dtype=torch.float64, device=self.device)
        self.gamma_mat = torch.tensor(self.params['gamma_mat'], dtype=torch.float64, device=self.device)
        self.r_val = float(self.params['r'])

        # Precompute pseudo-inverse matrix for rho(V_t): (\sigma_0^T \Sigma_0^{-1})^T
        Sigma_0 = torch.matmul(self.sigma_0, self.sigma_0.t())
        self.pseudo_inv_sigma_0_T = torch.matmul(self.sigma_0.t(), torch.linalg.pinv(Sigma_0)).t()

        self.lambda_wall = lambda_wall
        self.phi_threshold = phi_threshold if phi_threshold is not None else 0.1 * self.radius

        self.current_epoch = 0
        self.history = {
            'epoch': [], 'loss': [], 'penalty_loss': [], 'yt_mean': [], 'yt_std': [],
            'grad_norm_lambda': [], 'grad_norm_p0': [], 'p0': [],
            'learning_rate_lambda': [], 'learning_rate_p0': [],
            'freeze_p0': [], 'freeze_lambda': []
        }
        self.optimizer_state = None
        self.scheduler_state = None

        print("=" * 75)
        print(f"[System] Initialized Log-Space Online Solver (with Skorokhod Compensation)")
        print("-" * 75)
        print(f" • Initial log value check_P(0)         : {initial_p0:+11.6f}")
        print(f" • Initial physical value P(0)          : {math.exp(initial_p0):10.6f}")
        print(f" • lambda_wall (boundary penalty weight): {self.lambda_wall}")
        print(f" • phi_threshold (projection threshold) : {self.phi_threshold:.4f}")
        print("=" * 75 + "\n")

    def _geometry_aware_projection(self, Lambda_raw, V_t):
        norms = torch.norm(V_t, p=2, dim=1, keepdim=True)
        d_v = torch.clamp(self.radius - norms, min=0.0)
        phi_d = torch.clamp(1.0 - d_v / self.phi_threshold, min=0.0) ** 3
        n_v = V_t / (norms + 1e-8)

        Lambda_active = Lambda_raw[:, :self.d_dim]
        n_tilde = torch.matmul(n_v, self.inv_sigma_V_T)
        inner_prod = torch.sum(Lambda_active * n_tilde, dim=1, keepdim=True)
        n_mapped_back = torch.matmul(n_v, self.sigma_V)
        correction = phi_d * inner_prod * n_mapped_back

        Lambda_proj_active = Lambda_active - correction
        if Lambda_raw.shape[1] > self.d_dim:
            Lambda_proj = torch.cat([Lambda_proj_active, Lambda_raw[:, self.d_dim:]], dim=1)
        else:
            Lambda_proj = Lambda_proj_active

        step_penalty = torch.sum(correction ** 2, dim=1)
        return Lambda_proj, step_penalty

    def _compute_physical_prior_rho(self, V_t):
        """Compute state-dependent market price of risk prior rho(V_t)."""
        mu_V = self.mu_base + torch.matmul(V_t, self.mu_load.t())
        B_t = mu_V - self.r_val

        # Lower-bound clamp aligned with the drift equation
        lin_gamma_V = torch.clamp(1.0 + torch.matmul(V_t, self.gamma_mat.t()), min=1e-6)
        B_prime = B_t / lin_gamma_V

        rho_v = torch.matmul(B_prime, self.pseudo_inv_sigma_0_T)
        return rho_v

    def forward_simulation(self, V_paths, dW_paths, Xi_paths):
        batch_size = V_paths.shape[0]
        check_P_clamped = torch.clamp(self.P0, min=self.log_p_lower, max=self.log_p_upper)
        check_P_t = check_P_clamped.expand(batch_size, 1)

        V_0 = V_paths[:, 0, :]
        rho_0 = self._compute_physical_prior_rho(V_0)

        # Call step-0 sub-network
        check_Lambda_raw = self.lambda_net(0, V_0, rho_0)

        total_penalty = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        total_kkt_res = torch.zeros(batch_size, dtype=torch.float64, device=self.device)

        for i in range(self.N_steps):
            V_curr = V_paths[:, i, :]
            dW_curr = dW_paths[:, i, :]
            delta_xi_curr = Xi_paths[:, i, :] # Current step boundary reflection displacement

            check_Lambda_proj, step_penalty = self._geometry_aware_projection(check_Lambda_raw, V_curr)
            total_penalty += step_penalty

            if check_Lambda_proj.shape[1] < dW_curr.shape[1]:
                zeros = torch.zeros(check_Lambda_proj.shape[0], dW_curr.shape[1] - check_Lambda_proj.shape[1],
                                    dtype=check_Lambda_proj.dtype, device=check_Lambda_proj.device)
                check_Lambda_full = torch.cat([check_Lambda_proj, zeros], dim=1)
            else:
                check_Lambda_full = check_Lambda_proj

            current_t_val = i * self.dt
            drift = self.sre_equation(current_t_val, check_P_t, check_Lambda_full, V_curr)
            total_kkt_res += self.sre_equation._step_kkt_res

            diffusion_term = torch.sum(check_Lambda_full * dW_curr, dim=1, keepdim=True)

            # ==============================================================================
            # Skorokhod local-time compensation
            # ==============================================================================
            # 1. Extract active factor adjoint component Lambda_active
            check_Lambda_active = check_Lambda_full[:, :self.d_dim]
            # 2. Map back to spatial gradient: \nabla P = \Lambda @ \sigma_V^{-1}
            grad_P = torch.matmul(check_Lambda_active, self.inv_sigma_V)
            # 3. Inner product compensation: \langle \nabla P, \Delta \xi \rangle
            skorokhod_comp = torch.sum(grad_P * delta_xi_curr, dim=1, keepdim=True)

            # 4. Forward Euler update with reflection compensation
            check_P_t = check_P_t + drift * self.dt + diffusion_term + skorokhod_comp
            # ==============================================================================

            check_P_t = torch.clamp(check_P_t, max=math.log(100.0))

            if i < self.N_steps - 1:
                V_next = V_paths[:, i + 1, :]
                rho_next = self._compute_physical_prior_rho(V_next)
                # Call step-(i+1) sub-network
                check_Lambda_raw = self.lambda_net(i + 1, V_next, rho_next)

        self._current_batch_kkt = torch.mean(total_kkt_res).item()
        return check_P_t, total_penalty

    def train_model(self, env_generator, v0, additional_epochs=1200,
                    lr_lambda=1e-3, lr_p0=5e-2,
                    freeze_p0=False, freeze_lambda=False, std_penalty_multiplier=10,
                    reset_optim_state=True, patience=30):

        self.P0.requires_grad = not freeze_p0
        for p in self.lambda_net.parameters():
            p.requires_grad = not freeze_lambda

        param_groups = [
            {'params': [p for p in self.lambda_net.parameters() if p.requires_grad], 'lr': lr_lambda},
            {'params': [self.P0] if self.P0.requires_grad else [], 'lr': lr_p0}
        ]
        param_groups = [g for g in param_groups if len(g['params']) > 0]

        optimizer = optim.Adam(param_groups)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=patience)

        start_epoch = self.current_epoch

        if reset_optim_state:
            print(f"[Pipeline] Optimizer state reset: cleared historical momentum.")
        else:
            if self.optimizer_state is not None:
                saved_state = copy.deepcopy(self.optimizer_state)
                current_groups_count = len(optimizer.param_groups)
                saved_groups_count = len(saved_state['param_groups'])

                if saved_groups_count == 1 and current_groups_count == 2:
                    p0_group_template = copy.deepcopy(optimizer.state_dict()['param_groups'][1])
                    num_params_in_g0 = len(saved_state['param_groups'][0]['params'])
                    p0_group_template['params'] = [num_params_in_g0]
                    saved_state['param_groups'].append(p0_group_template)
                elif saved_groups_count == 2 and current_groups_count == 1:
                    saved_state['param_groups'].pop()
                    p0_idx = len(saved_state['param_groups'][0]['params'])
                    if p0_idx in saved_state['state']:
                        del saved_state['state'][p0_idx]

                optimizer.load_state_dict(saved_state)
                idx = 0
                if not freeze_lambda and len(optimizer.param_groups) > idx:
                    optimizer.param_groups[idx]['lr'] = lr_lambda
                    idx += 1
                if not freeze_p0 and len(optimizer.param_groups) > idx:
                    optimizer.param_groups[idx]['lr'] = lr_p0
                if self.scheduler_state is not None:
                    scheduler.load_state_dict(self.scheduler_state)
                    scheduler.num_bad_epochs = 0
                print(f"[Pipeline] Checkpoint resumed: restored and aligned optimizer momentum state.")

        scheduler.best = float('inf')
        scheduler.num_bad_epochs = 0
        scheduler.cooldown_counter = 0

        target_epoch = start_epoch + additional_epochs
        status_str = "Fix_P0" if freeze_p0 else "Fix_Lambda" if freeze_lambda else "Joint_Opt"

        print("\n" + "=" * 125)
        print(f" Starting Log-Space Training [Skorokhod Compensated] | Mode: {status_str} | VarWeight: {std_penalty_multiplier} | Target Epoch: {target_epoch}")
        print("-" * 125)
        print(f"{'Epoch':<7} | {'TotalLoss':<11} | {'Train_loss':<11} |{'BSDE_Loss':<11} | {'WallPen':<9} | "
              f"{'KKT_Err':<9} | "
              f"{'P_T Mean':<9} | {'P_T Std':<9} | {'Grad_Lam':<9} | {'Grad_P0':<9} | "
              f"{'LR_Lambda':<10} | {'P(0)':<10}")
        print("-" * 125)

        try:
            loss_buffer = []
            for epoch in range(start_epoch, target_epoch):
                if os.path.exists("stop.txt"):
                    print(f"\n[External Control] Detected stop.txt signal, executing safe exit...")
                    os.remove("stop.txt")
                    break

                if not freeze_lambda: self.lambda_net.train()
                else: self.lambda_net.eval()

                # Simulate forward trajectories with reflection increments
                V_tensor, _, dW_S_tensor, Xi_tensor = env_generator.simulate(
                    v0=v0, return_full_path=True, auto_analyze=False, verbose=False
                )

                V_batch = V_tensor.detach().permute(1, 0, 2).contiguous().to(dtype=torch.float64)
                dW_batch = dW_S_tensor.detach().permute(1, 0, 2).contiguous().to(dtype=torch.float64)
                Xi_batch = Xi_tensor.detach().permute(1, 0, 2).contiguous().to(dtype=torch.float64)

                optimizer.zero_grad()

                # Forward pass with boundary reflection displacement
                check_P_T_pred, path_penalties = self.forward_simulation(V_batch, dW_batch, Xi_batch)

                kkt_err = getattr(self, '_current_batch_kkt', 0.0)

                check_P_T_mean = torch.mean(check_P_T_pred)
                check_P_T_var = torch.var(check_P_T_pred)

                bsde_loss_unweighted = (check_P_T_mean - 0.0)**2 + check_P_T_var
                penalty_loss = self.lambda_wall * torch.mean(path_penalties)

                train_loss = (check_P_T_mean - 0.0)**2 + std_penalty_multiplier * check_P_T_var + penalty_loss

                train_loss.backward()

                if not freeze_lambda:
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.lambda_net.parameters() if p.requires_grad], max_norm=20.0)
                if not freeze_p0:
                    torch.nn.utils.clip_grad_norm_([self.P0], max_norm=10.0)

                optimizer.step()

                loss_buffer.append(bsde_loss_unweighted.item())
                if len(loss_buffer) > 10:
                    loss_buffer.pop(0)

                current_lr_lambda = optimizer.param_groups[0]['lr'] if not freeze_lambda else 0.0
                current_lr_p0 = optimizer.param_groups[-1]['lr'] if not freeze_p0 else 0.0

                if (epoch - start_epoch) == 10:
                    scheduler.best = float('inf')
                    scheduler.num_bad_epochs = 0
                    scheduler.cooldown_counter = 0

                recorded_loss = float('nan')
                if len(loss_buffer) >= 10:
                    smoothed_loss = sum(loss_buffer) / len(loss_buffer)
                    recorded_loss = smoothed_loss
                if (epoch - start_epoch) >= 10:
                    old_lrs = [g['lr'] for g in optimizer.param_groups]
                    scheduler.step(smoothed_loss)
                    new_lrs = [g['lr'] for g in optimizer.param_groups]
                    if any(n < o for n, o in zip(new_lrs, old_lrs)):
                        scheduler.best = float('inf')

                # === Diagnostics and evaluation logging ===
                if epoch % 10 == 0 or epoch == target_epoch - 1:
                    with torch.no_grad():
                        self.lambda_net.eval()
                        # Validation simulation
                        V_eval, _, dW_eval, Xi_eval = env_generator.simulate(
                            v0=v0, return_full_path=True, auto_analyze=False, verbose=False
                        )
                        V_eval_batch = V_eval.detach().permute(1, 0, 2).contiguous().to(dtype=torch.float64)
                        dW_eval_batch = dW_eval.detach().permute(1, 0, 2).contiguous().to(dtype=torch.float64)
                        Xi_eval_batch = Xi_eval.detach().permute(1, 0, 2).contiguous().to(dtype=torch.float64)

                        check_P_T_eval, _ = self.forward_simulation(V_eval_batch, dW_eval_batch, Xi_eval_batch)
                        mean_check_p = check_P_T_eval.mean().item()
                        std_check_p = check_P_T_eval.std().item()

                    current_check_p0 = torch.clamp(self.P0, min=self.log_p_lower, max=self.log_p_upper).item()
                    grad_norm_lambda = sum(p.grad.norm().item() ** 2 for p in self.lambda_net.parameters()
                                           if p.grad is not None) ** 0.5
                    grad_norm_p0 = self.P0.grad.norm().item() if self.P0.grad is not None else 0.0

                    self.history['epoch'].append(epoch)
                    self.history['loss'].append(recorded_loss)
                    self.history['penalty_loss'].append(penalty_loss.item())
                    self.history['yt_mean'].append(mean_check_p)
                    self.history['yt_std'].append(std_check_p)
                    self.history['grad_norm_lambda'].append(grad_norm_lambda)
                    self.history['grad_norm_p0'].append(grad_norm_p0)
                    self.history['p0'].append(current_check_p0)
                    self.history['learning_rate_lambda'].append(current_lr_lambda)
                    self.history['learning_rate_p0'].append(current_lr_p0)
                    self.history['freeze_p0'].append(freeze_p0)
                    self.history['freeze_lambda'].append(freeze_lambda)

                    if len(loss_buffer) >= 10:
                        print(f"{epoch:<7d} | {recorded_loss:<11.4e} | {train_loss.item():<11.4e} |{bsde_loss_unweighted.item():<11.4e} | {penalty_loss.item():<9.2e} | "
                              f"{kkt_err:<9.2e} | "
                              f"{mean_check_p:<+9.5f} | {std_check_p:<9.5f} | {grad_norm_lambda:<9.2e} | {grad_norm_p0:<9.2e} | "
                              f"{current_lr_lambda:<10.2e} | {math.exp(current_check_p0):<10.6f}")

                self.current_epoch = epoch + 1

        except KeyboardInterrupt:
            print(f"\n[Manual Interrupt] Signal captured, state locked at epoch {self.current_epoch}.")

        self.optimizer_state = optimizer.state_dict()
        self.scheduler_state = scheduler.state_dict()
        print("-" * 125)
        print(f"========== Stage Training Completed (Total Epochs: {self.current_epoch}) ==========\n")
        self.export_history_to_csv()
        return torch.clamp(self.P0, min=self.log_p_lower, max=self.log_p_upper).item()

    def export_history_to_csv(self, csv_path='sre_logspace_history.csv'):
        if not self.history['epoch']: return
        df = pd.DataFrame(self.history)
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f">>> [Data Export] Exported log-space training history to: {csv_path}")

    def save_model(self, file_path='sre_logspace_model.pt', csv_path='sre_logspace_history.csv'):
        clamped_check_p0 = torch.clamp(self.P0, min=self.log_p_lower, max=self.log_p_upper).item()
        save_dict = {
            'lambda_net_state': self.lambda_net.state_dict(), 'P0': clamped_check_p0,
            'env_params': self.params, 'current_epoch': self.current_epoch,
            'history': self.history, 'optimizer_state': self.optimizer_state, 'scheduler_state': self.scheduler_state
        }
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        torch.save(save_dict, file_path)
        print(f">>> [Model Saved] Network parameters and optimizer states saved to: {file_path}")
        self.export_history_to_csv(csv_path)

    def load_model(self, file_path='sre_logspace_model.pt'):
        if not os.path.exists(file_path): raise FileNotFoundError(f"Checkpoint file not found: {file_path}")
        checkpoint = torch.load(file_path, map_location=self.device, weights_only=False)
        self.lambda_net.load_state_dict(checkpoint['lambda_net_state'])
        self.P0.data = torch.tensor([checkpoint['P0']], dtype=torch.float64, device=self.device)
        self.current_epoch = checkpoint['current_epoch']

        loaded_history = checkpoint['history']
        self.history = {k: list(loaded_history[k]) for k in self.history.keys()}
        self.optimizer_state = checkpoint['optimizer_state']
        self.scheduler_state = checkpoint['scheduler_state']

        print(f">>> [Model Loaded] Restored checkpoint state! Current progress: Epoch {self.current_epoch}")
        print(f" • check_P(0) = {self.P0.item():+10.6f} (P(0) = {math.exp(self.P0.item()):10.6f})")
        return checkpoint['env_params']