import torch
import torch.nn as nn

class SubNetwork(nn.Module):
    """
    Single-step sub-network for the Deep BSDE solver.
    """
    def __init__(self, d_dim, M_dim, radius, hidden_layers=None):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [64, 64]

        self.d_dim = d_dim
        self.M_dim = M_dim
        self.radius = float(radius)

        # Input features: V_t(d) + ||V_t||(1) + d(v)(1) +
        #                 alert(1) + multi_alert(1) + n_v(d) + V_t^2(d) + rho_v(M)
        # Total dimension = 3*d + M + 4
        input_dim = 3 * self.d_dim + self.M_dim + 4

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.Tanh())
            current_dim = hidden_dim

        self.net = nn.Sequential(*layers).to(torch.float64)

        # Output head
        self.lambda_head = nn.Linear(current_dim, self.d_dim).to(torch.float64)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Small uniform initialization for the output head
        nn.init.uniform_(self.lambda_head.weight, -1e-4, 1e-4)

    def forward(self, V_t, rho_v):
        # 1. Geometric distance to boundary
        V_norm = torch.norm(V_t, p=2, dim=1, keepdim=True)
        d_v = torch.clamp(self.radius - V_norm, min=0.0)

        # 2. Unit normal vector
        n_v = V_t / (V_norm + 1e-8)

        # 3. Multi-scale boundary proximity features
        boundary_alert = torch.exp(-2.5 * d_v)
        multi_scale_alert = torch.exp(-5.0 * d_v)

        # 4. Normalized quadratic features
        V_sq = (V_t ** 2) / (self.radius ** 2 + 1e-8)

        # Concatenate physical and geometric features
        x = torch.cat([
            V_t, V_norm, d_v,
            boundary_alert, multi_scale_alert,
            n_v, V_sq, rho_v
        ], dim=1)

        # MLP forward pass
        features = self.net(x)

        Lambda_pred = self.lambda_head(features)

        # Clamp output to prevent gradient explosion
        Lambda_pred = torch.clamp(Lambda_pred, min=-10.0, max=10.0)

        return Lambda_pred


class LambdaNetwork(nn.Module):
    """
    Time-step decoupled Deep BSDE network architecture.
    Instantiates an independent SubNetwork for each discrete time step.
    """
    def __init__(self, num_steps, d_dim, M_dim, radius, hidden_layers=None):
        super().__init__()
        self.num_steps = num_steps
        self.d_dim = d_dim
        self.M_dim = M_dim

        # Independent sub-network for each discrete time step
        self.sub_networks = nn.ModuleList([
            SubNetwork(d_dim, M_dim, radius, hidden_layers) for _ in range(num_steps)
        ])

    def forward(self, step_idx, V_t, rho_v):
        """
        Evaluate the sub-network corresponding to the discrete time index step_idx.
        """
        return self.sub_networks[step_idx](V_t, rho_v)