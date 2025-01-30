import torch
import torch.nn as nn

class MeteoEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.cnn(x)

class MaskEncoder(nn.Module):
    def __init__(self, n_masks=3, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_masks, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, masks):
        return self.cnn(masks)

class CoordProcessor(nn.Module):
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.coord_net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x, y, t):
        t_expanded = t.view(-1, 1, 1).expand(-1, x.shape[1], x.shape[2])
        coords = torch.stack([x, y, t_expanded], dim=-1)
        processed_coords = self.coord_net(coords)
        return processed_coords.permute(0, 3, 1, 2)

class ClimatePINN(nn.Module):
    def __init__(self, hidden_dim=64, initial_re=100.0):
        super().__init__()
        # Encoders
        self.meteo_encoder = MeteoEncoder(in_channels=2, hidden_dim=hidden_dim)
        self.mask_encoder = MaskEncoder(hidden_dim=hidden_dim)
        self.coord_processor = CoordProcessor(hidden_dim=hidden_dim)
        
        # Combine features
        self.feature_combiner = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1),
            nn.Tanh(),
            nn.Conv2d(hidden_dim * 2, 3, kernel_size=3, padding=1)
        )
        
        # Learnable Reynolds number parameter
        self.log_re = nn.Parameter(torch.log(torch.tensor(initial_re)))
    
    def get_reynolds_number(self):
        # Return Reynolds number in a differentiable way
        # Using exp ensures Re stays positive
        return torch.exp(self.log_re)
    
    def forward(self, meteo_inputs, masks, coords):
        meteo_features = self.meteo_encoder(meteo_inputs)
        mask_features = self.mask_encoder(masks)
        coord_features = self.coord_processor(coords['lat'], coords['lon'], coords['time'])
        
        combined = torch.cat([
            meteo_features,
            mask_features,
            coord_features
        ], dim=1)
        
        outputs = self.feature_combiner(combined)
        
        return {
            't2m': outputs[:, 0],
            'u10': outputs[:, 1],
            'v10': outputs[:, 2]
        }
    
    def compute_physics_loss(self, coords, predictions):
        """
        Compute the physics-informed loss based on Navier-Stokes equations
        
        Args:
            coords: Dict containing 'lat', 'lon', 'time' tensors with require_grad=True
            predictions: Dict containing 'u10', 'v10' predictions
        """

        x, y, t = coords['lon'], coords['lat'], coords['time']
        u_pred, v_pred = predictions['u10'], predictions['v10']
        
        # First derivatives
        u_x = torch.autograd.grad(outputs=u_pred, inputs=x, grad_outputs=torch.ones_like(u_pred),
                                create_graph=True, retain_graph=True)[0]
        u_y = torch.autograd.grad(outputs=u_pred, inputs=y, grad_outputs=torch.ones_like(u_pred),
                                create_graph=True, retain_graph=True)[0]
        u_t = torch.autograd.grad(outputs=u_pred, inputs=t, grad_outputs=torch.ones_like(u_pred),
                                create_graph=True, retain_graph=True)[0]
        
        v_x = torch.autograd.grad(outputs=v_pred, inputs=x, grad_outputs=torch.ones_like(v_pred),
                                create_graph=True, retain_graph=True)[0]
        v_y = torch.autograd.grad(outputs=v_pred, inputs=y, grad_outputs=torch.ones_like(v_pred),
                                create_graph=True, retain_graph=True)[0]
        v_t = torch.autograd.grad(outputs=v_pred, inputs=t, grad_outputs=torch.ones_like(v_pred),
                                create_graph=True, retain_graph=True)[0]
        
        # Second derivatives
        u_xx = torch.autograd.grad(outputs=u_x, inputs=x, grad_outputs=torch.ones_like(u_x),
                                 create_graph=True, retain_graph=True)[0]
        u_yy = torch.autograd.grad(outputs=u_y, inputs=y, grad_outputs=torch.ones_like(u_y),
                                 create_graph=True, retain_graph=True)[0]
        
        v_xx = torch.autograd.grad(outputs=v_x, inputs=x, grad_outputs=torch.ones_like(v_x),
                                 create_graph=True, retain_graph=True)[0]
        v_yy = torch.autograd.grad(outputs=v_y, inputs=y, grad_outputs=torch.ones_like(v_y),
                                 create_graph=True, retain_graph=True)[0]
        
        # Get current Reynolds number
        Re = self.get_reynolds_number()
        
        # Compute residuals
        e1 = u_x + u_y  # Continuity equation
        e2 = u_t + (u_pred * u_x + v_pred * u_y) - (1/Re) * (u_xx + u_yy)  # x-momentum
        e3 = v_t + (u_pred * v_x + v_pred * v_y) - (1/Re) * (v_xx + v_yy)  # y-momentum
        
        # Combine losses (you can adjust the weights if needed)
        physics_loss = torch.mean(e1**2 + e2**2 + e3**2)
        
        return physics_loss