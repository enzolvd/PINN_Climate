import torch
import torch.nn as nn

class ResBlock(torch.nn.Module):
    """Define a Res connection block for encoder decoder"""
    def __init__(self, in_channel):        
        super().__init__()
        
        self.conv_block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels = in_channel,
                            out_channels = in_channel,
                            kernel_size=3,
                            stride=1,
                            padding='same'),
                    torch.nn.BatchNorm2d(in_channel),
                    torch.nn.ReLU(),
                    torch.nn.Conv2d(in_channels = in_channel,
                            out_channels = in_channel,
                            kernel_size=3,
                            stride=1,
                            padding='same'),
                    torch.nn.BatchNorm2d(in_channel)                         
        )
        self.act = torch.nn.ReLU()

    def forward(self, input):
        out = self.conv_block(input)
        out += input 
        out = self.act(out)
        return out
    
class MeteoEncoder(nn.Module):
    def __init__(self, in_channels=2, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh(),
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.cnn(x)

class MaskEncoder(nn.Module):
    def __init__(self, n_masks=3, hidden_dim=64):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(n_masks, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.Tanh(),
            ResBlock(hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
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
        coords = torch.stack([x, y, t], dim=-1)
        processed_coords = self.coord_net(coords)
        return processed_coords.permute(0, 3, 1, 2)

class ClimatePINN(nn.Module):
    def __init__(self, hidden_dim=64, initial_re=100.0, device='cpu'):
        super().__init__()
        self.device = device
        
        # Initialize components
        self.meteo_encoder = MeteoEncoder(in_channels=2, hidden_dim=hidden_dim)
        self.mask_encoder = MaskEncoder(hidden_dim=hidden_dim)
        self.coord_processor = CoordProcessor(hidden_dim=hidden_dim)
        
        # Combine features
        self.feature_combiner = nn.Sequential(
            nn.Conv2d(hidden_dim * 3, hidden_dim * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.Tanh(),
            ResBlock(hidden_dim*2),
            nn.Conv2d(hidden_dim * 2, 3, kernel_size=3, padding=1)
        )
        
        # Learnable Reynolds number parameter
        self.log_re = nn.Parameter(torch.log(torch.tensor(initial_re, device=device)))
        
        # Loss function
        self.MSE = nn.MSELoss()
        
        # Move model to device
        self.to(device)
        self.re_momentum = 0.7
        self.previous_re = None

    def get_reynolds_number(self):
        clamped_log_re = torch.clamp(self.log_re, 
                                    min=torch.log(torch.tensor(50.0, device=self.device)), 
                                    max=torch.log(torch.tensor(1e5, device=self.device)))
        current_re = torch.exp(clamped_log_re)
        
        if self.previous_re is None:
            self.previous_re = current_re
        
        smoothed_re = self.re_momentum * self.previous_re + (1 - self.re_momentum) * current_re
        self.previous_re = smoothed_re.detach()
        
        return smoothed_re
    
    def forward(self, meteo_inputs, masks, coords, compute_physics=True):     
        # Get original coordinates
        x = coords[0].requires_grad_(True)  # [32, 32, 64]
        y = coords[1].requires_grad_(True)  # [32, 32, 64]
        t = coords[2].requires_grad_(True)  # [32]
        
        # Expand time to match spatial dimensions
        t_expanded = t.view(-1, 1, 1).expand(-1, x.shape[1], x.shape[2])  # [32, 32, 64]
        
        # Multi head encoder
        meteo_features = self.meteo_encoder(meteo_inputs)
        mask_features = self.mask_encoder(masks)
        coord_features = self.coord_processor(x, y, t_expanded)  # Use expanded time here
        
        # Combine
        combined = torch.cat([
            meteo_features,
            mask_features,
            coord_features
        ], dim=1) 

        # Decode
        outputs = self.feature_combiner(combined)
        
        predictions = {
            'output': outputs
        }

        if compute_physics:
            u_pred = outputs[:, 1]  # u10
            v_pred = outputs[:, 2]  # v10

            # First derivatives (using t_expanded for time derivatives)
            u_x = torch.autograd.grad(u_pred, x, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            u_y = torch.autograd.grad(u_pred, y, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            u_t = torch.autograd.grad(u_pred, t_expanded, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            
            v_x = torch.autograd.grad(v_pred, x, grad_outputs=torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]
            v_y = torch.autograd.grad(v_pred, y, grad_outputs=torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]
            v_t = torch.autograd.grad(v_pred, t_expanded, grad_outputs=torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]
            
            # Second derivatives
            u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_pred), create_graph=True, retain_graph=True)[0]
            
            v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]
            v_yy = torch.autograd.grad(v_y, y, grad_outputs=torch.ones_like(v_pred), create_graph=True, retain_graph=True)[0]
            
            # Get Reynolds number
            Re = self.get_reynolds_number()
            
            # Compute residuals
            e1 = u_x + u_y  # Continuity equation
            e2 = u_t + (u_pred * u_x + v_pred * u_y) - (1/Re) * (u_xx + u_yy)  # x-momentum
            e3 = v_t + (u_pred * v_x + v_pred * v_y) - (1/Re) * (v_xx + v_yy)  # y-momentum
            
            physics_loss = {
                            'e1': self.MSE(e1,torch.zeros_like(e1)), 
                            'e2': self.MSE(e2,torch.zeros_like(e2)), 
                            'e3': self.MSE(e3,torch.zeros_like(e3))
                            }
            predictions['physics_loss'] = physics_loss
            
        return predictions