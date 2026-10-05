import torch
import torch.nn as nn
import torch.optim as optim

class PINN(nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden_layers = nn.Sequential(
            nn.Linear(1, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, t):
        return self.hidden_layers(t)
    
def damped_harmonic_oscillator_loss(model, t, m, c, k):
    x = model(t)
    
    x_t = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    x_tt = torch.autograd.grad(x_t, t, torch.ones_like(x_t), create_graph=True)[0]
    
    residual = m * x_tt + c * x_t + k * x
    
    return torch.mean(residual**2)


# Parameters
m = 1.0  # mass
c = 0.1  # damping coefficient
k = 2.0  # stiffness
epochs = 10000
learning_rate = 1e-3

# Initialize the model and optimizer
model = PINN()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    optimizer.zero_grad()
    
    # Randomly sample points from the domain
    t = torch.rand(100, 1, requires_grad=True)
    
    loss = damped_harmonic_oscillator_loss(model, t, m, c, k)
    loss.backward()
    optimizer.step()
    
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

print('Training complete.')

import numpy as np
import matplotlib.pyplot as plt

t_vals = np.linspace(0, 10, 100)
t_tensor = torch.tensor(t_vals, dtype=torch.float32).reshape(-1, 1)

with torch.no_grad():
    x_pred = model(t_tensor).numpy()

plt.plot(t_vals, x_pred, label='Predicted x(t)')
plt.xlabel('Time t')
plt.ylabel('Displacement x(t)')
plt.title('Predicted Displacement for Damped Harmonic Oscillator')
plt.legend()
plt.show()
