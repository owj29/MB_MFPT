import numpy as np
import torch
from torch import nn, optim
import matplotlib.pyplot as plt

#spib encoder
class SPIB_encoder(nn.Module):
    def __init__(self, input_dim, latent_dim=3, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        return self.net(x)

# spib training loop
def train_spib(trajectories, lag_steps, latent_dim=3, max_epochs=20, lr=1e-3):
    T, S, P, dim = trajectories.shape
    D = P * dim
    flat = trajectories.reshape(T * S, D)

    # Normalize coordinates
    mean = flat.mean(axis=0, keepdims=True)
    std  = flat.std(axis=0, keepdims=True) + 1e-6
    flat_norm = (flat - mean) / std

    # make training pairs
    X_t  = flat_norm[:-lag_steps]
    X_tp = flat_norm[lag_steps:]

    X_t  = torch.tensor(X_t, dtype=torch.float32)
    X_tp = torch.tensor(X_tp, dtype=torch.float32)

    # use the spib model
    encoder = SPIB_encoder(input_dim=D, latent_dim=latent_dim)
    predictor = nn.Linear(latent_dim, latent_dim)

    params = list(encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(params, lr=lr)
    loss_fn = nn.MSELoss()

    #training loop
    for epoch in range(max_epochs):
        optimizer.zero_grad()

        z_t = encoder(X_t)
        z_tp_pred = predictor(z_t)

        with torch.no_grad():
            z_tp = encoder(X_tp)

        loss = loss_fn(z_tp_pred, z_tp)
        loss.backward()
        optimizer.step()

        if epoch % 5 == 0:
            print(f"Epoch {epoch}, SPIB loss: {loss.item():.6f}")

    return encoder, mean, std


# build the MSM
def build_MSM(trajectories, n_bins, lag_time, dt, latent_dim):
    """
    trajectories: (T, S, P, dim)
    bins latent space (not raw coords)
    """
    T, S, P, dim = trajectories.shape
    D = P * dim
    lag_steps = int(lag_time / dt)

    # train SPIB
    encoder, mean, std = train_spib(
        trajectories,
        lag_steps=lag_steps,
        latent_dim=latent_dim,
        max_epochs=40
    )
    encoder.eval()

    #encode all the frames into the SPIB latent space
    flat = trajectories.reshape(T * S, D)
    flat_norm = (flat - mean) / std

    flat_norm = torch.tensor(flat_norm, dtype=torch.float32)
    with torch.no_grad():
        Z = encoder(flat_norm).cpu().numpy()  # (T*S, latent_dim)

    Z = Z.reshape(T, S, latent_dim)

    # bin in latent space
    bin_edges = []
    for d in range(latent_dim):
        lo = Z[..., d].min()
        hi = Z[..., d].max()
        bin_edges.append(np.linspace(lo, hi, n_bins + 1))

    multipliers = (n_bins ** np.arange(latent_dim)[::-1]).astype(int)
    n_states = n_bins ** latent_dim
    
    # counts to build the MSM
    all_start = []
    all_final = []

    for s in range(S):
        ztraj = Z[:, s, :]
        bins = np.zeros((T, latent_dim), dtype=int)
        valid = np.ones(T, dtype=bool)

        for d in range(latent_dim):
            bd = np.digitize(ztraj[:, d], bin_edges[d]) - 1
            bins[:, d] = bd
            valid &= (bd >= 0) & (bd < n_bins)

            pair_valid = valid[:-lag_steps] & valid[lag_steps:]

        start_bins = bins[:-lag_steps][pair_valid]
        final_bins = bins[lag_steps:][pair_valid]

        start_idx = (start_bins * multipliers).sum(axis=1)
        final_idx = (final_bins * multipliers).sum(axis=1)

        all_start.append(start_idx)
        all_final.append(final_idx)

    all_start = np.concatenate(all_start)
    all_final = np.concatenate(all_final)
    print(f"Collected transition pairs: {all_start.size}")

    edges = np.arange(n_states + 1)
    transitions, _, _ = np.histogram2d(all_start, all_final, bins=(edges, edges))

    T = transitions / np.maximum(transitions.sum(axis=1, keepdims=True), 1e-10)

    return T, encoder


#make some random test trajectories
def generate_random_trajectories(steps, write_every, num_simulations,
                                 num_particles=2, dimension=2,
                                 seed=None):
    """
    Generate a random trajectory tensor with shape:
    (num_time_steps, num_simulations, num_particles, dimension)

    Parameters
    ----------
    steps : int
        Total number of MD steps (not counting step 0).
    write_every : int
        Interval between recorded frames.
    num_simulations : int
        Number of parallel simulations.
    num_particles : int
        Number of particles per simulation (default: 2)
    dimension : int
        Spatial dimension (default: 2)
    seed : int or None
        Random seed for reproducibility

    Returns
    -------
    positions : np.ndarray
        Array of shape (T, S, P, D)
    """

    if seed is not None:
        np.random.seed(seed)

    # Number of saved time steps (including t=0)
    num_time_steps = steps // write_every + 1

    # Generate random positions
    positions = np.random.randn(num_time_steps,
                                num_simulations,
                                num_particles,
                                dimension)

    return positions


positions = generate_random_trajectories(
    steps=1000,
    write_every=10,
    num_simulations=5,   # for example
    num_particles=6,
    dimension=3,
    seed=1
)

msm, _ = build_MSM(positions, 5, 0.1, 0.05, 2)
plt.figure(figsize=(6, 5))
plt.imshow(msm, cmap='viridis', interpolation='nearest')
plt.colorbar(label="Probability")
plt.title('thing')
plt.xlabel("To state")
plt.ylabel("From state")

# Ensure equal aspect for square matrix
plt.gca().set_aspect('equal', adjustable='box')

plt.tight_layout()
plt.show()




