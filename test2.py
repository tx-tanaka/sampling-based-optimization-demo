import torch
import matplotlib.pyplot as plt

# Parameters
n = 2
alpha = 100.0
N = 100_000_000
M = 1_000  # number of X_tilde samples


def cost(X: torch.Tensor) -> torch.Tensor:
    x1 = X[:, 0]
    x2 = X[:, 1]

    y1 = 20 * x1 - 10
    y2 = 20 * x2 - 10

    return (
        4 * y1**4
        + 4 * y1**3 * y2
        - 7 * y1**2 * y2**2
        - 2 * y1 * y2**3
        + 10 * y2**4
    )


def main(seed: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Step 1: Generate X(i) ~ Unif([0,1]^2)
    X = torch.rand((N, n), device=device)

    # Step 2: Compute desirability scores
    C = cost(X)
    r = torch.exp(-C / alpha)
    r_sum = torch.sum(r)

    # Step 3 modified: Generate 1000 samples d ~ Unif([0, r_sum])
    d = torch.rand(M, device=device) * r_sum

    # Step 4 modified: Find corresponding indices for all d's
    cumulative_r = torch.cumsum(r, dim=0)
    idx = torch.searchsorted(cumulative_r, d)

    # Step 5 modified: Select 1000 samples X_tilde
    X_tilde = X[idx]

    print(f"r_sum = {r_sum.item():.6e}")
    print(f"First 5 selected samples:\n{X_tilde[:5].detach().cpu().numpy()}")

    # Move selected samples to CPU for plotting
    X_tilde_cpu = X_tilde.detach().cpu().numpy()

    # Contour plot of C over [0,1] x [0,1]
    grid_size = 400
    x1_grid = torch.linspace(0, 1, grid_size)
    x2_grid = torch.linspace(0, 1, grid_size)

    X1, X2 = torch.meshgrid(x1_grid, x2_grid, indexing="ij")
    grid_points = torch.stack([X1.reshape(-1), X2.reshape(-1)], dim=1).to(device)

    C_grid = cost(grid_points).reshape(grid_size, grid_size)
    C_grid_cpu = C_grid.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    contour = plt.contourf(
        X1.numpy(),
        X2.numpy(),
        C_grid_cpu,
        levels=50,
    )
    plt.colorbar(contour, label="C(x1, x2)")

    plt.scatter(
        X_tilde_cpu[:, 0],
        X_tilde_cpu[:, 1],
        s=8,
        c="red",
        alpha=0.7,
        label=r"$\tilde{X}$ samples",
    )

    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Contour plot of C with sampled X_tilde points")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main(seed=0)