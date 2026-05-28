import torch

# User-specified function f: R -> R
# Edit this function as needed.
def f(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x) + x**2


def monte_carlo_expectation(num_samples: int = 1_000_000, seed: int = 0) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # X ~ Uniform[0, 1]
    x = torch.rand(num_samples, device=device)

    # Monte Carlo estimate: E[f(X)] ≈ (1/N) sum_i f(X_i)
    estimate = f(x).mean()

    return estimate.item()


if __name__ == "__main__":
    estimate = monte_carlo_expectation(num_samples=1_000_000)
    print(f"Estimated E[f(X)] = {estimate:.8f}")