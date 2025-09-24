class PPOHealthCallback:
    def on_compute_ratios(self, ratios):
        import torch
        r_std = torch.std(ratios).item() if hasattr(ratios, "std") else 0.0
        r_mean = torch.mean(ratios).item() if hasattr(ratios, "mean") else 1.0
        print(f"[PPOHealth] ratio_mean={r_mean:.3f} ratio_std={r_std:.3f}")
        if r_std < 1e-4:
            raise RuntimeError("Degenerate PPO ratios (std ~ 0). Check old/new logprob plumbing.")
