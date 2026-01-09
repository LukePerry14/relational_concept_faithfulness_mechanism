import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

def visualize_robust_rbf():
    # Range: from -10 to +10 standard deviations (Gammas)
    z = torch.linspace(-10, 10, 1000)
    z.requires_grad = True
    
    # 1. Standard RBF (Gaussian)
    # Penalty = 0.5 * z^2
    penalty_rbf = 0.5 * z**2
    
    # 2. Robust RBF (Huber-style)
    # Beta = 1.0 (Transition at 1 sigma)
    beta = 1.0
    abs_z = torch.abs(z)
    penalty_robust = torch.where(
        abs_z < beta,
        0.5 * abs_z**2,                 # Quadratic
        beta * abs_z - (0.5 * beta**2)  # Linear
    )
    
    # 3. Calculate Gradients
    penalty_rbf.sum().backward(retain_graph=True)
    grad_rbf = z.grad.clone()
    z.grad.zero_()
    
    penalty_robust.sum().backward()
    grad_robust = z.grad.clone()
    
    # 4. Calculate Similarity Scores (e^-penalty)
    # This shows "Mass Contribution"
    sim_rbf = torch.exp(-penalty_rbf).detach()
    sim_robust = torch.exp(-penalty_robust).detach()
    
    # --- PLOTTING ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: The Penalty Landscape (The "Valley")
    ax1.plot(z.detach(), penalty_rbf.detach(), 'r--', label='Standard RBF (Gaussian)')
    ax1.plot(z.detach(), penalty_robust.detach(), 'b-', linewidth=2, label='Robust RBF (Huber)')
    ax1.set_title("1. The Penalty Landscape")
    ax1.set_xlabel("Z-Score (Distance / Gamma)")
    ax1.set_ylabel("Penalty")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: The Gradient (The "Push")
    ax2.plot(z.detach(), grad_rbf, 'r--', label='RBF Gradient (Explodes!)')
    ax2.plot(z.detach(), grad_robust, 'b-', linewidth=2, label='Robust Gradient (Constant)')
    ax2.set_title("2. The Gradient Signal")
    ax2.set_xlabel("Z-Score")
    ax2.set_ylabel("Gradient Strength")
    ax2.set_ylim(-5, 5)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: The Similarity Score (The "Mass")
    ax3.plot(z.detach(), sim_rbf, 'r--', label='RBF Mass')
    ax3.plot(z.detach(), sim_robust, 'b-', linewidth=2, label='Robust Mass')
    ax3.set_title("3. Mass Contribution (Similarity)")
    ax3.set_xlabel("Z-Score")
    ax3.set_ylabel("Similarity (0 to 1)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    visualize_robust_rbf()