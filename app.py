import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time
from matplotlib import cm

st.title("SGF vs SGD: Importance of discretization with 2D Trajectories")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    epsilon = st.slider(
    "Learning Rate (ε)",
    min_value=0.000001,
    max_value=0.8,
    value=0.00001,
    step=0.00001,
    format="%.7f",
    help="Learning rate controls step size in gradient updates."
)
    batch_size = st.slider("Batch Size (m)", min_value=1, max_value=100, value=32, step=1)
    steps = st.slider("Steps", min_value=10, max_value=300, value=100, step=10)
    speed = st.slider("Animation Speed (sec per frame)", 0.01, 0.5, 0.05, 0.01,
                      help="Lower value = faster animation")
    start = st.button("Start Animation")
    reset = st.button("Reset")

# Internal state to control animation progress
if "anim_step" not in st.session_state or reset:
    st.session_state.anim_step = 0
    st.session_state.running = False

if start:
    st.session_state.running = True
    st.session_state.anim_step = 0

# Generate synthetic data (cache if needed)
n, p, sigma = 500, 2, 0.5 # dimensions = 2 for visualization because then we are just taking the projections so the trajectories are not accurate for 2D projects. 
np.random.seed(42)
X = np.random.randn(n, p)
beta_star = np.random.randn(p)
y = X @ beta_star + sigma * np.random.randn(n)

def run_sgd(X, y, beta_init, lr, batch_size, steps):
    n, p = X.shape
    beta = beta_init.copy()
    beta_hist = []
    for _ in range(steps):
        idx = np.random.choice(n, batch_size, replace=False)
        X_batch, y_batch = X[idx], y[idx]
        grad = X_batch.T @ (y_batch - X_batch @ beta) / batch_size
        beta += lr * grad
        beta_hist.append(beta.copy())
    return np.array(beta_hist)

def run_sgf_em(X, y, beta_init, epsilon, batch_size, steps):
    n, p = X.shape
    beta = beta_init.copy()
    beta_hist = []
    for _ in range(steps):
        full_grad = X.T @ (y - X @ beta) / n
        batch_idx = np.random.choice(n, batch_size, replace=False)
        X_batch = X[batch_idx]
        y_batch = y[batch_idx]
        grads = np.array([X_batch[i] * (y_batch[i] - X_batch[i] @ beta) for i in range(batch_size)])
        Σ_hat = np.cov(grads.T) + 1e-6 * np.eye(p)
        try:
            Σ_sqrt = np.linalg.cholesky(Σ_hat)
        except np.linalg.LinAlgError:
            Σ_sqrt = np.linalg.svd(Σ_hat)[0]
        z_k = np.random.randn(p)
        beta += epsilon * full_grad + epsilon * (Σ_sqrt @ z_k)
        beta_hist.append(beta.copy())
    return np.array(beta_hist)

if st.session_state.running:
    
    beta_0 = np.random.randn(p) # Random initial point
    sgd_path = run_sgd(X, y, beta_0, epsilon, batch_size, steps)
    sgf_path = run_sgf_em(X, y, beta_0, epsilon, batch_size, steps)

    # Prepare 2D loss contour as before
    beta_range = 3
    grid_points = 50
    b1 = np.linspace(-beta_range, beta_range, grid_points)
    b2 = np.linspace(-beta_range, beta_range, grid_points)
    B1, B2 = np.meshgrid(b1, b2)

    def loss_on_grid(b1_, b2_):
        beta_full = np.zeros(p)
        beta_full[0] = b1_
        beta_full[1] = b2_
        res = y - X @ beta_full
        return 0.5 * np.mean(res**2)

    Z = np.array([[loss_on_grid(x, y) for x in b1] for y in b2])

    sgd_proj = sgd_path[:, :2]
    sgf_proj = sgf_path[:, :2]
    beta_star_proj = beta_star[:2]

    placeholder_err = st.empty()
    placeholder_traj = st.empty()
    progress_bar = st.progress(0)
    step_info = st.empty()
    
    for i in range(st.session_state.anim_step, steps):
        sgd_err = np.linalg.norm(sgd_path[:i+1] - beta_star, axis=1)
        sgf_err = np.linalg.norm(sgf_path[:i+1] - beta_star, axis=1)

        # Error plot
        fig_err, ax_err = plt.subplots(figsize=(7, 4))
        ax_err.plot(sgd_err, label="SGD", color="#d62728", linewidth=3)
        ax_err.plot(sgf_err, label="SGF (Euler-Maruyama)", color="#1f77b4", linewidth=3)
        ax_err.set_xlabel("Steps")
        ax_err.set_ylabel(r"$\|\beta^{(t)} - \beta^*\|$")
        ax_err.set_title(f"Convergence to True β* (Step {i+1}) & ε={epsilon:.5f}")
        ax_err.legend()
        ax_err.grid(True)
        placeholder_err.pyplot(fig_err)
        plt.close(fig_err)

        # Trajectory plot
        fig_traj, ax_traj = plt.subplots(figsize=(7, 6))
        contour = ax_traj.contourf(B1, B2, Z, levels=30, cmap=cm.viridis)
        ax_traj.plot(sgd_proj[:i+1, 0], sgd_proj[:i+1, 1], label="SGD path", color="#ff7f0e", linewidth=2)
        ax_traj.plot(sgf_proj[:i+1, 0], sgf_proj[:i+1, 1], label="SGF path", color="#2ca02c", linewidth=2)
        ax_traj.scatter(*beta_star_proj, color="yellow", marker="*", s=150, label="True β*")
        ax_traj.set_xlabel("β[0]")
        ax_traj.set_ylabel("β[1]")
        ax_traj.set_title("Trajectory over Loss Contour")
        ax_traj.legend()
        ax_traj.grid(True)
        fig_traj.colorbar(contour, ax=ax_traj, label="Loss")
        placeholder_traj.pyplot(fig_traj)
        plt.close(fig_traj)

        progress_bar.progress((i+1)/steps)
        step_info.text(f"Step {i+1} of {steps}")

        st.session_state.anim_step = i + 1

        time.sleep(speed)

    st.session_state.running = False
    progress_bar.empty()
    step_info.empty()
else:
    st.info("Set parameters and click 'Start Animation' to begin.")
