import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("SGD vs SGF: Coefficient Path Evolution")

# Sidebar
with st.sidebar:
    st.header("Controls")
    epsilon = st.slider(
        "Learning Rate (ε)",
        min_value=0.000001,
        max_value=0.8,
        value=0.00001,
        step=0.00001,
        format="%.7f"
    )
    batch_size = st.slider("Batch Size", 1, 100, 32)
    steps = st.slider("Steps", 10, 300, 100, step=10)
    speed = st.slider("Animation Speed (sec/frame)", 0.01, 0.5, 0.05, 0.01)
    start = st.button("Start Animation")
    reset = st.button("Reset")

if "anim_step" not in st.session_state or reset:
    st.session_state.anim_step = 0
    st.session_state.running = False

if start:
    st.session_state.running = True
    st.session_state.anim_step = 0

n, p, sigma = 500, 5, 0.5
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
        Σ_hat = np.cov(grads.T)
        
        try:
            Σ_sqrt = np.linalg.cholesky(Σ_hat)
        except np.linalg.LinAlgError:
            Σ_sqrt = np.linalg.svd(Σ_hat)[0]
            
        z_k = np.random.randn(p)
        beta += epsilon * full_grad + epsilon * (Σ_sqrt @ z_k)
        beta_hist.append(beta.copy())
        
    return np.array(beta_hist)

if st.session_state.running:
    beta_0 = np.random.randn(p)
    sgd_path = run_sgd(X, y, beta_0, epsilon, batch_size, steps)
    sgf_path = run_sgf_em(X, y, beta_0, epsilon, batch_size, steps)

    placeholder = st.empty()
    progress_bar = st.progress(0)
    step_info = st.empty()

    for i in range(st.session_state.anim_step, steps):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

        # SGD path
        for j in range(p):
            ax1.plot(range(i+1), sgd_path[:i+1, j], label=f"β{j}")
        ax1.set_title("SGD Coefficient Paths")
        ax1.set_xlabel("Steps")
        ax1.set_ylabel("β value")
        ax1.grid(True)
        ax1.legend(ncol=2, fontsize=8)

        # SGF path
        for j in range(p):
            ax2.plot(range(i+1), sgf_path[:i+1, j], label=f"β{j}")
        ax2.set_title("SGF Coefficient Paths")
        ax2.set_xlabel("Steps")
        ax2.grid(True)
        ax2.legend(ncol=2, fontsize=8)

        fig.suptitle(f"Coefficient Evolution at Step {i+1}", fontsize=14)
        placeholder.pyplot(fig)
        plt.close(fig)

        progress_bar.progress((i+1)/steps)
        step_info.text(f"Step {i+1} of {steps}")
        st.session_state.anim_step = i + 1
        time.sleep(speed)

    st.session_state.running = False
    progress_bar.empty()
    step_info.empty()
else:
    st.info("Set parameters and click 'Start Animation'.")
