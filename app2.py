import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("Brownian Motion Approximation")

# User controls
levels = st.slider("Number of Schauder levels", 1, 10, 5)
animation_speed = st.slider("Animation speed (seconds per frame)", 0.001, 1.0, 0.05)
if st.button("Generate New Brownian Path"):
    st.session_state['new_path'] = True
else:
    if 'new_path' not in st.session_state:
        st.session_state['new_path'] = True

# Time grid setup
N = 512
T = 1.0
t = np.linspace(0, T, N)
dt = t[1] - t[0]

# Schauder function definition
def schauder_function(k, n, t):
    left = k / 2**n
    mid = (2*k + 1) / 2**(n+1)
    right = (k + 1) / 2**n
    val = np.zeros_like(t)
    mask_rise = (t >= left) & (t < mid)
    val[mask_rise] = (t[mask_rise] - left) / (mid - left)
    mask_fall = (t >= mid) & (t < right)
    val[mask_fall] = (right - t[mask_fall]) / (right - mid)
    return val * 2**(n/2)

@st.cache_data
def build_basis(levels, t):
    basis_funcs = []
    for n in range(levels):
        for k in range(2**n):
            basis_funcs.append(schauder_function(k, n, t))
    return np.array(basis_funcs).T

# Generate or reuse Brownian path
if st.session_state['new_path']:
    dW = np.random.randn(N) * np.sqrt(dt)
    W = np.cumsum(dW)
    W -= W[0]
    st.session_state['W'] = W
    st.session_state['new_path'] = False
else:
    W = st.session_state['W']

# Build basis matrix
B = build_basis(levels, t)

# Solve least squares for coefficients
coeffs, _, _, _ = np.linalg.lstsq(B, W, rcond=None)

# Animation
placeholder = st.empty()
W_approx = np.zeros_like(t)

for i in range(len(coeffs)):
    W_approx += coeffs[i] * B[:, i]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, W, label="True Brownian Motion", color="black", linewidth=1)
    ax.plot(t, W_approx, label=f"Schauder Approximation (terms={i+1})", color="purple", linewidth=2)
    ax.set_ylim(W.min()*1.1, W.max()*1.1)
    ax.set_xlabel("Time")
    ax.set_ylabel("W(t)")
    ax.set_title("Schauder basis approximation of Brownian Motion")
    ax.legend()
    ax.grid(True)

    placeholder.pyplot(fig)
    plt.close(fig)
    time.sleep(animation_speed)

