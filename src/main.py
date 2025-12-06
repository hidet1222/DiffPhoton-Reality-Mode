import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os

# ==========================================
# 設定 (Configuration)
# ==========================================
<<<<<<< HEAD
N = 64 # 行列サイズ
CROSSTALK_LEVEL = 0.20   # crosstalk レベル
PHASE_ERROR_STD = 0.20   # 位相誤差の標準偏差
STEPS = 8000             # 最適化ステップ数
=======
N = 64  # 行列サイズ
CROSSTALK_LEVEL = 0.15   # 15%の漏れ
PHASE_ERROR_STD = 0.15   # 製造誤差
STEPS = 8000             # ステップ数を少し増やす
>>>>>>> 7c00c8ed29099987cd97160ddf3ecc00285e9aaf
# ==========================================


def create_scalable_engine(size, crosstalk, phase_error):
    num_layers = size 
    def count_params():
        p_count = 0
        for layer in range(num_layers * 2):
            is_odd_layer = (layer % 2 == 1)
            start_idx = 1 if is_odd_layer else 0
            for i in range(start_idx, size - 1, 2):
                p_count += 2
        p_count += size
        return p_count

    total_params = count_params()
    print(f"   Building {size}x{size} Mesh...")
    print(f"   -> Required Voltage Parameters: {total_params}")

    def init_fabrication_errors(key):
        return jax.random.normal(key, shape=(total_params,)) * phase_error

    def phase_shifter(voltage, error):
        phi = (voltage * jnp.pi) + error
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def mzi(v_theta, v_phi, e_theta, e_phi):
        PS_phi = phase_shifter(v_phi, e_phi)
        DC = directional_coupler()
        PS_theta = phase_shifter(v_theta, e_theta)
        return jnp.dot(DC, jnp.dot(PS_theta, jnp.dot(DC, PS_phi)))

    def apply_crosstalk(voltages):
        if crosstalk == 0.0: return voltages
        s = voltages.shape[0]
        leak = jnp.eye(s, k=1) * crosstalk + jnp.eye(s, k=-1) * crosstalk
        return jnp.dot(jnp.eye(s) + leak, voltages)

    @jit
    def simulate_mesh(params, static_errors):
        real_params = apply_crosstalk(params)
        U = jnp.eye(size, dtype=complex)
        p_idx = 0
        
        for layer in range(num_layers * 2):
            is_odd_layer = (layer % 2 == 1)
            start_idx = 1 if is_odd_layer else 0
            
            for i in range(start_idx, size - 1, 2):
                theta = real_params[p_idx];   e_theta = static_errors[p_idx]
                phi   = real_params[p_idx+1]; e_phi   = static_errors[p_idx+1]
                p_idx += 2
                
                m = mzi(theta, phi, e_theta, e_phi)
                slice_U = jax.lax.dynamic_slice(U, (i, 0), (2, size))
                new_slice = jnp.dot(m, slice_U)
                U = jax.lax.dynamic_update_slice(U, new_slice, (i, 0))
                
        out_phases = real_params[p_idx : p_idx + size]
        out_errors = static_errors[p_idx : p_idx + size]
        phase_mat = jnp.diag(jnp.exp(1j * (out_phases * jnp.pi + out_errors)))
        
        return jnp.dot(phase_mat, U)

    return simulate_mesh, init_fabrication_errors, total_params

def run_simulation():
    print(f"🚀 DiffPhoton: Large Scale Simulation (N={N}) with Scheduler")
    
    mesh_fn, error_gen_fn, num_params = create_scalable_engine(N, CROSSTALK_LEVEL, PHASE_ERROR_STD)
    
    key = jax.random.PRNGKey(42)
    random_mat = jax.random.normal(key, (N, N)) + 1j * jax.random.normal(key, (N, N))
    target_U, _ = jnp.linalg.qr(random_mat)
    
    fab_key, train_key = jax.random.split(key)
    static_errors = error_gen_fn(fab_key)

    @jit
    def loss_fn(params):
        current_U = mesh_fn(params, static_errors)
        diff = current_U - target_U
        return jnp.sum(jnp.abs(diff)**2)

    params = jax.random.uniform(train_key, shape=(num_params,), minval=-1.0, maxval=1.0)
    
    # ★ 修正ポイント: 学習率スケジューラを導入
    # 0.05 からスタートして、徐々に 0.001 まで下げる
    schedule = optax.cosine_decay_schedule(init_value=0.05, decay_steps=STEPS, alpha=0.02)
    optimizer = optax.adam(learning_rate=schedule)
    
    opt_state = optimizer.init(params)
    
    loss_history = []
    
    print(f"   Optimizing {num_params} parameters...", end="", flush=True)
    
    for i in range(STEPS):
        val, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(val)
        
        if i % (STEPS // 10) == 0:
            print(f"[{val:.3f}]", end="", flush=True)
            
    print(" Done!")
    print(f"   Final Loss: {loss_history[-1]:.8f}")

    # --- 可視化 ---
    final_U = mesh_fn(params, static_errors)
    
    if not os.path.exists('output'): os.makedirs('output')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss
    axes[0].plot(loss_history)
    axes[0].set_yscale('log')
    axes[0].set_title(f"Optimization History (N={N})")
    axes[0].set_xlabel("Steps")
    axes[0].set_ylabel("Matrix Distance")
    
    # 2. Target
    axes[1].imshow(jnp.abs(target_U), cmap='magma')
    axes[1].set_title("Target Matrix (Amplitude)")
    
    # 3. Result
    axes[2].imshow(jnp.abs(final_U), cmap='magma')
    axes[2].set_title(f"Reproduced Matrix\nLoss={loss_history[-1]:.1e}")
    
    output_path = f"output/result_{N}x{N}.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Image Saved: {os.path.abspath(output_path)}")

if __name__ == "__main__":
<<<<<<< HEAD
    run_simulation()
=======
    run_simulation()
>>>>>>> 7c00c8ed29099987cd97160ddf3ecc00285e9aaf
