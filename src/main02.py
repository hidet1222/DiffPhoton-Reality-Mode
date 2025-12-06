import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os
import time

# ==========================================
# ⚙️ 設定: カリキュラム学習 (Curriculum Learning)
# ==========================================
N = 16
DAC_BITS = 12
CROSSTALK_LEVEL = 0.15
PHASE_ERROR_STD = 0.15
STEPS = 20000          # ステップ数を増やしてじっくりやる
STE_START_STEP = 5000  # ★重要: 最初の5000歩は量子化しない！
# ==========================================

def create_curriculum_engine(size, crosstalk_val):
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

    def init_fabrication_errors(key):
        return jax.random.normal(key, shape=(total_params,)) * PHASE_ERROR_STD

    # STE (Straight-Through Estimator)
    def quantize_ste(x, bits):
        scale = 2**bits - 1
        x_norm = (jnp.tanh(x) + 1.0) / 2.0
        x_int = jnp.round(x_norm * scale)
        x_quant_norm = x_int / scale
        x_quant = x_quant_norm * 2.0 - 1.0
        return x + jax.lax.stop_gradient(x_quant - x)

    # 連続値モード (カリキュラム初期用)
    def continuous_activation(x):
        # tanhだけかけて範囲を制限
        return jnp.tanh(x)

    def apply_crosstalk(voltages):
        if crosstalk_val == 0.0: return voltages
        s = voltages.shape[0]
        leak = jnp.eye(s, k=1) * crosstalk_val + jnp.eye(s, k=-1) * crosstalk_val
        return jnp.dot(jnp.eye(s) + leak, voltages)

    def phase_shifter(voltage_param, error, use_ste):
        # ★カリキュラム分岐: STEを使うか、連続値を使うか
        v_eff = jax.lax.cond(
            use_ste,
            lambda v: quantize_ste(v, DAC_BITS),     # Phase 2: Quantized
            lambda v: continuous_activation(v),      # Phase 1: Continuous
            voltage_param
        )
        phi = (v_eff * jnp.pi) + error
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def mzi(v_theta, v_phi, e_theta, e_phi, use_ste):
        PS_phi = phase_shifter(v_phi, e_phi, use_ste)
        DC = directional_coupler()
        PS_theta = phase_shifter(v_theta, e_theta, use_ste)
        return jnp.dot(DC, jnp.dot(PS_theta, jnp.dot(DC, PS_phi)))

    @jit
    def simulate_mesh(params, static_errors, use_ste):
        # Note: In this refined model, we apply crosstalk to the raw params first?
        # Ideally: Param -> (STE/Cont) -> Crosstalk -> Phase
        # Let's align with the Phase Shifter logic.
        # To make it simple: We apply STE/Cont inside phase_shifter.
        # But crosstalk happens physically. 
        # Correct Order: Param -> (STE/Cont) -> Crosstalk -> Phase
        
        # 1. Activation (Quantized or Continuous)
        p_act = jax.lax.cond(
            use_ste,
            lambda p: quantize_ste(p, DAC_BITS),
            lambda p: continuous_activation(p),
            params
        )
        
        # 2. Crosstalk
        p_leak = apply_crosstalk(p_act)
        
        U = jnp.eye(size, dtype=complex)
        p_idx = 0
        
        for layer in range(num_layers * 2):
            is_odd_layer = (layer % 2 == 1)
            start_idx = 1 if is_odd_layer else 0
            for i in range(start_idx, size - 1, 2):
                theta_val = p_leak[p_idx];   e_theta = static_errors[p_idx]
                phi_val   = p_leak[p_idx+1]; e_phi   = static_errors[p_idx+1]
                p_idx += 2
                
                # 直接計算
                phi_t = (theta_val * jnp.pi) + e_theta
                ps_t = jnp.array([[jnp.exp(1j * phi_t), 0], [0, 1.0 + 0j]])
                
                phi_p = (phi_val * jnp.pi) + e_phi
                ps_p = jnp.array([[jnp.exp(1j * phi_p), 0], [0, 1.0 + 0j]])
                
                DC = directional_coupler()
                m = jnp.dot(DC, jnp.dot(ps_t, jnp.dot(DC, ps_p)))
                
                slice_U = jax.lax.dynamic_slice(U, (i, 0), (2, size))
                new_slice = jnp.dot(m, slice_U)
                U = jax.lax.dynamic_update_slice(U, new_slice, (i, 0))
        
        out_phases = p_leak[p_idx : p_idx + size]
        out_errors = static_errors[p_idx : p_idx + size]
        phase_mat = jnp.diag(jnp.exp(1j * (out_phases * jnp.pi + out_errors)))
        return jnp.dot(phase_mat, U)

    return simulate_mesh, init_fabrication_errors, total_params

def run_simulation():
    print(f"DiffPhoton: CURRICULUM VICTORY (N={N})")
    print(f"   Strategy: Warmup (Float) -> Hard (12-bit STE)")
    
    mesh_fn, error_gen_fn, num_params = create_curriculum_engine(N, CROSSTALK_LEVEL)
    
    key = jax.random.PRNGKey(42)
    target_mat_key, train_key, fab_key = jax.random.split(key, 3)
    
    random_mat = jax.random.normal(target_mat_key, (N, N)) + 1j * jax.random.normal(target_mat_key, (N, N))
    target_U, _ = jnp.linalg.qr(random_mat)
    static_errors = error_gen_fn(fab_key)
    
    params = jax.random.uniform(train_key, shape=(num_params,), minval=-1.0, maxval=1.0)
    
    # Optimizer with Gradient Clipping (勾配爆発を防ぐ)
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, 
        peak_value=0.002, # 学習率をかなり下げる (N=64は繊細)
        warmup_steps=2000, 
        decay_steps=STEPS, 
        end_value=1e-5 
    )
    # Clip by global norm: 勾配が大きすぎたらカットする
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(learning_rate=schedule)
    )
    opt_state = optimizer.init(params)

    @jit
    def loss_fn(p, use_ste_flag):
        U_est = mesh_fn(p, static_errors, use_ste_flag)
        return jnp.sum(jnp.abs(U_est - target_U)**2)

    print(f"   Training started (Steps={STEPS})...")
    loss_history = []

    for step in range(STEPS):
        # カリキュラム制御: 最初の5000歩はSTEを使わない
        use_ste = (step >= STE_START_STEP)
        
        val, grads = value_and_grad(loss_fn)(params, use_ste)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(val)
        
        if step % 1000 == 0:
            mode = "Quantized" if use_ste else "Continuous"
            print(f"   Step {step:05d}: Loss={val:.4f} [{mode}]")

    # ==========================================
    # Final Validation
    # ==========================================
    print(f"\nFinal Validation (Real Chip Mode)...")
    
    # 完全に量子化モードで評価
    final_U_real = mesh_fn(params, static_errors, True)
    loss_real = jnp.sum(jnp.abs(final_U_real - target_U)**2)
    
    print(f"   Real Loss: {loss_real:.6f}")
    
    if loss_real < 1.0:
        print("   🏆 VICTORY! Curriculum Learning worked.")
    else:
        print("   ⚠️ Optimization failed.")

    # Visualize
    if not os.path.exists('output'): os.makedirs('output')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(loss_history)
    # STE開始地点に縦線を入れる
    axes[0].axvline(x=STE_START_STEP, color='r', linestyle='--', label='STE Start')
    axes[0].legend()
    axes[0].set_yscale('log'); axes[0].set_title('Curriculum Training')
    
    axes[1].imshow(jnp.abs(target_U), cmap='magma'); axes[1].set_title('Target')
    axes[2].imshow(jnp.abs(final_U_real), cmap='magma'); axes[2].set_title(f'Result\nLoss={loss_real:.4f}')
    
    plt.savefig('output/curriculum_victory.png')
    print("Evidence: output/curriculum_victory.png")

if __name__ == "__main__":
    run_simulation()