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
# 設定: 最終決戦仕様 (Single Wavelength)
# ==========================================
N = 64                # 大規模シミュレーション (N=64)
DAC_BITS = 12         # 本番環境 (4096段階)
TRAIN_NOISE_BITS = 10 # 学習時ノイズ (10bit相当のノイズで特訓)
STEPS = 6000          # じっくり学習
TARGET_WL = 1550e-9   # 単一波長に集中！
# ==========================================

def create_engine(size):
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

    # 単一波長なので分散項は固定(=1.0)
    # 物理モデルとして波長項は残すが、今回は固定値で計算
    def phase_shifter(voltage):
        phi = voltage * jnp.pi 
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])

    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])

    def mzi(v_theta, v_phi):
        PS_phi = phase_shifter(v_phi)
        DC = directional_coupler()
        PS_theta = phase_shifter(v_theta)
        return jnp.dot(DC, jnp.dot(PS_theta, jnp.dot(DC, PS_phi)))

    @jit
    def simulate_mesh(params):
        U = jnp.eye(size, dtype=complex)
        p_idx = 0
        for layer in range(num_layers * 2):
            is_odd_layer = (layer % 2 == 1)
            start_idx = 1 if is_odd_layer else 0
            for i in range(start_idx, size - 1, 2):
                theta = params[p_idx]; phi = params[p_idx+1]
                p_idx += 2
                m = mzi(theta, phi)
                slice_U = jax.lax.dynamic_slice(U, (i, 0), (2, size))
                new_slice = jnp.dot(m, slice_U)
                U = jax.lax.dynamic_update_slice(U, new_slice, (i, 0))
        
        out_phases = params[p_idx : p_idx + size]
        phase_mat = jnp.diag(jnp.exp(1j * out_phases * jnp.pi))
        return jnp.dot(phase_mat, U)

    return simulate_mesh, total_params

# ノイズ注入 (QAT用)
def inject_noise(params, key, bits):
    step = 2.0 / (2**bits)
    noise = jax.random.uniform(key, shape=params.shape, minval=-step/2, maxval=step/2)
    return params + noise

def run_simulation():
    print(f" DiffPhoton: Final Battle (N={N}, Single-WL, QAT)")
    print(f"   Target: Perfect inference with {DAC_BITS}-bit DAC constraints.")
    
    start_time = time.time()
    mesh_fn, num_params = create_engine(N)
    print(f"   Parameters to optimize: {num_params}")
    
    key = jax.random.PRNGKey(42)
    target_mat_key, train_key = jax.random.split(key)
    
    # ターゲット行列 (ユニタリ)
    random_mat = jax.random.normal(target_mat_key, (N, N)) + 1j * jax.random.normal(target_mat_key, (N, N))
    target_U, _ = jnp.linalg.qr(random_mat)
    
    params = jax.random.uniform(train_key, shape=(num_params,), minval=-1.0, maxval=1.0)
    
    # スケジューラ: Warmupで初期崩壊を防ぎ、最後はCosine Decayで微調整
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0, peak_value=0.01, warmup_steps=500, decay_steps=STEPS, end_value=0.00005
    )
    optimizer = optax.adam(learning_rate=schedule)
    opt_state = optimizer.init(params)

    # ★ QAT Loss関数
    @jit
    def loss_fn(p, key):
        # 10bit相当のノイズを入れて学習 (本番12bitより少し厳しく)
        noisy_p = inject_noise(p, key, bits=TRAIN_NOISE_BITS)
        U_est = mesh_fn(noisy_p)
        return jnp.sum(jnp.abs(U_est - target_U)**2)

    print(f"   Training started with {TRAIN_NOISE_BITS}-bit noise injection...")
    loss_history = []

    for step in range(STEPS):
        iter_key = jax.random.fold_in(train_key, step)
        val, grads = value_and_grad(loss_fn)(params, iter_key)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(val)
        
        if step % 500 == 0:
            print(f"   Step {step:04d}: Loss = {val:.4f}")

    print(f" Training Done in {time.time()-start_time:.1f}s.")

    # ==========================================
    # ⚖️ 運命のリアリティチェック (12bit DAC)
    # ==========================================
    print(f"\n🔬 Reality Check (Simulating 12-bit DAC hardware)...")
    
    # 1. 理想状態 (Float64/32)
    final_U_ideal = mesh_fn(params)
    loss_ideal = jnp.sum(jnp.abs(final_U_ideal - target_U)**2)

    # 2. 量子化状態 (12bit Int DAC)
    DAC_LEVELS = 2**DAC_BITS
    # -1.0 ~ 1.0 を 0 ~ 4095 にマップして丸める
    params_clipped = jnp.clip(params, -1.0, 1.0)
    params_normalized = (params_clipped + 1.0) / 2.0
    params_quantized_int = jnp.round(params_normalized * (DAC_LEVELS - 1))
    params_quantized = (params_quantized_int / (DAC_LEVELS - 1)) * 2.0 - 1.0
    
    final_U_real = mesh_fn(params_quantized)
    loss_real = jnp.sum(jnp.abs(final_U_real - target_U)**2)
    
    print(f"   Ideal Loss (No Quantization) : {loss_ideal:.6f}")
    print(f"   Real Loss (12-bit DAC)       : {loss_real:.6f}")
    
    if loss_ideal > 0:
        ratio = loss_real / loss_ideal
        print(f"   Degradation Ratio            : {ratio:.2f}x")
    
    if loss_real < 1.0:
        print("   🎉 VICTORY! The design is robust enough for mass production.")
    else:
        print("   ⚠️ Warning: Still fragile. Need more layers or better optimization.")

    # ==========================================
    # 結果の可視化
    # ==========================================
    if not os.path.exists('output'): os.makedirs('output')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss Curve
    axes[0].plot(loss_history, color='tab:blue', alpha=0.8)
    axes[0].set_yscale('log')
    axes[0].set_title(f'QAT Training (N={N})\nNoise Level: {TRAIN_NOISE_BITS}-bit')
    axes[0].set_xlabel('Steps')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True, which="both", alpha=0.3)
    
    # 2. Target Matrix
    im1 = axes[1].imshow(jnp.abs(target_U), cmap='magma')
    axes[1].set_title('Target Matrix (Amplitude)')
    plt.colorbar(im1, ax=axes[1])
    
    # 3. Reproduced Matrix (Quantized)
    im2 = axes[2].imshow(jnp.abs(final_U_real), cmap='magma')
    axes[2].set_title(f'Real Chip Simulation (12-bit DAC)\nLoss={loss_real:.4f}')
    plt.colorbar(im2, ax=axes[2])
    
    output_path = 'output/single_wl_victory.png'
    plt.savefig(output_path, dpi=300)
    print(f"Evidence Saved: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_simulation()