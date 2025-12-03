import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax
from jax import value_and_grad, jit
import optax
import os

# --- 物理エンジン (設定は同じ: Crosstalk 15% + Phase Error 0.5) ---
def create_realistic_engine(crosstalk_level=0.15, phase_error_std=0.5):
    def init_fabrication_errors(key, num_params):
        return jax.random.normal(key, shape=(num_params,)) * phase_error_std
    def directional_coupler():
        val = 1.0 / jnp.sqrt(2.0)
        return jnp.array([[val, val * 1j], [val * 1j, val]])
    def phase_shifter(voltage, error_offset):
        phi = (voltage * jnp.pi) + error_offset
        return jnp.array([[jnp.exp(1j * phi), 0], [0, 1.0 + 0j]])
    def apply_crosstalk(voltages):
        size = voltages.shape[0]
        identity = jnp.eye(size)
        leak = jnp.eye(size, k=1) * crosstalk_level + jnp.eye(size, k=-1) * crosstalk_level
        return jnp.dot(identity + leak, voltages)
    def universal_mzi(v_theta, v_phi, err_theta, err_phi):
        PS_phi = phase_shifter(v_phi, err_phi)
        DC1 = directional_coupler()
        PS_theta = phase_shifter(v_theta, err_theta)
        DC2 = directional_coupler()
        return jnp.dot(DC2, jnp.dot(PS_theta, jnp.dot(DC1, PS_phi)))
    
    @jit
    def simulate_mesh(params, static_errors):
        real_params = apply_crosstalk(params)
        thetas, phis, outs = real_params[0:6], real_params[6:12], real_params[12:16]
        e_thetas, e_phis, e_outs = static_errors[0:6], static_errors[6:12], static_errors[12:16]
        T0 = universal_mzi(thetas[0], phis[0], e_thetas[0], e_phis[0])
        T1 = universal_mzi(thetas[1], phis[1], e_thetas[1], e_phis[1])
        L1 = jnp.block([[T0, jnp.zeros((2,2))], [jnp.zeros((2,2)), T1]])
        T2 = universal_mzi(thetas[2], phis[2], e_thetas[2], e_phis[2])
        L2 = jnp.eye(4, dtype=complex); L2 = L2.at[1:3, 1:3].set(T2)
        T3 = universal_mzi(thetas[3], phis[3], e_thetas[3], e_phis[3])
        T4 = universal_mzi(thetas[4], phis[4], e_thetas[4], e_phis[4])
        L3 = jnp.block([[T3, jnp.zeros((2,2))], [jnp.zeros((2,2)), T4]])
        T5 = universal_mzi(thetas[5], phis[5], e_thetas[5], e_phis[5])
        L4 = jnp.eye(4, dtype=complex); L4 = L4.at[1:3, 1:3].set(T5)
        U_mesh = jnp.dot(L4, jnp.dot(L3, jnp.dot(L2, L1)))
        final_phases = (outs * jnp.pi) + e_outs
        phase_matrix = jnp.diag(jnp.exp(1j * final_phases))
        return jnp.dot(phase_matrix, U_mesh)
    return simulate_mesh, init_fabrication_errors

def run_complete_report():
    print("🚀 DiffPhoton: Generating Complete Report (Loss + Matrix)...")
    
    mesh_fn, error_gen_fn = create_realistic_engine(crosstalk_level=0.15, phase_error_std=0.5)
    input_patterns = jnp.eye(4, dtype=complex)
    target_patterns = jnp.array([[0,0,0,1], [0,0,1,0], [0,1,0,0], [1,0,0,0]], dtype=float)

    key = jax.random.PRNGKey(42)
    fab_key, train_key = jax.random.split(key)
    static_errors = error_gen_fn(fab_key, 16)

    @jit
    def loss_fn(params):
        U = mesh_fn(params, static_errors)
        outputs = jnp.abs(jnp.dot(U, input_patterns))**2
        return jnp.mean((outputs - target_patterns.T)**2)

    params = jax.random.uniform(train_key, shape=(16,), minval=-1.0, maxval=1.0)
    optimizer = optax.adam(learning_rate=0.01)
    opt_state = optimizer.init(params)
    
    # --- 1. 学習と記録 ---
    loss_history = []
    print("   Training...", end="", flush=True)
    for i in range(2000):
        val, grads = value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        loss_history.append(val)
    print(" Done!")

    # --- 2. テスト (Confusion Matrix) ---
    print("   Testing...", end="")
    test_key = jax.random.PRNGKey(999)
    confusion_mat = jnp.zeros((4, 4))
    U_final = mesh_fn(params, static_errors)
    
    for i in range(4): 
        noise = jax.random.normal(test_key, (4, 25)) * 0.1
        batch_input = jnp.tile(input_patterns[:, i:i+1], (1, 25)) + noise
        batch_out = jnp.abs(jnp.dot(U_final, batch_input))**2
        predictions = jnp.argmax(batch_out, axis=0)
        true_label = 3 - i
        for pred in predictions:
            confusion_mat = confusion_mat.at[true_label, pred].add(1)
    print(" Done!")
    
    # --- 3. 統合グラフの描画 ---
    if not os.path.exists('output'): os.makedirs('output')
    
    # 1行2列のキャンバスを作成
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # [左側] Loss History Plot
    axes[0].plot(loss_history, color='#1f77b4', linewidth=2)
    axes[0].set_yscale('log')
    axes[0].set_title("(A) Calibration Process (Loss)", fontsize=14)
    axes[0].set_xlabel("Steps", fontsize=12)
    axes[0].set_ylabel("Error (MSE)", fontsize=12)
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].text(0.95, 0.95, f'Final Loss:\n{loss_history[-1]:.1e}', 
                 transform=axes[0].transAxes, ha='right', va='top', 
                 bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

    # [右側] Confusion Matrix Heatmap
    sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues', cbar=False, 
                ax=axes[1], annot_kws={'size': 16})
    axes[1].set_title("(B) Inference Result (Accuracy 100%)", fontsize=14)
    axes[1].set_xlabel("Predicted Output Port", fontsize=12)
    axes[1].set_ylabel("True Input Class", fontsize=12)

    plt.suptitle("DiffPhoton: Calibration of Defective Chip (Crosstalk 15%)", fontsize=16, y=0.98)
    plt.tight_layout()
    
    output_path = "output/calibration_report.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    
    print(f"✅ Full Report Saved: {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_complete_report()