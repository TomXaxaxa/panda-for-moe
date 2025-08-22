import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # Required for 3D plotting

# --- 1. Configuration Area ---
# Please modify these two paths
INPUT_DATA_DIR = "./predictions/AnyVariate"  # 存放 .npy 文件的文件夹
PLOT_OUTPUT_DIR = "./plots/AnyVariate"     # 保存生成图像的文件夹
# -----------------------------


def plot_2d_samples(true_data, pred_data, system_name, timestamp, output_dir):
    """
    Randomly selects 3 dimensions for a system and plots the true vs. predicted values
    in 3 subplots.
    """
    # Expected shape for true_data and pred_data: (num_samples, num_timesteps, num_channels)
    if true_data.ndim != 3:
        print(f"  [Skipping 2D Plot] Data for '{system_name}' has incorrect dimensions. Expected 3 dimensions.")
        return

    num_samples, num_timesteps, num_channels = true_data.shape

    # Randomly select one sample to visualize
    sample_index = random.randint(0, num_samples - 1)
    
    # Randomly select 3 dimension indices
    if num_channels < 3:
        print(f"  [Warning] System '{system_name}' has {num_channels} dimensions (< 3). Visualizing all available dimensions.")
        dim_indices = list(range(num_channels))
    else:
        dim_indices = sorted(random.sample(range(num_channels), 3))

    if not dim_indices:
        print(f"  [Skipping 2D Plot] No dimensions to visualize for '{system_name}'.")
        return

    # Create a figure with a subplot for each selected dimension
    fig, axes = plt.subplots(len(dim_indices), 1, figsize=(16, 4 * len(dim_indices)), sharex=True)
    if len(dim_indices) == 1: # Handle case with only one dimension
        axes = [axes]

    for i, dim_index in enumerate(dim_indices):
        ax = axes[i]
        # Plot Ground Truth
        ax.plot(true_data[sample_index, :, dim_index], label=f'Ground Truth (Dim {dim_index})', color='royalblue')
        # Plot Prediction
        ax.plot(pred_data[sample_index, :, dim_index], label=f'Prediction (Dim {dim_index})', color='darkorange', linestyle='--')
        
        ax.set_title(f"System: {system_name} - Dimension {dim_index} Comparison")
        ax.set_ylabel("Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(f'Univariate Comparison - {system_name} ({timestamp})\n(Sample Index: {sample_index})', fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save the figure
    save_path = os.path.join(output_dir, f"{timestamp}_{system_name}_2D_samples.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  - 2D plot saved to: {save_path}")


def plot_3d_trajectory(true_data, pred_data, system_name, timestamp, output_dir):
    """
    Plots the true and predicted trajectories in 3D space using the first 3 dimensions.
    """
    # Check if there are enough dimensions for a 3D plot
    if true_data.shape[-1] < 3:
        print(f"  [Skipping 3D Plot] System '{system_name}' has {true_data.shape[-1]} dimensions, which is less than 3.")
        return

    # Randomly select one sample to visualize
    sample_index = random.randint(0, true_data.shape[0] - 1)
    
    # Extract the first three dimensions
    true_traj = true_data[sample_index, :, :3]
    pred_traj = pred_data[sample_index, :, :3]

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the ground truth trajectory
    ax.plot(true_traj[:, 0], true_traj[:, 1], true_traj[:, 2], label='Ground Truth Trajectory', color='royalblue', alpha=0.8)
    # Plot the predicted trajectory
    ax.plot(pred_traj[:, 0], pred_traj[:, 1], pred_traj[:, 2], label='Predicted Trajectory', color='darkorange', linestyle='--')

    # Mark the starting point
    ax.scatter(true_traj[0, 0], true_traj[0, 1], true_traj[0, 2], color='green', s=100, label='Start Point', marker='o')

    ax.set_xlabel("Dimension 0")
    ax.set_ylabel("Dimension 1")
    ax.set_zlabel("Dimension 2")
    ax.set_title(f'3D Phase Space Reconstruction - {system_name} ({timestamp})\n(Sample Index: {sample_index})', fontsize=14)
    ax.legend()
    ax.grid(True)

    # Save the figure
    save_path = os.path.join(output_dir, f"{timestamp}_{system_name}_3D_trajectory.png")
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"  - 3D plot saved to: {save_path}")


def visualize_all_results(input_dir, output_dir):
    """
    Main function to scan the input directory and run visualization tasks for all found data pairs.
    """
    print(f"Scanning data directory: {input_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Find all prediction files
    pred_files = glob.glob(os.path.join(input_dir, "*_preds.npy"))

    if not pred_files:
        print("Error: No '_preds.npy' files found in the specified directory. Please check the path.")
        return

    print(f"Found {len(pred_files)} sets of prediction data. Starting visualization...")

    for pred_filepath in pred_files:
        # Infer the ground truth file path from the prediction file path
        true_filepath = pred_filepath.replace("_preds.npy", "_trues.npy")
        
        # Parse timestamp and system name from the filename
        base_name = os.path.basename(pred_filepath).replace("_preds.npy", "")
        timestamp, system_name = base_name.split('_', 1)

        print(f"\nProcessing system: '{system_name}' (Timestamp: {timestamp})")

        if not os.path.exists(true_filepath):
            print(f"  [Warning] Corresponding ground truth file not found: {true_filepath}. Skipping this system.")
            continue

        try:
            # Load the data
            pred_data = np.load(pred_filepath)
            true_data = np.load(true_filepath)
            
            # Run both visualization tasks
            plot_2d_samples(true_data, pred_data, system_name, timestamp, output_dir)
            plot_3d_trajectory(true_data, pred_data, system_name, timestamp, output_dir)

        except Exception as e:
            print(f"  [Error] An error occurred while processing {pred_filepath}: {e}")

    print("\nAll visualization tasks are complete!")
    print(f"Plots have been saved to: {output_dir}")


if __name__ == "__main__":
    visualize_all_results(INPUT_DATA_DIR, PLOT_OUTPUT_DIR)