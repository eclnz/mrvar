#!/usr/bin/env python3

import argparse
import numpy as np
import pickle
from pathlib import Path
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('variance_visualization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def plot_variance_slices(group_data, output_dir, slice_idx=None, n_cols=5, log_scale=False):
    """
    Plot sagittal slices from variance maps with consistent color scaling.
    
    Args:
        group_data: Dictionary containing variance maps and metadata
        output_dir: Directory to save the plots
        slice_idx: Index of sagittal slice to plot (if None, uses middle slice)
        n_cols: Number of columns in the plot grid
        log_scale: Whether to use log scale for visualization
    """
    variance_maps = group_data['variance_maps']
    scan_order = group_data['scan_order']
    
    # Determine number of rows needed
    n_scans = len(scan_order)
    n_rows = (n_scans + n_cols - 1) // n_cols
    
    # If slice_idx not provided, use middle slice
    if slice_idx is None:
        slice_idx = variance_maps.shape[3] // 2
    
    # Get global min and max for consistent color scaling
    if log_scale:
        # Set minimum to log(100) since we expect values > 100
        vmin = np.log10(100)  # log(100) ≈ 2
        vmax = np.log10(np.max(variance_maps) + 1e-10)
        logger.info(f"Global log variance range: [{vmin:.2f}, {vmax:.2f}]")
    else:
        vmin = np.min(variance_maps)
        vmax = np.max(variance_maps)
        logger.info(f"Global variance range: [{vmin:.2f}, {vmax:.2f}]")
    
    # Create figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
    axes = axes.flatten()
    
    # Plot each scan
    for idx, scan_id in enumerate(scan_order):
        ax = axes[idx]
        
        # Get sagittal slice
        sagittal_slice = np.transpose(variance_maps[idx, :, :, slice_idx], (1, 0))
        
        if log_scale:
            # Apply log transform and clip values below log(100)
            sagittal_slice = np.log10(sagittal_slice + 1e-10)
            sagittal_slice = np.clip(sagittal_slice, vmin, None)  # clip below log(100)
        
        # Plot with consistent colorbar and flip y-axis
        im = ax.imshow(sagittal_slice, cmap='viridis', aspect='auto', origin='lower',
                      vmin=vmin, vmax=vmax)
        ax.set_title(f"{scan_id}\nSlice {slice_idx}", fontsize=8)
        ax.axis('off')
        
        # Add colorbar with appropriate formatting
        if log_scale:
            # Create custom ticks for the colorbar showing actual variance values
            ticks = np.linspace(vmin, vmax, 5)
            tick_labels = [f"{10**tick:.1f}" for tick in ticks]
            cbar = plt.colorbar(im, ax=ax, shrink=0.7, ticks=ticks)
            cbar.ax.set_yticklabels(tick_labels)
        else:
            plt.colorbar(im, ax=ax, shrink=0.7)
    
    # Hide unused subplots
    for idx in range(len(scan_order), len(axes)):
        axes[idx].axis('off')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    scale_type = "log" if log_scale else "linear"
    output_path = Path(output_dir) / f"sagittal_slices_slice{slice_idx}_{scale_type}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved {scale_type} scale sagittal slice plot to {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Visualize variance maps with linear and log scales')
    parser.add_argument('group_data_path', type=str, help='Path to group variance maps pickle file')
    parser.add_argument('--output-dir', type=str, default='variance_plots',
                      help='Output directory for plots')
    parser.add_argument('--slice-idx', type=int, default=None,
                      help='Index of sagittal slice to plot (default: middle slice)')
    parser.add_argument('--n-cols', type=int, default=5,
                      help='Number of columns in plot grid')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load group data
        with open(args.group_data_path, 'rb') as f:
            group_data = pickle.load(f)
        
        # Generate both linear and log scale plots
        plot_variance_slices(group_data, output_dir, args.slice_idx, args.n_cols, log_scale=False)
        plot_variance_slices(group_data, output_dir, args.slice_idx, args.n_cols, log_scale=True)
        
        logger.info("Visualization completed successfully")
        
    except Exception as e:
        logger.error(f"Error in visualization process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 