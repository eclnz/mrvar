#!/usr/bin/env python3

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('variance_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def clip_extremes(variance_map: np.ndarray, clip_percentile: float = 99.9) -> np.ndarray:
    """Clip extreme outliers while preserving original variance values.
    
    Args:
        variance_map: Input variance map
        clip_percentile: Percentile to clip at (default: 99.9)
    
    Returns:
        Clipped variance map with original units preserved
    """
    # Clip high values
    upper_threshold = np.percentile(variance_map, clip_percentile)
    n_above = np.sum(variance_map > upper_threshold)
    if n_above > 0:
        logger.info(f"Clipping {n_above} values above {upper_threshold:.2f}")
        variance_map = np.clip(variance_map, 0, upper_threshold)
    
    logger.info(f"Variance range: [{np.min(variance_map):.2f}, {np.max(variance_map):.2f}]")
    return variance_map

def process_nifti(input_path, output_path, downsample=0, clip_percentile=99.9):
    """Process a NIfTI file to calculate temporal variance.
    
    Args:
        input_path: Path to input NIfTI file
        output_path: Path for output NIfTI file
        downsample: Downsampling level (0 = no downsampling, 1 = 2x, 2 = 4x, etc.)
        clip_percentile: Percentile to clip at (default: 99.9)
    """
    # Load NIfTI file
    logger.info(f"Loading NIfTI file: {input_path}")
    nifti_img = nib.load(input_path)
    data = nifti_img.get_fdata()
    original_shape = data.shape[:3]  # Store original spatial dimensions
    logger.info(f"Input data shape: {data.shape}")
    
    # Downsample if needed
    if downsample > 0:
        logger.info(f"Downsampling with factor {downsample}")
        from pyqa.resize import Resize
        resizer = Resize(downsample_factor=downsample)
        data = resizer.downsample(data)
        logger.info(f"Downsampled shape: {data.shape}")
    
    # Calculate variance over time
    logger.info("Calculating temporal variance")
    temporal_var = np.var(data, axis=-1)  # shape: (height, width, depth)
    logger.info(f"Raw variance range: [{np.min(temporal_var):.2f}, {np.max(temporal_var):.2f}]")
    
    # Clip extreme values
    logger.info("Clipping extreme values")
    temporal_var = clip_extremes(temporal_var, clip_percentile)
    
    # Create output directory if it doesn't exist
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save spatial variance as NIfTI
    output_nifti = nib.Nifti1Image(temporal_var, nifti_img.affine, nifti_img.header)
    nib.save(output_nifti, output_path)
    logger.info(f"Saved variance map to: {output_path}")
    logger.info(f"Output shape: {temporal_var.shape}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description='Calculate temporal variance for NIfTI images')
    parser.add_argument('input_path', type=str, help='Path to input NIfTI file')
    parser.add_argument('output_path', type=str, help='Path for output NIfTI file')
    parser.add_argument('--downsample', type=int, default=0,
                      help='Downsampling level (0 = no downsampling, 1 = 2x, 2 = 4x, etc.)')
    parser.add_argument('--clip-percentile', type=float, default=99.9,
                      help='Percentile to clip at (default: 99.9)')
    
    args = parser.parse_args()
    
    try:
        # Process NIfTI file
        output_path = process_nifti(args.input_path, args.output_path, args.downsample, args.clip_percentile)
        
        print(f"\nProcessing complete!")
        print(f"Input: {args.input_path}")
        print(f"Downsample level: {args.downsample}")
        print(f"Clip percentile: {args.clip_percentile}")
        print(f"Output: {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        raise

if __name__ == "__main__":
    main() 