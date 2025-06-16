#!/usr/bin/env python3

import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import logging
from scipy.ndimage import zoom
import pickle
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('variance_collection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def resample_image(img_data, target_shape):
    """Resample image to target shape using zoom."""
    current_shape = img_data.shape
    zoom_factors = [t/c for t, c in zip(target_shape, current_shape)]
    return zoom(img_data, zoom_factors, order=1)

def collect_variance_maps(bids_dir: str):
    """Collect all variance maps into a group-level object."""
    bids_path = Path(bids_dir)
    derivatives_path = bids_path / "derivatives" / "aMRI"
    
    if not derivatives_path.exists():
        raise ValueError(f"Derivatives directory not found at {derivatives_path}")
    
    # Find all processed files
    processed_files = list(derivatives_path.rglob("*_raw_variance_downsampled.nii.gz"))
    
    if not processed_files:
        raise ValueError("No variance maps found in derivatives directory")
    
    logger.info(f"Found {len(processed_files)} variance maps to process")
    
    # Get target shape from first scan
    first_scan = nib.load(processed_files[0])
    target_shape = first_scan.shape
    logger.info(f"Using target shape from first scan: {target_shape}")
    
    # Dictionary to store processed images and metadata
    group_data = {
        'variance_maps': None,  # Will be a 4D array (n_subjects, *target_shape)
        'metadata': {},
        'scan_order': [],  # List to maintain scan order
        'target_shape': target_shape
    }
    
    # Initialize array to store all variance maps
    n_scans = len(processed_files)
    group_data['variance_maps'] = np.zeros((n_scans, *target_shape))
    
    for idx, file_path in enumerate(tqdm(processed_files, desc="Processing variance maps")):
        try:
            # Load NIfTI file
            nifti_img = nib.load(file_path)
            img_data = nifti_img.get_fdata()
            
            # Get subject and session info from path
            rel_path = file_path.relative_to(derivatives_path)
            subject = rel_path.parts[0]
            session = rel_path.parts[1]
            
            # Create unique identifier
            scan_id = f"{subject}_{session}"
            
            # Store scan order
            group_data['scan_order'].append(scan_id)
            
            # Store original shape
            original_shape = img_data.shape
            
            # Resample if shape doesn't match target
            if img_data.shape != target_shape:
                logger.info(f"Resampling {scan_id} from {original_shape} to {target_shape}")
                img_data = resample_image(img_data, target_shape)
            
            # Store in group array
            group_data['variance_maps'][idx] = img_data
            
            # Store metadata
            group_data['metadata'][scan_id] = {
                'original_shape': original_shape,
                'final_shape': img_data.shape,
                'file_path': str(file_path),
                'affine': nifti_img.affine,
                'header': nifti_img.header,
                'array_index': idx
            }
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            continue
    
    return group_data

def save_scan_order(group_data, output_dir):
    """Save scan order information to a text file."""
    order_file = Path(output_dir) / "scan_order.txt"
    with open(order_file, 'w') as f:
        f.write("Index\tSubject_Session\n")
        for idx, scan_id in enumerate(group_data['scan_order']):
            f.write(f"{idx}\t{scan_id}\n")
    logger.info(f"Saved scan order to {order_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect variance maps into group-level object')
    parser.add_argument('bids_dir', type=str, help='Path to BIDS directory')
    parser.add_argument('--output-dir', type=str, default='group_variance',
                      help='Output directory for group data (default: group_variance)')
    
    args = parser.parse_args()
    
    try:
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect variance maps
        group_data = collect_variance_maps(args.bids_dir)
        
        # Save group data to pickle file
        pickle_path = output_dir / "group_variance_maps.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(group_data, f)
        
        # Save scan order
        save_scan_order(group_data, output_dir)
        
        # Print summary
        print("\nProcessing Summary:")
        print(f"Total variance maps collected: {len(group_data['scan_order'])}")
        print(f"Target shape: {group_data['target_shape']}")
        print(f"Group array shape: {group_data['variance_maps'].shape}")
        print(f"Output directory: {output_dir}")
        print(f"Group data saved to: {pickle_path}")
        print(f"Scan order saved to: {output_dir}/scan_order.txt")
        
    except Exception as e:
        logger.error(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    main() 