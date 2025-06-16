"""
Quality Control Pipeline for Unaligned 4D MRI Scans using Variance Maps.
This module implements a pipeline to detect low-quality scans using variance-based features.
"""

import os
import numpy as np
import nibabel as nib
from scipy import stats
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Tuple, Optional
from bids import BIDS, Scan
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QC4DPipeline:
    def __init__(self, bids_path: str, output_folder: str):
        """
        Initialize the QC pipeline.
        
        Args:
            bids_path: Path to BIDS dataset
            output_folder: Path to save output files
        """
        self.bids = BIDS(bids_path)
        self.output_folder = Path(output_folder)
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize feature storage
        self.global_features = []
        self.texture_features = []
        self.scan_ids = []
        
        # Set up plotting style
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
    def extract_variance_maps(self) -> List[str]:
        """Extract variance maps from 4D scans."""
        variance_map_paths = []
        
        for subject in self.bids.subjects:
            for session in subject.sessions:
                for scan in session.scans:
                    try:
                        # Check if scan is 4D
                        if len(scan.shape) != 4:
                            logger.info(f"Skipping {scan.scan_name} - not a 4D scan")
                            continue
                            
                        # Load 4D scan
                        data = scan.get_data()
                        
                        # Compute variance over time axis
                        var_map = np.var(data, axis=3)
                        
                        # Create output path in BIDS derivatives structure
                        scan_id = f"{subject.get_name()}_{session.get_name()}_{scan.scan_name}"
                        output_path = self.output_folder / "variance_maps" / f"{scan_id}_variance.nii.gz"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Save variance map
                        var_nii = nib.Nifti1Image(var_map, scan.header.get_best_affine())
                        nib.save(var_nii, output_path)
                        
                        variance_map_paths.append(str(output_path))
                        self.scan_ids.append(scan_id)
                        logger.info(f"Processed {scan_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing {scan.scan_name}: {str(e)}")
                
        return variance_map_paths
    
    def extract_global_features(self, var_map: np.ndarray) -> Dict:
        """Extract global statistical features from variance map."""
        # Flatten and remove background (assuming background is 0)
        brain_mask = var_map > 0
        var_flat = var_map[brain_mask]
        
        features = {
            'mean': np.mean(var_flat),
            'std': np.std(var_flat),
            'min': np.min(var_flat),
            'max': np.max(var_flat),
            'skewness': stats.skew(var_flat),
            'kurtosis': stats.kurtosis(var_flat)
        }
        
        # Add percentiles
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        for p in percentiles:
            features[f'p{p}'] = np.percentile(var_flat, p)
            
        # Compute entropy
        hist, _ = np.histogram(var_flat, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zero bins
        features['entropy'] = -np.sum(hist * np.log(hist))
        
        # High variance ratio
        median = np.median(var_flat)
        high_var_mask = var_flat > (median * 2)
        features['high_variance_ratio'] = np.mean(high_var_mask)
        
        return features
    
    def extract_texture_features(self, var_map: np.ndarray) -> Dict:
        """Extract texture features using GLCM and LBP."""
        # Quantize to 32 gray levels
        var_norm = (var_map - var_map.min()) / (var_map.max() - var_map.min())
        var_quant = (var_norm * 31).astype(np.uint8)
        
        # GLCM features
        distances = [1]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        glcm = graycomatrix(var_quant, distances, angles, 32, symmetric=True, normed=True)
        
        features = {}
        for prop in ['contrast', 'homogeneity', 'correlation', 'energy']:
            feature = graycoprops(glcm, prop)
            features[f'glcm_{prop}_mean'] = np.mean(feature)
            features[f'glcm_{prop}_std'] = np.std(feature)
            
        # LBP features
        lbp = local_binary_pattern(var_quant, P=8, R=1, method='uniform')
        lbp_hist, _ = np.histogram(lbp, bins=10, density=True)
        features['lbp_entropy'] = -np.sum(lbp_hist * np.log(lbp_hist + 1e-10))
        
        return features
    
    def process_variance_maps(self, variance_map_paths: List[str]):
        """Process all variance maps and extract features."""
        for path in variance_map_paths:
            try:
                # Load variance map
                var_map = nib.load(path).get_fdata()
                
                # Extract features
                global_feats = self.extract_global_features(var_map)
                texture_feats = self.extract_texture_features(var_map)
                
                self.global_features.append(global_feats)
                self.texture_features.append(texture_feats)
                
            except Exception as e:
                logger.error(f"Error processing {path}: {str(e)}")
    
    def merge_features(self) -> pd.DataFrame:
        """Merge global and texture features into a single DataFrame."""
        global_df = pd.DataFrame(self.global_features)
        texture_df = pd.DataFrame(self.texture_features)
        
        # Add scan IDs
        global_df['scan_id'] = self.scan_ids
        texture_df['scan_id'] = self.scan_ids
        
        # Merge features
        merged_df = pd.merge(global_df, texture_df, on='scan_id')
        return merged_df
    
    def cluster_scans(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Perform dimensionality reduction and clustering."""
        # Prepare features
        X = features_df.drop('scan_id', axis=1).values
        X_scaled = StandardScaler().fit_transform(X)
        
        # UMAP reduction
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        
        # HDBSCAN clustering
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10)
        labels = clusterer.fit_predict(X_umap)
        
        return X_umap, labels
    
    def flag_poor_quality(self, features_df: pd.DataFrame, labels: np.ndarray) -> List[str]:
        """Flag potentially poor quality scans based on cluster analysis."""
        flagged_scans = []
        
        # Add cluster labels to features
        features_df['cluster'] = labels
        
        # Analyze each cluster
        for cluster in np.unique(labels):
            if cluster == -1:  # Skip noise points
                continue
                
            cluster_data = features_df[features_df['cluster'] == cluster]
            
            # Compute z-scores for key features
            key_features = ['entropy', 'kurtosis', 'high_variance_ratio']
            z_scores = stats.zscore(cluster_data[key_features])
            
            # Flag scans with multiple high z-scores
            high_z_mask = np.sum(np.abs(z_scores) > 2.5, axis=1) >= 2
            flagged_scans.extend(cluster_data[high_z_mask]['scan_id'].tolist())
        
        return flagged_scans
    
    def plot_umap_results(self, X_umap: np.ndarray, labels: np.ndarray, features_df: pd.DataFrame):
        """
        Create visualization plots for UMAP results and cluster assignments.
        
        Args:
            X_umap: UMAP coordinates
            labels: Cluster labels
            features_df: DataFrame containing features
        """
        derivatives_path = self.output_folder / "qc_pipeline"
        derivatives_path.mkdir(parents=True, exist_ok=True)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: UMAP scatter plot with cluster colors
        scatter = ax1.scatter(X_umap[:, 0], X_umap[:, 1], 
                            c=labels, cmap='viridis', 
                            alpha=0.6, s=100)
        ax1.set_title('UMAP Projection with Cluster Assignments')
        ax1.set_xlabel('UMAP 1')
        ax1.set_ylabel('UMAP 2')
        
        # Add legend for clusters
        legend1 = ax1.legend(*scatter.legend_elements(),
                           title="Clusters", loc="upper right")
        ax1.add_artist(legend1)
        
        # Plot 2: Feature distributions by cluster
        # Select key features for visualization
        key_features = ['entropy', 'kurtosis', 'high_variance_ratio']
        
        # Create box plots for each feature
        feature_data = []
        for feature in key_features:
            for label in np.unique(labels):
                if label == -1:  # Skip noise points
                    continue
                values = features_df[features_df['cluster'] == label][feature]
                feature_data.append({
                    'Feature': feature,
                    'Cluster': f'Cluster {label}',
                    'Value': values
                })
        
        feature_df = pd.DataFrame(feature_data)
        
        # Create box plot
        sns.boxplot(data=feature_df, x='Cluster', y='Value', hue='Feature', ax=ax2)
        ax2.set_title('Feature Distributions by Cluster')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Value')
        plt.xticks(rotation=45)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(derivatives_path / 'umap_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create additional plot for flagged scans
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get flagged scan indices
        flagged_mask = features_df['scan_id'].isin(self.flag_poor_quality(features_df, labels))
        
        # Plot all points
        scatter = ax.scatter(X_umap[:, 0], X_umap[:, 1], 
                           c='gray', alpha=0.3, s=50,
                           label='Normal Scans')
        
        # Plot flagged points
        ax.scatter(X_umap[flagged_mask, 0], X_umap[flagged_mask, 1],
                  c='red', alpha=0.8, s=100,
                  label='Flagged Scans')
        
        ax.set_title('UMAP Projection with Flagged Scans')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(derivatives_path / 'flagged_scans_visualization.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Created visualization plots in qc_pipeline directory")

    def run_pipeline(self) -> Tuple[pd.DataFrame, List[str]]:
        """Run the complete QC pipeline."""
        logger.info("Starting QC pipeline...")
        
        # Step 1: Extract variance maps
        variance_map_paths = self.extract_variance_maps()
        
        # Step 2 & 3: Extract features
        self.process_variance_maps(variance_map_paths)
        
        # Step 4: Merge features
        features_df = self.merge_features()
        
        # Step 5: Dimensionality reduction and clustering
        X_umap, labels = self.cluster_scans(features_df)
        
        # Step 6: Flag poor quality scans
        flagged_scans = self.flag_poor_quality(features_df, labels)
        
        # Step 7: Create visualizations
        self.plot_umap_results(X_umap, labels, features_df)
        
        # Save results in BIDS derivatives structure
        derivatives_path = self.output_folder / "qc_pipeline"
        derivatives_path.mkdir(parents=True, exist_ok=True)
        
        # Save features
        features_df.to_csv(derivatives_path / 'features.csv', index=False)
        
        # Save flagged scans
        with open(derivatives_path / 'flagged_scans.txt', 'w') as f:
            f.write('\n'.join(flagged_scans))
            
        # Save UMAP coordinates and cluster labels
        umap_df = pd.DataFrame({
            'scan_id': self.scan_ids,
            'umap_x': X_umap[:, 0],
            'umap_y': X_umap[:, 1],
            'cluster': labels
        })
        umap_df.to_csv(derivatives_path / 'umap_coordinates.csv', index=False)
            
        logger.info(f"Pipeline completed. Found {len(flagged_scans)} potentially poor quality scans.")
        
        return features_df, flagged_scans

def main():
    """Example usage of the QC pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QC Pipeline for 4D MRI Scans')
    parser.add_argument('bids_path', help='Path to BIDS dataset')
    parser.add_argument('output_folder', help='Path to save output files')
    
    args = parser.parse_args()
    
    pipeline = QC4DPipeline(args.bids_path, args.output_folder)
    features_df, flagged_scans = pipeline.run_pipeline()
    
    print(f"\nFlagged scans ({len(flagged_scans)}):")
    for scan in flagged_scans:
        print(f"- {scan}")

if __name__ == '__main__':
    main() 