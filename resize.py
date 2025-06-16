import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional, Union

class Crop:
    def __init__(self, threshold: Union[float, str] = 0.1):
        self.threshold = threshold
        self._crop_slices = None
        self._original_shape = None
    
    def _get_threshold_value(self, volume: np.ndarray) -> float:
        if isinstance(self.threshold, str) and self.threshold == 'mean':
            return np.abs(volume).mean()
        return float(self.threshold)
    
    def _find_bounding_box(self, volume: np.ndarray) -> tuple[slice, ...]:
        # Determine whether volume is 3D or 4D
        if volume.ndim == 3:
            # 3D spatial volume
            data = np.abs(volume)
            original_ndim = 3
        elif volume.ndim == 4:
            data = np.abs(volume).mean(axis=-1)
            original_ndim = 4
        else:
            raise ValueError("Volume must be 3D or 4D (x, y, z[, t])")
        threshold = self._get_threshold_value(volume)
        mask = data > threshold
        slices: list[slice] = []
        for axis in range(3):
            # Project mask along other spatial axes
            proj_axes = tuple(i for i in range(3) if i != axis)
            proj = np.any(mask, axis=proj_axes)
            indices = np.where(proj)[0]
            if len(indices) > 0:
                start, end = indices[0], indices[-1] + 1
            else:
                start, end = 0, volume.shape[axis]
            slices.append(slice(start, end))
        if original_ndim == 4:
            slices.append(slice(None))
        return tuple(slices)
    
    def crop(self, volume: np.ndarray) -> np.ndarray:
        """Crop the volume based on the threshold."""
        if self._crop_slices is not None:
            return volume[self._crop_slices]
        
        # Store original shape for restoration
        self._original_shape = volume.shape
        
        # For 4D volumes, crop only the spatial dimensions
        if volume.ndim == 4:
            # Get the first time point for thresholding
            first_time = volume[..., 0]
            # Compute threshold value
            if isinstance(self.threshold, str):
                if self.threshold == 'mean':
                    threshold_value = np.mean(first_time)
                else:
                    raise ValueError(f"Unknown threshold string: {self.threshold}")
            else:
                threshold_value = self.threshold
            # Compute crop region based on threshold
            mask = first_time > threshold_value
            if not np.any(mask):
                return volume
            
            # Get the bounding box of the mask
            indices = np.where(mask)
            start_x, end_x = np.min(indices[0]), np.max(indices[0]) + 1
            start_y, end_y = np.min(indices[1]), np.max(indices[1]) + 1
            start_z, end_z = np.min(indices[2]), np.max(indices[2]) + 1
            
            # Store only spatial slices
            self._crop_slices = (slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z))
            
            # Return cropped volume (including time dimension)
            return volume[self._crop_slices + (slice(None),)]
        else:
            # For 3D volumes, crop all dimensions
            if isinstance(self.threshold, str):
                if self.threshold == 'mean':
                    threshold_value = np.mean(volume)
                else:
                    raise ValueError(f"Unknown threshold string: {self.threshold}")
            else:
                threshold_value = self.threshold
            mask = volume > threshold_value
            if not np.any(mask):
                return volume
            
            indices = np.where(mask)
            start_x, end_x = np.min(indices[0]), np.max(indices[0]) + 1
            start_y, end_y = np.min(indices[1]), np.max(indices[1]) + 1
            start_z, end_z = np.min(indices[2]), np.max(indices[2]) + 1
            
            self._crop_slices = (slice(start_x, end_x), slice(start_y, end_y), slice(start_z, end_z))
            return volume[self._crop_slices]

    def apply_crop(self, volume: np.ndarray) -> np.ndarray:
        if self._crop_slices is None:
            raise ValueError("No crop region computed. Call crop() first.")
        return volume[self._crop_slices]
    
    def restore(self, cropped_volume: np.ndarray) -> np.ndarray:
        """Restore the cropped volume to its original size."""
        if self._crop_slices is None or self._original_shape is None:
            raise ValueError("No crop region computed. Call crop() first.")
        
        # Create output array with original spatial dimensions
        restored = np.zeros(self._original_shape[:3], dtype=cropped_volume.dtype)
        
        # Restore only spatial dimensions
        restored[self._crop_slices] = cropped_volume
        
        return restored


class Resize:
    def __init__(self, downsample_factor: int = 0):
        self.downsample_factor = downsample_factor
        self._original_shape = None
        self._downsample_factor = downsample_factor  # Direct downsampling factor
        
    @staticmethod
    def _resize_to_shape(volume: np.ndarray, target_shape: tuple) -> np.ndarray:
        current_shape = volume.shape
        slices = tuple(
            slice(0, min(c, t)) for c, t in zip(current_shape, target_shape)
        )
        cropped = volume[slices]
        # Pad if needed
        pad_width = [(0, max(0, t - cropped.shape[i])) for i, t in enumerate(target_shape)]
        return np.pad(cropped, pad_width, mode='edge')

    def downsample(self, volume: np.ndarray) -> np.ndarray:
        if self._downsample_factor == 0:  # 0 means no downsampling
            return volume

        self._original_shape = volume.shape
        f = self._downsample_factor

        if volume.ndim == 4:
            # For 4D volumes, only downsample spatial dimensions
            return volume[::f, ::f, ::f, :].copy()
        else:
            # For 3D volumes, downsample all dimensions
            return volume[::f, ::f, ::f].copy()

    def restore(self, downsampled_volume: np.ndarray, target_shape: Optional[tuple] = None, order: Optional[int] = 3) -> np.ndarray:
        if self._downsample_factor == 0:  # 0 means no downsampling
            return downsampled_volume
            
        # Handle interpolation case
        if order is not None:
            zoom_factors = [t / s for s, t in zip(downsampled_volume.shape[:3], target_shape[:3])]
            if downsampled_volume.ndim == 4:
                return np.stack([
                    zoom(downsampled_volume[..., t], zoom_factors, order=order)
                    for t in range(downsampled_volume.shape[3])
                ], axis=-1)
            else:
                return zoom(downsampled_volume, zoom_factors, order=order)
            
        # Get target shape
        original_shape = target_shape if target_shape is not None else self._original_shape
        if original_shape is None:
            raise ValueError("No shape information available. Either provide target_shape or call downsample() first.")
            
        # Handle 4D volumes by preserving time dimension
        is_4d = downsampled_volume.ndim == 4
        spatial_shape = original_shape[:3] if len(original_shape) > 3 else original_shape
        target_shape = spatial_shape + (downsampled_volume.shape[3],) if is_4d else spatial_shape
        
        # Upsample by repeating
        f = self._downsample_factor
        upsampled = downsampled_volume.repeat(f, axis=0).repeat(f, axis=1).repeat(f, axis=2)
        
        # Trim or pad to match target shape
        return Resize._resize_to_shape(upsampled, target_shape)