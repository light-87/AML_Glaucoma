"""
Memory Efficient Data Loading

Optimized data loading for large datasets with memory management.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union, Callable, Generator
import gc
import psutil
import logging

from glaucoma_detection.logger import get_logger, ERROR_CODES, log_exception_handler

logger = get_logger(__name__)

def memory_stats() -> Dict[str, float]:
    """Get current memory statistics.
    
    Returns:
        Dictionary with memory statistics
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    
    return {
        "rss_mb": mem_info.rss / (1024 * 1024),  # Resident Set Size in MB
        "vms_mb": mem_info.vms / (1024 * 1024),  # Virtual Memory Size in MB
        "percent": process.memory_percent(),
        "system_available_gb": psutil.virtual_memory().available / (1024 * 1024 * 1024)
    }

class CachedImage:
    """Simple image cache with reference counting."""
    
    def __init__(self, image_data):
        """Initialize with image data.
        
        Args:
            image_data: Image data array
        """
        self.data = image_data
        self.ref_count = 1
    
    def acquire(self):
        """Increment reference count."""
        self.ref_count += 1
    
    def release(self) -> bool:
        """Decrement reference count and return True if it reaches zero.
        
        Returns:
            True if reference count is zero (can be removed from cache)
        """
        self.ref_count -= 1
        return self.ref_count <= 0

class MemoryEfficientDataset(Dataset):
    """Memory-efficient dataset for glaucoma images that implements caching."""
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 target_size: Tuple[int, int] = (224, 224), 
                 augment: bool = False, 
                 mode: str = 'segmentation',
                 normalization: str = 'imagenet',
                 max_cache_size: int = 1000,
                 prefetch_size: int = 100,
                 cache_dir: Optional[str] = None):
        """Initialize the dataset with caching options.
        
        Args:
            data: DataFrame with image/mask paths
            target_size: Target size for resizing
            augment: Whether to apply data augmentation
            mode: 'segmentation' or 'classification'
            normalization: Normalization method
            max_cache_size: Maximum number of images to cache in memory
            prefetch_size: Number of images to prefetch
            cache_dir: Directory to store disk cache (if None, no disk cache)
        """
        self.data = data
        self.target_size = target_size
        self.augment = augment
        self.mode = mode
        self.max_cache_size = max_cache_size
        self.prefetch_size = min(prefetch_size, len(data))
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Set up cache
        self.image_cache = {}
        self.mask_cache = {}
        self.last_accessed = []
        
        # Define transforms based on mode and augmentation
        self.transforms = self._get_transforms(normalization, augment)
        
        # Create disk cache directory if needed
        if self.cache_dir:
            self.cache_dir.mkdir(exist_ok=True, parents=True)
            logger.info(f"Disk cache enabled at {self.cache_dir}")
        
        # Prefetch data if prefetch_size > 0
        if self.prefetch_size > 0:
            self._prefetch_data()
    
    def _get_transforms(self, normalization: str, augment: bool) -> A.Compose:
        """Get the appropriate transforms.
        
        Args:
            normalization: Normalization method
            augment: Whether to apply augmentation
            
        Returns:
            Albumentations composition of transforms
        """
        # Set normalization parameters
        if normalization == 'imagenet':
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean, std = None, None
        
        if augment:
            return A.Compose([
                A.RandomRotate90(),
                A.Flip(),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
                A.GaussianBlur(blur_limit=(3, 5)),
                A.Normalize(mean=mean, std=std) if mean else A.Normalize(),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Normalize(mean=mean, std=std) if mean else A.Normalize(),
                ToTensorV2()
            ])
    
    def _prefetch_data(self):
        """Prefetch a batch of data into memory cache."""
        logger.info(f"Prefetching {self.prefetch_size} images into memory cache")
        
        indices = np.random.choice(len(self.data), self.prefetch_size, replace=False)
        for idx in indices:
            self._get_cached_image(idx)
            if self.mode == 'segmentation':
                self._get_cached_mask(idx)
    
    def _get_cached_image(self, idx: int) -> np.ndarray:
        """Get image from cache or load it.
        
        Args:
            idx: Index of the image
            
        Returns:
            Image data as numpy array
        """
        if idx in self.image_cache:
            # Update access record and increment reference count
            self.image_cache[idx].acquire()
            if idx in self.last_accessed:
                self.last_accessed.remove(idx)
            self.last_accessed.append(idx)
            return self.image_cache[idx].data
        
        # Load image
        row = self.data.iloc[idx]
        image_path = row['image_path']
        
        # Try loading from disk cache first
        disk_cache_path = self._get_disk_cache_path(image_path, "image")
        if disk_cache_path and disk_cache_path.exists():
            try:
                image = np.load(disk_cache_path)
                logger.debug(f"Loaded image from disk cache: {disk_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to load image from disk cache: {str(e)}")
                image = self._load_and_preprocess_image(image_path)
        else:
            image = self._load_and_preprocess_image(image_path)
        
        # Save to disk cache if enabled
        if disk_cache_path and not disk_cache_path.exists():
            try:
                np.save(disk_cache_path, image)
                logger.debug(f"Saved image to disk cache: {disk_cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save image to disk cache: {str(e)}")
        
        # Manage memory cache size
        if len(self.image_cache) >= self.max_cache_size and self.last_accessed:
            # Remove oldest accessed image
            oldest_idx = self.last_accessed.pop(0)
            if oldest_idx in self.image_cache and self.image_cache[oldest_idx].release():
                del self.image_cache[oldest_idx]
                logger.debug(f"Removed image {oldest_idx} from memory cache")
        
        # Add to memory cache
        self.image_cache[idx] = CachedImage(image)
        self.last_accessed.append(idx)
        
        return image
    
    def _get_cached_mask(self, idx: int) -> np.ndarray:
        """Get mask from cache or load it.
        
        Args:
            idx: Index of the mask
            
        Returns:
            Mask data as numpy array
        """
        if idx in self.mask_cache:
            # Update access record
            if idx in self.last_accessed:
                self.last_accessed.remove(idx)
            self.last_accessed.append(idx)
            self.mask_cache[idx].acquire()
            return self.mask_cache[idx].data
        
        # Load mask
        row = self.data.iloc[idx]
        mask_path = row.get('mask_path')
        
        if not mask_path or not Path(mask_path).exists():
            # Create empty mask if not available
            mask = np.zeros((1, *self.target_size), dtype=np.float32)
        else:
            # Try loading from disk cache first
            disk_cache_path = self._get_disk_cache_path(mask_path, "mask")
            if disk_cache_path and disk_cache_path.exists():
                try:
                    mask = np.load(disk_cache_path)
                    logger.debug(f"Loaded mask from disk cache: {disk_cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to load mask from disk cache: {str(e)}")
                    mask = self._load_and_preprocess_mask(mask_path)
            else:
                mask = self._load_and_preprocess_mask(mask_path)
            
            # Save to disk cache if enabled
            if disk_cache_path and not disk_cache_path.exists():
                try:
                    np.save(disk_cache_path, mask)
                    logger.debug(f"Saved mask to disk cache: {disk_cache_path}")
                except Exception as e:
                    logger.warning(f"Failed to save mask to disk cache: {str(e)}")
        
        # Manage cache size
        if len(self.mask_cache) >= self.max_cache_size and self.last_accessed:
            # Remove oldest accessed mask
            oldest_idx = self.last_accessed.pop(0)
            if oldest_idx in self.mask_cache and self.mask_cache[oldest_idx].release():
                del self.mask_cache[oldest_idx]
                logger.debug(f"Removed mask {oldest_idx} from memory cache")
        
        # Add to cache
        self.mask_cache[idx] = CachedImage(mask)
        self.last_accessed.append(idx)
        
        return mask
    
    def _get_disk_cache_path(self, file_path: str, prefix: str) -> Optional[Path]:
        """Get the path for disk cache.
        
        Args:
            file_path: Original file path
            prefix: Prefix for the cache file ('image' or 'mask')
            
        Returns:
            Path to the cache file or None if disk cache is disabled
        """
        if not self.cache_dir:
            return None
        
        file_hash = str(hash(file_path))
        return self.cache_dir / f"{prefix}_{file_hash}.npy"
    
    def _load_and_preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Load image efficiently with OpenCV
            image = cv2.imread(image_path)
            if image is None:
                logger.warning(f"Failed to load image with OpenCV: {image_path}")
                # Try with PIL as fallback
                pil_image = Image.open(image_path)
                image = np.array(pil_image)
                if pil_image.mode == 'RGB':
                    # Convert from RGB to BGR (OpenCV format)
                    image = image[:, :, ::-1]
            else:
                # Convert from BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize efficiently
            image = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
            
            return image
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}", exc_info=True)
            # Return a blank image in case of error
            return np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
    
    def _load_and_preprocess_mask(self, mask_path: str) -> np.ndarray:
        """Load and preprocess a mask.
        
        Args:
            mask_path: Path to the mask file
            
        Returns:
            Preprocessed mask as numpy array
        """
        try:
            # Load mask efficiently with OpenCV (grayscale)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logger.warning(f"Failed to load mask with OpenCV: {mask_path}")
                # Try with PIL as fallback
                pil_mask = Image.open(mask_path).convert('L')
                mask = np.array(pil_mask)
            
            # Normalize to 0-1 range if needed
            if mask.max() > 1:
                mask = mask / 255.0
                
            # Ensure mask is binary (this is the key fix)
            mask = (mask > 0.5).astype(np.float32)
            
            # Resize efficiently
            mask = cv2.resize(mask, self.target_size, interpolation=cv2.INTER_NEAREST)
            
            # Add channel dimension
            mask = np.expand_dims(mask, axis=0)
            
            return mask
        except Exception as e:
            logger.error(f"Error loading mask {mask_path}: {str(e)}", exc_info=True)
            # Return a blank mask in case of error
            return np.zeros((1, self.target_size[1], self.target_size[0]), dtype=np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Tuple of (image, mask/label) as tensors
        """
        # Get image and mask from cache or load them
        image = self._get_cached_image(idx)
        
        if self.mode == 'segmentation':
            mask = self._get_cached_mask(idx)
            
            # Apply transforms
            if self.transforms:
                # For segmentation, we need to ensure mask is 2D for albumentations
                mask_2d = mask[0]  # Get the first channel (remove channel dimension)
                transformed = self.transforms(image=image, mask=mask_2d)
                image = transformed["image"]  # This is now a torch tensor
                mask = transformed["mask"].unsqueeze(0)  # Add channel dimension back
                # Ensure mask remains binary after transforms (this is important)
                mask = (mask > 0.5).float()
            
            return image, mask
        else:
            # Classification mode
            row = self.data.iloc[idx]
            label = torch.tensor([row.get('label', 0)], dtype=torch.float32)
            
            # Apply transforms (image only)
            if self.transforms:
                transformed = self.transforms(image=image)
                image = transformed["image"]  # This is now a torch tensor
            
            return image, label
    
    def cleanup(self):
        """Clean up memory cache and release resources."""
        self.image_cache.clear()
        self.mask_cache.clear()
        self.last_accessed.clear()
        gc.collect()
        logger.info("Cleared dataset cache and collected garbage")

class BatchPrefetchLoader:
    """Loader that prefetches batches in a background thread."""
    
    def __init__(self, dataloader: DataLoader, prefetch_batches: int = 2):
        """Initialize with dataloader.
        
        Args:
            dataloader: Source DataLoader
            prefetch_batches: Number of batches to prefetch
        """
        from queue import Queue
        from threading import Thread
        
        self.dataloader = dataloader
        self.prefetch_batches = prefetch_batches
        self.queue = Queue(maxsize=prefetch_batches)
        self.iterator = iter(dataloader)
        self.stopped = False
        
        # Start the prefetch thread
        self.prefetch_thread = Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_thread.start()
    
    def _prefetch_loop(self):
        """Background thread loop to prefetch batches."""
        try:
            while not self.stopped:
                try:
                    batch = next(self.iterator)
                    self.queue.put(batch)
                except StopIteration:
                    # If we've reached the end, create a new iterator
                    self.iterator = iter(self.dataloader)
                    batch = next(self.iterator)
                    self.queue.put(batch)
        except Exception as e:
            logger.error(f"Error in prefetch thread: {str(e)}", exc_info=True)
    
    def __iter__(self):
        """Return self as iterator."""
        return self
    
    def __next__(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the next batch.
        
        Returns:
            Next batch from the dataloader
            
        Raises:
            StopIteration: When no more batches are available
        """
        if self.stopped:
            raise StopIteration
        
        batch = self.queue.get()
        return batch
    
    def __len__(self) -> int:
        """Get the number of batches.
        
        Returns:
            Number of batches in the dataloader
        """
        return len(self.dataloader)
    
    def stop(self):
        """Stop the prefetch thread."""
        self.stopped = True
        # Wait for the thread to finish
        self.prefetch_thread.join(timeout=1.0)

def create_memory_efficient_data_loader(
    data: pd.DataFrame,
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (224, 224),
    augment: bool = False,
    mode: str = 'segmentation',
    shuffle: bool = True,
    max_cache_size: int = 1000,
    prefetch_size: int = 100,
    cache_dir: Optional[str] = None,
    prefetch_batches: int = 2
) -> DataLoader:
    """Create a memory-efficient data loader.
    
    Args:
        data: DataFrame with image/mask paths
        batch_size: Batch size
        num_workers: Number of worker processes
        target_size: Target size for resizing
        augment: Whether to apply data augmentation
        mode: 'segmentation' or 'classification'
        shuffle: Whether to shuffle the data
        max_cache_size: Maximum number of images to cache in memory
        prefetch_size: Number of images to prefetch
        cache_dir: Directory to store disk cache
        prefetch_batches: Number of batches to prefetch
        
    Returns:
        DataLoader with memory efficiency features
    """
    # Log memory stats before creating dataset
    mem_before = memory_stats()
    logger.info(f"Memory before dataset creation: {mem_before['rss_mb']:.2f} MB")
    
    # Create dataset
    dataset = MemoryEfficientDataset(
        data=data,
        target_size=target_size,
        augment=augment,
        mode=mode,
        max_cache_size=max_cache_size,
        prefetch_size=prefetch_size,
        cache_dir=cache_dir
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Log memory stats after dataset creation
    mem_after = memory_stats()
    logger.info(f"Memory after dataset creation: {mem_after['rss_mb']:.2f} MB")
    logger.info(f"Memory increase: {mem_after['rss_mb'] - mem_before['rss_mb']:.2f} MB")
    
    # Wrap with batch prefetch loader if requested
    if prefetch_batches > 0:
        return BatchPrefetchLoader(dataloader, prefetch_batches)
    else:
        return dataloader