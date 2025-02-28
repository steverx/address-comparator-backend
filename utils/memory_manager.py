import gc
import os
import psutil
import logging
import tracemalloc
import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Any, Callable, TypeVar
import functools
import time

logger = logging.getLogger(__name__)

T = TypeVar('T')  # For generic function type hints

class MemoryManager:
    """
    Advanced memory management for data processing operations.
    Tracks memory usage, provides optimization suggestions, and handles cleanup.
    """
    
    def __init__(self, 
                threshold_pct: float = 80.0, 
                enable_tracemalloc: bool = True,
                log_level: int = logging.INFO):
        """
        Initialize memory manager.
        
        Args:
            threshold_pct: Memory usage threshold percentage to trigger warnings
            enable_tracemalloc: Whether to enable detailed memory allocation tracking
            log_level: Logging level for memory manager
        """
        self.threshold_pct = threshold_pct
        self.enable_tracemalloc = enable_tracemalloc
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Initialize memory tracking
        self.process = psutil.Process(os.getpid())
        
        if enable_tracemalloc and not tracemalloc.is_tracing():
            tracemalloc.start()
            self.logger.info("Memory allocation tracking enabled")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get current memory usage statistics.
        
        Returns:
            Dictionary with memory stats
        """
        # System memory
        system_memory = psutil.virtual_memory()
        
        # Process memory
        process_memory = self.process.memory_info()
        
        return {
            'system': {
                'total': system_memory.total,
                'available': system_memory.available,
                'used': system_memory.used,
                'percent': system_memory.percent,
            },
            'process': {
                'rss': process_memory.rss,  # Resident Set Size
                'vms': process_memory.vms,  # Virtual Memory Size
                'percent': self.process.memory_percent(),
            },
            'timestamp': time.time()
        }
    
    def log_memory_usage(self, label: str = "Memory usage") -> Dict[str, Any]:
        """
        Log current memory usage with optional label.
        
        Args:
            label: Label for log message
            
        Returns:
            Memory usage statistics
        """
        memory_stats = self.get_memory_usage()
        
        self.logger.info(
            f"{label}: Process: {memory_stats['process']['rss'] / (1024 * 1024):.2f} MB "
            f"({memory_stats['process']['percent']:.1f}%), "
            f"System: {memory_stats['system']['percent']:.1f}% used"
        )
        
        # Warning if memory usage is high
        if memory_stats['system']['percent'] > self.threshold_pct:
            self.logger.warning(
                f"High memory usage detected: {memory_stats['system']['percent']:.1f}% "
                f"(threshold: {self.threshold_pct}%)"
            )
        
        return memory_stats
    
    def take_snapshot(self) -> Optional[tracemalloc.Snapshot]:
        """
        Take a memory snapshot if tracemalloc is enabled.
        
        Returns:
            Memory snapshot or None if tracemalloc is disabled
        """
        if not self.enable_tracemalloc:
            self.logger.warning("tracemalloc is not enabled, cannot take snapshot")
            return None
        
        return tracemalloc.take_snapshot()
    
    def compare_snapshots(self, 
                        snapshot1: tracemalloc.Snapshot, 
                        snapshot2: tracemalloc.Snapshot,
                        key_type: str = 'lineno',
                        limit: int = 10) -> None:
        """
        Compare two memory snapshots and log differences.
        
        Args:
            snapshot1: First snapshot
            snapshot2: Second snapshot
            key_type: Type of statistics to show ('lineno', 'traceback', or 'filename')
            limit: Number of top memory allocations to show
        """
        if not self.enable_tracemalloc:
            self.logger.warning("tracemalloc is not enabled")
            return
        
        top_stats = snapshot2.compare_to(snapshot1, key_type)
        
        self.logger.info(f"Top {limit} memory allocations:")
        for stat in top_stats[:limit]:
            self.logger.info(stat)
    
    def cleanup_dataframes(self, 
                         dataframes: List[pd.DataFrame], 
                         collect_garbage: bool = True) -> None:
        """
        Clean up dataframes and optionally trigger garbage collection.
        
        Args:
            dataframes: List of dataframes to clean up
            collect_garbage: Whether to trigger garbage collection
        """
        if dataframes:
            for df in dataframes:
                del df
        
        if collect_garbage:
            gc.collect()
            self.log_memory_usage("Memory after cleanup")
    
    def optimize_numeric_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize memory usage by converting numeric columns to smaller dtypes.
        
        Args:
            df: Input dataframe
            
        Returns:
            Optimized dataframe
        """
        df_optimized = df.copy()
        start_mem = df.memory_usage().sum() / 1024**2
        
        # Integer columns
        int_columns = df.select_dtypes(include=['int']).columns
        for col in int_columns:
            col_min = df[col].min()
            col_max = df[col].max()
            
            # Convert to smallest possible int type
            if col_min >= 0:
                if col_max < 256:
                    df_optimized[col] = df[col].astype(np.uint8)
                elif col_max < 65536:
                    df_optimized[col] = df[col].astype(np.uint16)
                elif col_max < 4294967296:
                    df_optimized[col] = df[col].astype(np.uint32)
                else:
                    df_optimized[col] = df[col].astype(np.uint64)
            else:
                if col_min > -128 and col_max < 128:
                    df_optimized[col] = df[col].astype(np.int8)
                elif col_min > -32768 and col_max < 32768:
                    df_optimized[col] = df[col].astype(np.int16)
                elif col_min > -2147483648 and col_max < 2147483648:
                    df_optimized[col] = df[col].astype(np.int32)
                else:
                    df_optimized[col] = df[col].astype(np.int64)
        
        # Float columns
        float_columns = df.select_dtypes(include=['float']).columns
        for col in float_columns:
            df_optimized[col] = df[col].astype(np.float32)
        
        # Calculate memory savings
        end_mem = df_optimized.memory_usage().sum() / 1024**2
        reduction = (start_mem - end_mem) / start_mem * 100
        
        self.logger.info(f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB ({reduction:.1f}% reduction)")
        
        return df_optimized
    
    def memory_profile(self, func: Callable[..., T]) -> Callable[..., T]:
        """
        Decorator to profile memory usage of a function.
        
        Args:
            func: Function to profile
            
        Returns:
            Wrapper function that profiles memory usage
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Log memory before
            self.log_memory_usage(f"Memory before {func.__name__}")
            
            # Take snapshot before if tracemalloc enabled
            snapshot_before = self.take_snapshot() if self.enable_tracemalloc else None
            
            # Call function
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            # Log memory after
            self.log_memory_usage(f"Memory after {func.__name__}")
            
            # Compare snapshots if tracemalloc enabled
            if self.enable_tracemalloc and snapshot_before:
                snapshot_after = self.take_snapshot()
                self.logger.info(f"Memory allocation differences in {func.__name__}:")
                self.compare_snapshots(snapshot_before, snapshot_after)
            
            self.logger.info(f"Function {func.__name__} executed in {end_time - start_time:.2f} seconds")
            
            return result
        
        return wrapper
    
    def __del__(self):
        """Clean up tracemalloc when object is destroyed."""
        if self.enable_tracemalloc:
            tracemalloc.stop()
    
    def get_memory_stats(self):
        """Get current memory usage statistics."""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent
        }


# Usage example for the DataProcessor class
class DataProcessor:
    def __init__(self):
        self.memory_manager = MemoryManager(threshold_pct=85.0, enable_tracemalloc=True)
        
        # Decorate process_chunk method
        self.process_chunk = self.memory_manager.memory_profile(self.process_chunk)
    
    def load_and_validate_file(self, file, file_key: str) -> pd.DataFrame:
        """Loads, validates and optimizes an uploaded file."""
        df = super().load_and_validate_file(file, file_key)
        self.memory_manager.log_memory_usage(f"Memory after loading {file_key}")
        
        # Optimize memory usage
        df_optimized = self.memory_manager.optimize_numeric_dtypes(df)
        return df_optimized