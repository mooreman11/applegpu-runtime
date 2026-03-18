use crate::error::{GpuError, Result};

// Note: MemoryTracker uses plain usize (not AtomicUsize) because LazyRuntime
// is behind a Mutex<LazyRuntime> in the Python layer, which serializes all access.

/// Configurable resource limits for the GPU runtime.
#[derive(Debug, Clone)]
pub struct ResourceLimits {
    /// Maximum size of a single tensor in bytes (0 = unlimited).
    pub max_tensor_size_bytes: usize,
    /// Maximum total GPU memory usage across all tensors in bytes (0 = unlimited).
    pub max_total_memory_bytes: usize,
    /// Maximum number of live tensors (0 = unlimited).
    pub max_tensor_count: usize,
}

impl ResourceLimits {
    /// Default limits: 1GB per tensor, 8GB total, 100000 tensors.
    /// Generous defaults to avoid quota errors during multi-epoch training
    /// with large models. Use set_limits() or env vars to restrict.
    pub fn default_limits() -> Self {
        ResourceLimits {
            max_tensor_size_bytes: 1024 * 1024 * 1024,
            max_total_memory_bytes: 8_usize * 1024 * 1024 * 1024,
            max_tensor_count: 100_000,
        }
    }

    /// No limits (unlimited).
    pub fn unlimited() -> Self {
        ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 0,
            max_tensor_count: 0,
        }
    }

    /// Load limits from environment variables, falling back to defaults.
    pub fn from_env() -> Self {
        let mut limits = Self::default_limits();

        if let Ok(val) = std::env::var("APPLEGPU_MAX_TENSOR_SIZE_MB") {
            if let Ok(mb) = val.parse::<usize>() {
                limits.max_tensor_size_bytes = mb * 1024 * 1024;
            }
        }
        if let Ok(val) = std::env::var("APPLEGPU_MAX_MEMORY_MB") {
            if let Ok(mb) = val.parse::<usize>() {
                limits.max_total_memory_bytes = mb * 1024 * 1024;
            }
        }
        if let Ok(val) = std::env::var("APPLEGPU_MAX_TENSORS") {
            if let Ok(n) = val.parse::<usize>() {
                limits.max_tensor_count = n;
            }
        }

        limits
    }
}

/// Tracks current GPU memory usage.
pub struct MemoryTracker {
    current_bytes: usize,
    current_count: usize,
}

impl MemoryTracker {
    pub fn new() -> Self {
        MemoryTracker {
            current_bytes: 0,
            current_count: 0,
        }
    }

    /// Check if allocating `size_bytes` would exceed limits.
    pub fn check_allocation(&self, size_bytes: usize, limits: &ResourceLimits) -> Result<()> {
        if limits.max_tensor_size_bytes > 0 && size_bytes > limits.max_tensor_size_bytes {
            return Err(GpuError::ResourceLimitExceeded(format!(
                "Tensor size {} bytes exceeds limit of {} bytes ({} MB)",
                size_bytes, limits.max_tensor_size_bytes,
                limits.max_tensor_size_bytes / (1024 * 1024)
            )));
        }

        if limits.max_total_memory_bytes > 0 {
            if self.current_bytes + size_bytes > limits.max_total_memory_bytes {
                return Err(GpuError::ResourceLimitExceeded(format!(
                    "Total GPU memory would exceed limit: current {} + new {} > limit {} bytes ({} MB)",
                    self.current_bytes, size_bytes, limits.max_total_memory_bytes,
                    limits.max_total_memory_bytes / (1024 * 1024)
                )));
            }
        }

        if limits.max_tensor_count > 0 {
            if self.current_count >= limits.max_tensor_count {
                return Err(GpuError::ResourceLimitExceeded(format!(
                    "Tensor count {} would exceed limit of {}",
                    self.current_count + 1, limits.max_tensor_count
                )));
            }
        }

        Ok(())
    }

    pub fn track_alloc(&mut self, size_bytes: usize) {
        self.current_bytes += size_bytes;
        self.current_count += 1;
    }

    pub fn track_free(&mut self, size_bytes: usize) {
        self.current_bytes = self.current_bytes.saturating_sub(size_bytes);
        self.current_count = self.current_count.saturating_sub(1);
    }

    pub fn memory_usage(&self) -> usize {
        self.current_bytes
    }

    pub fn tensor_count(&self) -> usize {
        self.current_count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_limits_are_reasonable() {
        let limits = ResourceLimits::default_limits();
        assert_eq!(limits.max_tensor_size_bytes, 1024 * 1024 * 1024);
        assert_eq!(limits.max_total_memory_bytes, 8_usize * 1024 * 1024 * 1024);
        assert_eq!(limits.max_tensor_count, 100_000);
    }

    #[test]
    fn unlimited_allows_anything() {
        let limits = ResourceLimits::unlimited();
        let tracker = MemoryTracker::new();
        assert!(tracker.check_allocation(usize::MAX / 2, &limits).is_ok());
    }

    #[test]
    fn tensor_size_limit_enforced() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 1024,
            max_total_memory_bytes: 0,
            max_tensor_count: 0,
        };
        let tracker = MemoryTracker::new();
        assert!(tracker.check_allocation(512, &limits).is_ok());
        assert!(tracker.check_allocation(2048, &limits).is_err());
    }

    #[test]
    fn total_memory_limit_enforced() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 1024,
            max_tensor_count: 0,
        };
        let mut tracker = MemoryTracker::new();
        tracker.track_alloc(512);
        assert!(tracker.check_allocation(256, &limits).is_ok());
        assert!(tracker.check_allocation(1024, &limits).is_err());
    }

    #[test]
    fn tensor_count_limit_enforced() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 0,
            max_tensor_count: 2,
        };
        let mut tracker = MemoryTracker::new();
        assert!(tracker.check_allocation(64, &limits).is_ok());
        tracker.track_alloc(64);
        assert!(tracker.check_allocation(64, &limits).is_ok());
        tracker.track_alloc(64);
        assert!(tracker.check_allocation(64, &limits).is_err());
    }

    #[test]
    fn track_free_decreases_usage() {
        let mut tracker = MemoryTracker::new();
        tracker.track_alloc(1024);
        tracker.track_alloc(2048);
        assert_eq!(tracker.memory_usage(), 3072);
        assert_eq!(tracker.tensor_count(), 2);
        tracker.track_free(1024);
        assert_eq!(tracker.memory_usage(), 2048);
        assert_eq!(tracker.tensor_count(), 1);
    }

    #[test]
    fn track_free_underflow_does_not_panic() {
        let mut tracker = MemoryTracker::new();
        tracker.track_alloc(100);
        // Free more than allocated — should not panic
        tracker.track_free(200);
        assert_eq!(tracker.memory_usage(), 0);
        assert_eq!(tracker.tensor_count(), 0);
    }
}
