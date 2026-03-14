/// GPU runtime error type.
#[derive(Debug)]
pub enum GpuError {
    /// Metal device not available (e.g. headless CI, no GPU)
    DeviceNotAvailable,
    /// Backend not initialized
    BackendNotInitialized,
    /// Invalid tensor specification
    InvalidTensor(String),
    /// Buffer allocation failed
    BufferAllocationFailed(usize),
    /// Compute operation failed
    ComputeFailed(String),
    /// Graph evaluation error
    GraphError(String),
    /// Resource limit exceeded
    ResourceLimitExceeded(String),
    /// Container not found in scheduler
    ContainerNotFound(String),
    /// Container is paused
    ContainerPaused(String),
    /// Container resource quota exceeded
    ContainerQuotaExceeded(String),
    /// Job not found in scheduler
    JobNotFound(String),
    /// Job submission rejected (queue full)
    AdmissionRejected(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceNotAvailable => write!(f, "Metal GPU device not available"),
            GpuError::BackendNotInitialized => write!(f, "Backend not initialized, call init_backend() first"),
            GpuError::InvalidTensor(msg) => write!(f, "Invalid tensor: {}", msg),
            GpuError::BufferAllocationFailed(size) => write!(f, "Failed to allocate GPU buffer of {} bytes", size),
            GpuError::ComputeFailed(msg) => write!(f, "Compute failed: {}", msg),
            GpuError::GraphError(msg) => write!(f, "Graph error: {}", msg),
            GpuError::ResourceLimitExceeded(msg) => write!(f, "Resource limit exceeded: {}", msg),
            GpuError::ContainerNotFound(msg) => write!(f, "Container not found: {}", msg),
            GpuError::ContainerPaused(msg) => write!(f, "Container paused: {}", msg),
            GpuError::ContainerQuotaExceeded(msg) => write!(f, "Container quota exceeded: {}", msg),
            GpuError::JobNotFound(msg) => write!(f, "Job not found: {}", msg),
            GpuError::AdmissionRejected(msg) => write!(f, "Admission rejected: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

pub type Result<T> = std::result::Result<T, GpuError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn error_display() {
        let e = GpuError::DeviceNotAvailable;
        assert_eq!(e.to_string(), "Metal GPU device not available");
    }

    #[test]
    fn error_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GpuError>();
    }

    #[test]
    fn scheduler_error_display() {
        let e = GpuError::ContainerNotFound("container 5".to_string());
        assert!(e.to_string().contains("container 5"));

        let e = GpuError::ContainerPaused("container 3".to_string());
        assert!(e.to_string().contains("paused"));

        let e = GpuError::ContainerQuotaExceeded("container 1: memory".to_string());
        assert!(e.to_string().contains("quota"));

        let e = GpuError::JobNotFound("job 42".to_string());
        assert!(e.to_string().contains("42"));

        let e = GpuError::AdmissionRejected("container 2: queue full".to_string());
        assert!(e.to_string().contains("queue full"));
    }
}
