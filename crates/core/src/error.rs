/// GPU runtime error type.
#[derive(Debug)]
pub enum GpuError {
    /// Metal device not available (e.g. headless CI, no GPU)
    DeviceNotAvailable,
    /// Backend not initialized
    BackendNotInitialized,
    /// Invalid tensor specification
    InvalidTensor(String),
}

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuError::DeviceNotAvailable => write!(f, "Metal GPU device not available"),
            GpuError::BackendNotInitialized => write!(f, "Backend not initialized, call init_backend() first"),
            GpuError::InvalidTensor(msg) => write!(f, "Invalid tensor: {}", msg),
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
}
