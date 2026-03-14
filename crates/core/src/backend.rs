use crate::device::Device;
use crate::error::{GpuError, Result};
use once_cell::sync::OnceCell;

/// Available GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    /// Direct MLX → Metal execution (default, high performance)
    Mlx,
    /// VM-mediated Metal execution via Apple Virtualization Framework
    Vm,
}

impl std::str::FromStr for Backend {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "mlx" => Ok(Backend::Mlx),
            "vm" => Ok(Backend::Vm),
            other => Err(format!("Unknown backend: {}", other)),
        }
    }
}

/// Global runtime state.
static RUNTIME: OnceCell<Runtime> = OnceCell::new();

/// Runtime holds the initialized backend and device.
pub struct Runtime {
    pub backend: Backend,
    pub device: Device,
    /// Socket path for VM backend IPC (None for MLX backend).
    pub socket_path: Option<String>,
}

/// Initialize the GPU backend. Reads `APPLEGPU_BACKEND` env var,
/// defaults to MLX. Can only be called once.
pub fn init_backend() -> Result<&'static Runtime> {
    RUNTIME.get_or_try_init(|| {
        let backend = std::env::var("APPLEGPU_BACKEND")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(Backend::Mlx);

        let socket_path = if backend == Backend::Vm {
            Some(std::env::var("APPLEGPU_SOCKET")
                .unwrap_or_else(|_| crate::ipc::default_socket_path()))
        } else {
            None
        };

        let device = Device::new()?;

        Ok(Runtime { backend, device, socket_path })
    })
}

/// Get the runtime if already initialized.
pub fn get_runtime() -> Result<&'static Runtime> {
    RUNTIME.get().ok_or(GpuError::BackendNotInitialized)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_from_str() {
        assert_eq!("mlx".parse::<Backend>().unwrap(), Backend::Mlx);
        assert_eq!("MLX".parse::<Backend>().unwrap(), Backend::Mlx);
        assert_eq!("vm".parse::<Backend>().unwrap(), Backend::Vm);
        assert_eq!("VM".parse::<Backend>().unwrap(), Backend::Vm);
        assert!("invalid".parse::<Backend>().is_err());
    }

    // Note: init_backend() uses OnceCell so it can only be tested once per process.
    // The integration test in tests/ handles the full init flow.
}
