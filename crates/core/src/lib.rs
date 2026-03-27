pub mod backend;
pub mod backend_ffi;
pub mod compiled_graph;
pub mod buffer;
pub mod compute;
pub mod device;
pub mod eager;
pub mod eager_ffi;
pub mod error;
pub mod ffi;
pub mod fusion;
pub mod graph;
pub mod ipc;
pub mod kernel_templates;
pub mod limits;
pub mod lazy;
pub mod ops;
pub mod pool;
pub mod remote_eager;
pub mod scheduler;
pub mod serial;
pub mod tensor;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert_eq!(super::version(), "0.9.0");
    }
}
