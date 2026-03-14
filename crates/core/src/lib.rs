pub mod backend;
pub mod buffer;
pub mod compute;
pub mod device;
pub mod error;
pub mod ffi;
pub mod scheduler;
pub mod tensor;

pub fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_version() {
        assert_eq!(super::version(), "0.1.0");
    }
}
