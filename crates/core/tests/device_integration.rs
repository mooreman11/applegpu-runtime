use applegpu_core::device::Device;

#[test]
fn metal_device_name_contains_apple() {
    match Device::new() {
        Ok(device) => {
            let name = device.name();
            // Apple Silicon device names contain "Apple"
            assert!(
                name.contains("Apple"),
                "Expected device name to contain 'Apple', got: {}",
                name
            );
        }
        Err(_) => {
            // No Metal GPU available (CI) — skip
        }
    }
}

#[test]
fn multiple_devices_independent() {
    let d1 = Device::new();
    let d2 = Device::new();
    match (d1, d2) {
        (Ok(dev1), Ok(dev2)) => {
            assert_eq!(dev1.name(), dev2.name());
        }
        _ => {
            // No GPU — skip
        }
    }
}
