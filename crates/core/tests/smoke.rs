use applegpu_core;

#[test]
fn integration_version() {
    assert!(!applegpu_core::version().is_empty());
}
