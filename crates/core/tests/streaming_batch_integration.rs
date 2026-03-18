use applegpu_core::compute;
use applegpu_core::device::Device;
use std::sync::Mutex;

// Streaming batch uses global static state, so tests must run serially.
static TEST_LOCK: Mutex<()> = Mutex::new(());

fn get_device() -> Option<Device> {
    Device::new().ok()
}

#[test]
fn streaming_batch_basic_lifecycle() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let queue = compute::get_shared_queue(&device);
    assert!(!queue.is_null());
    assert!(!compute::streaming_is_active());
    compute::begin_streaming_batch(queue);
    assert!(compute::streaming_is_active());
    compute::flush_streaming_batch();
    assert!(compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_idempotent_begin() {
    let _guard = TEST_LOCK.lock().unwrap();
    let device = match get_device() {
        Some(d) => d,
        None => return,
    };
    let queue = compute::get_shared_queue(&device);
    compute::begin_streaming_batch(queue);
    compute::begin_streaming_batch(queue); // no-op
    assert!(compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_end_when_inactive() {
    let _guard = TEST_LOCK.lock().unwrap();
    assert!(!compute::streaming_is_active());
    compute::end_streaming_batch();
    assert!(!compute::streaming_is_active());
}

#[test]
fn streaming_batch_flush_when_inactive() {
    let _guard = TEST_LOCK.lock().unwrap();
    assert!(!compute::streaming_is_active());
    compute::flush_streaming_batch();
    assert!(!compute::streaming_is_active());
}
