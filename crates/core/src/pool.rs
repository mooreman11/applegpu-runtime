use std::collections::HashMap;

use crate::buffer::Buffer;
use crate::device::Device;
use crate::error::Result;

pub struct PoolStats {
    pub hits: u64,
    pub misses: u64,
    pub pooled_bytes: usize,
    pub bucket_count: usize,
}

pub struct BufferPool {
    buckets: HashMap<usize, Vec<Buffer>>,
    pooled_bytes: usize,
    max_pooled_bytes: usize,
    hits: u64,
    misses: u64,
}

impl BufferPool {
    pub fn new(max_pooled_bytes: usize) -> Self {
        BufferPool {
            buckets: HashMap::new(),
            pooled_bytes: 0,
            max_pooled_bytes,
            hits: 0,
            misses: 0,
        }
    }

    /// Acquire a buffer of at least `size` bytes. Rounds up to next power of two.
    /// Reuses a pooled buffer if available (hit), otherwise allocates fresh (miss).
    pub fn acquire(&mut self, device: &Device, size: usize) -> Result<Buffer> {
        let bucketed = size.next_power_of_two();
        if let Some(bufs) = self.buckets.get_mut(&bucketed) {
            if let Some(buf) = bufs.pop() {
                self.pooled_bytes -= buf.len();
                self.hits += 1;
                return Ok(buf);
            }
        }
        self.misses += 1;
        Buffer::new(device, bucketed)
    }

    /// Return a buffer to the pool for future reuse.
    /// Drops non-power-of-two buffers and buffers that would exceed the watermark.
    pub fn release(&mut self, buffer: Buffer) {
        let len = buffer.len();
        if !len.is_power_of_two() {
            // Drop non-power-of-two buffers (not from pool)
            return;
        }
        if self.pooled_bytes + len > self.max_pooled_bytes {
            // Over watermark — drop
            return;
        }
        self.pooled_bytes += len;
        self.buckets.entry(len).or_default().push(buffer);
    }

    /// Clear all pooled buffers.
    pub fn drain(&mut self) {
        self.buckets.clear();
        self.pooled_bytes = 0;
    }

    /// Update the maximum pooled bytes watermark.
    /// If the new limit is lower, evicts from the largest bucket first.
    pub fn set_max_pooled_bytes(&mut self, max: usize) {
        self.max_pooled_bytes = max;
        while self.pooled_bytes > self.max_pooled_bytes {
            // Find the largest bucket
            let largest = match self.buckets.keys().max().copied() {
                Some(k) => k,
                None => break,
            };
            if let Some(bufs) = self.buckets.get_mut(&largest) {
                if let Some(buf) = bufs.pop() {
                    self.pooled_bytes -= buf.len();
                } else {
                    self.buckets.remove(&largest);
                }
            }
            // Clean up empty buckets
            if self.buckets.get(&largest).map_or(false, |v| v.is_empty()) {
                self.buckets.remove(&largest);
            }
        }
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        PoolStats {
            hits: self.hits,
            misses: self.misses,
            pooled_bytes: self.pooled_bytes,
            bucket_count: self.buckets.len(),
        }
    }

    /// Total bytes currently pooled.
    pub fn pooled_bytes(&self) -> usize {
        self.pooled_bytes
    }

    /// Pre-warm the pool by allocating and releasing buffers for given sizes.
    /// Skips zero-size entries.
    pub fn prewarm(&mut self, device: &Device, sizes: &[usize]) -> Result<()> {
        for &size in sizes {
            if size == 0 {
                continue;
            }
            let buf = self.acquire(device, size)?;
            self.release(buf);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_device() -> Option<Device> {
        Device::new().ok()
    }

    #[test]
    fn test_acquire_returns_sufficient_size() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 100).unwrap();
        assert!(buf.len() >= 100);
    }

    #[test]
    fn test_acquire_rounds_to_power_of_two() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 100).unwrap();
        assert_eq!(buf.len(), 128); // next power of two of 100
    }

    #[test]
    fn test_release_acquire_reuses_buffer() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 256).unwrap();
        let ptr = buf.contents() as usize;
        pool.release(buf);
        let buf2 = pool.acquire(&device, 256).unwrap();
        let ptr2 = buf2.contents() as usize;
        assert_eq!(ptr, ptr2, "Reused buffer should have the same pointer");
    }

    #[test]
    fn test_watermark_eviction() {
        let device = match get_device() { Some(d) => d, None => return };
        // Set watermark very small — buffer won't be pooled
        let mut pool = BufferPool::new(64);
        let buf = pool.acquire(&device, 128).unwrap();
        pool.release(buf);
        assert_eq!(pool.pooled_bytes(), 0, "Buffer should be dropped, not pooled");
    }

    #[test]
    fn test_drain_clears_pool() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        let buf = pool.acquire(&device, 256).unwrap();
        pool.release(buf);
        assert!(pool.pooled_bytes() > 0);
        pool.drain();
        assert_eq!(pool.pooled_bytes(), 0);
    }

    #[test]
    fn test_non_power_of_two_dropped_on_release() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        // Create a buffer with a non-power-of-two size directly
        let buf = Buffer::new(&device, 100).unwrap();
        pool.release(buf);
        assert_eq!(pool.pooled_bytes(), 0, "Non-power-of-two buffer should be dropped");
    }

    #[test]
    fn test_stats_track_hits_misses() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        // First acquire is a miss
        let buf = pool.acquire(&device, 256).unwrap();
        assert_eq!(pool.stats().misses, 1);
        assert_eq!(pool.stats().hits, 0);
        // Release and re-acquire is a hit
        pool.release(buf);
        let _buf2 = pool.acquire(&device, 256).unwrap();
        assert_eq!(pool.stats().hits, 1);
        assert_eq!(pool.stats().misses, 1);
    }

    #[test]
    fn test_prewarm_populates_buckets() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        pool.prewarm(&device, &[256, 512]).unwrap();
        assert!(pool.pooled_bytes() > 0);
        // Acquiring 256 should be a hit now
        let _buf = pool.acquire(&device, 256).unwrap();
        assert_eq!(pool.stats().hits, 1);
    }

    #[test]
    fn test_set_max_pooled_bytes_evicts() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        pool.prewarm(&device, &[256, 512, 1024]).unwrap();
        let before = pool.pooled_bytes();
        assert!(before > 0);
        // Shrink watermark to 256 — should evict larger buffers
        pool.set_max_pooled_bytes(256);
        assert!(pool.pooled_bytes() <= 256);
    }

    #[test]
    fn test_acquire_empty_pool_allocates_fresh() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        // Pool is empty, so this must allocate fresh
        let buf = pool.acquire(&device, 64).unwrap();
        assert_eq!(buf.len(), 64);
        assert_eq!(pool.stats().misses, 1);
        assert_eq!(pool.stats().hits, 0);
    }

    #[test]
    fn test_prewarm_skips_zero_size() {
        let device = match get_device() { Some(d) => d, None => return };
        let mut pool = BufferPool::new(1024 * 1024);
        pool.prewarm(&device, &[0, 128]).unwrap();
        // Only the 128 entry should be pooled (rounded to 128)
        assert_eq!(pool.pooled_bytes(), 128);
    }
}
