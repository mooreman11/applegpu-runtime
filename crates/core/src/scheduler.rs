use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::time::Instant;

use crate::error::{GpuError, Result};
use crate::limits::{MemoryTracker, ResourceLimits};

/// Unique identifier for a container in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ContainerId(pub u64);

impl ContainerId {
    /// The default container ID (used for backward compatibility).
    pub const DEFAULT: ContainerId = ContainerId(0);
}

impl fmt::Display for ContainerId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "container-{}", self.0)
    }
}

/// Unique identifier for a job in the scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct JobId(pub u64);

impl fmt::Display for JobId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "job-{}", self.0)
    }
}

/// Job priority level. Lower numeric value = higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Priority {
    High = 0,
    Normal = 1,
    Low = 2,
}

/// Configuration for a container.
#[derive(Debug, Clone)]
pub struct ContainerConfig {
    pub priority: Priority,
    pub max_memory_bytes: usize,
    pub max_tensor_count: usize,
    /// Maximum size of a single tensor in bytes. 0 means inherit from global limits.
    pub max_tensor_size_bytes: usize,
    pub max_pending_jobs: usize,
}

/// Runtime state of a container.
pub struct ContainerState {
    pub id: ContainerId,
    pub config: ContainerConfig,
    pub limits: ResourceLimits,
    pub tracker: MemoryTracker,
    pub tensor_ids: HashSet<u64>,
    pub pending_jobs: usize,
    pub total_jobs_completed: u64,
    pub cumulative_exec_time_ns: u64,
    pub last_scheduled_at: Option<Instant>,
    pub created_at: Instant,
    pub paused: bool,
}

/// A job submitted to the scheduler.
#[derive(Debug)]
pub struct Job {
    pub id: JobId,
    pub container_id: ContainerId,
    pub target_tensor_id: u64,
    pub submitted_at: Instant,
    pub priority: Priority,
}

/// Status of a job in the scheduler.
#[derive(Debug)]
pub enum JobStatus {
    Queued,
    Running { started_at: Instant },
    Completed { tensor_id: u64, exec_time_ns: u64 },
    Failed { error: String },
}

/// Multi-container GPU scheduler with priority queues and fairness.
pub struct Scheduler {
    pub global_limits: ResourceLimits,
    pub global_tracker: MemoryTracker,
    pub containers: HashMap<ContainerId, ContainerState>,
    /// Per-priority sub-queues: index 0=High, 1=Normal, 2=Low.
    /// Each maps container ID to that container's job queue at that priority.
    pub queues: [HashMap<ContainerId, VecDeque<Job>>; 3],
    /// Job ID -> (container ID, status)
    pub jobs: HashMap<JobId, (ContainerId, JobStatus)>,
    pub tensor_owners: HashMap<u64, ContainerId>,
    next_container_id: u64,
    next_job_id: u64,
    pub starvation_threshold_ns: u64,
}

impl Scheduler {
    /// Create a new scheduler with the given global resource limits.
    /// Automatically registers the default container (ID 0) with full global limits.
    pub fn new(global_limits: ResourceLimits) -> Self {
        let mut sched = Scheduler {
            global_limits: global_limits.clone(),
            global_tracker: MemoryTracker::new(),
            containers: HashMap::new(),
            queues: [HashMap::new(), HashMap::new(), HashMap::new()],
            jobs: HashMap::new(),
            tensor_owners: HashMap::new(),
            next_container_id: 1,
            next_job_id: 0,
            starvation_threshold_ns: 10_000_000_000, // 10 seconds
        };

        // Register the default container with global limits
        let default_state = ContainerState {
            id: ContainerId::DEFAULT,
            config: ContainerConfig {
                priority: Priority::Normal,
                max_memory_bytes: global_limits.max_total_memory_bytes,
                max_tensor_count: global_limits.max_tensor_count,
                max_tensor_size_bytes: global_limits.max_tensor_size_bytes,
                max_pending_jobs: usize::MAX,
            },
            limits: global_limits,
            tracker: MemoryTracker::new(),
            tensor_ids: HashSet::new(),
            pending_jobs: 0,
            total_jobs_completed: 0,
            cumulative_exec_time_ns: 0,
            last_scheduled_at: None,
            created_at: Instant::now(),
            paused: false,
        };
        sched.containers.insert(ContainerId::DEFAULT, default_state);

        sched
    }

    /// Create a new scheduler with a custom starvation threshold (for testing).
    pub fn with_starvation_threshold(global_limits: ResourceLimits, threshold_ns: u64) -> Self {
        let mut sched = Self::new(global_limits);
        sched.starvation_threshold_ns = threshold_ns;
        sched
    }

    /// Number of registered containers (including default).
    pub fn container_count(&self) -> usize {
        self.containers.len()
    }

    /// Returns (memory_bytes, tensor_count) for a container, or None if not found.
    pub fn container_usage(&self, id: ContainerId) -> Option<(usize, usize)> {
        self.containers.get(&id).map(|c| {
            (c.tracker.memory_usage(), c.tracker.tensor_count())
        })
    }

    /// Returns (memory_bytes, tensor_count) across all containers.
    pub fn global_usage(&self) -> (usize, usize) {
        (self.global_tracker.memory_usage(), self.global_tracker.tensor_count())
    }

    /// Total number of queued jobs across all priorities and containers.
    pub fn queue_depth(&self) -> usize {
        self.queues.iter()
            .flat_map(|q| q.values())
            .map(|dq| dq.len())
            .sum()
    }

    /// Number of pending jobs for a specific container across all priority levels.
    pub fn pending_job_count(&self, id: ContainerId) -> usize {
        self.queues.iter()
            .filter_map(|q| q.get(&id))
            .map(|dq| dq.len())
            .sum()
    }

    /// Register a new container with the given configuration.
    /// Validates that the sum of all container quotas does not exceed global limits.
    pub fn register_container(&mut self, config: ContainerConfig) -> Result<ContainerId> {
        // Sum existing non-default container quotas. The default container is a
        // shared pool that dynamically uses remaining capacity, so it is excluded
        // from the conservative quota check.
        let total_memory: usize = self.containers.values()
            .filter(|c| c.id != ContainerId::DEFAULT)
            .map(|c| c.config.max_memory_bytes)
            .sum();
        let total_tensors: usize = self.containers.values()
            .filter(|c| c.id != ContainerId::DEFAULT)
            .map(|c| c.config.max_tensor_count)
            .sum();

        // Check memory quota
        if self.global_limits.max_total_memory_bytes > 0 {
            if total_memory.saturating_add(config.max_memory_bytes) > self.global_limits.max_total_memory_bytes {
                return Err(GpuError::ContainerQuotaExceeded(format!(
                    "Adding container with {} bytes quota would exceed global limit of {} bytes (current sum: {})",
                    config.max_memory_bytes, self.global_limits.max_total_memory_bytes, total_memory
                )));
            }
        }

        // Check tensor count quota
        if self.global_limits.max_tensor_count > 0 {
            if total_tensors.saturating_add(config.max_tensor_count) > self.global_limits.max_tensor_count {
                return Err(GpuError::ContainerQuotaExceeded(format!(
                    "Adding container with {} tensor quota would exceed global limit of {} (current sum: {})",
                    config.max_tensor_count, self.global_limits.max_tensor_count, total_tensors
                )));
            }
        }

        let id = ContainerId(self.next_container_id);
        self.next_container_id += 1;

        // Build ResourceLimits for this container. Inherit max_tensor_size_bytes from global if 0.
        let max_tensor_size = if config.max_tensor_size_bytes == 0 {
            self.global_limits.max_tensor_size_bytes
        } else {
            config.max_tensor_size_bytes
        };

        let limits = ResourceLimits {
            max_tensor_size_bytes: max_tensor_size,
            max_total_memory_bytes: config.max_memory_bytes,
            max_tensor_count: config.max_tensor_count,
        };

        let state = ContainerState {
            id,
            config,
            limits,
            tracker: MemoryTracker::new(),
            tensor_ids: HashSet::new(),
            pending_jobs: 0,
            total_jobs_completed: 0,
            cumulative_exec_time_ns: 0,
            last_scheduled_at: None,
            created_at: Instant::now(),
            paused: false,
        };

        self.containers.insert(id, state);
        Ok(id)
    }

    /// Deregister a container. Returns the list of tensor IDs that were owned by it.
    /// Cannot deregister the default container. Rejects if container has running jobs.
    pub fn deregister_container(&mut self, id: ContainerId) -> Result<Vec<u64>> {
        if id == ContainerId::DEFAULT {
            return Err(GpuError::ContainerNotFound(
                "Cannot deregister the default container".to_string(),
            ));
        }

        let container = self.containers.get(&id).ok_or_else(|| {
            GpuError::ContainerNotFound(format!("{}", id))
        })?;

        // Check for running jobs
        for (_job_id, (cid, status)) in &self.jobs {
            if *cid == id {
                if matches!(status, JobStatus::Running { .. }) {
                    return Err(GpuError::ContainerNotFound(format!(
                        "Cannot deregister {} with running jobs", id
                    )));
                }
            }
        }

        // Collect owned tensor IDs
        let owned_tensors: Vec<u64> = container.tensor_ids.iter().copied().collect();

        // Decrement global tracker by container's aggregate usage
        let mem = container.tracker.memory_usage();
        let count = container.tracker.tensor_count();
        for _ in 0..count {
            // We don't know individual sizes, so we'll do a bulk subtraction below
        }
        // Bulk subtract memory from global tracker
        if mem > 0 {
            // Use track_free for each tensor to decrement count, but we need to handle
            // the memory as a lump sum. Since track_free does saturating_sub for both,
            // we can call it once with total memory and then adjust count.
            // Actually, let's just call track_free count times with distributed sizes.
            // Simpler: directly manipulate via repeated calls.
            // The cleanest approach: free each tensor individually.
            // But we don't track individual tensor sizes in ContainerState.
            // For now, free the total memory in one call per "tensor" for count adjustment.
        }
        // Since MemoryTracker only exposes track_free(size_bytes) which decrements both
        // bytes and count by 1 each call, we need count calls.
        // We'll distribute memory evenly for the first (count-1) calls and remainder for last.
        if count > 0 {
            let per_tensor = mem / count;
            let remainder = mem % count;
            for i in 0..count {
                let size = if i == count - 1 { per_tensor + remainder } else { per_tensor };
                self.global_tracker.track_free(size);
            }
        }

        // Remove tensor ownership entries
        for tid in &owned_tensors {
            self.tensor_owners.remove(tid);
        }

        // Drain queued jobs for this container
        for queue in &mut self.queues {
            queue.remove(&id);
        }

        // Remove job entries for this container
        self.jobs.retain(|_, (cid, _)| *cid != id);

        // Remove the container
        self.containers.remove(&id);

        Ok(owned_tensors)
    }

    /// Update global resource limits. Also updates the default container's limits and config.
    pub fn update_global_limits(&mut self, limits: ResourceLimits) {
        self.global_limits = limits.clone();

        if let Some(default) = self.containers.get_mut(&ContainerId::DEFAULT) {
            default.config.max_memory_bytes = limits.max_total_memory_bytes;
            default.config.max_tensor_count = limits.max_tensor_count;
            default.config.max_tensor_size_bytes = limits.max_tensor_size_bytes;
            default.limits = limits;
        }
    }

    /// Pause a container. Paused containers cannot have new jobs scheduled.
    pub fn pause_container(&mut self, id: ContainerId) -> Result<()> {
        let container = self.containers.get_mut(&id).ok_or_else(|| {
            GpuError::ContainerNotFound(format!("{}", id))
        })?;
        container.paused = true;
        Ok(())
    }

    /// Resume a paused container.
    pub fn resume_container(&mut self, id: ContainerId) -> Result<()> {
        let container = self.containers.get_mut(&id).ok_or_else(|| {
            GpuError::ContainerNotFound(format!("{}", id))
        })?;
        container.paused = false;
        Ok(())
    }

    // ── Task 7: Resource tracking ──────────────────────────────────────

    /// Atomically check limits and register a tensor for a container.
    pub fn allocate_tensor(&mut self, container_id: ContainerId, tensor_id: u64, size_bytes: usize) -> Result<()> {
        let container = self.containers.get(&container_id)
            .ok_or_else(|| GpuError::ContainerNotFound(container_id.to_string()))?;

        // Check per-container limits
        container.tracker.check_allocation(size_bytes, &container.limits)
            .map_err(|_| GpuError::ContainerQuotaExceeded(format!(
                "{}: would exceed container memory/tensor limits", container_id
            )))?;

        // Check global limits
        self.global_tracker.check_allocation(size_bytes, &self.global_limits)?;

        // Both checks passed — update both trackers
        let container = self.containers.get_mut(&container_id).unwrap();
        container.tracker.track_alloc(size_bytes);
        container.tensor_ids.insert(tensor_id);
        self.global_tracker.track_alloc(size_bytes);
        self.tensor_owners.insert(tensor_id, container_id);

        Ok(())
    }

    /// Free a tensor. Looks up the owning container. No-op if tensor is unknown.
    pub fn free_tensor(&mut self, tensor_id: u64, size_bytes: usize) {
        if let Some(container_id) = self.tensor_owners.remove(&tensor_id) {
            if let Some(container) = self.containers.get_mut(&container_id) {
                container.tracker.track_free(size_bytes);
                container.tensor_ids.remove(&tensor_id);
            }
            self.global_tracker.track_free(size_bytes);
        }
    }

    /// Look up which container owns a tensor.
    pub fn tensor_owner(&self, tensor_id: u64) -> Option<ContainerId> {
        self.tensor_owners.get(&tensor_id).copied()
    }

    // ── Task 8: Job submission ─────────────────────────────────────────

    pub fn submit(&mut self, container_id: ContainerId, target_tensor_id: u64) -> Result<JobId> {
        let container = self.containers.get(&container_id)
            .ok_or_else(|| GpuError::ContainerNotFound(container_id.to_string()))?;

        if container.paused {
            return Err(GpuError::ContainerPaused(container_id.to_string()));
        }

        if container.pending_jobs >= container.config.max_pending_jobs {
            return Err(GpuError::AdmissionRejected(format!(
                "{}: pending jobs {} >= max {}",
                container_id, container.pending_jobs, container.config.max_pending_jobs
            )));
        }

        let job_id = JobId(self.next_job_id);
        self.next_job_id += 1;

        let priority = container.config.priority;
        let job = Job {
            id: job_id,
            container_id,
            target_tensor_id,
            submitted_at: Instant::now(),
            priority,
        };

        let tier = priority as usize;
        self.queues[tier]
            .entry(container_id)
            .or_insert_with(VecDeque::new)
            .push_back(job);

        self.jobs.insert(job_id, (container_id, JobStatus::Queued));
        self.containers.get_mut(&container_id).unwrap().pending_jobs += 1;

        Ok(job_id)
    }

    pub fn job_status(&self, job_id: JobId) -> Option<&(ContainerId, JobStatus)> {
        self.jobs.get(&job_id)
    }

    // ── Task 9: Job scheduling with fairness ───────────────────────────

    pub fn next_job(&mut self) -> Option<Job> {
        // Starvation prevention: promote starved lower-tier jobs
        self.promote_starved_jobs();

        // Scan tiers High -> Normal -> Low
        for tier_idx in 0..3 {
            if self.queues[tier_idx].is_empty() {
                continue;
            }

            // Find the non-paused container with the lowest cumulative_exec_time_ns
            let mut best_container: Option<ContainerId> = None;
            let mut best_time = u64::MAX;

            for (&cid, queue) in self.queues[tier_idx].iter() {
                if queue.is_empty() {
                    continue;
                }
                if let Some(container) = self.containers.get(&cid) {
                    if container.paused {
                        continue;
                    }
                    if container.cumulative_exec_time_ns < best_time {
                        best_time = container.cumulative_exec_time_ns;
                        best_container = Some(cid);
                    }
                }
            }

            if let Some(cid) = best_container {
                let queue = self.queues[tier_idx].get_mut(&cid).unwrap();
                if let Some(job) = queue.pop_front() {
                    if queue.is_empty() {
                        self.queues[tier_idx].remove(&cid);
                    }

                    let job_id = job.id;
                    self.jobs.insert(job_id, (cid, JobStatus::Running {
                        started_at: Instant::now(),
                    }));

                    if let Some(container) = self.containers.get_mut(&cid) {
                        container.last_scheduled_at = Some(Instant::now());
                    }

                    return Some(job);
                }
            }
        }

        None
    }

    fn promote_starved_jobs(&mut self) {
        let now = Instant::now();
        let threshold = self.starvation_threshold_ns;

        // Check tiers Low (2) and Normal (1) for starvation
        for tier_idx in (1..=2).rev() {
            let mut to_promote: Vec<ContainerId> = Vec::new();

            for (&cid, queue) in &self.queues[tier_idx] {
                if queue.is_empty() {
                    continue;
                }
                if let Some(container) = self.containers.get(&cid) {
                    if container.paused {
                        continue;
                    }
                    let reference_time = container.last_scheduled_at
                        .unwrap_or(container.created_at);
                    let elapsed = now.duration_since(reference_time).as_nanos() as u64;
                    let starved = elapsed > threshold;
                    if starved {
                        to_promote.push(cid);
                    }
                }
            }

            // Promote oldest job from each starved container
            for cid in to_promote {
                if let Some(queue) = self.queues[tier_idx].get_mut(&cid) {
                    if let Some(mut job) = queue.pop_front() {
                        if queue.is_empty() {
                            self.queues[tier_idx].remove(&cid);
                        }
                        let target_tier = tier_idx - 1;
                        job.priority = match target_tier {
                            0 => Priority::High,
                            1 => Priority::Normal,
                            _ => Priority::Low,
                        };
                        self.queues[target_tier]
                            .entry(cid)
                            .or_insert_with(VecDeque::new)
                            .push_back(job);
                    }
                }
            }
        }
    }

    // ── Task 10: Job completion and failure ─────────────────────────────

    pub fn complete_job(&mut self, job_id: JobId, exec_time_ns: u64) -> Result<()> {
        let (container_id, status) = self.jobs.get(&job_id)
            .ok_or_else(|| GpuError::JobNotFound(job_id.to_string()))?;

        if !matches!(status, JobStatus::Running { .. }) {
            return Err(GpuError::JobNotFound(format!(
                "{} is not in Running state", job_id
            )));
        }

        let container_id = *container_id;

        self.jobs.insert(job_id, (container_id, JobStatus::Completed {
            tensor_id: 0,
            exec_time_ns,
        }));

        if let Some(container) = self.containers.get_mut(&container_id) {
            container.cumulative_exec_time_ns += exec_time_ns;
            container.total_jobs_completed += 1;
            container.pending_jobs = container.pending_jobs.saturating_sub(1);
        }

        Ok(())
    }

    pub fn fail_job(&mut self, job_id: JobId, error: String) -> Result<()> {
        let (container_id, status) = self.jobs.get(&job_id)
            .ok_or_else(|| GpuError::JobNotFound(job_id.to_string()))?;

        if !matches!(status, JobStatus::Running { .. }) {
            return Err(GpuError::JobNotFound(format!(
                "{} is not in Running state", job_id
            )));
        }

        let container_id = *container_id;

        self.jobs.insert(job_id, (container_id, JobStatus::Failed { error }));

        if let Some(container) = self.containers.get_mut(&container_id) {
            container.pending_jobs = container.pending_jobs.saturating_sub(1);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::ResourceLimits;

    fn test_limits() -> ResourceLimits {
        ResourceLimits {
            max_tensor_size_bytes: 1024 * 1024,
            max_total_memory_bytes: 10 * 1024 * 1024,
            max_tensor_count: 100,
        }
    }

    fn test_config() -> ContainerConfig {
        ContainerConfig {
            priority: Priority::Normal,
            max_memory_bytes: 2 * 1024 * 1024,
            max_tensor_count: 20,
            max_tensor_size_bytes: 0,
            max_pending_jobs: 10,
        }
    }

    // Task 4 tests
    #[test]
    fn test_new_creates_default_container() {
        let limits = ResourceLimits::default_limits();
        let sched = Scheduler::new(limits);
        assert_eq!(sched.container_count(), 1);
        assert!(sched.container_usage(ContainerId::DEFAULT).is_some());
    }

    #[test]
    fn test_global_usage_starts_at_zero() {
        let sched = Scheduler::new(test_limits());
        assert_eq!(sched.global_usage(), (0, 0));
    }

    #[test]
    fn container_id_display() {
        assert_eq!(ContainerId(5).to_string(), "container-5");
        assert_eq!(ContainerId::DEFAULT.to_string(), "container-0");
    }

    #[test]
    fn job_id_display() {
        assert_eq!(JobId(42).to_string(), "job-42");
    }

    // Task 5 tests
    #[test]
    fn test_register_container() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        assert_eq!(sched.container_count(), 2);
        assert!(sched.container_usage(id).is_some());
    }

    #[test]
    fn test_register_exceeding_global_capacity() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 1024 * 1024,
            max_total_memory_bytes: 10 * 1024 * 1024,
            max_tensor_count: 100,
        };
        let mut sched = Scheduler::new(limits);
        // Request more memory than global allows
        let config = ContainerConfig {
            priority: Priority::Normal,
            max_memory_bytes: 11 * 1024 * 1024,
            max_tensor_count: 50,
            max_tensor_size_bytes: 0,
            max_pending_jobs: 10,
        };
        assert!(sched.register_container(config).is_err());
    }

    #[test]
    fn test_deregister_container() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        let tensors = sched.deregister_container(id).unwrap();
        assert!(tensors.is_empty());
        assert_eq!(sched.container_count(), 1);
    }

    #[test]
    fn test_cannot_deregister_default() {
        let mut sched = Scheduler::new(test_limits());
        assert!(sched.deregister_container(ContainerId::DEFAULT).is_err());
    }

    // Task 6 tests
    #[test]
    fn test_pause_resume() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.pause_container(id).unwrap();
        assert!(sched.containers.get(&id).unwrap().paused);
        sched.resume_container(id).unwrap();
        assert!(!sched.containers.get(&id).unwrap().paused);
    }

    #[test]
    fn test_pause_unknown_container() {
        let mut sched = Scheduler::new(test_limits());
        assert!(sched.pause_container(ContainerId(99)).is_err());
    }

    // Task 7 tests
    #[test]
    fn test_allocate_tensor_per_container() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.allocate_tensor(id, 1, 1024).unwrap();
        assert_eq!(sched.container_usage(id), Some((1024, 1)));
        assert_eq!(sched.global_usage(), (1024, 1));
    }

    #[test]
    fn test_allocate_tensor_exceeds_container_quota() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 10_000,
            max_tensor_count: 100,
        };
        let mut sched = Scheduler::new(limits);
        let config = ContainerConfig {
            priority: Priority::Normal,
            max_memory_bytes: 500,
            max_tensor_count: 10,
            max_tensor_size_bytes: 0,
            max_pending_jobs: 10,
        };
        let id = sched.register_container(config).unwrap();
        sched.allocate_tensor(id, 1, 400).unwrap();
        assert!(sched.allocate_tensor(id, 2, 200).is_err());
    }

    #[test]
    fn test_free_tensor() {
        let mut sched = Scheduler::new(test_limits());
        sched.allocate_tensor(ContainerId::DEFAULT, 1, 1024).unwrap();
        assert_eq!(sched.global_usage(), (1024, 1));
        sched.free_tensor(1, 1024);
        assert_eq!(sched.global_usage(), (0, 0));
    }

    #[test]
    fn test_free_unknown_tensor() {
        let mut sched = Scheduler::new(test_limits());
        sched.free_tensor(999, 1024);
        assert_eq!(sched.global_usage(), (0, 0));
    }

    #[test]
    fn test_container_isolation() {
        let limits = ResourceLimits {
            max_tensor_size_bytes: 0,
            max_total_memory_bytes: 10_000,
            max_tensor_count: 100,
        };
        let mut sched = Scheduler::new(limits);
        let config_a = ContainerConfig {
            priority: Priority::Normal,
            max_memory_bytes: 500,
            max_tensor_count: 10,
            max_tensor_size_bytes: 0,
            max_pending_jobs: 10,
        };
        let config_b = ContainerConfig {
            priority: Priority::Normal,
            max_memory_bytes: 500,
            max_tensor_count: 10,
            max_tensor_size_bytes: 0,
            max_pending_jobs: 10,
        };
        let a = sched.register_container(config_a).unwrap();
        let b = sched.register_container(config_b).unwrap();
        sched.allocate_tensor(a, 1, 500).unwrap();
        assert!(sched.allocate_tensor(a, 2, 100).is_err());
        assert!(sched.allocate_tensor(b, 3, 400).is_ok());
    }

    #[test]
    fn test_register_tensor_ownership() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.allocate_tensor(id, 42, 256).unwrap();
        assert!(sched.containers.get(&id).unwrap().tensor_ids.contains(&42));
        assert_eq!(sched.tensor_owner(42), Some(id));
    }

    // Task 8 tests
    #[test]
    fn test_submit_job() {
        let mut sched = Scheduler::new(test_limits());
        let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
        assert_eq!(sched.queue_depth(), 1);
        assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Queued))));
    }

    #[test]
    fn test_submit_rejects_paused() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.pause_container(id).unwrap();
        assert!(sched.submit(id, 42).is_err());
    }

    #[test]
    fn test_submit_admission_control() {
        let mut sched = Scheduler::new(test_limits());
        let config = ContainerConfig {
            priority: Priority::Normal,
            max_memory_bytes: 2 * 1024 * 1024,
            max_tensor_count: 20,
            max_tensor_size_bytes: 0,
            max_pending_jobs: 2,
        };
        let id = sched.register_container(config).unwrap();
        sched.submit(id, 1).unwrap();
        sched.submit(id, 2).unwrap();
        assert!(sched.submit(id, 3).is_err());
    }

    // Task 9 tests
    #[test]
    fn test_priority_ordering() {
        let mut sched = Scheduler::new(test_limits());
        let high_config = ContainerConfig {
            priority: Priority::High, max_memory_bytes: 1024*1024,
            max_tensor_count: 10, max_tensor_size_bytes: 0, max_pending_jobs: 10,
        };
        let low_config = ContainerConfig {
            priority: Priority::Low, max_memory_bytes: 1024*1024,
            max_tensor_count: 10, max_tensor_size_bytes: 0, max_pending_jobs: 10,
        };
        let low_id = sched.register_container(low_config).unwrap();
        let high_id = sched.register_container(high_config).unwrap();
        sched.submit(low_id, 1).unwrap();
        sched.submit(high_id, 2).unwrap();
        let job = sched.next_job().unwrap();
        assert_eq!(job.container_id, high_id);
    }

    #[test]
    fn test_fairness_within_tier() {
        let mut sched = Scheduler::new(test_limits());
        let config_a = ContainerConfig {
            priority: Priority::Normal, max_memory_bytes: 1024*1024,
            max_tensor_count: 10, max_tensor_size_bytes: 0, max_pending_jobs: 10,
        };
        let config_b = ContainerConfig {
            priority: Priority::Normal, max_memory_bytes: 1024*1024,
            max_tensor_count: 10, max_tensor_size_bytes: 0, max_pending_jobs: 10,
        };
        let a = sched.register_container(config_a).unwrap();
        let b = sched.register_container(config_b).unwrap();
        sched.containers.get_mut(&a).unwrap().cumulative_exec_time_ns = 1000;
        sched.containers.get_mut(&b).unwrap().cumulative_exec_time_ns = 100;
        sched.submit(a, 1).unwrap();
        sched.submit(b, 2).unwrap();
        let job = sched.next_job().unwrap();
        assert_eq!(job.container_id, b);
    }

    #[test]
    fn test_pause_skips_container() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.submit(id, 1).unwrap();
        sched.pause_container(id).unwrap();
        assert!(sched.next_job().is_none());
    }

    #[test]
    fn test_next_job_empty_queue() {
        let mut sched = Scheduler::new(test_limits());
        assert!(sched.next_job().is_none());
    }

    #[test]
    fn test_starvation_prevention() {
        use std::time::Duration;

        let limits = test_limits();
        // Use a large default threshold so normal scheduling isn't affected
        let mut sched = Scheduler::new(limits);
        let high_config = ContainerConfig {
            priority: Priority::High, max_memory_bytes: 1024*1024,
            max_tensor_count: 10, max_tensor_size_bytes: 0, max_pending_jobs: 10,
        };
        let low_config = ContainerConfig {
            priority: Priority::Low, max_memory_bytes: 1024*1024,
            max_tensor_count: 10, max_tensor_size_bytes: 0, max_pending_jobs: 10,
        };
        let high_id = sched.register_container(high_config).unwrap();
        let low_id = sched.register_container(low_config).unwrap();
        sched.submit(low_id, 1).unwrap();
        sched.submit(high_id, 2).unwrap();

        // High priority job runs first (no starvation yet, threshold is 10s)
        let job1 = sched.next_job().unwrap();
        assert_eq!(job1.container_id, high_id);
        sched.complete_job(job1.id, 1000).unwrap();

        // Submit another high-priority job
        sched.submit(high_id, 3).unwrap();

        // Now simulate starvation: set the low container's created_at far in the past
        // and lower the threshold so the low container is detected as starved.
        sched.starvation_threshold_ns = 1; // 1 nanosecond threshold
        sched.containers.get_mut(&low_id).unwrap().created_at =
            Instant::now() - Duration::from_millis(100);

        // Low-priority job should get promoted and win by fairness (exec_time 0 < 1000)
        let job2 = sched.next_job().unwrap();
        assert_eq!(job2.container_id, low_id);
    }

    // Task 10 tests
    #[test]
    fn test_job_status_transitions() {
        let mut sched = Scheduler::new(test_limits());
        let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
        assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Queued))));
        let _job = sched.next_job().unwrap();
        assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Running { .. }))));
        sched.complete_job(job_id, 5000).unwrap();
        assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Completed { .. }))));
    }

    #[test]
    fn test_complete_job_not_running() {
        let mut sched = Scheduler::new(test_limits());
        let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
        assert!(sched.complete_job(job_id, 5000).is_err());
    }

    #[test]
    fn test_complete_already_completed() {
        let mut sched = Scheduler::new(test_limits());
        let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
        let _job = sched.next_job().unwrap();
        sched.complete_job(job_id, 5000).unwrap();
        assert!(sched.complete_job(job_id, 5000).is_err());
    }

    #[test]
    fn test_fail_job() {
        let mut sched = Scheduler::new(test_limits());
        let job_id = sched.submit(ContainerId::DEFAULT, 42).unwrap();
        let _job = sched.next_job().unwrap();
        sched.fail_job(job_id, "eval error".to_string()).unwrap();
        assert!(matches!(sched.job_status(job_id), Some(&(_, JobStatus::Failed { .. }))));
    }

    #[test]
    fn test_deregister_rejects_running_jobs() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.submit(id, 42).unwrap();
        let _job = sched.next_job().unwrap();
        assert!(sched.deregister_container(id).is_err());
    }

    #[test]
    fn test_deregister_drains_queued_jobs() {
        let mut sched = Scheduler::new(test_limits());
        let id = sched.register_container(test_config()).unwrap();
        sched.submit(id, 1).unwrap();
        sched.submit(id, 2).unwrap();
        assert_eq!(sched.queue_depth(), 2);
        sched.deregister_container(id).unwrap();
        assert_eq!(sched.queue_depth(), 0);
    }
}
