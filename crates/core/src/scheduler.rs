use std::fmt;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn container_id_display() {
        assert_eq!(ContainerId(5).to_string(), "container-5");
        assert_eq!(ContainerId::DEFAULT.to_string(), "container-0");
    }

    #[test]
    fn container_id_is_copy_hash() {
        let id = ContainerId(1);
        let id2 = id; // Copy
        assert_eq!(id, id2);

        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(id);
        assert!(set.contains(&id));
    }

    #[test]
    fn job_id_display() {
        assert_eq!(JobId(42).to_string(), "job-42");
    }
}
