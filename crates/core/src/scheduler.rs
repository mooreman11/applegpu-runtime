/// Multi-container GPU scheduler.
pub struct Scheduler;

impl Scheduler {
    pub fn new() -> Self {
        Scheduler
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scheduler_creates() {
        let _s = Scheduler::new();
    }
}
