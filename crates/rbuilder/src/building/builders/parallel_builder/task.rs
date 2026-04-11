use std::{cmp::Ordering, time::Instant};

use super::ConflictGroup;

/// ConflictTask provides a task for resolving a [ConflictGroup] with a specific [Algorithm].
#[derive(Debug, Clone)]
pub struct ConflictTask {
    pub group_idx: usize,
    pub algorithm: Algorithm,
    pub priority: TaskPriority,
    pub group: ConflictGroup,
    pub created_at: Instant,
}

/// TaskPriority provides a priority for a [ConflictTask].
#[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug)]
pub enum TaskPriority {
    Low = 0,
    Medium = 1,
    High = 2,
}

impl TaskPriority {
    pub fn display(&self) -> &str {
        match self {
            TaskPriority::Low => "Low",
            TaskPriority::Medium => "Medium",
            TaskPriority::High => "High",
        }
    }
}

/// [PartialEq] [Eq] [PartialOrd] [Ord] are the traits that are required for a [ConflictTask] to be used in a priority queue.
impl PartialEq for ConflictTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for ConflictTask {}

impl PartialOrd for ConflictTask {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ConflictTask {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority first, then earlier created_at
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| self.created_at.cmp(&other.created_at))
    }
}

/// Algorithm provides an algorithm for resolving a [ConflictGroup].
/// Initially these are all algorithms that produce a sequence of orders to execute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Algorithm {
    /// `Greedy` checks the following ordrerings: max profit, mev gas price
    Greedy,
    /// `GreedyFast` is a faster version of `Greedy` that handles nonces during sequence generation
    GreedyFast,
    /// `HeapGreedy` checks the following ordrerings: max profit, mev gas price using a heap like the ordering builder
    GreedyHeap,
    /// `ReverseGreedy` checks the reverse greedy orderings: e.g. min profit, min mev gas price first
    ReverseGreedy,
    /// `Length` checks the length based orderings
    Length,
    /// `AllPermutations` checks all possible permutations of the group.
    AllPermutations,
    /// `Random` checks random permutations of the group.
    Random { seed: u64, count: usize },
    /// `Genetic` uses a genetic algorithm to find near-optimal orderings.
    Genetic { population: usize, crossover_rate: f64, mutation_rate: f64, max_generations: usize, time_ms: u64, seed: u64, num_islands: usize, migration_interval: usize, w_choice: f64, early_stopping_generations: usize, temp_tight_low: f64, temp_tight_high: f64, temp_broad_low: f64, temp_broad_high: f64, tight_fraction: f64 },
    /// `RandomImproved` checks nonce-valid random permutations of the group only.
    RandomImproved { seed: u64, count: usize },
    /// `DexDirectionBalanced` scores orders by `profit_eth - lambda * impact` where impact
    /// is a pool-popularity-weighted sum of price displacements.
    DexDirectionBalanced { alpha: f64, lambda: f64 },
}

impl Algorithm {
    pub fn display(&self) -> &str {
        match self {
            Algorithm::Greedy => "Greedy",
            Algorithm::GreedyFast => "GreedyFast",
            Algorithm::GreedyHeap => "HeapGreedy",
            Algorithm::ReverseGreedy => "ReverseGreedy",
            Algorithm::Length => "Length",
            Algorithm::AllPermutations => "AllPermutations",
            Algorithm::Random { .. } => "Random",
            Algorithm::Genetic { .. } => "Genetic",
            Algorithm::RandomImproved { .. } => "RandomImproved",
            Algorithm::DexDirectionBalanced { .. } => "DexDirectionBalanced",
        }
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_priority_ordering() {
        assert!(TaskPriority::Low < TaskPriority::Medium);
        assert!(TaskPriority::Medium < TaskPriority::High);
        assert!(TaskPriority::Low < TaskPriority::High);
    }

    #[test]
    fn test_task_priority_display() {
        assert_eq!(TaskPriority::Low.display(), "Low");
        assert_eq!(TaskPriority::Medium.display(), "Medium");
        assert_eq!(TaskPriority::High.display(), "High");
    }

    #[test]
    fn test_task_priority_equality() {
        assert_eq!(TaskPriority::Low, TaskPriority::Low);
        assert_ne!(TaskPriority::Low, TaskPriority::Medium);
        assert_ne!(TaskPriority::Low, TaskPriority::High);
    }

    // to-do: test equal priority ordering by created_at
}
