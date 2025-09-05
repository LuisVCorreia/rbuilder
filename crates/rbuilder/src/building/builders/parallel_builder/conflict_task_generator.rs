use crate::primitives::SimulatedOrder;
use ahash::{HashMap, HashSet};
use alloy_primitives::{utils::format_ether, U256};
use crossbeam_queue::SegQueue;
use itertools::Itertools;
use std::{sync::Arc, time::Instant};
use tracing::trace;
use super::nonce_handling::{
    compute_interleaving_stats, NonceLayout, is_simple_chain,
    ALL_PERMS_CAP as MULTINOMIAL_ALL_PERMS_THRESHOLD,
};
use super::{
    task::ConflictTask, Algorithm, ConflictGroup, ConflictResolutionResultPerGroup, GroupId,
    ResolutionResult, TaskPriority, TaskQueue,
};
use std::sync::mpsc as std_mpsc;

const THRESHOLD_FOR_SIGNIFICANT_CHANGE: u64 = 20;
const NUMBER_OF_TOP_ORDERS_TO_CONSIDER_FOR_SIGNIFICANT_CHANGE: usize = 10;
const NUMBER_OF_RANDOM_TASKS: usize = 50;

/// Manages conflicts and updates for conflict groups, coordinating with a worker pool to process tasks.
pub struct ConflictTaskGenerator {
    existing_groups: HashMap<GroupId, ConflictGroup>,
    task_queue: TaskQueue,
    group_result_sender: std_mpsc::Sender<ConflictResolutionResultPerGroup>,
}

impl ConflictTaskGenerator {
    /// Creates a new [ConflictTaskGenerator] instance responsible for generating and managing tasks for conflict groups.
    ///
    /// # Arguments
    ///
    /// * `task_queue` - The queue to store the generated tasks.
    /// * `group_result_sender` - The sender to send the results of the conflict resolution.
    pub fn new(
        task_queue: TaskQueue,
        group_result_sender: std_mpsc::Sender<ConflictResolutionResultPerGroup>,
    ) -> Self {
        Self {
            existing_groups: HashMap::default(),
            task_queue,
            group_result_sender,
        }
    }

    /// Processes a vector of new conflict groups, updating existing groups or creating new tasks as necessary.
    ///
    /// # Arguments
    ///
    /// * `new_groups` - A vector of new [ConflictGroup]s to process.
    pub fn process_groups(&mut self, new_groups: Vec<ConflictGroup>) {
        let mut sorted_groups = new_groups;
        sorted_groups.sort_by(|a, b| b.orders.len().cmp(&a.orders.len()));

        let mut processed_groups = HashSet::default();
        for new_group in sorted_groups {
            // If the group has already been processed, skip it
            if self.has_been_processed(&new_group.id, &processed_groups) {
                continue;
            }

            // Process the group
            self.process_single_group(new_group.clone());

            // Add the new group and all its subsets to the processed groups
            self.add_processed_groups(&new_group, &mut processed_groups);

            // Remove all subset groups
            if new_group.conflicting_group_ids.len() > 0 {
                self.remove_conflicting_subset_groups(&new_group);
            }
        }
    }

    /// Adds the group and all the conflicting group subsets to the processed groups
    fn add_processed_groups(
        &mut self,
        new_group: &ConflictGroup,
        processed_groups: &mut HashSet<GroupId>,
    ) {
        // Add all subset groups to the processed groups
        let subset_ids: Vec<GroupId> = self
            .existing_groups
            .keys()
            .filter(|&id| new_group.conflicting_group_ids.contains(id))
            .cloned()
            .collect();
        for id in subset_ids {
            processed_groups.insert(id);
        }

        // Add the new group to the processed groups
        processed_groups.insert(new_group.id);
    }

    /// Checks if a group has been processed
    fn has_been_processed(&self, group_id: &GroupId, processed_groups: &HashSet<GroupId>) -> bool {
        processed_groups.contains(group_id)
    }

    /// Removes the subset groups from the existing groups and cancels the tasks for those groups
    fn remove_conflicting_subset_groups(&mut self, superset_group: &ConflictGroup) {
        let subset_ids: Vec<GroupId> = self
            .existing_groups
            .keys()
            .filter(|&id| superset_group.conflicting_group_ids.contains(id))
            .cloned()
            .collect();

        trace!(groups = ?subset_ids,"Removing subset groups");
        for id in subset_ids {
            self.existing_groups.remove(&id);
            self.cancel_tasks_for_group(id);
        }
    }

    /// Processes a single order group, determining whether to update an existing group or create new tasks.
    ///
    /// This method checks if the group has changed since it was last processed. If there are no changes,
    /// it skips further processing. For changed groups, it determines the appropriate priority and
    /// either updates existing tasks or creates new ones.
    ///
    /// # Arguments
    ///
    /// * `new_group` - The new `ConflictGroup` to process.
    fn process_single_group(&mut self, new_group: ConflictGroup) {
        let group_id = new_group.id;

        let should_process = match self.existing_groups.get(&group_id) {
            Some(existing_group) => self.group_has_changed(&new_group, existing_group),
            None => true,
        };

        if should_process {
            if new_group.orders.len() == 1 {
                self.process_single_order_group(group_id, &new_group);
            } else {
                self.process_multi_order_group(group_id, &new_group);
            }
            self.existing_groups.insert(group_id, new_group);
        }
    }

    /// Processes a multi-order group, determining whether to update existing tasks or create new ones and the corresponding priority
    /// We give updates higher priority if they are significant changes
    ///
    /// # Arguments
    ///
    /// * `group_id` - The ID of the group to process.
    /// * `new_group` - The new `ConflictGroup` to process.
    fn process_multi_order_group(&mut self, group_id: GroupId, new_group: &ConflictGroup) {
        let priority = if let Some(existing_group) = self.existing_groups.get(&group_id) {
            if self.has_significant_changes(new_group, existing_group) {
                TaskPriority::High
            } else {
                TaskPriority::Medium
            }
        } else {
            TaskPriority::High
        };
        trace!(
            group = group_id,
            order_count = new_group.orders.len(),
            profit =
                format_ether(self.sum_top_n_profits(&new_group.orders, new_group.orders.len())),
            priority = priority.display(),
            "Processing multi order group"
        );
        if self.existing_groups.contains_key(&group_id) {
            self.update_tasks(group_id, new_group, priority);
        } else {
            self.create_new_tasks(new_group, priority);
        }
    }

    /// Processes a single order group, sending the result to the worker pool.
    ///
    /// # Arguments
    ///
    /// * `group_id` - The ID of the group to process.
    /// * `group` - The `ConflictGroup` to process.
    fn process_single_order_group(&mut self, group_id: GroupId, group: &ConflictGroup) {
        let sequence_of_orders = ResolutionResult::new(
            group.orders[0].sim_value.coinbase_profit,
            group.orders[0].sim_value.gas_used,
            vec![(0, group.orders[0].sim_value.coinbase_profit, group.orders[0].sim_value.gas_used)],
        );
        // We ignore the error since it means "receiver disconnected" and we expect the caller will detect the cancellation and stop calling us.
        let _ = self
            .group_result_sender
            .send((group_id, (sequence_of_orders, group.clone())));
    }

    /// Determines if there are any changes between a new group and an existing group.
    ///
    /// This method compares the number of orders and each individual order in both groups
    /// to detect any changes.
    ///
    /// # Arguments
    ///
    /// * `new_group` - The new `ConflictGroup` to compare.
    /// * `existing_group` - The existing `ConflictGroup` to compare against.
    ///
    /// # Returns
    ///
    /// `true` if there are any changes between the groups, `false` otherwise.
    fn group_has_changed(&self, new_group: &ConflictGroup, existing_group: &ConflictGroup) -> bool {
        // Compare the number of orders
        if new_group.orders.len() != existing_group.orders.len() {
            return true;
        }

        // Compare each order in the groups
        for (new_order, existing_order) in new_group.orders.iter().zip(existing_group.orders.iter())
        {
            if new_order != existing_order {
                return true;
            }
        }

        false
    }

    /// Determines if there are significant changes between a new group and an existing group.
    ///
    /// # Arguments
    ///
    /// * `new_group` - The new `ConflictGroup` to compare.
    /// * `existing_group` - The existing `ConflictGroup` to compare against.
    ///
    /// # Returns
    ///
    /// `true` if there are significant changes, `false` otherwise.
    fn has_significant_changes(
        &self,
        new_group: &ConflictGroup,
        existing_group: &ConflictGroup,
    ) -> bool {
        let new_top_profit = self.sum_top_n_profits(
            &new_group.orders,
            NUMBER_OF_TOP_ORDERS_TO_CONSIDER_FOR_SIGNIFICANT_CHANGE,
        );
        let existing_top_profit = self.sum_top_n_profits(
            &existing_group.orders,
            NUMBER_OF_TOP_ORDERS_TO_CONSIDER_FOR_SIGNIFICANT_CHANGE,
        );

        if new_top_profit == existing_top_profit {
            return false;
        }

        if new_top_profit > existing_top_profit {
            // if new_top_profit is more than 20% higher than existing_top_profit, we consider it a significant change
            let threshold = existing_top_profit * U256::from(THRESHOLD_FOR_SIGNIFICANT_CHANGE)
                / U256::from(100);
            if new_top_profit - existing_top_profit > threshold {
                return true;
            }
        }
        false
    }

    /// Calculates the sum of the top N profits from a slice of simulated orders.
    ///
    /// # Arguments
    ///
    /// * `orders` - A slice of `SimulatedOrder`s to process.
    /// * `n` - The number of top profits to sum.
    ///
    /// # Returns
    ///
    /// The sum of the top N profits as a `U256`.
    fn sum_top_n_profits(&self, orders: &[Arc<SimulatedOrder>], n: usize) -> U256 {
        orders
            .iter()
            .map(|o| o.sim_value.coinbase_profit)
            .sorted_by(|a, b| b.cmp(a))
            .take(n)
            .sum()
    }

    /// Updates tasks for a given group, cancelling existing tasks and creating new ones.
    ///
    /// # Arguments
    ///
    /// * `group_id` - The ID of the group to update.
    /// * `new_group` - The new `ConflictGroup` to use for task creation.
    /// * `priority` - The priority to assign to the new tasks.
    fn update_tasks(
        &mut self,
        group_id: GroupId,
        new_group: &ConflictGroup,
        priority: TaskPriority,
    ) {
        trace!(
            group = group_id,
            priority = priority.display(),
            "Updating tasks",
        );
        // Cancel existing tasks for this grou
        self.cancel_tasks_for_group(group_id);

        // Create new tasks for the updated group
        self.create_new_tasks(new_group, priority);
    }

    /// Cancels all tasks for a given group
    fn cancel_tasks_for_group(&mut self, group_id: GroupId) {
        let temp_queue = SegQueue::new();

        while let Some(task) = self.task_queue.pop() {
            if task.group_idx != group_id {
                temp_queue.push(task);
            }
        }

        while let Some(task) = temp_queue.pop() {
            self.task_queue.push(task);
        }
    }

    /// Creates and spawns new tasks for a given order group.
    ///
    /// # Arguments
    ///
    /// * `new_group` - The `ConflictGroup` to create tasks for.
    /// * `priority` - The priority to assign to the tasks.
    fn create_new_tasks(&mut self, new_group: &ConflictGroup, priority: TaskPriority) {
        let tasks = get_tasks_for_group(new_group, priority);
        for task in tasks {
            self.task_queue.push(task);
        }
    }
}

use std::sync::OnceLock;
static RB_ALLOWLIST: OnceLock<HashMap<u64, HashSet<usize>>> = OnceLock::new();

fn selected_groups() -> &'static HashMap<u64, HashSet<usize>> {
    RB_ALLOWLIST.get_or_init(|| {
        let mut m: HashMap<u64, HashSet<usize>> = HashMap::default();

        // helper to insert
        let mut ins = |bn: u64, gid: usize| { m.entry(bn).or_default().insert(gid); };

        ins(22274264, 104);
        ins(20314472, 21);
        ins(21626181, 10);
        ins(20311195, 4);
        ins(22489877, 9);
        ins(22485725, 43);
        ins(22485138, 5);
        ins(20100348, 3);
        ins(19874363, 2);
        ins(20969654, 13);
        ins(19657221, 115);
        ins(22712795, 38);
        ins(22712795, 32);
        ins(21195356, 125);
        ins(21195356, 22);
        ins(19660788, 147);
        ins(19660788, 158);
        ins(20967865, 21);
        ins(21854682, 57);
        ins(21854682, 65);
        ins(21408989, 27);
        ins(21405709, 5);
        ins(21191186, 0);
        ins(22273666, 14);
        ins(22273968, 0);
        ins(21628569, 21);
        ins(21628569, 5);
        ins(22713092, 36);
        ins(20098565, 59);
        ins(20097670, 4);
        ins(19871974, 9);
        ins(20752606, 7);
        ins(22053356, 0);
        ins(21195654, 11);
        ins(20096476, 22);
        ins(20533017, 55);
        ins(20099460, 81);
        ins(19662577, 40);
        ins(22277248, 96);
        ins(20758891, 84);
        ins(20758891, 2);
        ins(20534208, 58);
        ins(20534208, 7);
        ins(22055148, 3);
        ins(21627080, 0);
        ins(21627080, 23);
        ins(19664061, 28);
        ins(19664061, 49);
        ins(19877353, 47);
        ins(21850521, 69);
        ins(19663766, 24);
        ins(21630657, 5);
        ins(19878248, 6);
        ins(21407499, 13);
        ins(20094404, 14);
        ins(21408395, 6);
        ins(21406904, 25);
        ins(22708631, 13);
        ins(21191485, 3);
        ins(20970547, 21);
        ins(20969056, 18);
        ins(20314771, 10);
        ins(20967265, 5);
        ins(22276948, 5);
        ins(20093805, 45);
        ins(20095290, 179);
        ins(22051261, 13);
        ins(22276053, 0);
        ins(21194167, 18);
        ins(21631553, 71);
        ins(21191783, 15);
        ins(22490176, 31);
        ins(19876754, 0);
        ins(22274564, 1);

        m
    })
}

#[inline]
fn is_selected_group(block_number: u64, group_id: usize) -> bool {
    selected_groups()
        .get(&block_number)
        .map_or(false, |set| set.contains(&group_id))
}

/// Optional, non-breaking wrapper: pass `Some(block_number)` to filter,
/// or `None` to behave like the legacy `get_tasks_for_group`.
pub fn get_tasks_for_group_filtered(
    group: &ConflictGroup,
    priority: TaskPriority,
    block_number: Option<u64>,
) -> Vec<ConflictTask> {
    if let Some(bn) = block_number {
        if !is_selected_group(bn, group.id) {
            // Skip this group entirely
            return vec![];
        }
    }
    // Fall back to the existing policy
    get_tasks_for_group(group, priority)
}


/// Generates a vector of conflict tasks for a given order group.
///
/// # Arguments
///
/// * `group` - The `ConflictGroup` to create tasks for.
/// * `priority` - The priority to assign to the tasks.
///
/// # Returns
///
/// A vector of `ConflictTask`s for the given group.
pub fn get_tasks_for_group(group: &ConflictGroup, priority: TaskPriority) -> Vec<ConflictTask> {
    let mut tasks = vec![];
    let created_at = Instant::now();

    if let Some(_layout) = NonceLayout::from_group(&group) {
        if is_simple_chain(&group) {
            // Single sender: only AllPermutations (no need for Greedy/others)
            tasks.push(ConflictTask {
                group_idx: group.id,
                algorithm: Algorithm::AllPermutations,
                priority,
                group: group.clone(),
                created_at,
            });
            return tasks;
        }

        // Always try Greedy first (fast baseline)
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::Greedy,
            priority,
            group: group.clone(),
            created_at,
        });

        if let Some(stats) = compute_interleaving_stats(group) {
            let ln_cap = (MULTINOMIAL_ALL_PERMS_THRESHOLD as f64).ln();
            let small = stats.ln_with_choice <= ln_cap + 1e-12;

            println!("Group {}: {} orders, multinomial {}, ln={}{}",
                group.id,
                group.orders.len(),
                stats.ln_multinomial,
                stats.ln_with_choice,
                if small { "" } else { " (too large)" }
            );

            if small {
                tasks.push(ConflictTask {
                    group_idx: group.id,
                    algorithm: Algorithm::AllPermutations,
                    priority,
                    group: group.clone(),
                    created_at,
                });
            } else {
                if group.id == 107 {
                    tasks.push(ConflictTask {
                        group_idx: group.id,
                        algorithm: Algorithm::Genetic {
                            population: 30,
                            crossover_rate: 0.9,
                            mutation_rate: 0.2,
                            tourn_k: 3,
                            max_generations: 50,
                            time_ms: 6000,
                            seed: group.id as u64,
                        },
                        priority: TaskPriority::Medium,
                        group: group.clone(),
                        created_at,
                    });
                }

                // tasks.push(ConflictTask {
                //     group_idx: group.id,
                //     algorithm: Algorithm::Genetic {
                //         population: 20,
                //         crossover_rate: 0.9,
                //         mutation_rate: 0.4,
                //         tourn_k: 2,
                //         max_generations: 100,
                //         time_ms: 3000,
                //         seed: group.id as u64,
                //     },
                //     priority: TaskPriority::Medium,
                //     group: group.clone(),
                //     created_at,
                // });

                // tasks.push(ConflictTask {
                //         group_idx: group.id,
                //         algorithm: Algorithm::ExhaustiveStreaming {
                //             time_ms: 5_000, // 1 hour
                //             top_k: 20,
                //         },
                //         priority: TaskPriority::Low,
                //         group: group.clone(),
                //         created_at,
                //     });

                tasks.push(ConflictTask {
                    group_idx: group.id,
                    algorithm: Algorithm::Random {
                        seed: group.id as u64,
                        count: NUMBER_OF_RANDOM_TASKS,
                    },
                    priority: TaskPriority::Low,
                    group: group.clone(),
                    created_at,
                });

                // tasks.push(ConflictTask {
                //     group_idx: group.id,
                //     algorithm: Algorithm::RandomChain {
                //         seed: group.id as u64,
                //         count: NUMBER_OF_RANDOM_TASKS,
                //     },
                //     priority: TaskPriority::Low,
                //     group: group.clone(),
                //     created_at,
                // });

                tasks.push(ConflictTask {
                    group_idx: group.id,
                    algorithm: Algorithm::RandomImproved {
                        seed: group.id as u64,
                        count: NUMBER_OF_RANDOM_TASKS,
                    },
                    priority: TaskPriority::Low,
                    group: group.clone(),
                    created_at,
                });

                // tasks.push(ConflictTask {
                //     group_idx: group.id,
                //     algorithm: Algorithm::Length,
                //     priority: TaskPriority::Low,
                //     group: group.clone(),
                //     created_at,
                // });
                // tasks.push(ConflictTask {
                //     group_idx: group.id,
                //     algorithm: Algorithm::ReverseGreedy,
                //     priority: TaskPriority::Low,
                //     group: group.clone(),
                //     created_at,
                // });
            }
            return tasks;
        }
    }

    // Fallback: legacy policy (bundles/multi-tx orders / cannot extract sender+nonce)
    tasks.push(ConflictTask {
        group_idx: group.id,
        algorithm: Algorithm::Greedy,
        priority,
        group: group.clone(),
        created_at,
    });

    if group.orders.len() <= 3 {
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::AllPermutations,
            priority,
            group: group.clone(),
            created_at,
        });
    } else {
        // Random
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::Random {
                seed: group.id as u64,
                count: NUMBER_OF_RANDOM_TASKS,
            },
            priority: TaskPriority::Low,
            group: group.clone(),
            created_at,
        });
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::Length,
            priority: TaskPriority::Low,
            group: group.clone(),
            created_at,
        });
        tasks.push(ConflictTask {
            group_idx: group.id,
            algorithm: Algorithm::ReverseGreedy,
            priority: TaskPriority::Low,
            group: group.clone(),
            created_at,
        });
    }

    tasks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::{
        MempoolTx, Order, SimValue, SimulatedOrder, TransactionSignedEcRecoveredWithBlobs,
    };
    use alloy_consensus::TxLegacy;
    use alloy_primitives::{Address, TxHash, B256, U256};
    use reth::primitives::{Transaction, TransactionSigned};
    use reth_primitives::Recovered;
    use std::sync::Arc;

    use crate::building::evm_inspector::{SlotKey, UsedStateTrace};
    use std::sync::mpsc;

    struct DataGenerator {
        last_used_id: u64,
    }
    impl DataGenerator {
        pub fn new() -> DataGenerator {
            DataGenerator { last_used_id: 0 }
        }

        pub fn create_u64(&mut self) -> u64 {
            self.last_used_id += 1;
            self.last_used_id
        }

        pub fn create_u256(&mut self) -> U256 {
            U256::from(self.create_u64())
        }

        pub fn create_b256(&mut self) -> B256 {
            B256::from(self.create_u256())
        }

        pub fn create_hash(&mut self) -> TxHash {
            TxHash::from(self.create_u256())
        }

        pub fn create_tx(&mut self) -> Recovered<TransactionSigned> {
            Recovered::new_unchecked(
                TransactionSigned::new(
                    Transaction::Legacy(TxLegacy::default()),
                    alloy_primitives::Signature::test_signature(),
                    self.create_hash(),
                ),
                Address::default(),
            )
        }

        pub fn create_order(
            &mut self,
            read: Option<&SlotKey>,
            write: Option<&SlotKey>,
            coinbase_profit: U256,
        ) -> Arc<SimulatedOrder> {
            let mut trace = UsedStateTrace::default();
            if let Some(read) = read {
                trace
                    .read_slot_values
                    .insert(read.clone(), self.create_b256());
            }
            if let Some(write) = write {
                trace
                    .written_slot_values
                    .insert(write.clone(), self.create_b256());
            }

            let sim_value = SimValue {
                coinbase_profit,
                ..Default::default()
            };

            Arc::new(SimulatedOrder {
                order: Order::Tx(MempoolTx {
                    tx_with_blobs: TransactionSignedEcRecoveredWithBlobs::new_no_blobs(
                        self.create_tx(),
                    )
                    .unwrap(),
                }),
                used_state_trace: Some(trace),
                sim_value,
            })
        }
    }

    // Helper function to create an order group
    fn create_conflict_group(
        id: GroupId,
        orders: Vec<Arc<SimulatedOrder>>,
        conflicting_ids: HashSet<GroupId>,
    ) -> ConflictGroup {
        ConflictGroup {
            id,
            orders: Arc::new(orders),
            conflicting_group_ids: Arc::new(conflicting_ids.into_iter().collect()),
        }
    }

    fn create_task_queue() -> Arc<SegQueue<ConflictTask>> {
        Arc::new(SegQueue::new())
    }

    fn create_task_generator() -> ConflictTaskGenerator {
        let (sender, _receiver) = mpsc::channel();
        ConflictTaskGenerator::new(create_task_queue(), sender)
    }

    #[test]
    fn test_process_single_group_new() {
        let mut conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();
        let group = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(100)),
                data_generator.create_order(None, None, U256::from(200)),
            ],
            HashSet::default(),
        );
        conflict_manager.process_single_group(group.clone());

        assert_eq!(conflict_manager.existing_groups.len(), 1);
        assert!(conflict_manager.existing_groups.contains_key(&1));
        assert_eq!(conflict_manager.task_queue.len(), 2);
    }

    #[test]
    fn test_process_single_group_update() {
        let mut conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();

        let initial_group = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(100)),
                data_generator.create_order(None, None, U256::from(200)),
            ],
            HashSet::default(),
        );
        conflict_manager.existing_groups.insert(1, initial_group);

        let updated_group = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(200)),
                data_generator.create_order(None, None, U256::from(300)),
            ],
            HashSet::default(),
        );
        conflict_manager.process_single_group(updated_group);

        assert_eq!(conflict_manager.existing_groups.len(), 1);
        assert_eq!(conflict_manager.task_queue.len(), 2);
    }

    #[test]
    fn test_conflicting_groups_remove_tasks() {
        let mut conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();

        let mut group2_conflicts_with_group1 = HashSet::default();
        group2_conflicts_with_group1.insert(1_usize);

        let group1 = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(100)),
                data_generator.create_order(None, None, U256::from(200)),
            ],
            HashSet::default(),
        );
        let group2 = create_conflict_group(
            2,
            vec![
                data_generator.create_order(None, None, U256::from(200)),
                data_generator.create_order(None, None, U256::from(300)),
            ],
            group2_conflicts_with_group1,
        );

        conflict_manager.process_groups(vec![group1, group2]);

        assert_eq!(conflict_manager.existing_groups.len(), 1);
        assert!(conflict_manager.existing_groups.contains_key(&2));
        assert_eq!(conflict_manager.task_queue.len(), 2);
    }

    #[test]
    fn test_process_groups() {
        let mut conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();

        let mut group2_conflicts_with_group1 = HashSet::default();
        group2_conflicts_with_group1.insert(1_usize);

        let groups = vec![
            create_conflict_group(
                1,
                vec![
                    data_generator.create_order(None, None, U256::from(100)),
                    data_generator.create_order(None, None, U256::from(200)),
                ],
                HashSet::default(),
            ),
            create_conflict_group(
                2,
                vec![
                    data_generator.create_order(None, None, U256::from(200)),
                    data_generator.create_order(None, None, U256::from(300)),
                ],
                group2_conflicts_with_group1,
            ),
            create_conflict_group(
                3,
                vec![
                    data_generator.create_order(None, None, U256::from(300)),
                    data_generator.create_order(None, None, U256::from(400)),
                ],
                HashSet::default(),
            ),
        ];

        conflict_manager.process_groups(groups);

        assert_eq!(conflict_manager.existing_groups.len(), 2);
        assert!(conflict_manager.existing_groups.contains_key(&2));
        assert!(conflict_manager.existing_groups.contains_key(&3));
        assert_eq!(conflict_manager.task_queue.len(), 4);
    }

    #[test]
    fn test_has_significant_changes() {
        let conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();

        let group1 = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(100)),
                data_generator.create_order(None, None, U256::from(90)),
                data_generator.create_order(None, None, U256::from(80)),
            ],
            HashSet::default(),
        );

        let group2 = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(150)),
                data_generator.create_order(None, None, U256::from(140)),
                data_generator.create_order(None, None, U256::from(130)),
            ],
            HashSet::default(),
        );

        assert!(conflict_manager.has_significant_changes(&group2, &group1));

        let group3 = create_conflict_group(
            1,
            vec![
                data_generator.create_order(None, None, U256::from(110)),
                data_generator.create_order(None, None, U256::from(100)),
                data_generator.create_order(None, None, U256::from(90)),
            ],
            HashSet::default(),
        );

        assert!(!conflict_manager.has_significant_changes(&group3, &group1));
    }

    #[test]
    fn has_group_changed() {
        let conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();

        let order1 = data_generator.create_order(None, None, U256::from(100));

        let group1 = create_conflict_group(1, vec![order1.clone()], HashSet::default());

        let group2 = create_conflict_group(
            1,
            vec![data_generator.create_order(None, None, U256::from(200))],
            HashSet::default(),
        );

        assert!(conflict_manager.group_has_changed(&group2, &group1));

        let group3 = create_conflict_group(1, vec![order1], HashSet::default());

        assert!(!conflict_manager.group_has_changed(&group3, &group1));
    }

    #[test]
    fn test_single_order_group() {
        let mut conflict_manager = create_task_generator();

        let mut data_generator = DataGenerator::new();

        let group = create_conflict_group(
            1,
            vec![data_generator.create_order(None, None, U256::from(100))],
            HashSet::default(),
        );

        conflict_manager.process_single_group(group);

        assert_eq!(conflict_manager.existing_groups.len(), 1);
        assert!(conflict_manager.existing_groups.contains_key(&1));

        // Should not create new tasks
        assert_eq!(conflict_manager.task_queue.len(), 0);
    }
}
