use alloy_primitives::utils::format_ether;
use eyre::Result;
use reth_provider::StateProvider;
use std::{
    sync::{mpsc as std_mpsc, Arc},
    thread,
    time::Instant,
};
use tokio_util::sync::CancellationToken;
use tracing::{trace, warn};
use std::sync::atomic::{AtomicUsize, Ordering};

use super::{
    conflict_resolvers::{AlgoRecord, GAGenRecord, ResolverContext},
    conflict_task_generator::get_tasks_for_group,
    simulation_cache::SharedSimulationCache, ConflictGroup, ConflictResolutionResultPerGroup,
    ConflictTask, GroupId, ResolutionResult, TaskPriority,
};
use crate::{building::{BlockBuildingContext, ThreadBlockBuildingContext, builders::parallel_builder::TaskQueueReceiver}, provider::StateProviderFactory, utils::elapsed_ms};


pub struct ConflictResolvingPool<P> {
    task_queue: TaskQueueReceiver,
    group_result_sender: std_mpsc::Sender<ConflictResolutionResultPerGroup>,
    cancellation_token: CancellationToken,
    ctx: BlockBuildingContext,
    provider: P,
    simulation_cache: Arc<SharedSimulationCache>,
    num_threads: usize,
    outstanding: Option<Arc<AtomicUsize>>,
    safe_sorting_only: bool,
}

impl<P> ConflictResolvingPool<P>
where
    P: StateProviderFactory + Clone + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        num_threads: usize,
        task_queue: TaskQueueReceiver,
        safe_sorting_only: bool,
        group_result_sender: std_mpsc::Sender<ConflictResolutionResultPerGroup>,
        cancellation_token: CancellationToken,
        ctx: BlockBuildingContext,
        provider: P,
        simulation_cache: Arc<SharedSimulationCache>,
    ) -> Self {
        Self {
            task_queue,
            group_result_sender,
            safe_sorting_only,
            cancellation_token,
            ctx,
            provider,
            simulation_cache,
            num_threads,
            outstanding: None,
        }
    }

    pub fn with_outstanding_counter(mut self, counter: Arc<AtomicUsize>) -> Self {
        self.outstanding = Some(counter);
        self
    }


    pub fn start(&self) -> eyre::Result<()> {
        for _ in 0..self.num_threads {
            let task_queue = self.task_queue.clone();
            let cancellation_token = self.cancellation_token.clone();
            let group_result_sender = self.group_result_sender.clone();
            let simulation_cache = self.simulation_cache.clone();
            let ctx = self.ctx.clone();
            let outstanding = self.outstanding.clone();

            let block_state: Arc<dyn StateProvider> = self
                .provider
                .history_by_block_hash(self.ctx.attributes.parent)?
                .into();
            thread::spawn(move || {
                let mut local_ctx = ThreadBlockBuildingContext::default();
                loop {
                    match task_queue.recv_timeout(std::time::Duration::from_millis(100)) {
                        Ok(task) => {
                            if cancellation_token.is_cancelled() {
                                if let Some(ref o) = outstanding { o.fetch_sub(1, Ordering::AcqRel); }
                                return;
                            }
                            let task_start = Instant::now();
                            let processed = Self::process_task(
                                task,
                                &ctx,
                                &mut local_ctx,
                                block_state.clone(),
                                cancellation_token.clone(),
                                Arc::clone(&simulation_cache),
                            );
                            if let Some(ref o) = outstanding { o.fetch_sub(1, Ordering::AcqRel); }
                            if let Ok((task_id, result, _algo_record, _ga_records)) = processed {
                                match group_result_sender.send((task_id, result)) {
                                    Ok(_) => {
                                        trace!(
                                            task_id = %task_id,
                                            time_taken_ms = %elapsed_ms(task_start),
                                            "Conflict resolving: successfully sent group result"
                                        );
                                    }
                                    Err(err) => {
                                        warn!(
                                            task_id = %task_id,
                                            error = ?err,
                                            time_taken_ms = %elapsed_ms(task_start),
                                            "Conflict resolving: failed to send group result"
                                        );
                                        return;
                                    }
                                }
                            }
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                            if cancellation_token.is_cancelled() { return; }
                        }
                        Err(crossbeam::channel::RecvTimeoutError::Disconnected) => return,
                    }
                }
            });
        }
        Ok(())
    }

    fn process_task(
        task: ConflictTask,
        ctx: &BlockBuildingContext,
        local_ctx: &mut ThreadBlockBuildingContext,
        state: Arc<dyn StateProvider>,
        cancellation_token: CancellationToken,
        simulation_cache: Arc<SharedSimulationCache>,
    ) -> Result<(GroupId, (ResolutionResult, ConflictGroup), AlgoRecord, Vec<GAGenRecord>)> {
        let mut merging_context = ResolverContext::new(
            state,
            ctx.clone(),
            cancellation_token.clone(),
            simulation_cache,
        );
        let task_id = task.group_idx;
        let task_group = task.group.clone();
        let task_algo = task.algorithm;

        match merging_context.run_conflict_task(task, local_ctx) {
            Ok((sequence_of_orders, algo_record, ga_records)) => {
                trace!(
                    task_type = ?task_algo,
                    group_id = task_id,
                    profit = format_ether(sequence_of_orders.total_profit),
                    order_count = sequence_of_orders.sequence_of_orders.len(),
                    "Successfully ran conflict task"
                );
                Ok((task_id, (sequence_of_orders, task_group), algo_record, ga_records))
            }
            Err(err) => {
                // Fast patch/heuristic to fix excessive tracing.
                // TODO: Use good errors.
                if !cancellation_token.is_cancelled() {
                    warn!(
                        group_id = task_id,
                        err = ?err,
                        "Error running conflict task for group_idx",
                    );
                }
                Err(err)
            }
        }
    }

    pub fn process_groups_backtest(
        &mut self,
        new_groups: Vec<ConflictGroup>,
        ctx: &BlockBuildingContext,
        state: Arc<dyn StateProvider>,
        simulation_cache: Arc<SharedSimulationCache>,
    ) -> (Vec<(GroupId, (ResolutionResult, ConflictGroup))>, Vec<AlgoRecord>, Vec<GAGenRecord>) {
        let mut results = Vec::new();
        let mut algo_records: Vec<AlgoRecord> = Vec::new();
        let mut ga_records: Vec<GAGenRecord> = Vec::new();
        let mut local_ctx = ThreadBlockBuildingContext::default();
        for new_group in new_groups {
            let tasks = get_tasks_for_group(&new_group, TaskPriority::High, self.safe_sorting_only);
            for task in tasks {
                let simulation_cache = Arc::clone(&simulation_cache);
                let result = Self::process_task(
                    task,
                    ctx,
                    &mut local_ctx,
                    state.clone(),
                    CancellationToken::new(),
                    simulation_cache,
                );
                if let Ok((gid, res_grp, algo_rec, ga_recs)) = result {
                    results.push((gid, res_grp));
                    algo_records.push(algo_rec);
                    ga_records.extend(ga_recs);
                }
            }
        }
        (results, algo_records, ga_records)
    }
}
