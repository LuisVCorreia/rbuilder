use ahash::HashMap;
use alloy_consensus::Transaction;
use alloy_primitives::{b256, Address, Log, B256, U256};
use alloy_rpc_types::AccessList;
use reth_primitives::{Recovered, TransactionSigned};
use revm::{
    bytecode::opcode,
    context::ContextTr,
    inspector::JournalExt,
    interpreter::{interpreter_types::Jumps, CallInputs, CallOutcome, Interpreter},
    Inspector,
};
use revm_inspectors::access_list::AccessListInspector;

/// Uniswap V2 Swap event topic:
/// Swap(address sender, uint256 amount0In, uint256 amount1In, uint256 amount0Out, uint256 amount1Out, address to)
pub const UNI_V2_SWAP_TOPIC: B256 =
    b256!("0xd78ad95fa46c994b6551d0da85fc275fe613ce37657fb8d5e3d130840159d822");

/// Uniswap V3 Swap event topic:
/// Swap(address sender, address recipient, int256 amount0, int256 amount1, uint160 sqrtPriceX96, uint128 liquidity, int24 tick)
pub const UNI_V3_SWAP_TOPIC: B256 =
    b256!("0xc42079f94a6350d7e6235f29174924f928cc2ac818eb64fed8004e115fbcca67");

/// Protocol kind for a DEX pool, used to know which storage slot holds the price.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolKind {
    /// Uniswap V2 (or compatible fork): price is encoded as reserves in storage slot 8.
    UniV2,
    /// Uniswap V3 (or compatible fork): price is encoded as sqrtPriceX96 in storage slot 0.
    UniV3,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SlotKey {
    pub address: Address,
    pub key: B256,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
/// UsedStateTrace is an execution trace of the given order
/// Limitations:
/// * `written_slot_values`, `received_amount` and `sent_amount` are not correct if transaction reverts
pub struct UsedStateTrace {
    /// read slot values contains first read
    pub read_slot_values: HashMap<SlotKey, B256>,
    /// write slot values contains last write
    pub written_slot_values: HashMap<SlotKey, B256>,
    /// balance of first read
    pub read_balances: HashMap<Address, U256>,
    /// number of `wei` sent or received during execution
    pub received_amount: HashMap<Address, U256>,
    pub sent_amount: HashMap<Address, U256>,
    pub created_contracts: Vec<Address>,
    pub destructed_contracts: Vec<Address>,
    /// DEX pools touched by this order (identified from Swap event logs).
    /// Combined with `read_slot_values` (first read = price before all txs) and
    /// `written_slot_values` (last write = price after all txs), this gives the
    /// exact before/after price for each pool across the entire order.
    pub touched_pools: HashMap<Address, PoolKind>,
}

impl UsedStateTrace {
    /// Order of appending traces matters. We assume that "other" trace comes after previously appended traces.
    /// We keep track of first read and last write operations.
    pub fn append_trace(&mut self, other: &UsedStateTrace) {
        for (read_slot, read_value) in &other.read_slot_values {
            if self.read_slot_values.contains_key(read_slot) {
                continue;
            }
            self.read_slot_values.insert(read_slot.clone(), *read_value);
        }

        self.written_slot_values
            .extend(other.written_slot_values.clone());

        for (address, balance) in &other.read_balances {
            if self.read_balances.contains_key(address) {
                continue;
            }
            self.read_balances.insert(*address, *balance);
        }

        for (address, received_amount) in &other.received_amount {
            *self.received_amount.entry(*address).or_default() += received_amount;
        }

        for (address, sent_amount) in &other.sent_amount {
            *self.sent_amount.entry(*address).or_default() += sent_amount;
        }

        self.created_contracts
            .extend(other.created_contracts.clone());

        for address in &other.destructed_contracts {
            if self.destructed_contracts.contains(address) {
                continue;
            }
            self.destructed_contracts.push(*address);
        }

        for (addr, kind) in &other.touched_pools {
            self.touched_pools.entry(*addr).or_insert(*kind);
        }
    }

    pub fn clear(&mut self) {
        self.read_slot_values.clear();
        self.written_slot_values.clear();
        self.read_balances.clear();
        self.received_amount.clear();
        self.sent_amount.clear();
        self.created_contracts.clear();
        self.destructed_contracts.clear();
        self.touched_pools.clear();
    }

    /// Returns true if for every pool touched by this order the price after all txs is exactly
    /// equal to the price before any tx touched the pool.
    ///
    /// Uses `read_slot_values` (first-read-wins across the bundle = pre-bundle price) and
    /// `written_slot_values` (last-write-wins across the bundle = post-bundle price).
    /// This is correct even when the same pool is touched by multiple txs in the bundle.
    ///
    /// Orders with no touched pools are considered price-neutral (vacuously true).
    pub fn is_price_neutral(&self) -> bool {
        let mask112: U256 = (U256::from(1u64) << 112) - U256::from(1u64);

        for (pool, kind) in &self.touched_pools {
            match kind {
                PoolKind::UniV3 => {
                    let key = SlotKey { address: *pool, key: B256::ZERO };
                    let before = self.read_slot_values.get(&key)
                        .map(|v| U256::from_be_bytes(v.0) & sqrt_price_mask());
                    let after = self.written_slot_values.get(&key)
                        .map(|v| U256::from_be_bytes(v.0) & sqrt_price_mask());
                    // No write entry means the SSTORE hook saw no net change — treat as neutral.
                    // TODO: maybe we should not treat these as neutral, can get top of block
                    if let (Some(b), Some(a)) = (before, after) {
                        if b != a {
                            return false;
                        }
                    }
                }
                PoolKind::UniV2 => {
                    let key = SlotKey {
                        address: *pool,
                        key: B256::from(U256::from(8u64).to_be_bytes::<32>()),
                    };
                    let extract = |v: &B256| -> (U256, U256) {
                        let s = U256::from_be_bytes(v.0);
                        (s & mask112, (s >> 112) & mask112) // (reserve0, reserve1)
                    };
                    let before = self.read_slot_values.get(&key).map(extract);
                    let after = self.written_slot_values.get(&key).map(extract);
                    if let (Some((r0b, r1b)), Some((r0a, r1a))) = (before, after) {
                        // Cross-multiply to avoid lossy division: same price iff r1b*r0a == r1a*r0b
                        if r1b * r0a != r1a * r0b {
                            return false;
                        }
                    }
                }
            }
        }
        true // TODO: Might change this to false, we're only interested in bundles that touch pools, everything else is ordered greedily as per usual
    }
}

#[derive(Debug, Clone, Default)]
enum NextStepAction {
    #[default]
    None,
    ReadSloadKeyResult(B256),
    ReadBalanceResult(Address),
}

#[derive(Debug)]
struct UsedStateEVMInspector<'a> {
    next_step_action: NextStepAction,
    used_state_trace: &'a mut UsedStateTrace,
}

impl<'a> UsedStateEVMInspector<'a> {
    fn new(used_state_trace: &'a mut UsedStateTrace) -> Self {
        Self {
            next_step_action: NextStepAction::None,
            used_state_trace,
        }
    }

    /// This method is used to mark nonce change as a slot read / write.
    /// Txs with the same nonce are in conflict and origin address is EOA that does not have storage.
    /// We convert nonce change to the slot 0 read and write of the signer
    fn use_tx_nonce(&mut self, tx: &Recovered<TransactionSigned>) {
        self.used_state_trace.read_slot_values.insert(
            SlotKey {
                address: tx.signer(),
                key: Default::default(),
            },
            U256::from(tx.nonce()).into(),
        );
        self.used_state_trace.written_slot_values.insert(
            SlotKey {
                address: tx.signer(),
                key: Default::default(),
            },
            U256::from(tx.nonce() + 1).into(),
        );
    }
}

impl<CTX> Inspector<CTX> for UsedStateEVMInspector<'_>
where
    CTX: ContextTr<Journal: JournalExt>,
{
    fn step(&mut self, interpreter: &mut Interpreter, _context: &mut CTX) {
        match std::mem::take(&mut self.next_step_action) {
            NextStepAction::ReadSloadKeyResult(slot) => {
                if let Ok(value) = interpreter.stack.peek(0) {
                    let value = B256::from(value.to_be_bytes());
                    let key = SlotKey {
                        address: interpreter.input.target_address,
                        key: slot,
                    };
                    self.used_state_trace
                        .read_slot_values
                        .entry(key)
                        .or_insert(value);
                }
            }
            NextStepAction::ReadBalanceResult(addr) => {
                if let Ok(value) = interpreter.stack.peek(0) {
                    self.used_state_trace
                        .read_balances
                        .entry(addr)
                        .or_insert(value);
                }
            }
            NextStepAction::None => {}
        }
        match interpreter.bytecode.opcode() {
            opcode::SLOAD => {
                if let Ok(slot) = interpreter.stack.peek(0) {
                    let slot = B256::from(slot.to_be_bytes());
                    self.next_step_action = NextStepAction::ReadSloadKeyResult(slot);
                }
            }
            opcode::SSTORE => {
                if let (Ok(slot), Ok(value)) =
                    (interpreter.stack.peek(0), interpreter.stack.peek(1))
                {
                    let written_value = B256::from(value.to_be_bytes());
                    let key = SlotKey {
                        address: interpreter.input.target_address,
                        key: B256::from(slot.to_be_bytes()),
                    };
                    // if we write the same value that we read as the first read we don't have a write
                    if let Some(read_value) = self.used_state_trace.read_slot_values.get(&key) {
                        if read_value == &written_value {
                            self.used_state_trace.written_slot_values.remove(&key);
                            return;
                        }
                    }
                    self.used_state_trace
                        .written_slot_values
                        .insert(key, written_value);
                }
            }
            opcode::BALANCE => {
                if let Ok(addr) = interpreter.stack.peek(0) {
                    let addr = Address::from_word(B256::from(addr.to_be_bytes()));
                    self.next_step_action = NextStepAction::ReadBalanceResult(addr);
                }
            }
            opcode::SELFBALANCE => {
                let addr = interpreter.input.target_address;
                self.next_step_action = NextStepAction::ReadBalanceResult(addr);
            }
            _ => (),
        }
    }

    fn call(&mut self, _context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        if let Some(transfer_value) = inputs.transfer_value() {
            if !transfer_value.is_zero() {
                *self
                    .used_state_trace
                    .sent_amount
                    .entry(inputs.transfer_from())
                    .or_default() += transfer_value;
                *self
                    .used_state_trace
                    .received_amount
                    .entry(inputs.transfer_to())
                    .or_default() += transfer_value;
            }
        }
        None
    }

    fn create_end(
        &mut self,
        _context: &mut CTX,
        _: &revm::interpreter::CreateInputs,
        outcome: &mut revm::interpreter::CreateOutcome,
    ) {
        if let Some(addr) = outcome.address {
            self.used_state_trace.created_contracts.push(addr);
        }
    }

    fn selfdestruct(&mut self, contract: Address, target: Address, value: U256) {
        // selfdestruct can be called multiple times during transaction execution
        if self
            .used_state_trace
            .destructed_contracts
            .contains(&contract)
        {
            return;
        }
        self.used_state_trace.destructed_contracts.push(contract);
        if !value.is_zero() {
            *self
                .used_state_trace
                .sent_amount
                .entry(contract)
                .or_default() += value;
            *self
                .used_state_trace
                .received_amount
                .entry(target)
                .or_default() += value;
        }
    }
}

#[derive(Debug)]
pub struct RBuilderEVMInspector<'a> {
    access_list_inspector: AccessListInspector,
    used_state_inspector: Option<UsedStateEVMInspector<'a>>,
}

impl<'a> RBuilderEVMInspector<'a> {
    pub fn new(
        tx: &Recovered<TransactionSigned>,
        used_state_trace: Option<&'a mut UsedStateTrace>,
    ) -> Self {
        let access_list_inspector =
            AccessListInspector::new(tx.access_list().cloned().unwrap_or_default());

        let mut used_state_inspector = used_state_trace.map(UsedStateEVMInspector::new);
        if let Some(i) = &mut used_state_inspector {
            i.use_tx_nonce(tx);
        }

        Self {
            access_list_inspector,
            used_state_inspector,
        }
    }

    pub fn into_access_list(self) -> AccessList {
        self.access_list_inspector.into_access_list()
    }

    /// Record which DEX pools were touched during execution by inspecting the swap event logs.
    /// Must be called after EVM execution and before dropping the inspector.
    pub fn process_execution_logs(&mut self, logs: &[Log]) {
        let Some(inspector) = &mut self.used_state_inspector else {
            return;
        };
        for log in logs {
            let Some(topic0) = log.topics().first() else {
                continue;
            };
            if *topic0 == UNI_V3_SWAP_TOPIC {
                inspector.used_state_trace.touched_pools
                    .entry(log.address)
                    .or_insert(PoolKind::UniV3);
            } else if *topic0 == UNI_V2_SWAP_TOPIC {
                inspector.used_state_trace.touched_pools
                    .entry(log.address)
                    .or_insert(PoolKind::UniV2);
            }
        }
    }
}

/// Mask for the lower 160 bits (sqrtPriceX96 field in Uni v3 slot 0).
#[inline]
fn sqrt_price_mask() -> U256 {
    (U256::from(1u64) << 160) - U256::from(1u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b256(v: U256) -> B256 {
        B256::from(v.to_be_bytes::<32>())
    }

    fn slot0_key(pool: Address) -> SlotKey {
        SlotKey { address: pool, key: B256::ZERO }
    }

    fn slot8_key(pool: Address) -> SlotKey {
        SlotKey { address: pool, key: b256(U256::from(8u64)) }
    }

    fn pack_reserves(reserve0: U256, reserve1: U256) -> B256 {
        let mask112: U256 = (U256::from(1u64) << 112) - U256::from(1u64);
        b256((reserve1 & mask112) << 112 | (reserve0 & mask112))
    }

    // ── is_price_neutral ─────────────────────────────────────────────────────

    #[test]
    fn test_v3_neutral_no_write() {
        // Pool touched but no slot written → price unchanged → neutral
        let pool = Address::repeat_byte(0x01);
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.read_slot_values.insert(slot0_key(pool), b256(U256::from(1_000_000u64)));
        // No write → is_price_neutral returns true
        assert!(trace.is_price_neutral());
    }

    #[test]
    fn test_v3_neutral_same_sqrt_price() {
        // Both read and written slot have the same sqrtPriceX96 → neutral
        let pool = Address::repeat_byte(0x02);
        let p = U256::from(1_000_000u64);
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.read_slot_values.insert(slot0_key(pool), b256(p));
        trace.written_slot_values.insert(slot0_key(pool), b256(p));
        assert!(trace.is_price_neutral());
    }

    #[test]
    fn test_v3_impacting_different_sqrt_price() {
        let pool = Address::repeat_byte(0x03);
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.read_slot_values.insert(slot0_key(pool), b256(U256::from(1_000_000u64)));
        trace.written_slot_values.insert(slot0_key(pool), b256(U256::from(1_010_000u64)));
        assert!(!trace.is_price_neutral());
    }

    #[test]
    fn test_v3_neutral_backrun_restores_price() {
        // Simulate append_trace result for a user tx + backrun on V3:
        //   read_slot_values[slot0] = P0  (first read, from user tx)
        //   written_slot_values[slot0] = P0  (last write, from backrun restoring the price)
        let pool = Address::repeat_byte(0x04);
        let p0 = U256::from(1_000_000u64);
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.read_slot_values.insert(slot0_key(pool), b256(p0));
        trace.written_slot_values.insert(slot0_key(pool), b256(p0));
        assert!(trace.is_price_neutral());
    }

    #[test]
    fn test_v2_neutral_same_reserves() {
        let pool = Address::repeat_byte(0x05);
        let (r0, r1) = (U256::from(100_000u64), U256::from(200_000u64));
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV2);
        trace.read_slot_values.insert(slot8_key(pool), pack_reserves(r0, r1));
        trace.written_slot_values.insert(slot8_key(pool), pack_reserves(r0, r1));
        assert!(trace.is_price_neutral());
    }

    #[test]
    fn test_v2_impacting_changed_reserves() {
        let pool = Address::repeat_byte(0x06);
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV2);
        trace.read_slot_values.insert(slot8_key(pool), pack_reserves(U256::from(100u64), U256::from(200u64)));
        trace.written_slot_values.insert(slot8_key(pool), pack_reserves(U256::from(110u64), U256::from(181u64)));
        assert!(!trace.is_price_neutral());
    }

    #[test]
    fn test_v2_neutral_cross_multiply_exact_cancel() {
        // Backrun exactly restores reserves to original values.
        // After append_trace: read=original, write=original → neutral.
        let pool = Address::repeat_byte(0x07);
        let (r0, r1) = (U256::from(100u64), U256::from(200u64));
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV2);
        trace.read_slot_values.insert(slot8_key(pool), pack_reserves(r0, r1));
        trace.written_slot_values.insert(slot8_key(pool), pack_reserves(r0, r1));
        assert!(trace.is_price_neutral());
    }

    #[test]
    fn test_no_touched_pools_is_neutral() {
        // Empty trace → vacuously neutral
        assert!(UsedStateTrace::default().is_price_neutral());
    }

    // ── append_trace ─────────────────────────────────────────────────────────

    #[test]
    fn test_append_trace_accumulates_touched_pools() {
        let pool_a = Address::repeat_byte(0x10);
        let pool_b = Address::repeat_byte(0x11);

        let mut trace1 = UsedStateTrace::default();
        trace1.touched_pools.insert(pool_a, PoolKind::UniV3);

        let mut trace2 = UsedStateTrace::default();
        trace2.touched_pools.insert(pool_b, PoolKind::UniV2);
        trace2.touched_pools.insert(pool_a, PoolKind::UniV3); // duplicate — kept

        trace1.append_trace(&trace2);
        assert_eq!(trace1.touched_pools.len(), 2);
        assert_eq!(trace1.touched_pools[&pool_a], PoolKind::UniV3);
        assert_eq!(trace1.touched_pools[&pool_b], PoolKind::UniV2);
    }

    #[test]
    fn test_clear_resets_touched_pools() {
        let pool = Address::repeat_byte(0x20);
        let mut trace = UsedStateTrace::default();
        trace.touched_pools.insert(pool, PoolKind::UniV3);
        trace.clear();
        assert!(trace.touched_pools.is_empty());
    }
}

impl<'a, CTX> Inspector<CTX> for RBuilderEVMInspector<'a>
where
    CTX: ContextTr<Journal: JournalExt>,
    UsedStateEVMInspector<'a>: Inspector<CTX>,
{
    #[inline]
    fn step(&mut self, interp: &mut Interpreter, context: &mut CTX) {
        self.access_list_inspector.step(interp, context);
        if let Some(used_state_inspector) = &mut self.used_state_inspector {
            used_state_inspector.step(interp, context);
        }
    }

    #[inline]
    fn call(&mut self, context: &mut CTX, inputs: &mut CallInputs) -> Option<CallOutcome> {
        if let Some(used_state_inspector) = &mut self.used_state_inspector {
            used_state_inspector.call(context, inputs)
        } else {
            None
        }
    }

    #[inline]
    fn create_end(
        &mut self,
        context: &mut CTX,
        inputs: &revm::interpreter::CreateInputs,
        outcome: &mut revm::interpreter::CreateOutcome,
    ) {
        if let Some(used_state_inspector) = &mut self.used_state_inspector {
            used_state_inspector.create_end(context, inputs, outcome);
        }
    }

    #[inline]
    fn selfdestruct(&mut self, contract: Address, target: Address, value: U256) {
        if let Some(used_state_inspector) = &mut self.used_state_inspector {
            used_state_inspector.selfdestruct(contract, target, value);
        }
    }
}
