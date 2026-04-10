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

    /// Returns the relative price displacement for each pool touched by this order.
    ///
    /// For V3: |sqrtPriceX96_after - sqrtPriceX96_before| / sqrtPriceX96_before.
    /// For V2: |price_after - price_before| / price_before, where price = reserve1 / reserve0.
    ///
    /// Returns 0.0 for a pool when there is no write (price unchanged) or no read.
    /// Orders with no touched pools return an empty map.
    pub fn price_displacement(&self) -> HashMap<Address, f64> {
        let mut result = HashMap::default();
        let mask112: U256 = (U256::from(1u64) << 112) - U256::from(1u64);

        for (pool, kind) in &self.touched_pools {
            let displacement = match kind {
                PoolKind::UniV3 => {
                    let key = SlotKey { address: *pool, key: B256::ZERO };
                    let mask = sqrt_price_mask(); // lower 160 bits
                    let before = self.read_slot_values.get(&key)
                        .map(|v| u256_to_f64(U256::from_be_bytes(v.0) & mask));
                    let after = self.written_slot_values.get(&key)
                        .map(|v| u256_to_f64(U256::from_be_bytes(v.0) & mask));
                    match (before, after) {
                        (Some(b), Some(a)) if b > 0.0 => (a - b).abs() / b,
                        _ => 0.0,
                    }
                }
                PoolKind::UniV2 => {
                    let key = SlotKey {
                        address: *pool,
                        key: B256::from(U256::from(8u64).to_be_bytes::<32>()),
                    };
                    let extract = |v: &B256| -> (f64, f64) {
                        let s = U256::from_be_bytes(v.0);
                        // reserves packed as (reserve1 << 112 | reserve0), each 112 bits
                        let r0 = u256_to_f64(s & mask112);
                        let r1 = u256_to_f64((s >> 112) & mask112);
                        (r0, r1)
                    };
                    let before = self.read_slot_values.get(&key).map(extract);
                    let after = self.written_slot_values.get(&key).map(extract);
                    match (before, after) {
                        (Some((r0b, r1b)), Some((r0a, r1a))) if r0b > 0.0 && r0a > 0.0 => {
                            let price_before = r1b / r0b;
                            let price_after = r1a / r0a;
                            (price_after - price_before).abs() / price_before
                        }
                        _ => 0.0,
                    }
                }
            };
            result.insert(*pool, displacement);
        }
        result
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

/// Convert a U256 to f64 via its little-endian u64 limbs.
/// Precision is limited to ~15 significant digits, which is acceptable for displacement ratios.
/// Safe for the full uint160 range (sqrtPriceX96 max ~1.46e48) unlike u128 which overflows at ~3.4e38.
#[inline]
fn u256_to_f64(v: U256) -> f64 {
    const TWO64: f64 = 1.844_674_407_370_955_2e19; // 2^64
    let limbs = v.as_limbs(); // [lo0, lo1, lo2, lo3] little-endian u64s
    limbs[0] as f64
        + limbs[1] as f64 * TWO64
        + limbs[2] as f64 * (TWO64 * TWO64)
        + limbs[3] as f64 * (TWO64 * TWO64 * TWO64)
}

#[cfg(test)]
mod tests {
    use super::*;

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
