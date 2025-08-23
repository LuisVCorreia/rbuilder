use crate::building::evm_inspector::UsedStateTrace;

/// Trait to trace ANY use of an EVM instance for metrics
pub trait SimulationTracer {
    /// En EVM instance executed a tx consuming gas.
    /// This includes reverting transactions.
    fn add_gas_used(&mut self, _gas: u64) {}

    /// If tracer returns true tx_commit will call add_used_state_trace with the given transaction trace.
    fn should_collect_used_state_trace(&self) -> bool {
        false
    }

    fn add_used_state_trace(&mut self, _trace: &UsedStateTrace) {}

    fn get_used_state_tracer(&self) -> Option<&UsedStateTrace> {
        None
    }
}

impl SimulationTracer for () {}

#[derive(Debug, Default, Clone)]
pub struct GasUsedSimulationTracer {
    pub used_gas: u64,
}

impl SimulationTracer for GasUsedSimulationTracer {
    fn add_gas_used(&mut self, gas: u64) {
        self.used_gas += gas;
    }
}

/// Tracer that accumulates gas and used state.
#[derive(Debug)]
pub struct AccumulatorSimulationTracer {
    pub used_gas: u64,
    pub used_state_trace: UsedStateTrace,
}

impl AccumulatorSimulationTracer {
    pub fn new() -> Self {
        Self {
            used_gas: 0,
            used_state_trace: UsedStateTrace::default(),
        }
    }
}

impl Default for AccumulatorSimulationTracer {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulationTracer for AccumulatorSimulationTracer {
    fn add_gas_used(&mut self, gas: u64) {
        self.used_gas += gas;
    }

    fn should_collect_used_state_trace(&self) -> bool {
        true
    }

    fn add_used_state_trace(&mut self, trace: &UsedStateTrace) {
        self.used_state_trace.append_trace(trace);
    }

    fn get_used_state_tracer(&self) -> Option<&UsedStateTrace> {
        // // Open the file in append mode. This will create the file if it doesn't exist,
        // // and add new content to the end if it does.
        // if let Ok(mut file) = OpenOptions::new()
        //     .create(true)
        //     .append(true)
        //     .open("state_trace_rbuilder.txt")
        // {
        //     // Use writeln! to write formatted strings to the file.
        //     // We ignore the Result of the write operation for this debugging purpose.
        //     let _ = writeln!(file, "\n--- Dumping UsedStateTrace ---");

        //     if !self.used_state_trace.read_slot_values.is_empty() {
        //         let _ = writeln!(file, "\n[Read Slots]");
        //         for (slot_key, value) in self.used_state_trace.read_slot_values.iter() {
        //             let _ = writeln!(file, "  - Address: {:?}, Slot: {:?}, Value: {:?}", slot_key.address, slot_key.key, value);
        //         }
        //     }

        //     if !self.used_state_trace.written_slot_values.is_empty() {
        //         let _ = writeln!(file, "\n[Written Slots]");
        //         for (slot_key, value) in self.used_state_trace.written_slot_values.iter() {
        //             let _ = writeln!(file, "  - Address: {:?}, Slot: {:?}, Value: {:?}", slot_key.address, slot_key.key, value);
        //         }
        //     }

        //     if !self.used_state_trace.read_balances.is_empty() {
        //         let _ = writeln!(file, "\n[Read Balances]");
        //         for (address, balance) in self.used_state_trace.read_balances.iter() {
        //             let _ = writeln!(file, "  - Address: {:?}, Balance: {:?}", address, balance);
        //         }
        //     }

        //     if !self.used_state_trace.received_amount.is_empty() {
        //         let _ = writeln!(file, "\n[Received Amounts (Wei)]");
        //         for (address, amount) in self.used_state_trace.received_amount.iter() {
        //             let _ = writeln!(file, "  - Address: {:?}, Amount: {:?}", address, amount);
        //         }
        //     }

        //     if !self.used_state_trace.sent_amount.is_empty() {
        //         let _ = writeln!(file, "\n[Sent Amounts (Wei)]");
        //         for (address, amount) in self.used_state_trace.sent_amount.iter() {
        //             let _ = writeln!(file, "  - Address: {:?}, Amount: {:?}", address, amount);
        //         }
        //     }

        //     if !self.used_state_trace.created_contracts.is_empty() {
        //         let _ = writeln!(file, "\n[Created Contracts]");
        //         for address in &self.used_state_trace.created_contracts {
        //             let _ = writeln!(file, "  - Address: {:?}", address);
        //         }
        //     }

        //     if !self.used_state_trace.destructed_contracts.is_empty() {
        //         let _ = writeln!(file, "\n[Destructed Contracts]");
        //         for address in &self.used_state_trace.destructed_contracts {
        //             let _ = writeln!(file, "  - Address: {:?}", address);
        //         }
        //     }

        //     let _ = writeln!(file, "--- End of Trace Dump ---\n");
        // } else {
        //     // If the file can't be opened, print an error to the console as a fallback.
        //     println!("Error: Could not open or write to state_trace_rbuilder.txt");
        // }

        // The function's primary purpose (returning the trace) remains unchanged.
        Some(&self.used_state_trace)
    }
}
