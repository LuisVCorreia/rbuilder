use osqp::{CscMatrix, Problem, Settings, Status};
use alloy_primitives::{Address, U256};
use std::collections::HashMap;

const SCALE: f64 = 1.0;
const EPS_EDGE_UP: f64 = 1e-9;

fn u256_to_f64(u: U256) -> f64 {
    // Clamp to u128 to avoid precision/overflow
    match u128::try_from(u) {
        Ok(v) => (v as f64) / SCALE,
        Err(_) => {
            println!("u256_to_f64: value {} exceeded u128 max, clamping", u);
            (u128::MAX as f64) / SCALE
        }
    }
}

fn f64_to_u256(x: f64) -> U256 {
    if x <= 0.0 { return U256::ZERO; }
    let x_scaled = x * SCALE;
    let flooored = x_scaled.floor();
    let frac = x_scaled - flooored;

    // Round up if very close to next integer
    let y = if frac >= 1.0 - EPS_EDGE_UP { flooored + 1.0 } else { flooored };

    let y_u128 = if y.is_finite() && y >= 0.0 { y as u128 } else { 0 };
    U256::from(y_u128)
}

fn dense_zero_row_major(m: usize, n: usize) -> Vec<f64> {
    vec![0.0; m * n]
}
fn set_row_major(m: usize, n: usize, data: &mut [f64], row: usize, col: usize, val: f64) {
    debug_assert!(row < m && col < n);
    data[row * n + col] = val;
}

pub fn osqp_project_refunds(
    flat_tax: &[U256],
    realized_value: &[U256],
    identity_addrs: &[Address],
    joint_block_value_delta_all: &[(Vec<Address>, U256)],
    total_cap: U256,
) -> Option<Vec<U256>> {
    let n = flat_tax.len();
    if n == 0 { return Some(vec![]); }

    let mut idx_of: HashMap<Address, usize> = HashMap::default();
    for (i, id) in identity_addrs.iter().enumerate() {
        idx_of.insert(*id, i);
    }

    let mut subset_specs: Vec<(Vec<usize>, f64)> = Vec::new();
    for (subset_addrs, provided_cap) in joint_block_value_delta_all {
        let mut subset_idx: Vec<usize> = Vec::with_capacity(subset_addrs.len());
        let mut ok = true;
        for a in subset_addrs {
            if let Some(&ix) = idx_of.get(a) { subset_idx.push(ix); } else { ok = false; break; }
        }
        if !ok || subset_idx.len() < 2 { continue; }

        subset_idx.sort();
        subset_idx.dedup();

        // mu_S = min(sum b_i, provided_cap)
        let sum_b: U256 = subset_idx.iter().map(|&i| realized_value[i]).sum();
        let cap = std::cmp::min(sum_b, *provided_cap);
        subset_specs.push((subset_idx, u256_to_f64(cap)));
    }

    let m_sub = subset_specs.len();
    let m_nonneg = n;
    let m_total = 1;
    let m = m_sub + m_nonneg + m_total;

    let mut a_dense = dense_zero_row_major(m, n);
    // subset rows
    for (row, (idxs, _)) in subset_specs.iter().enumerate() {
        for &j in idxs {
            set_row_major(m, n, &mut a_dense, row, j, 1.0);
        }
    }
    // non-negative rows
    for i in 0..n {
        set_row_major(m, n, &mut a_dense, m_sub + i, i, 1.0);
    }
    // total-sum row
    for j in 0..n {
        set_row_major(m, n, &mut a_dense, m - 1, j, 1.0);
    }
    let a_csc: CscMatrix<'static> = CscMatrix::from_row_iter_dense(m, n, a_dense);

    let mut p_dense = dense_zero_row_major(n, n);
    for i in 0..n { set_row_major(n, n, &mut p_dense, i, i, 2.0); }
    let p_csc: CscMatrix<'static> = CscMatrix::from_row_iter_dense(n, n, p_dense).into_upper_tri();

    // q = -2 phi
    let q: Vec<f64> = flat_tax.iter().map(|&u| -2.0 * u256_to_f64(u)).collect();

    let mut lower: Vec<f64> = Vec::with_capacity(m);
    let mut upper: Vec<f64> = Vec::with_capacity(m);

    // subset rows: [0, mu_S]
    for (_, cap) in &subset_specs {
        lower.push(0.0);
        upper.push(*cap);
    }
    // non-negative rows: [0, +inf]
    for _ in 0..m_nonneg {
        lower.push(0.0);
        upper.push(f64::INFINITY);
    }
    // total-sum row: [0, total_cap]
    lower.push(0.0);
    upper.push(u256_to_f64(total_cap));


    let settings = Settings::default()
        .verbose(true)  // TODO: Remove this in production
        .polishing(true)
        .scaled_termination(false);

    let mut problem = Problem::new(p_csc, &q, a_csc, &lower, &upper, &settings).ok()?;

    let x0: Vec<f64> = flat_tax.iter().map(|&u| u256_to_f64(u)).collect();
    problem.warm_start_x(&x0);

    let status = problem.solve();

    // Accept any status that carries a Solution
    let solution = match status {
        Status::Solved(ref sol)
        | Status::SolvedInaccurate(ref sol)
        | Status::MaxIterationsReached(ref sol)
        | Status::TimeLimitReached(ref sol) => {
            sol
        }
        _ => {
            return None;
        }
    };

    let x = solution.x();
    let r_u256: Vec<U256> = x.iter().copied().map(f64_to_u256).collect();
    Some(r_u256)
}
