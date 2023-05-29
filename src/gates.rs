use crate::utils::{sigmoid, rand_float};

// OR-gate
const OR_GATE: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
];

// AND-gate
const AND_GATE: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 1.0, 1.0],
];

// NAND-gate
const NAND_GATE: [[f64; 3]; 4] = [
    [0.0, 0.0, 1.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];

// XOR-gate
const XOR_GATE: [[f64; 3]; 4] = [
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 1.0],
    [1.0, 1.0, 0.0],
];

const TRAIN_DATA: [[f64; 3]; 4] = AND_GATE;

const MAX_RANGE: f64 = 10.0;
const EPS: f64 = 1e-1;
const RATE: f64 = 1e-1;

pub fn run() {
    let mut weight1 = rand_float(MAX_RANGE);
    let mut weight2 = rand_float(MAX_RANGE);
    let mut bias = rand_float(5.0);

    for x in 0..1_000_000 {
        let c = cost(weight1, weight2, bias);

        let diff_weight1 = (cost(weight1 + EPS, weight2, bias) - c) / EPS;
        let diff_weight2 = (cost(weight1, weight2 + EPS, bias) - c) / EPS;
        let diff_bias = (cost(weight1, weight2, bias + EPS) - c) / EPS;

        weight1 -= RATE * diff_weight1;
        weight2 -= RATE * diff_weight2;
        bias -= RATE * diff_bias;

        // println!("cost={:}, w1={:}, w2={:}, bias={:}", c, weight1, weight2, bias);
    }
    println!("--------------------------------------------");
    println!(
        "weight1: {:}, weight2: {:}, bias: {:}, cost: {:}",
        weight1,
        weight2,
        bias,
        cost(weight1, weight2, bias)
    );

    for data in TRAIN_DATA {
        println!(
            "{:} | {:} => {:}",
            data[0],
            data[1],
            sigmoid((weight1 * data[0]) + (weight2 * data[1]) + bias)
        )
    }
}

fn cost(weight1: f64, weight2: f64, bias: f64) -> f64 {
    let mut result = 0.0;

    for i in TRAIN_DATA.iter() {
        let x1 = i[0];
        let x2 = i[1];
        let y = sigmoid(x1 * weight1 + x2 * weight2 + bias);

        let distance = y - i[2];

        result += distance * distance;
    }
    result /= TRAIN_DATA.len() as f64;
    return result;
}
