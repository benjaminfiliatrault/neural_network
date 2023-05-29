use crate::utils::rand_float;

const TRAIN_DATA: [[f64; 2]; 5] = [[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];

const MAX_RANGE: f64 = 10.0;
const EPS: f64 = 1e-3;
const RATE: f64 = 1e-2;

pub fn run() {
    let mut weight = rand_float(MAX_RANGE);
    let mut bias = rand_float(5.0);

    for x in 0..500 {
        let c = cost(weight, bias);
        let diff_weight = (cost(weight + EPS, bias) - c) / EPS;
        let diff_bias = (cost(weight, bias + EPS) - c) / EPS;
        weight -= RATE * diff_weight;
        bias -= RATE * diff_bias;
        println!("cost={:}, w={:}, bias={:}", c, weight, bias);
    }
    println!("----------------------");
    println!("weight: {:}, bias: {:}", weight, bias);
}

fn cost(weight: f64, bias: f64) -> f64 {
    let mut result = 0.0;

    for i in TRAIN_DATA.iter() {
        let x = i[0];
        let y = x * weight + bias;

        let distance = y - i[1];

        result += distance * distance;
    }
    result /= TRAIN_DATA.len() as f64;
    return result;
}