use crate::{
    data::{AND_GATE, NAND_GATE, OR_GATE, XOR_GATE},
    utils::{rand_float, sigmoid},
};

#[derive(Clone, Copy)]
struct Xor {
    // First Layer
    or_w1: f64,
    or_w2: f64,
    or_b: f64,
    nand_w1: f64,
    nand_w2: f64,
    nand_b: f64,

    // Second Layer
    and_w1: f64,
    and_w2: f64,
    and_b: f64,
}

const MAX_RANGE: f64 = 0.2;
const EPS: f64 = 1e-2;
const RATE: f64 = 1e-1;
const TRAIN_DATA: [[f64; 3]; 4] = XOR_GATE;

fn forward(modal: Xor, x1: f64, x2: f64) -> f64 {
    let a = sigmoid(((modal.or_w1 * x1) + (modal.or_w2 * x2)) + modal.or_b);
    let b = sigmoid(((modal.nand_w1 * x1) + (modal.nand_w2 * x2)) + modal.nand_b);
    return sigmoid(((a * modal.and_w1) + (b * modal.and_w2)) + modal.and_b);
}

fn cost(modal: Xor) -> f64 {
    let mut result = 0.0;

    for i in TRAIN_DATA.iter() {
        let x1 = i[0];
        let x2 = i[1];
        let y = forward(modal, x1, x2);

        let distance = y - i[2];

        result += distance * distance;
    }
    result /= TRAIN_DATA.len() as f64;
    return result;
}

fn rand_xor() -> Xor {
    return Xor {
        or_w1: rand_float(MAX_RANGE),
        or_w2: rand_float(MAX_RANGE),
        or_b: rand_float(MAX_RANGE),
        nand_w1: rand_float(MAX_RANGE),
        nand_w2: rand_float(MAX_RANGE),
        nand_b: rand_float(MAX_RANGE),
        and_w1: rand_float(MAX_RANGE),
        and_w2: rand_float(MAX_RANGE),
        and_b: rand_float(MAX_RANGE),
    };
}

fn print_xor(modal: Xor) {
    println!(
        "\n{0: <10} | {1: <30} | {2: <30} | {3: <10}",
        "Gate", "Weight 1", "Weight 2", "Bias"
    );
    println!("{:-<120}", "-");
    println!(
        "{0: <10} | {1: <30} | {2: <30} | {3: <10}",
        "OR", modal.or_w1, modal.or_w2, modal.or_b
    );
    println!(
        "{0: <10} | {1: <30} | {2: <30} | {3: <10}",
        "NAND", modal.nand_w1, modal.nand_w2, modal.nand_b
    );
    println!(
        "{0: <10} | {1: <30} | {2: <30} | {3: <10}\n",
        "AND", modal.and_w1, modal.and_w2, modal.and_b
    );
}

fn finite_diff(mut m: Xor, eps: f64) -> Xor {
    let c = cost(m);
    let mut g: Xor = Xor {
        or_w1: 0.0,
        or_w2: 0.0,
        or_b: 0.0,
        nand_w1: 0.0,
        nand_w2: 0.0,
        nand_b: 0.0,
        and_w1: 0.0,
        and_w2: 0.0,
        and_b: 0.0,
    };

    // Dumb but it works... stay with
    // me on this one. We works with float
    // I do not want error, I could have done
    // minus EPS but... meh
    let mut saved: f64;

    saved = m.or_w1;
    m.or_w1 += eps;
    g.or_w1 = (cost(m) - c) / eps;
    m.or_w1 = saved;

    saved = m.or_w2;
    m.or_w2 += eps;
    g.or_w2 = (cost(m) - c) / eps;
    m.or_w2 = saved;

    saved = m.or_b;
    m.or_b += eps;
    g.or_b = (cost(m) - c) / eps;
    m.or_b = saved;

    saved = m.nand_w1;
    m.nand_w1 += eps;
    g.nand_w1 = (cost(m) - c) / eps;
    m.nand_w1 = saved;

    saved = m.nand_w2;
    m.nand_w2 += eps;
    g.nand_w2 = (cost(m) - c) / eps;
    m.nand_w2 = saved;

    saved = m.nand_b;
    m.nand_b += eps;
    g.nand_b = (cost(m) - c) / eps;
    m.nand_b = saved;

    saved = m.and_w1;
    m.and_w1 += eps;
    g.and_w1 = (cost(m) - c) / eps;
    m.and_w1 = saved;

    saved = m.and_w2;
    m.and_w2 += eps;
    g.and_w2 = (cost(m) - c) / eps;
    m.and_w2 = saved;

    saved = m.and_b;
    m.and_b += eps;
    g.and_b = (cost(m) - c) / eps;
    m.and_b = saved;

    return g;
}

fn learn(mut modal: Xor, g: Xor, rate: f64) -> Xor {
    modal.or_w1 -= rate * g.or_w1;
    modal.or_w2 -= rate * g.or_w2;
    modal.or_b -= rate * g.or_b;
    modal.nand_w1 -= rate * g.nand_w1;
    modal.nand_w2 -= rate * g.nand_w2;
    modal.nand_b -= rate * g.nand_b;
    modal.and_w1 -= rate * g.and_w1;
    modal.and_w2 -= rate * g.and_w2;
    modal.and_b -= rate * g.and_b;

    return modal;
}

pub fn run() {
    let mut modal = rand_xor();
    let mut cost_val = cost(modal);
    let accepted_cost = 0.000009;


    while cost_val > accepted_cost {
        let mut g = finite_diff(modal, EPS);
        modal = learn(modal, g, RATE);
        cost_val = cost(modal);
    }

    print_xor(modal);

    println!("Layer 1a - OR Gate");
    println!("{:-<30}", "-");
    for data in TRAIN_DATA {
        println!(
            "{:} | {:} => {:}",
            data[0],
            data[1],
            sigmoid((modal.or_w1 * data[0]) + (modal.or_w2 * data[1]) + modal.or_b)
        )
    }

    println!("\nLayer 1b - NAND Gate");
    println!("{:-<30}", "-");
    for data in TRAIN_DATA {
        println!(
            "{:} | {:} => {:}",
            data[0],
            data[1],
            sigmoid((modal.nand_w1 * data[0]) + (modal.nand_w2 * data[1]) + modal.nand_b)
        )
    }

    println!("\nLayer 2 - AND Gate");
    println!("{:-<30}", "-");
    for data in TRAIN_DATA {
        println!(
            "{:} | {:} => {:}",
            data[0],
            data[1],
            sigmoid((modal.and_w1 * data[0]) + (modal.and_w2 * data[1]) + modal.and_b)
        )
    }

    println!("\nXOR Gate");
    println!("{:-<30}", "-");
    for data in TRAIN_DATA {
        println!(
            "{:} | {:} => {:}",
            data[0],
            data[1],
            forward(modal, data[0], data[1])
        )
    }

    println!();
}
