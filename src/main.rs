#![allow(dead_code, unused, special_module_name)]
use lib::matrix::Matrix;
use std::time::Instant;

mod data;
mod lib;

#[derive(Debug, Clone)]
struct Xor {
    activation_0: Matrix,
    // Layer 1
    weights_1: Matrix,
    biases_1: Matrix,
    activation_1: Matrix,
    // Layer 2
    weights_2: Matrix,
    biases_2: Matrix,
    activation_2: Matrix,
}

fn xor_alloc() -> Xor {
    return Xor {
        activation_0: Matrix::allocate(1, 2),
        weights_1: Matrix::allocate(2, 2),
        biases_1: Matrix::allocate(1, 2),
        activation_1: Matrix::allocate(1, 2),
        weights_2: Matrix::allocate(2, 1),
        biases_2: Matrix::allocate(1, 1),
        activation_2: Matrix::allocate(1, 1),
    };
}

fn forward_xor(modal: &mut Xor) {
    // Layer 1 forwarding inputs
    Matrix::multiply(
        &mut modal.activation_1,
        &modal.activation_0,
        &modal.weights_1,
    );
    Matrix::add(&mut modal.activation_1, &modal.biases_1);
    Matrix::sigmoid(&mut modal.activation_1);

    // forwarding layer 1 to layer 2
    Matrix::multiply(
        &mut modal.activation_2,
        &modal.activation_1,
        &modal.weights_2,
    );
    Matrix::add(&mut modal.activation_2, &modal.biases_2);
    Matrix::sigmoid(&mut modal.activation_2);
}

fn loss(mut modal: &mut Xor, inputs: &Matrix, output: &Matrix) -> f64 {
    assert!(inputs.rows == output.rows);
    assert!(output.cols == modal.activation_2.cols);

    let mut loss = 0.0;
    let in_rows = inputs.rows;
    let out_cols = output.cols;

    for row in 0..in_rows {
        let x = Matrix::row(&inputs, row);
        let y = Matrix::row(&output, row);

        Matrix::copy(&mut modal.activation_0, &x);
        forward_xor(&mut modal);

        for col in 0..out_cols {
            let mut diff = Matrix::get_at(&modal.activation_2, 0, col) - Matrix::get_at(&y, 0, col);
            loss += diff * diff;
        }
    }

    return loss / in_rows as f64;
}

fn finite_diff(modal: &mut Xor, gradient: &mut Xor, eps: f64, inputs: &Matrix, outputs: &Matrix) {
    let mut saved: f64;
    let original_loss = loss(modal, &inputs, &outputs);

    // Oh boii here we go again with copy pasting to
    // visually remember myself how it works... no loop maybe next time
    for row in 0..modal.weights_1.rows {
        for col in 0..modal.weights_1.cols {
            saved = Matrix::get_at(&modal.weights_1, row, col);

            modal.weights_1.data[Matrix::at(modal.weights_1.stride, row, col)] += eps;

            gradient.weights_1.data[Matrix::at(modal.weights_1.stride, row, col)] =
                (loss(modal, &inputs, &outputs) - original_loss) / eps;

            modal.weights_1.data[Matrix::at(modal.weights_1.stride, row, col)] = saved;
        }
    }

    for row in 0..modal.biases_1.rows {
        for col in 0..modal.biases_1.cols {
            saved = Matrix::get_at(&modal.biases_1, row, col);
            modal.biases_1.data[Matrix::at(modal.biases_1.stride, row, col)] += eps;

            gradient.biases_1.data[Matrix::at(modal.biases_1.stride, row, col)] =
                (loss(modal, &inputs, &outputs) - original_loss) / eps;

            modal.biases_1.data[Matrix::at(modal.biases_1.stride, row, col)] = saved;
        }
    }

    for row in 0..modal.weights_2.rows {
        for col in 0..modal.weights_2.cols {
            saved = Matrix::get_at(&modal.weights_2, row, col);
            modal.weights_2.data[Matrix::at(modal.weights_2.stride, row, col)] += eps;

            gradient.weights_2.data[Matrix::at(modal.weights_2.stride, row, col)] =
                (loss(modal, &inputs, &outputs) - original_loss) / eps;

            modal.weights_2.data[Matrix::at(modal.weights_2.stride, row, col)] = saved;
        }
    }

    for row in 0..modal.biases_2.rows {
        for col in 0..modal.biases_2.cols {
            saved = Matrix::get_at(&modal.biases_2, row, col);
            modal.biases_2.data[Matrix::at(modal.biases_2.stride, row, col)] += eps;

            gradient.biases_2.data[Matrix::at(modal.biases_2.stride, row, col)] =
                (loss(modal, &inputs, &outputs) - original_loss) / eps;

            modal.biases_2.data[Matrix::at(modal.biases_2.stride, row, col)] = saved;
        }
    }
}

fn xor_learn(modal: &mut Xor, gradient: &mut Xor, rate: f64) {
    // Oh boii here we go again with copy pasting to
    // visually remember myself how it works... no loop maybe next time
    for row in 0..modal.weights_1.rows {
        for col in 0..modal.weights_1.cols {
            modal.weights_1.data[Matrix::at(modal.weights_1.stride, row, col)] -=
                gradient.weights_1.data[Matrix::at(modal.weights_1.stride, row, col)] * rate;
        }
    }

    for row in 0..modal.biases_1.rows {
        for col in 0..modal.biases_1.cols {
            modal.biases_1.data[Matrix::at(modal.biases_1.stride, row, col)] -=
                gradient.biases_1.data[Matrix::at(modal.biases_1.stride, row, col)] * rate;
        }
    }

    for row in 0..modal.weights_2.rows {
        for col in 0..modal.weights_2.cols {
            modal.weights_2.data[Matrix::at(modal.weights_2.stride, row, col)] -=
                gradient.weights_2.data[Matrix::at(modal.weights_2.stride, row, col)] * rate;
        }
    }

    for row in 0..modal.biases_2.rows {
        for col in 0..modal.biases_2.cols {
            modal.biases_2.data[Matrix::at(modal.biases_2.stride, row, col)] -=
                gradient.biases_2.data[Matrix::at(modal.biases_2.stride, row, col)] * rate;
        }
    }
}

fn main() {
    let now = Instant::now();

    let train_data = data::XOR_GATE;
    let stride = 3;
    let rows = train_data.len() as i32 / stride;

    let inputs = Matrix {
        rows,
        cols: 2,
        stride: 3,
        data: train_data.to_vec(),
    };

    let outputs = Matrix {
        rows,
        cols: 1,
        stride: 3,
        data: train_data.split_at(2).1.to_vec(),
    };

    let mut modal = xor_alloc();
    let mut gradient = xor_alloc();

    Matrix::fill_vec(&mut modal.activation_0, vec![0.0, 0.0]);

    // Layer 1 randomize values
    Matrix::fill_random(&mut modal.weights_1, Some(-5.0), Some(8.0));
    Matrix::fill_random(&mut modal.biases_1, Some(-5.0), Some(1.0));
    // Layer 2 randomize values
    Matrix::fill_random(&mut modal.weights_2, Some(0.0), Some(10.0));
    Matrix::fill_random(&mut modal.biases_2, Some(0.0), Some(5.0));

    // Fill the gradient to 0
    Matrix::fill(&mut gradient.weights_1, 0.0);
    Matrix::fill(&mut gradient.biases_1, 0.0);
    Matrix::fill(&mut gradient.weights_2, 0.0);
    Matrix::fill(&mut gradient.biases_2, 0.0);

    let eps = 1e-2;
    let learning_rate = 1e-2;
    let acceptable_cost = 0.002;

    let mut cost = loss(&mut modal, &inputs, &outputs);

    while cost > acceptable_cost {
        finite_diff(&mut modal, &mut gradient, eps, &inputs, &outputs);
        xor_learn(&mut modal, &mut gradient, learning_rate);
        println!("loss: {:}", cost);
        cost = loss(&mut modal, &inputs, &outputs);
    }

    println!("-------");

    for i in 0..2 {
        for j in 0..2 {
            modal.activation_0.data[Matrix::at(modal.activation_0.stride, 0, 0)] = i as f64;
            modal.activation_0.data[Matrix::at(modal.activation_0.stride, 0, 1)] = j as f64;

            forward_xor(&mut modal);

            println!("{:} | {:} => {:?}", i, j, modal.activation_2.data[0]);
        }
    }

    Matrix::print(&modal.weights_1, "Modal - weights_1");
    Matrix::print(&modal.biases_1, "Modal - biases_1");
    Matrix::print(&modal.weights_2, "Modal - weights_2");
    Matrix::print(&modal.biases_2, "Modal - biases_2");

    let end_elapsed = now.elapsed().as_micros() as f64 / 1e+6;
    println!("\nExecuted in {:?} seconds\n", end_elapsed);
}
