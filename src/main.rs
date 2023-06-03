#![allow(dead_code, unused, special_module_name)]

use lib::matrix::Matrix;

mod lib;

fn main() {
    let mut destination: Matrix = Matrix::allocate(2, 8);
    let mut primary: Matrix = Matrix::allocate(2, 2);
    let mut secondary: Matrix = Matrix::allocate(2, 8);

    Matrix::fill_vec(&mut primary, [1.0, 2.0, 3.0, 4.0].to_vec());
    // Matrix::fill_random(&mut secondary, None);

    // Matrix::add(&mut destination, &primary, &secondary);

    Matrix::print(&primary);
}
