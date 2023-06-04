use std::mem::size_of;

use super::utils::{rand_float, sigmoid};

/**
### The concept of the Matrix struct here is a continuous Vector in memory so
```
let input_data: Vec<f64> = [
     0.0, 0.0,
     0.0, 1.0,
     1.0, 0.0,
     1.0, 1.0,
];
let output_data: Vec<f64> = [
    0.0,
    1.0,
    1.0,
    0.0,
];
let input_matrix: Matrix = { rows: 4, cols: 2, data: input_data };
let output_matrix: Matrix = { rows: 4, cols: 1, data: output_data };
```
 */

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    // /** How far we need to move til the cols */
    pub stride: i32,
    // /** The ref of start of slice of the data in the vector */
    // pointer: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /** Allocate a given matrix size in memory */
    pub fn allocate(rows: i32, cols: i32) -> Matrix {
        let matrix = Matrix {
            rows,
            cols,
            stride: cols,
            data: Vec::with_capacity((rows * cols) as usize),
        };
        return matrix;
    }

    /** Fill a matrix with a defined value */
    pub fn fill(matrix: &mut Matrix, value: f64) {
        for i in 0..(matrix.rows * matrix.cols) {
            matrix.data.push(value)
        }
    }

    /** Fill a matrix with defined values  */
    pub fn fill_vec(matrix: &mut Matrix, values: Vec<f64>) {
        assert!(
            values.len() == (matrix.rows * matrix.cols) as usize,
            "Values vector passed is not the same size as the Matrix"
        );
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let at = Self::at(matrix.stride, i, j);
                let value = values[at];
                matrix.data.push(value);
            }
        }
    }

    /** Fill a matrix with random value */
    pub fn fill_random(matrix: &mut Matrix, min: Option<f64>, max: Option<f64>) {
        let max_random = match max {
            Some(x) => x,
            None => 10.0,
        };

        let min_random = match min {
            Some(x) => x,
            None => 0.0,
        };

        for i in 0..(matrix.rows * matrix.cols) {
            let value = rand_float(min_random, max_random);
            matrix.data.push(value)
        }
    }

    pub fn copy(mut destination: &mut Matrix, source: &Matrix) {
        assert!(
            destination.cols == source.cols && destination.rows == source.rows,
            "Destination & source should have the same dimensions"
        );
        for row in 0..source.rows {
            for col in 0..source.cols {
                let value = Self::get_at(source, row, col);
                destination.data[Self::at(destination.stride, row, col)] = value;
            }
        }
    }

    /**
     Sum two matrix into a destination matrix
     - **NOTE:** matrix_1 & matrix_2 needs to have the same size
    */
    pub fn add(matrix_1: &mut Matrix, matrix_2: &Matrix) {
        // Make sure both matrixes are the same size
        assert!(
            (matrix_1.rows == matrix_2.rows),
            "Primary & Secondary matrixes do not have the same rows size"
        );
        assert!(
            (matrix_1.cols == matrix_2.cols),
            "Primary & Secondary matrixes do not have the same size columns"
        );

        for row in 0..matrix_1.rows {
            for col in 0..matrix_1.cols {
                matrix_1.data[Self::at(matrix_1.stride, row, col)] =
                    Self::get_at(matrix_1, row, col) + Self::get_at(matrix_2, row, col)
            }
        }
    }

    /** Multiple two matrixes into an activation/destination Matrix */
    pub fn multiply(destination: &mut Matrix, matrix_1: &Matrix, matrix_2: &Matrix) {
        // Make sure both matrixes are multiplicable
        assert!(
            (matrix_1.cols == matrix_2.rows),
            "Primary & Secondary matrixes cannot be multiplied"
        );
        assert!(
            (destination.rows == matrix_1.rows),
            "Destination matrix doesn't have the same rows size as primary"
        );
        assert!(
            (destination.cols == matrix_2.cols),
            "Destination matrix doesn't have the same cols size as secondary"
        );

        // Make sure destination matrix has values in it
        if destination.data.len() == 0 {
            Self::fill(destination, 0.0);
        }

        let inner_size = matrix_1.cols;

        for row in 0..destination.rows {
            for col in 0..destination.cols {
                for inner in 0..inner_size {
                    destination.data[Self::at(destination.stride, row, col)] +=
                        Self::get_at(matrix_1, row, inner) * Self::get_at(matrix_2, inner, col)
                }
            }
        }
    }

    /** Apply sigmoid on the entire Matrix passed */
    pub fn sigmoid(mut matrix: &mut Matrix) {
        for row in 0..matrix.rows {
            for col in 0..matrix.cols {
                matrix.data[Self::at(matrix.stride, row, col)] =
                    sigmoid(matrix.data[Self::at(matrix.stride, row, col)])
            }
        }
    }

    /** Returns all the values of a given row */
    pub fn row(matrix: &Matrix, row: i32) -> Matrix {
        return Matrix {
            rows: 1,
            cols: matrix.cols,
            stride: matrix.stride,
            data: matrix.data
                [Self::at(matrix.stride, row, 0)..Self::at(matrix.stride, row, matrix.cols)]
                .to_vec(),
        };
    }

    pub fn col(matrix: &Matrix, col: i32) -> Matrix {
        todo!()
    }

    pub fn sub_matrix(matrix: &Matrix) {
        todo!()
    }

    /** Pretty print a given matrix */
    pub fn print(matrix: &Matrix, title: &str) {
        let trailing = 8;
        let nb_char_len = format!("{:#0.trailing$}", matrix.data[0]).chars().count();
        let spacing = (nb_char_len + (nb_char_len / 2)) as usize;
        let dash_nb = (nb_char_len * matrix.cols as usize) + 10;

        println!("\n{:^dash_nb$}", title);
        println!("{:-<dash_nb$}", "-");
        for row in 0..matrix.rows {
            for col in 0..matrix.cols {
                let value = Self::get_at(matrix, row, col);
                let formatted_value = format!("{value:.trailing$}");
                print!("{formatted_value:^spacing$}");
            }
            println!();
        }
        println!("{:-<dash_nb$}", "-");
    }

    /**
     Internal function to help get a given value at an index in the Matrix Vector
     let mat = Matrix {
        rows: 2,
        cols: 2
        data: [1,2,3,4]
     }

     then the Matrix is
        row_1: 1, 2
        row_2: 3, 4
    */
    pub fn at(stride: i32, row: i32, col: i32) -> usize {
        return ((row * stride) + col) as usize;
    }

    /**
     * Get an element in the Matrix at a given position
     */
    pub fn get_at(source: &Matrix, row: i32, col: i32) -> f64 {
        return source.data[Self::at(source.stride, row, col)];
    }
}

#[cfg(test)]
mod tests {
    use crate::lib::matrix::Matrix;

    #[test]
    fn matrix_multiply_2_x_2() {
        let result = [14.0, 19.0, 24.0, 33.0];

        let mut m1 = Matrix::allocate(2, 2);
        Matrix::fill_vec(
            &mut m1,
            vec![
                2.0, 3.0, //
                4.0, 5.0, //
            ],
        );

        let mut m2 = Matrix::allocate(2, 2);
        Matrix::fill_vec(
            &mut m2,
            vec![
                1.0, 2.0, //
                4.0, 5.0, //
            ],
        );

        let mut dest = Matrix::allocate(2, 2);

        Matrix::multiply(&mut dest, &m1, &m2);

        assert_eq!(dest.data, result);
    }

    #[test]
    fn matrix_add_2_x_2() {
        let result = [
            3.0, 4.0, //
            0.0, 2.0, //
        ];

        let mut m1 = Matrix::allocate(2, 2);
        Matrix::fill_vec(
            &mut m1,
            vec![
                1.0, 5.0, //
                -4.0, 3.0, //
            ],
        );

        let mut m2 = Matrix::allocate(2, 2);
        Matrix::fill_vec(
            &mut m2,
            vec![
                2.0, -1.0, //
                4.0, -1.0, //
            ],
        );

        Matrix::add(&mut m1, &m2);

        assert_eq!(m1.data, result);
    }
}
