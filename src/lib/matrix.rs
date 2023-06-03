use std::mem::size_of;

use super::utils::rand_float;

/**
### The concept of the Matrix struct here is a continuous Vector in memory so
```
let xor_data: Vec<f64> = [
     0.0, 0.0, 0.0,
     0.0, 1.0, 1.0,
     1.0, 0.0, 1.0,
     1.0, 1.0, 0.0,
];
let xor_matrix: Matrix = { rows: 4, cols: 3, data: xor_data };
```
 */

#[derive(Debug, Clone)]
pub struct Matrix {
    pub rows: i32,
    pub cols: i32,
    // /** How far we need to move til the cols */
    // stride: i32,
    // /** The ref of start of slice of the data in the vector */
    // pointer: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    /** Alloc a given matrix size in memory */
    pub fn allocate(rows: i32, cols: i32) -> Matrix {
        let matrix = Matrix {
            rows,
            cols,
            data: Vec::with_capacity((rows * cols) as usize),
        };
        return matrix;
    }

    /** Fill a matrix with defined value */
    pub fn fill(matrix: &mut Matrix, value: f64) {
        for i in 0..(matrix.rows * matrix.cols) {
            matrix.data.push(value)
        }
    }

    /** Fill a matrix with defined value */
    pub fn fill_vec(matrix: &mut Matrix, values: Vec<f64>) {
        assert!(
            values.len() == (matrix.rows * matrix.cols) as usize,
            "Values vector passed is not the same size as the Matrix"
        );
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let at = Self::at(matrix.cols, i, j);
                let value = values[at];
                matrix.data.push(value);
            }
        }
    }

    /** Fill a matrix with random value */
    pub fn fill_random(matrix: &mut Matrix, max: Option<f64>) {
        let max_random = match max {
            Some(x) => x,
            None => 10.0,
        };
        for i in 0..(matrix.rows * matrix.cols) {
            let value = rand_float(max_random);
            matrix.data.push(value)
        }
    }

    pub fn multiply(destination: Matrix, primary: Matrix, secondary: Matrix) -> Matrix {
        todo!()
    }

    /**
     Sum two matrix into a destination matrix
     - **NOTE:** Primary & Secondary needs to have the same size
    */
    pub fn add(mut destination: &mut Matrix, primary: &Matrix, secondary: &Matrix) {
        // Make sure both matrixes are the same size
        assert!(
            (primary.rows == secondary.rows),
            "Primary & Secondary matrixes do not have the same rows size"
        );
        assert!(
            (primary.cols == secondary.cols),
            "Primary & Secondary matrixes do not have the same size columns"
        );

        // To make sure it has values we can set to after
        Self::fill(destination, 0.0);

        for i in 0..primary.rows {
            for j in 0..primary.cols {
                destination.data[Self::at(primary.cols, i, j)] =
                    Self::get_at(primary, i, j) + Self::get_at(secondary, i, j)
            }
        }
    }

    /** Pretty print a given matrix */
    pub fn print(matrix: &Matrix) {
        let spacing = 25;
        let dash_nb = (matrix.data[0].to_string().chars().count() + 10) * matrix.cols as usize;

        println!("\n{:-<dash_nb$}", "-");
        for i in 0..matrix.rows {
            for j in 0..matrix.cols {
                let value = Self::get_at(matrix, i, j);
                print!("{value:<spacing$}");
            }
            println!();
        }
        println!("{:-<dash_nb$}\n", "-");
    }

    /**
     Internal function to help get a given value in the Matrix Vector
     let mat = Matrix {
        rows: 2,
        cols: 2
        data: [1,2,3,4]
     }

     then the Matrix is
        row_1: 1, 2
        row_2: 3, 4
    */
    fn at(cols: i32, row: i32, col: i32) -> usize {
        return ((row * cols) + col) as usize;
    }

    /**
     * Get an element in the Matrix at a given position
     */
    fn get_at(destination: &Matrix, row: i32, col: i32) -> f64 {
        return destination.data[Self::at(destination.cols, row, col)];
    }
}
