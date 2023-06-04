use rand::Rng;

pub fn rand_float(min:f64, max: f64) -> f64 {
    let mut rng = rand::thread_rng();
    return rng.gen_range(min..max);
}

pub fn sigmoid(value: f64) -> f64 {
    return 1.0 / (1.0 + f64::exp(-value));
}