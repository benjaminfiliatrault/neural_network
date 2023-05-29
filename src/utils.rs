use rand::Rng;

pub fn rand_float(max: f64) -> f64 {
    let mut rng = rand::thread_rng();
    return rng.gen_range(0.0..max);
}
