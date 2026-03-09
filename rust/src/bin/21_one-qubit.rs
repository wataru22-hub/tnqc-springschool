use num_complex::Complex64;

type C = Complex64;
type Vec2 = [C; 2];
type Mat2 = [[C; 2]; 2];

fn c(re: f64, im: f64) -> C {
    C::new(re, im)
}

fn clean(x: f64) -> f64 {
    if x.abs() < 1e-12 { 0.0 } else { x }
}

fn fmt_real(x: f64) -> String {
    let x = clean(x);
    if (x - x.round()).abs() < 1e-12 {
        format!("{}", x.round() as i64)
    } else {
        format!("{:.6}", x)
    }
}

fn fmt_complex(z: C) -> String {
    let re = clean(z.re);
    let im = clean(z.im);

    if im == 0.0 {
        fmt_real(re)
    } else if re == 0.0 {
        format!("{}im", fmt_real(im))
    } else {
        let sign = if im >= 0.0 { "+" } else { "" };
        format!("{}{}{}im", fmt_real(re), sign, fmt_real(im))
    }
}

fn print_vector<const N: usize>(name: &str, v: &[C; N]) {
    println!("{name} =");
    for x in v {
        println!("[{}]", fmt_complex(*x));
    }
    println!();
}

fn print_matrix<const R: usize, const COLUMNS: usize>(name: &str, m: &[[C; COLUMNS]; R]) {
    println!("{name} =");
    for row in m {
        let line = row
            .iter()
            .map(|&z| fmt_complex(z))
            .collect::<Vec<_>>()
            .join(", ");
        println!("[{}]", line);
    }
    println!();
}

fn mat_vec_mul<const R: usize, const COLUMNS: usize>(
    m: &[[C; COLUMNS]; R],
    v: &[C; COLUMNS],
) -> [C; R] {
    let zero = c(0.0, 0.0);
    let mut out = [zero; R];

    for i in 0..R {
        let mut sum = zero;
        for j in 0..COLUMNS {
            sum += m[i][j] * v[j];
        }
        out[i] = sum;
    }

    out
}

fn main() {
    // |0>, |1>
    let ket0: Vec2 = [c(1.0, 0.0), c(0.0, 0.0)];
    let ket1: Vec2 = [c(0.0, 0.0), c(1.0, 0.0)];
    print_vector("|0>", &ket0);
    print_vector("|1>", &ket1);

    // X gate
    let x: Mat2 = [
        [c(0.0, 0.0), c(1.0, 0.0)],
        [c(1.0, 0.0), c(0.0, 0.0)],
    ];
    print_matrix("X", &x);
    print_vector("X|0>", &mat_vec_mul(&x, &ket0));
    print_vector("X|1>", &mat_vec_mul(&x, &ket1));

    // Y gate
    let y: Mat2 = [
        [c(0.0, 0.0), c(0.0, -1.0)],
        [c(0.0, 1.0), c(0.0, 0.0)],
    ];
    print_matrix("Y", &y);
    print_vector("Y|0>", &mat_vec_mul(&y, &ket0));
    print_vector("Y|1>", &mat_vec_mul(&y, &ket1));

    // Z gate
    let z: Mat2 = [
        [c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(-1.0, 0.0)],
    ];
    print_matrix("Z", &z);
    print_vector("Z|0>", &mat_vec_mul(&z, &ket0));
    print_vector("Z|1>", &mat_vec_mul(&z, &ket1));

    // Hadamard gate
    let s = 1.0 / 2.0_f64.sqrt();
    let h: Mat2 = [
        [c(s, 0.0), c(s, 0.0)],
        [c(s, 0.0), c(-s, 0.0)],
    ];
    print_matrix("H", &h);
    print_vector("H|0>", &mat_vec_mul(&h, &ket0));
    print_vector("H|1>", &mat_vec_mul(&h, &ket1));
}
