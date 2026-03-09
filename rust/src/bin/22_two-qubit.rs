use num_complex::Complex64;

type C = Complex64;
type Vec4 = [C; 4];
type Mat4 = [[C; 4]; 4];

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

fn mat_mul<const R: usize, const K: usize, const COLUMNS: usize>(
    a: &[[C; K]; R],
    b: &[[C; COLUMNS]; K],
) -> [[C; COLUMNS]; R] {
    let zero = c(0.0, 0.0);
    let mut out = [[zero; COLUMNS]; R];

    for i in 0..R {
        for j in 0..COLUMNS {
            let mut sum = zero;
            for k in 0..K {
                sum += a[i][k] * b[k][j];
            }
            out[i][j] = sum;
        }
    }

    out
}

fn main() {
    // |00>, |01>, |10>, |11>
    let ket00: Vec4 = [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
    let ket01: Vec4 = [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)];
    let ket10: Vec4 = [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)];
    let ket11: Vec4 = [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)];
    print_vector("|00>", &ket00);
    print_vector("|01>", &ket01);
    print_vector("|10>", &ket10);
    print_vector("|11>", &ket11);

    // CX12 gate
    let cx12: Mat4 = [
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
    ];
    print_matrix("CX12", &cx12);
    print_vector("CX12|00>", &mat_vec_mul(&cx12, &ket00));
    print_vector("CX12|01>", &mat_vec_mul(&cx12, &ket01));
    print_vector("CX12|10>", &mat_vec_mul(&cx12, &ket10));
    print_vector("CX12|11>", &mat_vec_mul(&cx12, &ket11));

    // CX21 gate
    let cx21: Mat4 = [
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
    ];
    print_matrix("CX21", &cx21);
    print_vector("CX21|00>", &mat_vec_mul(&cx21, &ket00));
    print_vector("CX21|01>", &mat_vec_mul(&cx21, &ket01));
    print_vector("CX21|10>", &mat_vec_mul(&cx21, &ket10));
    print_vector("CX21|11>", &mat_vec_mul(&cx21, &ket11));

    // CZ gate
    let cz: Mat4 = [
        [c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)],
        [c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(-1.0, 0.0)],
    ];
    print_matrix("CZ", &cz);
    print_vector("CZ|00>", &mat_vec_mul(&cz, &ket00));
    print_vector("CZ|01>", &mat_vec_mul(&cz, &ket01));
    print_vector("CZ|10>", &mat_vec_mul(&cz, &ket10));
    print_vector("CZ|11>", &mat_vec_mul(&cz, &ket11));

    // Swap gate
    let swap = mat_mul(&mat_mul(&cx12, &cx21), &cx12);
    print_matrix("Swap", &swap);
    print_vector("Swap|00>", &mat_vec_mul(&swap, &ket00));
    print_vector("Swap|01>", &mat_vec_mul(&swap, &ket01));
    print_vector("Swap|10>", &mat_vec_mul(&swap, &ket10));
    print_vector("Swap|11>", &mat_vec_mul(&swap, &ket11));
}
