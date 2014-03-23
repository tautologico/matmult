
extern crate time; 

use std::vec_ng::Vec;
use std::iter::range_step;
use std::unstable::simd::f64x2;

use time::precise_time_s;

// matrices in column-major order
pub struct Matrix {
    m: uint,
    n: uint,

    entries: Vec<f64>
}

impl Matrix {
    #[inline]
    pub fn zero(m: uint, n: uint) -> Matrix {
        Matrix { m: m, n: n, entries: Vec::from_elem(m * n, 0.0) }
    }

    #[inline]
    pub fn get(&self, i: uint, j: uint) -> f64 {
        *self.entries.get(j * self.m + i)
    }

    #[inline]
    pub fn set(&mut self, i: uint, j: uint, v: f64) {
        let elref = self.entries.get_mut(j * self.m + i);
        *elref = v;
    }

    pub fn print(&self) {
        for i in range(0, self.m) {
            for j in range(0, self.n) {
                print!("{:f} ", self.get(i, j));
            }
            println!(" ");
        }
    }
}

fn mat_mul(m1: &Matrix, m2: &Matrix) -> Matrix {
    if m1.n != m2.m {
        fail!("Incompatible matrix sizes");
    }

    let mut res = Matrix::zero(m1.m, m2.n);
    for j in range(0, res.n) {
        for i in range(0, res.m) {
            let mut acc = 0.0;
            for p in range(0, m1.n) {
                acc += m1.get(i, p) * m2.get(p, j);
            }
            res.set(i, j, acc);
        }
    }

    res
}

#[inline]
fn set_pair(res: &mut Matrix, i: uint, j: uint, p: f64x2)
{
    match p {
        f64x2(e0, e1) => {
            let r0 = res.get(i,   j);
            let r1 = res.get(i+1, j);
            res.set(i,   j, r0 + e0);
            res.set(i+1, j, r1 + e1);
        }
    }
}

#[allow(experimental)]
fn mul_4x4_block(lhs: &Matrix, i: uint, rhs: &Matrix, j: uint, res: &mut Matrix)
{
    let mut c00_c10 = f64x2(0.0, 0.0);
    let mut c01_c11 = f64x2(0.0, 0.0);
    let mut c02_c12 = f64x2(0.0, 0.0);
    let mut c03_c13 = f64x2(0.0, 0.0);
    let mut c20_c30 = f64x2(0.0, 0.0);
    let mut c21_c31 = f64x2(0.0, 0.0);
    let mut c22_c32 = f64x2(0.0, 0.0);
    let mut c23_c33 = f64x2(0.0, 0.0);

    let mut a0p_a1p = f64x2(0.0, 0.0);
    let mut a2p_a3p = f64x2(0.0, 0.0);

    let mut bp0_bp0 = f64x2(0.0, 0.0); 
    let mut bp1_bp1 = f64x2(0.0, 0.0);
    let mut bp2_bp2 = f64x2(0.0, 0.0);
    let mut bp3_bp3 = f64x2(0.0, 0.0);

    for p in range(0, lhs.m) {
        a0p_a1p = f64x2(lhs.get(i,   p), lhs.get(i+1, p));
        a2p_a3p = f64x2(lhs.get(i+2, p), lhs.get(i+3, p));

        bp0_bp0 = f64x2(rhs.get(p,   j), rhs.get(p,   j));
        bp1_bp1 = f64x2(rhs.get(p, j+1), rhs.get(p, j+1));
        bp2_bp2 = f64x2(rhs.get(p, j+2), rhs.get(p, j+2));
        bp3_bp3 = f64x2(rhs.get(p, j+3), rhs.get(p, j+3));

        c00_c10 += a0p_a1p * bp0_bp0;
        c01_c11 += a0p_a1p * bp1_bp1;
        c02_c12 += a0p_a1p * bp2_bp2;
        c03_c13 += a0p_a1p * bp3_bp3;

        c20_c30 += a2p_a3p * bp0_bp0;
        c21_c31 += a2p_a3p * bp1_bp1;
        c22_c32 += a2p_a3p * bp2_bp2;
        c23_c33 += a2p_a3p * bp3_bp3;
    }
    
    set_pair(res, i,   j, c00_c10);
    set_pair(res, i, j+1, c01_c11);
    set_pair(res, i, j+2, c02_c12);
    set_pair(res, i, j+3, c03_c13);

    set_pair(res, i+2,   j, c20_c30);
    set_pair(res, i+2, j+1, c21_c31);
    set_pair(res, i+2, j+2, c22_c32);
    set_pair(res, i+2, j+3, c23_c33);
}


pub fn mat_mul_4x4(lhs: &Matrix, rhs: &Matrix) -> Matrix
{
    if lhs.n != rhs.m {
        fail!("Incompatible matrix sizes. LHS: {:?}, RHS: {:?}",
                   (lhs.m, lhs.n),
                   (rhs.m, rhs.n))
    }

    let mut res = Matrix::zero(lhs.m, rhs.n);

    for j in range_step(0, res.n, 4) {
        for i in range_step(0, res.m, 4) {
            mul_4x4_block(lhs, i, rhs, j, &mut res);
        }
    }
    res
}

fn test_mat_mul() {
    let mut m1 = Matrix { m: 2, n: 2, entries: Vec::from_elem(4, 0.25) };
    let mut m2 = Matrix { m: 2, n: 2, entries: Vec::from_elem(4, 4.2) };

    m1.set(0, 0, 4.2);
    m1.set(1, 0, 3.7);
    m2.set(0, 1, 2.3);
    m2.set(1, 1, 1.0);

    let m3 = mat_mul(&m1, &m2);

    m3.print();
}

fn gen_mat_1(m: uint, n: uint, start: f64, inc: f64) -> Matrix {
    let mut res = Matrix::zero(m, n);

    let mut acc = start;
    for j in range(0, n) {
        for i in range(0, m) {
            res.set(i, j, acc);
            acc += inc;
        }
    }
    res
}

fn test_mat_mul_4x4() {
    let m1 = gen_mat_1(4, 4, 0.0, 0.5);
    let m2 = gen_mat_1(4, 4, 4.25, 0.25);

    m1.print();
    m2.print(); 

    let m3 = mat_mul_4x4(&m1, &m2);

    m3.print();   
}

fn sum_matrix(m: &Matrix) -> f64 {
    let mut res = 0.0;
    for j in range(0, m.n) {
        for i in range(0, m.m) {
            res += m.get(i, j);
        }
    }
    res
}

#[allow(uppercase_variables)]
fn bench_mat_mul() {
    let N = 1200;
    let m1 = gen_mat_1(N, N, 0.0, 0.01);
    let m2 = gen_mat_1(N, N, 3.2, 0.02);

    let start = precise_time_s();
    let m3 = mat_mul(&m1, &m2);
    let stop = precise_time_s();

    let Nf = N as f64;
    let mulgflops = (2.0 * Nf * Nf * Nf) / 1e9;

    // avoid optimizer throwing m3 away
    //let s = sum_matrix(&m3);
    
    let time = stop - start;
    //println!("Sum: {:f}", s);
    println!("Matrix Multiply (loop): {:f}s, {:f} GFLOPS/s", time, mulgflops / time);
}

#[allow(uppercase_variables)]
fn bench_mat_mul_4x4() {
    let N = 1200;
    let m1 = gen_mat_1(N, N, 0.0, 0.01);
    let m2 = gen_mat_1(N, N, 3.2, 0.02);

    let start = precise_time_s();
    let m3 = mat_mul_4x4(&m1, &m2);
    let stop = precise_time_s();

    let Nf = N as f64;
    let mulgflops = (2.0 * Nf * Nf * Nf) / 1e9;

    // avoid optimizer throwing m3 away
    let s = sum_matrix(&m3);
    
    let time = stop - start;
    println!("Sum: {:f}", s);
    println!("Matrix Multiply (SIMD): {:f}s, {:f} GFLOPS/s", time, mulgflops / time);
}


fn main() {
    bench_mat_mul();

    bench_mat_mul_4x4();
}
