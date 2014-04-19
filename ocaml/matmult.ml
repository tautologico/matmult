
open Bigarray
let dim1 = Array2.dim1
let dim2 = Array2.dim2

type matrix = {
  m:    int;
  n:    int;
  elts: float array;
}

let zero m n =
  let elts = Array.make (m*n) 0.0 in
  { m; n; elts }

let get mat i j = mat.elts.(i * mat.n + j)

let set mat i j v = mat.elts.(i * mat.n + j) <- v

let print_matrix mat =
  for i = 0 to mat.m - 1 do
    for j = 0 to mat.n - 1 do
      Printf.printf "%5.2f " (get mat i j)
    done;
    Printf.printf "\n"
  done

let mat_mult m1 m2 =
  if m1.n <> m2.m then
    failwith "Incompatible matrix dimensions"
  else
    let res = zero m1.m m2.n in
    for i = 0 to res.m - 1 do
      for j = 0 to res.n - 1 do
        for p = 0 to m1.n - 1 do
          set res i j ((get res i j) +. (get m1 i p) *. (get m2 p j))
        done
      done
    done;
    res

(* Unroll inner loop at 2 iterations -- Only works for even sizes *)
let mat_mult_2 m1 m2 = 
  if m1.n <> m2.m then
    failwith "Incompatible matrix dimensions"
  else
    let res = zero m1.m m2.n in
    for i = 0 to res.m - 1 do
      for j = 0 to res.n - 1 do
        for p = 0 to (m1.n/2) - 1 do
          let p1 = 2 * p in
          let p2 = p1 + 1 in
          set res i j ((get res i j) +. (get m1 i p1) *. (get m2 p1 j));
          set res i j ((get res i j) +. (get m1 i p2) *. (get m2 p2 j))
        done
      done
    done;
    res

(* Unroll inner loop at 4 iterations -- Only works for sizes multiple of 4 *)
let mat_mult_4 m1 m2 = 
  if m1.n <> m2.m then
    failwith "Incompatible matrix dimensions"
  else
    let res = zero m1.m m2.n in
    for i = 0 to res.m - 1 do
      for j = 0 to res.n - 1 do
        for p = 0 to (m1.n/4) - 1 do
          let p1 = 4 * p in
          let p2 = p1 + 1 in
          let p3 = p2 + 1 in
          let p4 = p3 + 1 in
          set res i j ((get res i j) +. (get m1 i p1) *. (get m2 p1 j));
          set res i j ((get res i j) +. (get m1 i p2) *. (get m2 p2 j));
          set res i j ((get res i j) +. (get m1 i p3) *. (get m2 p3 j));
          set res i j ((get res i j) +. (get m1 i p4) *. (get m2 p4 j))
        done
      done
    done;
    res

let mat_mult_flops m1 m2 = 2.0 *. (float m1.m) *. (float m1.n) *. (float m2.n)

let b_mat_mult_flops m1 m2 = 2.0 *. (float (dim1 m1)) *. (float (dim2 m1)) *. (float (dim2 m2))

let b_mat_mult m1 m2 =
  if (dim2 m1) <> (dim1 m2) then
    failwith "Incompatible matrix dimensions (b_mat_mult)"
  else
    let res = Array2.create float64 c_layout (dim1 m1) (dim2 m2) in
    for i = 0 to (dim1 res) - 1 do
      for j = 0 to (dim2 res) - 1 do
        for p = 0 to (dim2 m1) - 1 do
          res.{i,j} <- res.{i,j} +. m1.{i,p} *. m2.{p,j}
        done
      done
    done;
    res

let gen_mat_1 m n start inc =
  let res = zero m n in
  Array.iteri (fun i _ -> res.elts.(i) <- start +. (float_of_int i) *. inc) res.elts;
  res

let gen_b_mat_1 m n start inc =
  let res = Array2.create float64 c_layout m n in
  let acc = ref start in
  for i = 0 to (dim1 res) - 1 do
    for j = 0 to (dim2 res) - 1 do
      res.{i,j} <- !acc;
      acc := !acc +. inc
    done
  done;
  res

let bench_mat_mult mf size =
  let m1 = gen_mat_1 size size 0.0 0.01 in
  let m2 = gen_mat_1 size size 3.2 0.02 in
  let start_time = Unix.gettimeofday () in
  let m3 = mf m1 m2 in
  let end_time = Unix.gettimeofday () in
  let elapsed = end_time -. start_time in
  let flops_sec = (mat_mult_flops m1 m2) /. (elapsed *. 1e9) in
  (elapsed, flops_sec)

let bench_b_mat_mult size =
  let m1 = gen_b_mat_1 size size 0.0 0.01 in
  let m2 = gen_b_mat_1 size size 3.2 0.02 in
  let start_time = Unix.gettimeofday () in
  let m3 = b_mat_mult m1 m2 in
  let end_time = Unix.gettimeofday () in
  let elapsed = end_time -. start_time in
  let flops_sec = (b_mat_mult_flops m1 m2) /. (elapsed *. 1e9) in
  (elapsed, flops_sec)

let bench_sizes low inc high f =
  let rec loop n =
    if n > high then ()
    else
      let elapsed, gflops = f n in
      Printf.printf "%d \t %5.3f \t %5.3f \n" n elapsed gflops;
      flush stdout;
      loop (n + inc)
  in
    loop low

let get_arg_opt i default = 
  if i < Array.length Sys.argv then
    int_of_string Sys.argv.(i)
  else
    default

let () =
  let low = get_arg_opt 1 200 in
  let inc = get_arg_opt 2 40 in
  let hi  = get_arg_opt 3 1200 in
  Printf.printf "*** Standard (array) gemm\n";
  Printf.printf "N \t time \t GFLOPS/s\n";
  bench_sizes low inc hi (bench_mat_mult mat_mult);

  Printf.printf "\n*** Unroll 2 gemm\n";
  Printf.printf "N \t time \t GFLOPS/s\n";
  bench_sizes low inc hi (bench_mat_mult mat_mult_2);

  Printf.printf "\n*** Unroll 4 gemm\n";
  Printf.printf "N \t time \t GFLOPS/s\n";
  bench_sizes low inc hi (bench_mat_mult mat_mult_4)

