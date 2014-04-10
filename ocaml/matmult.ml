open Core.Std
open Core_bench.Std

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

let mat_mult_flops m1 m2 = 2.0 *. (float m1.m) *. (float m1.n) *. (float m2.n)

let gen_mat_1 m n start inc =
  let res = zero m n in
  Array.iteri (fun i _ -> res.elts.(i) <- start +. (float_of_int i) *. inc) res.elts;
  res

let bench_mat_mult size =
  let m1 = gen_mat_1 size size 0.0 0.01 in
  let m2 = gen_mat_1 size size 3.2 0.02 in
  mat_mult m1 m2

let tests size =
  let test name f = Bench.Test.create f ~name in
  [
    test "array mat_mult"   (fun () -> ignore (bench_mat_mult size))
  ]

let m1 = { m = 2; n = 2; elts = [| 1.0; 2.0;
                                   3.0; 4.0 |]}

let m2 = { m = 2; n = 2; elts = [| 3.0; 1.0;
                                   5.0; 2.0 |]}

let () =
  tests 1200
  |> Bench.make_command
  |> Command.run
