open OUnit2

(* based on submodules/falconn/src/test/cpp_wrapper_test.cc *)
let basic_test_dense_1 _ =
  let p1 = [| 1.0; 0.0; 0.0; 0.0 |] in 
  let p2 = [| 0.6; 0.8; 0.0; 0.0 |] in 
  let p3 = [| 0.0; 0.0; 1.0; 0.0 |] in 
  let pmat = Falconn.input_of_array [| p1; p2; p3 |] in 
  let index = Falconn.create ~l:4 ~num_probes:4 pmat in
  let res1 = Falconn.find_nearest_neighbor index p1 in 
  let res2 = Falconn.find_nearest_neighbor index p2 in 
  let res3 = Falconn.find_nearest_neighbor index p3 in 
  let p4 = [| 0.0; 1.0; 0.0; 0.0 |] in 
  let res4 = Falconn.find_nearest_neighbor index p4 in 
  assert_equal res1 0;
  assert_equal res2 1;
  assert_equal res3 2;
  assert_equal res4 1

let basic_test_multi _ =
  let p1 = [| 1.0; 0.0; 0.0; 0.0 |] in 
  let p2 = [| 0.6; 0.8; 0.0; 0.0 |] in 
  let p3 = [| 0.0; 0.0; 0.0; 1.0 |] in 
  let p4 = [| 0.0; 0.0; 0.0; 1.0 |] in
  let pmat = Falconn.input_of_array [| p1; p2; p3; p4 |] in 
  let index = Falconn.create ~l:4 ~num_probes:4 pmat in
  let neighbors = Falconn.find_k_nearest_neighbors index 2 p3 in 
  let v1 = Array.get neighbors 0 in 
  let v2 = Array.get neighbors 1 in 
  let _ = Printf.printf "%d %d %d" (Array.length neighbors) v1 v2 in
  assert_equal (Array.length neighbors) 2;
  assert_bool "test 1" ((v1 == 2) || (v1 == 3));
  assert_bool "test 2" ((v2 == 2) || (v2 == 3))

let suite = 
  "suite">::: [
    "basic_test_dense_1" >:: basic_test_dense_1;
    "basic_test_multi" >:: basic_test_multi
  ]

let () =
  run_test_tt_main suite