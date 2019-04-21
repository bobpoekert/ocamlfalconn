open OUnit2

let basic_test_dense_1 _ =
  let p1 = [| 1.0; 0.0; 0.0; 0.0 |] in 
  let p2 = [| 0.6; 0.8; 0.0; 0.0 |] in 
  let p3 = [| 0.0; 0.0; 1.0; 0.0 |] in 
  let pmat = Falconn.input_of_array [| p1; p2; p3 |] in 
  let index = Falconn.create ~l:4 pmat in
  let res1 = Falconn.find_nearest_neighbor index p1 in 
  let res2 = Falconn.find_nearest_neighbor index p2 in 
  let res3 = Falconn.find_nearest_neighbor index p3 in 
  let p4 = [| 0.0; 1.0; 0.0; 0.0 |] in 
  let res4 = Falconn.find_nearest_neighbor index p4 in 
  assert_equal res1 0;
  assert_equal res2 1;
  assert_equal res3 2;
  assert_equal res4 1

let suite = 
  "suite">::: [
    "basic_test_dense_1" >:: basic_test_dense_1
  ]

let () =
  run_test_tt_main suite