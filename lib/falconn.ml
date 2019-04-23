open Bigarray

type _native_index
type distance_function = [`Negative_inner_product | `Euclidean_squared]
type input_array = (float, float32_elt, c_layout) Array2.t 

type index = (_native_index * input_array) (* falconn tries to access memory in input_array after create but we own the reference, so we need to keep the reference to the array alive as long as the reference to the index is alive *)

external _create : distance_function -> bool -> int -> int -> input_array -> _native_index = "call_index_create"
external _find_nearest_neighbor : _native_index -> float array -> int = "call_find_nearest_neighbor"
external _find_k_nearest_neighbors : _native_index -> int -> float array -> int array = "call_find_k_nearest_neighbors"
external get_dimension : _native_index -> int = "call_dimension"

let create 
  ?(dfunc=`Negative_inner_product)
  ?(is_dense=true)
  ?(l=20)
  ?(num_probes=0) (* zero indicates that parameter search should be run to find the optimal value *)
  dataset = 
  if num_probes != 0 && num_probes < l then
    raise (Invalid_argument (Printf.sprintf "num_probes must be >= l: %d for %d" num_probes l));
  let idx = _create dfunc is_dense l num_probes dataset in 
  (idx, dataset)

let check_dim idx q = 
  let idim = get_dimension idx in 
  let qdim = Array.length q in 
  if idim != qdim then
    raise (Invalid_argument (Printf.sprintf "input dimension does not match index dimension: %d for %d" qdim idim))

let find_nearest_neighbor idx q =
  let n_idx, _dat = idx in
  check_dim n_idx q;
  _find_nearest_neighbor n_idx q

let find_k_nearest_neighbors idx k q =
  let n_idx, _dat = idx in
  check_dim n_idx q;
  _find_k_nearest_neighbors n_idx k q

let input_of_array arr = 
  Array2.of_array float32 c_layout arr