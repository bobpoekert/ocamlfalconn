open Bigarray

type index
type distance_function = [`Negative_inner_product | `Euclidean_squared]
type input_array = (float, float32_elt, c_layout) Array2.t 

external _create : distance_function -> bool -> int -> int -> input_array -> index = "call_index_create"
external _find_nearest_neighbor : index -> float array -> int = "call_find_nearest_neighbor"
external _find_k_nearest_neighbors : index -> int -> float array -> int array = "call_find_k_nearest_neighbors"
external get_dimension : index -> int = "call_dimension"

let create 
  ?(dfunc=`Negative_inner_product)
  ?(is_dense=true)
  ?(l=20)
  ?(num_probes=0) (* zero indicates that parameter search should be run to find the optimal value *)
  dataset = 
  if num_probes != 0 && num_probes < l then
    raise (Invalid_argument (Printf.sprintf "num_probes must be >= l: %d for %d" num_probes l));
  _create dfunc is_dense l num_probes dataset

let check_dim idx q = 
  let idim = get_dimension idx in 
  let qdim = Array.length q in 
  if idim != qdim then
    raise (Invalid_argument (Printf.sprintf "input dimension does not match index dimension: %d for %d" qdim idim))

let find_nearest_neighbor idx q =
  check_dim idx q;
  _find_nearest_neighbor idx q

let find_k_nearest_neighbors idx k q =
  check_dim idx q;
  _find_k_nearest_neighbors idx k q

let input_of_array arr = 
  Array2.of_array float32 c_layout arr

let rec _array1_to_array res inp len off =
  if off < len then
    (
      Array.set res off (Array1.get inp off);
      _array1_to_array res inp len (off + 1)
    )