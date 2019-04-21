open Bigarray

type index
type distance_function = [`Negative_inner_product | `Euclidean_squared]
type input_array = (float, float32_elt, c_layout) Array2.t 

external _create : distance_function -> bool -> int -> input_array -> index = "call_index_create"
external find_nearest_neighbor : index -> float array -> int = "call_find_nearest_neighbor"
external find_k_nearest_neighbors : index -> float array -> int array = "call_find_k_nearest_neighbors"

let create 
  ?(dfunc=`Negative_inner_product)
  ?(is_dense=true)
  ?(l=20)
  dataset = 
  _create dfunc is_dense l dataset

let input_of_array arr = 
  Array2.of_array float32 c_layout arr