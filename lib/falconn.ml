open Bigarray

type index
type distance_function = [`Negative_inner_product | `Euclidean_squared]
type input_array = (float, float32_elt, c_layout) Array2.t 

external create : int -> distance_function -> bool -> int -> input_array -> index = "call_index_create"
external find_nearest_neighbor : index -> float array -> int = "call_find_nearest_neighbor"
external find_k_nearest_neighbors : index -> float array -> int array = "call_find_k_nearest_neighbors"

