
type index 
type distance_function 
type input_array 

val create : ?dfunc:distance_function -> ?is_dense:bool -> ?l:int -> input_array -> index 
val find_nearest_neighbor : index -> float array -> int 
val find_k_nearest_neighbors : index -> float array -> int array
val input_of_array : float array array -> input_array

