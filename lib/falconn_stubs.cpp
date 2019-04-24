
#include <falconn/lsh_nn_table.h>

#include <stdio.h>

#include <vector>

using std::vector;
using std::pair;
using std::make_pair;
using std::unique_ptr;

using falconn::construct_table;
using falconn::compute_number_of_hash_functions;
using falconn::DenseVector;
using falconn::DistanceFunction;
using falconn::LSHConstructionParameters;
using falconn::LSHFamily;
using falconn::LSHNearestNeighborTable;
using falconn::LSHNearestNeighborQueryPool;
using falconn::LSHNearestNeighborQuery;
using falconn::QueryStatistics;
using falconn::StorageHashTable;
using falconn::get_default_parameters;
using falconn::PlainArrayPointSet;

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/bigarray.h>
#include <caml/threads.h>

typedef int32_t Key;
typedef float Value;

typedef DenseVector<Value> Point;
typedef LSHNearestNeighborTable<Point> Table;
typedef LSHNearestNeighborQuery<Point> Cursor;
typedef PlainArrayPointSet<Value> InnerPlainArrayPointSet;

#define BEGIN_CXX_EX try {
#define END_CXX_EX } catch(std::exception &e) { caml_failwith(e.what()); } catch(...) { caml_failwith("unknown c++ exception");}

#define GIL_RELEASE_BEGIN caml_release_runtime_system(); try {
#define GIL_RELEASE_END } catch(...) { caml_acquire_runtime_system(); throw; } caml_acquire_runtime_system();

#define Index_val(v) (*((Index *) Data_custom_val(v)))

#define MIN(a, b) ((a > b) ? b : a)

#define xstr(a) str(a)
#define str(a) #a
#define camlassert(v) if (!(v)) caml_failwith("assert failed: " xstr(v))

#define NUM_PROBES_SAMPLE_SIZE 4096
void get_sample_indexes(size_t *res, size_t sample_size, size_t range) {
    size_t *buffer = (size_t *) malloc(sizeof(size_t) * range);
    for (size_t i=0; i < range; i++) {
        buffer[i] = i;
    }

    for (size_t i=range-1; i > 0; i++) {
        int j = rand() % i;
        size_t v = buffer[i];
        buffer[i] = buffer[j];
        buffer[j] = v;
    }

    for (size_t i=0; i < sample_size; i++) {
        res[i] = buffer[i];
    }

    free(buffer);
}

/*
* Generates answers for the queries using the (optimized) linear scan.
*/
void gen_answers(const vector<Point> &dataset, const vector<Point> &queries,
                    vector<int> *answers)
{
    answers->resize(queries.size());
    int outer_counter = 0;
    for (const auto &query : queries)
    {
        float best = -10.0;
        int inner_counter = 0;
        for (const auto &datapoint : dataset)
        {
            float score = query.dot(datapoint);
            if (score > best)
            {
                (*answers)[outer_counter] = inner_counter;
                best = score;
            }
            ++inner_counter;
        }
        ++outer_counter;
    }
}
/*
 * Computes the probability of success using a given number of probes.
 */
double evaluate_num_probes(LSHNearestNeighborTable<Point> *table,
                           const vector<Point> &queries,
                           const vector<int> &answers, int num_probes) {
  unique_ptr<LSHNearestNeighborQuery<Point>> query_object =
      table->construct_query_object(num_probes);
  int outer_counter = 0;
  int num_matches = 0;
  vector<Key> candidates;
  for (const auto &query : queries) {
    query_object->get_candidates_with_duplicates(query, &candidates);
    for (auto x : candidates) {
      if (x == answers[outer_counter]) {
        ++num_matches;
        break;
      }
    }
    ++outer_counter;
  }
  return (num_matches + 0.0) / (queries.size() + 0.0);
}

/*
* Finds the smallest number of probes that gives the probability of success
* at least 0.9 using binary search.
*/
int _find_num_probes(Table *table,
                        const vector<Point> &queries, const vector<int> &answers,
                        int start_num_probes)
{
    int num_probes = start_num_probes;
    while(num_probes < 100)
    {
        double precision = evaluate_num_probes(table, queries, answers, num_probes);
        if (precision >= 0.9)
        {
            break;
        }
        num_probes *= 2;
    }

    int r = num_probes;
    int l = r / 2;

    while (r - l > 1)
    {
        int num_probes = (l + r) / 2;
        double precision = evaluate_num_probes(table, queries, answers, num_probes);
        if (precision >= 0.9)
        {
            r = num_probes;
        }
        else
        {
            l = num_probes;
        }
    }

    return num_probes;
}

void random_sample(value inp, Point *res, size_t sample_size)
{
    int dim_y = Caml_ba_array_val(inp)->dim[0];
    int dim_x = Caml_ba_array_val(inp)->dim[1];
    float *inp_data = (float *) Caml_ba_data_val(inp);
    size_t *sample_idxes = (size_t *) malloc(sizeof(size_t) * sample_size);
    get_sample_indexes(sample_idxes, sample_size, dim_y);
    for (size_t i = 0; i < sample_size; i++)
    {
        Point p;
        p.resize(dim_x);
        size_t sample_off = i * dim_x;
        for (size_t j=0; j < dim_x; j++) {
            p[j] = inp_data[sample_off + j];
        }
        res[i] = p;
    }
    free(sample_idxes);
}

int find_num_probes(Table *table, value data, int start)
{
    size_t data_size = Caml_ba_array_val(data)->dim[0];
    vector<int> answers;
    vector<Point> sample_vec;
    if (data_size > NUM_PROBES_SAMPLE_SIZE) {
        Point *sample = (Point *) malloc(sizeof(Point) * NUM_PROBES_SAMPLE_SIZE);
        random_sample(data, sample, NUM_PROBES_SAMPLE_SIZE);
        sample_vec.assign(sample, sample + NUM_PROBES_SAMPLE_SIZE);
        free(sample);
    } else {
        size_t width = Caml_ba_array_val(data)->dim[1];
        float *inp_data = (float *) Caml_ba_data_val(data);
        sample_vec.resize(data_size);
        size_t off = 0;
        for (size_t y=0; y < data_size; y++) {
            Point p;
            for (size_t x; x < width; x++) {
                p[x] = inp_data[off];
                off++;
            }
            sample_vec[y] = p;
        }
    }
    
    gen_answers(sample_vec, sample_vec, &answers);

    return _find_num_probes(table, sample_vec, answers, start);
}

PlainArrayPointSet<float> unpack_bigarray(value inp) {
    camlassert((Caml_ba_array_val(inp)->flags & BIGARRAY_KIND_MASK) == CAML_BA_FLOAT32);
    camlassert(Caml_ba_array_val(inp)->num_dims == 2);
    size_t dim_y = Caml_ba_array_val(inp)->dim[0];
    size_t dim_x = Caml_ba_array_val(inp)->dim[1];
    float *inp_data = (float *) Caml_ba_data_val(inp);
    PlainArrayPointSet<float> res;
    res.num_points = dim_y;
    res.dimension = dim_x;
    res.data = inp_data;
    return res;
}

Table *construct_table_dense_float(value points, const LSHConstructionParameters &params, float **res_ptr) {
    camlassert((Caml_ba_array_val(points)->flags & BIGARRAY_KIND_MASK) == CAML_BA_FLOAT32);
    camlassert(Caml_ba_array_val(points)->num_dims == 2);
    size_t dim_y = Caml_ba_array_val(points)->dim[0];
    size_t dim_x = Caml_ba_array_val(points)->dim[1];
    float *inp_data = (float *) Caml_ba_data_val(points);
    PlainArrayPointSet<Value> converted;

    // defensive copy
    size_t data_size = dim_x * dim_y * sizeof(float);
    float *copied_data = (float *) malloc(data_size);
    memcpy(copied_data, inp_data, data_size);

    *res_ptr = copied_data;

    converted.num_points = dim_y;
    converted.dimension = dim_x;
    converted.data = copied_data;

    Table *res;
    GIL_RELEASE_BEGIN
    res = construct_table<Point, Key, InnerPlainArrayPointSet>(converted, params).release();
    GIL_RELEASE_END
    return res;
}

Point unpack_point(value _q, size_t width) {
    camlassert(Tag_val(_q) == Double_array_tag);
    camlassert(Wosize_val(_q) == width);
    Point p;
    p.resize(width);
    for (size_t i=0; i < width; i++) {
        p[i] = Double_field(_q, i);
    }
    return p;
}


extern "C" {

    typedef struct Index {
        Table *index;
        Cursor *cursor;
        size_t dimension;
        int num_probes;
        float *data_ptr;
    } Index;

    void index_finalize(value v) {
        Index index = Index_val(v);
        delete index.cursor;
        delete index.index;
        free(index.data_ptr);
    }

    void array_dims(value inp, size_t *width, size_t *height) {
        camlassert(Caml_ba_array_val(inp)->num_dims == 2);
        *width = Caml_ba_array_val(inp)->dim[1];
        *height = Caml_ba_array_val(inp)->dim[0];
    }

    static struct custom_operations index_ops = {
        (char *) "cheap.hella.falconn",
        index_finalize,
        custom_compare_default,
        custom_hash_default,
        custom_serialize_default,
        custom_deserialize_default
    };

    value _wrap_index(Index v) {
        value res = alloc_custom(&index_ops, sizeof(Index), 0, 1);
        Index_val(res) = v;
        return res;
    }


    value call_index_create(
        value _distance_function, value _is_dense, value _l, value _num_probes, value _dataset)
    {
        CAMLparam5(_distance_function, _is_dense, _l, _num_probes, _dataset);
        BEGIN_CXX_EX

        bool is_dense = Val_bool(_is_dense);
        long l = Val_long(_l);

        DistanceFunction distance_function;
        if (_distance_function == hash_variant("Negative_inner_product"))
        {
            distance_function = DistanceFunction::NegativeInnerProduct;
        }
        else if (_distance_function == hash_variant("Euclidean_squared"))
        {
            distance_function = DistanceFunction::EuclideanSquared;
        }
        else
        {
            caml_failwith("invalid distance function flag");
        }
        
        size_t dataset_dimension;
        size_t dataset_size;
        array_dims(_dataset, &dataset_dimension, &dataset_size);

        LSHConstructionParameters params = get_default_parameters<Point>(
            dataset_size, dataset_dimension, distance_function, is_dense);
        params.l = l;

        if (dataset_size < 1000) {
            params.num_setup_threads = 1;
        }

        float *data_ptr;
        Table *table = construct_table_dense_float(_dataset, params, &data_ptr);

        int inp_num_probes = Val_long(_num_probes);
        int num_probes;
        if (inp_num_probes == 0) {
            num_probes = find_num_probes(table, _dataset, l);
        } else {
            num_probes = inp_num_probes;
        }

        Index res;
        res.index = table;
        res.dimension = dataset_dimension;
        res.num_probes = num_probes;
        res.data_ptr = data_ptr;
        res.cursor = (table->construct_query_object(num_probes)).release();

        CAMLreturn(_wrap_index(res));
        END_CXX_EX
    }

    value call_find_nearest_neighbor(value _index, value _q) {
        CAMLparam2(_index, _q);
        BEGIN_CXX_EX
        Index index = Index_val(_index);
        Point q = unpack_point(_q, index.dimension);
        camlassert(q.size() == index.dimension);
        camlassert(q.size() > 0);
        Key res;
        GIL_RELEASE_BEGIN
        res = (index.cursor)->find_nearest_neighbor(q);
        GIL_RELEASE_END
        CAMLreturn(Val_long(res));
        END_CXX_EX
    }

    value call_find_k_nearest_neighbors(value _index, value _k, value _q) {
        CAMLparam3(_index, _k, _q);
        Index index = Index_val(_index);
        int k = Int_val(_k);
        BEGIN_CXX_EX
        Point q = unpack_point(_q, index.dimension);
        camlassert(q.size() == index.dimension);
        vector<Key> res_vec;
        GIL_RELEASE_BEGIN
        (index.cursor)->find_k_nearest_neighbors(q, k, &res_vec);
        GIL_RELEASE_END

        size_t res_size = MIN(res_vec.size(), k);
        value res = caml_alloc(res_size, 0);
        for (size_t i=0; i < res_size; i++ ){
            Store_field(res, i, Val_int(res_vec[i]));
        }

        CAMLreturn(res);
        END_CXX_EX
    }

    value call_dimension(value _index) {
        CAMLparam1(_index);
        Index index = Index_val(_index);
        long res = index.dimension;
        CAMLreturn(Val_long(res));
    }


}