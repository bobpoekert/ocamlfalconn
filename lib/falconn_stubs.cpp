
#include <falconn/lsh_nn_table.h>

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

#include <caml/mlvalues.h>
#include <caml/memory.h>
#include <caml/alloc.h>
#include <caml/custom.h>
#include <caml/fail.h>
#include <caml/bigarray.h>

typedef DenseVector<float> Point;
typedef LSHNearestNeighborTable<Point> Table;
typedef LSHNearestNeighborQueryPool<Point> Cursor;

// variant enums
const int NEGATIVE_INNER_PRODUCT = hash_variant("Negative_inner_product");
const int EUCLIDEAN_SQUARED = hash_variant("Euclidean_squared");

#define Index_val(v) (*((Index *) Data_custom_val(v)))

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
  vector<int32_t> candidates;
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
    for (;;)
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

void random_sample(vector<Point> inp, Point *res, size_t sample_size)
{
    size_t *sample_idxes = (size_t *) malloc(sizeof(size_t) * sample_size);
    size_t range = inp.size();
    get_sample_indexes(sample_idxes, sample_size, range);
    for (size_t i = 0; i < sample_size; i++)
    {
        res[i] = inp[sample_idxes[i]];
    }
    free(sample_idxes);
}

int find_num_probes(Table *table, vector<Point> data, int start)
{
    Point *sample = (Point *) malloc(sizeof(Point) * NUM_PROBES_SAMPLE_SIZE);
    random_sample(data, sample, NUM_PROBES_SAMPLE_SIZE);
    vector<Point> sample_vec;
    sample_vec.assign(sample, sample + NUM_PROBES_SAMPLE_SIZE);
    vector<int> answers;
    gen_answers(data, sample_vec, &answers);

    return _find_num_probes(table, sample_vec, answers, start);
}

Point unpack_point(value _q, size_t width) {
    Point p;
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
    } Index;

    void index_finalize(value v) {
        Index index = Index_val(v);
        delete index.cursor;
        delete index.index;
    }

    static struct custom_operations index_ops = {
        (char *) "fr.inria.caml.histograms",
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

    void load_dataset(value inp, vector<Point> *outp, int *n_dims)
    {
        camlassert((Caml_ba_array_val(inp)->flags & BIGARRAY_KIND_MASK) == CAML_BA_FLOAT32);
        int dim_x = Caml_ba_array_val(inp)->dim[0];
        int dim_y = Caml_ba_array_val(inp)->dim[1];

        *n_dims = dim_x;

        float *inp_data = (float *) Caml_ba_data_val(inp);

        outp->resize(dim_y);

        Point p;
        p.resize(dim_x);
        for (size_t y = 0; y < dim_y; y++)
        {
            size_t y_off = y * dim_y;
            for (size_t x = 0; x < dim_x; x++)
            {
                p[x] = inp_data[y_off + x];
            }
            p.normalize();
            outp->push_back(p);
        }
    }

    value call_index_create(
        value _dataset_size,
        value _distance_function, value _is_dense, value _l, value _dataset)
    {
        CAMLparam5(_dataset_size, _distance_function, _is_dense, _l, _dataset);

        long dataset_size = Val_long(_dataset_size);
        bool is_dense = Val_bool(_is_dense);
        long l = Val_long(_l);

        DistanceFunction distance_function;
        int distance_function_flag = Val_int(_distance_function);
        if (distance_function_flag == NEGATIVE_INNER_PRODUCT)
        {
            distance_function = DistanceFunction::NegativeInnerProduct;
        }
        else if (distance_function_flag == EUCLIDEAN_SQUARED)
        {
            distance_function = DistanceFunction::EuclideanSquared;
        }
        else
        {
            caml_failwith("invalid distance function flag");
        }
        
        vector<Point> dataset;
        int dataset_dimension;
        load_dataset(_dataset, &dataset, &dataset_dimension);

        LSHConstructionParameters params = get_default_parameters<Point>(
            dataset_size, dataset_dimension, distance_function, is_dense);
        params.l = l;

        Table *table = construct_table<Point>(dataset, params).release();

        int num_probes = find_num_probes(table, dataset, l);

        Index res;
        res.index = table;
        res.dimension = dataset_dimension;
        res.num_probes = num_probes;
        res.cursor = table->construct_query_pool(num_probes, -1, 1).release();

        CAMLreturn(_wrap_index(res));
        }

    value call_find_nearest_neighbor(value _index, value _q) {
        CAMLparam2(_index, _q);
        Index index = Index_val(_index);
        Point k = unpack_point(_q, index.dimension);
        int res = index.cursor->find_nearest_neighbor(k);
        CAMLreturn(Val_long(res));
    }

    value call_find_k_nearest_neighbors(value _index, value _q, value _k) {
        CAMLparam3(_index, _q, _k);
        Index index = Index_val(_index);
        int k = Val_int(_k);
        Point q = unpack_point(_q, index.dimension);

        vector<int> res_vec;
        index.cursor->find_k_nearest_neighbors(q, k, &res_vec);

        value res = caml_alloc(k, 0);
        for (size_t i=0; i < k; i++ ){
            Store_field(res, i, res_vec[i]);
        }

        CAMLreturn(res);
    }


}