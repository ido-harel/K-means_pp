#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdlib.h>
#include <math.h>

typedef struct DataSet {
    double *points;
    int rows; /* dimension */
    int cols; /* N */
} DataSet;

/* Function prototypes */
/* K-Means algorithms from HW1 */
double squared_distance(const double *point, const double *centroid, int cols);
int find_closest_centroid(const double *point, const double *centroids, int k, int cols);
void reset_clusters(double *sums, int *counts, int k, int cols);
void add_point_to_cluster(const double *point, int cluster_index, double *sums, int *counts, int cols);
int update_centroids(double *centroids, const double *sums, const int *counts, int k, int cols, double epsilon);
int run_kmeans(const DataSet *data, double *centroids, int k, int iter, double epsilon);

/* Convert a Python list-of-lists of numbers into a flat row-major C array.
   Sets *rows_out and *cols_out. Returns NULL (with a Python error set) on failure. */
static double *python_matrix_to_flat_array(PyObject *matrix_obj, int *rows_out, int *cols_out)
{
    Py_ssize_t rows, cols;
    Py_ssize_t i, j;
    double *arr;
    PyObject *row_obj, *item_obj;

    if(!PySequence_Check(matrix_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of sequences");
        return NULL;
    }
    rows = PySequence_Size(matrix_obj);
    if(rows <= 0) {
        PyErr_SetString(PyExc_ValueError, "Empty matrix");
        return NULL;
    }

    row_obj = PySequence_GetItem(matrix_obj, 0);
    if(row_obj == NULL) {
        return NULL;
    }
    if(!PySequence_Check(row_obj)) {
        Py_DECREF(row_obj);
        PyErr_SetString(PyExc_TypeError, "Expected a sequence of sequences");
        return NULL;
    }
    cols = PySequence_Size(row_obj);
    Py_DECREF(row_obj);

    arr = (double *) malloc((size_t)(rows * cols) * sizeof(double));
    if(arr == NULL) {
        PyErr_NoMemory();
        return NULL;
    }
    for(i=0; i<rows; i++){
        row_obj = PySequence_GetItem(matrix_obj, i);
        if(row_obj == NULL) {
            free(arr);
            return NULL;
        }
        if(!PySequence_Check(row_obj) || PySequence_Size(row_obj) != cols) {
            Py_DECREF(row_obj);
            free(arr);
            PyErr_SetString(PyExc_TypeError, "Ragged matrix");
            return NULL;
        }
        for(j=0; j<cols; j++){
            item_obj = PySequence_GetItem(row_obj, j);
            if(item_obj == NULL) {
                Py_DECREF(row_obj);
                free(arr);
                return NULL;
            }
            arr[i * cols + j] = PyFloat_AsDouble(item_obj);
            Py_DECREF(item_obj);
            if(PyErr_Occurred()) {
                Py_DECREF(row_obj);
                free(arr);
                return NULL;
            }
        }
        Py_DECREF(row_obj);
    }
    *rows_out = (int)rows;
    *cols_out = (int)cols;
    return arr;  
}

/* Convert a flat row-major C array into a Python list-of-lists of floats. */
static PyObject *flat_array_to_python_matrix(double *arr, int rows, int cols)
{
    PyObject *matrix;
    PyObject *row;
    PyObject *val;
    int i, j;

    matrix = PyList_New(rows);
    if(matrix == NULL) {
        return NULL;
    }

    for(i=0; i<rows; i++){
        row = PyList_New(cols);
        if(row == NULL) {
            Py_DECREF(matrix);
            return NULL;
        }
        for(j=0; j<cols; j++){
            val = PyFloat_FromDouble(arr[i * cols + j]);
            if(val == NULL) {
                Py_DECREF(row);
                Py_DECREF(matrix);
                return NULL;
            }
            PyList_SET_ITEM(row, j, val); /* steals reference to val */
        }
        PyList_SET_ITEM(matrix, i, row); /* steals reference to row */
    }
    return matrix;
}

/* Wrapper function for Python */
static PyObject *py_run_kmeans(PyObject *self, PyObject *args);

/* Function implementations */
double squared_distance(const double *point, const double *centroid, int cols){
    double sum = 0.0;
    double diff;
    int i;
    for(i=0; i< cols; i++){
        diff = point[i] - centroid[i];
        sum += diff * diff;
    }
    return sum;
}

int find_closest_centroid(const double *point, const double *centroids, int k, int cols){
    int i;
    double dst;
    double min_dst = squared_distance(point, centroids, cols);
    int closest = 0;
    for(i=1; i< k; i++){
        dst = squared_distance(point, centroids + i * cols, cols);
        if(dst < min_dst){
            min_dst = dst;
            closest = i;
        }
    }
    return closest;
}

void reset_clusters(double *sums, int *counts, int k, int cols)
{
    int i;
    int j;
    for (i = 0; i < k; i++) {
        counts[i] = 0;
        for (j = 0; j < cols; j++) {
            sums[i * cols + j] = 0.0;
        }
    }
}

void add_point_to_cluster(const double *point, int cluster_index, double *sums, int *counts, int cols)
{
    int j;
    for (j = 0; j < cols; j++) {
        sums[cluster_index * cols + j] += point[j];
    }
    counts[cluster_index]++;
}

int update_centroids(double *centroids, const double *sums, const int *counts, int k, int cols, double epsilon){
    int i;
    int j;
    double old_value;
    double new_value;
    double diff;
    double delta_squared;
    double epsilon_squared;
    int converged;

    converged = 1; 
    epsilon_squared = epsilon * epsilon;
    for (i = 0; i < k; i++) {
        delta_squared = 0.0;
        if (counts[i] == 0) continue;
        for (j = 0; j < cols; j++) {
            old_value = centroids[i * cols + j];
            new_value = sums[i * cols + j] / counts[i];
            diff = new_value - old_value;
            delta_squared += diff * diff;
            centroids[i * cols + j] = new_value;
        }
        if (delta_squared >= epsilon_squared) {
            converged = 0;
        }
    }
    return converged;
}

int run_kmeans(const DataSet *data, double *centroids, int k, int iter, double epsilon)
{
    double *sums;
    int *counts;
    const double *point;
    int iteration;
    int i;
    int closest;
    int converged;

    sums = (double *) calloc(k * data->cols, sizeof(double));
    counts = (int *) calloc(k, sizeof(int));

    if (sums == NULL || counts == NULL) {
        free(sums);
        free(counts);
        return 0;
    }
    for (iteration = 0; iteration < iter; iteration++) {
        reset_clusters(sums, counts, k, data->cols);
        for (i = 0; i < data->rows; i++) {
            point = data->points + i * data->cols;
            closest = find_closest_centroid(point, centroids, k, data->cols);
            add_point_to_cluster(point, closest, sums, counts, data->cols);
        }
        converged = update_centroids(centroids, sums, counts, k, data->cols, epsilon);
        if (converged) break;
    }
    free(sums);
    free(counts);
    return 1;
}

static PyObject *py_run_kmeans(PyObject *self, PyObject *args)
{
    int k, iter;
    double epsilon;
    PyObject *centroids_matrix_obj;
    double *centroids;
    int centroids_rows, centroids_cols;
    PyObject *data_matrix_obj;
    double *data_points;
    int rows, cols;
    DataSet data;

    PyObject *result;
    centroids = NULL;
    data_points = NULL;
    result = NULL;

    if(!PyArg_ParseTuple(args, "OOiid", &data_matrix_obj, &centroids_matrix_obj, &k, &iter, &epsilon)) {
        return NULL; 
    }
    centroids = python_matrix_to_flat_array(centroids_matrix_obj, &centroids_rows, &centroids_cols);

    if (centroids == NULL) 
        return NULL;

    data_points = python_matrix_to_flat_array(data_matrix_obj, &rows, &cols);

    if (data_points == NULL) {
        free(centroids);
        return NULL;
    }

    if (centroids_rows != k || centroids_cols != cols) {
        free(data_points);
        free(centroids);
        PyErr_SetString(PyExc_ValueError, "Invalid centroids shape");
        return NULL;
    }

    data.points = data_points;
    data.rows = rows;
    data.cols = cols;

    if (!run_kmeans(&data, centroids, k, iter, epsilon)) {
        free(data_points);
        free(centroids);
        PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
        return NULL;
    }

    result = flat_array_to_python_matrix(centroids, k, cols);
    free(data_points);
    free(centroids);

    return result;
}

static PyMethodDef kmeansMethods[] = {
    {"fit",
     (PyCFunction) py_run_kmeans,
     METH_VARARGS,
     PyDoc_STR(
         "fit(data_points, initial_centroids, k, iter, eps) -> list of centroids\n\n"
         "Run K-means on data_points starting from initial_centroids (the\n"
         "K-means++ result). k is the number of clusters, iter the maximum\n"
         "number of iterations, eps the convergence threshold. Returns the k\n"
         "final centroids as a list of lists of floats.")},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",
    "K-means clustering C extension for HW2 (K-means++).",
    -1,
    kmeansMethods,
    NULL,  /* m_slots */
    NULL,  /* m_traverse */
    NULL,  /* m_clear */
    NULL   /* m_free */
};

PyMODINIT_FUNC PyInit_mykmeanssp(void)
{
    return PyModule_Create(&kmeansmodule);
}
