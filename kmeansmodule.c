# define PY_SSIZE_T_CLEAN
# include <Python.h>
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

/* Conversion functions Python <-> C */
static double *python_matrix_to_flat_array(PyObject *matrix_obj, int *rows_out, int *cols_out);
static PyObject *flat_array_to_python_matrix(double *arr, int rows, int cols);

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
    int c_rows, c_cols;
    PyObject *data_matrix_obj;
    double *data_points;
    int rows, cols;

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

```

