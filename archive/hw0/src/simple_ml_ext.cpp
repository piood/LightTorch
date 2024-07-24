#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <vector>
#include <omp.h>

namespace py = pybind11;


void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_examples = m;
    int num_classes = k;
    int input_dimension = n;
    omp_set_num_threads(32);

    for(int start = 0; start < num_examples; start += (int)batch){
        int end = std::min(start + (int)batch, num_examples);
        int batch_size = end - start;

        // Compute the logits
        std::vector<float> logits(batch_size * num_classes);
        #pragma omp parallel for
        for(int i = 0; i < batch_size; i++){
            for(int j = 0; j < num_classes; j++){
                logits[i * num_classes + j] = 0;
                for(int l = 0; l < input_dimension; l++){
                    logits[i * num_classes + j] += X[(start + i) * input_dimension + l] * theta[l * num_classes + j];
                }
            }
        }

        // Compute the softmax
        std::vector<float> softmax(batch_size * num_classes);

        for(int i = 0; i < batch_size; i++){
            float max_logit = logits[i * num_classes];
            for(int j = 1; j < num_classes; j++){
                max_logit = std::max(max_logit, logits[i * num_classes + j]);
            }

            float sum = 0;
            for(int j = 0; j < num_classes; j++){
                softmax[i * num_classes + j] = std::exp(logits[i * num_classes + j] - max_logit);
                sum += softmax[i * num_classes + j];
            }

            for(int j = 0; j < num_classes; j++){
                softmax[i * num_classes + j] /= sum;
            }
        }

        // Compute the gradients
        std::vector<float> gradients(input_dimension * num_classes);

        for(int i = 0; i < batch_size; i++){
            for(int j = 0; j < num_classes; j++){
                #pragma omp parallel for
                for(int l = 0; l < input_dimension; l++){
                    gradients[l * num_classes + j] += X[(start + i) * input_dimension + l] * (softmax[i * num_classes + j] - (j == y[start + i]));
                }
            }
        }

        // Update the parameters
        #pragma omp parallel for
        for(int l = 0; l < input_dimension; l++){
            for(int j = 0; j < num_classes; j++){
                theta[l * num_classes + j] -= lr * gradients[l * num_classes + j] / batch_size;
            }
        }
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
