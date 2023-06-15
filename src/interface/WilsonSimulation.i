%module(directors="1") WilsonSimulation

%{
//#define SWIG_FILE_WITH_INIT
#include "WilsonSimulation/API.hpp"
%}

%include "stdint.i"
%include "std_string.i"
%include "std_shared_ptr.i"
%include "std_vector.i"
%include "exception.i"

%exception
{
    try {
        $action
    }
    catch (const std::runtime_error& e) {
        SWIG_exception(SWIG_RuntimeError, e.what());
    }
    catch (const std::invalid_argument& e) {
        SWIG_exception(SWIG_ValueError, e.what());
    }
    catch (const std::out_of_range& e) {
        SWIG_exception(SWIG_IndexError, e.what());
    }
    catch (...) {
        SWIG_exception(SWIG_RuntimeError, "unknown exception");
    }
}

//%include "numpy.i"
//%init %{
//  import_array();
//%}

//%define %np_vector_typemaps(DTYPE, NPY_DPTYE)

//namespace std {
//  // hmm...apparently telling SWIG to try to optimize this breaks it
//  // %typemap(out, fragment="NumPy_Fragments", optimal="1") vector<DTYPE> {
//  %typemap(out, fragment="NumPy_Fragments") vector<DTYPE> {
//    // create python array of appropriate shape
//    npy_intp sz = static_cast<npy_intp>($1.size());
//    npy_intp dims[] = {sz};
//    PyObject* out_array = PyArray_SimpleNew(1, dims, NPY_DPTYE);
//
//    if (! out_array) {
//      PyErr_SetString(PyExc_ValueError,
//                      "vector wrap: unable to create the output array.");
//      return NULL;
//    }
//
//    // copy data from vect into numpy array
//    DTYPE* out_data = (DTYPE*) array_data(out_array);
//    for (size_t i = 0; i < sz; i++) {
//      out_data[i] = static_cast<DTYPE>($1[i]);
//    }
//
//    $result = out_array;
//  }
//}
//
//%enddef

//%np_vector_typemaps(int, NPY_INT)
//%np_vector_typemaps(long, NPY_LONG)
//%np_vector_typemaps(float, NPY_FLOAT)
//%np_vector_typemaps(double, NPY_DOUBLE)

%template(VectorInt) std::vector<int32_t>;
%template(VectorVectorInt) std::vector<std::vector<int32_t>>;
%template(VectorDouble) std::vector<double>;
%template(VectorVectorDouble) std::vector<std::vector<double>>;
%template(VectorVectorVectorDouble) std::vector<std::vector<std::vector<double>>>;

//%apply (double* IN_ARRAY1, int DIM1) {(double* seq, int n)};
//%apply (double* IN_ARRAY2, int DIM1, int DIM2) {(double* seq, int n, int m)};
//%apply (int* IN_ARRAY1, int DIM1) {(int* seq, int n)};
//%apply (int* IN_ARRAY2, int DIM1, int DIM2) {(int* seq, int n, int m)};

//%apply (double* INPLACE_ARRAY1, int DIM1) {(double* inVec, int len)};

//%apply (double* IN_ARRAY1, int DIM1) {(double* v, int len)};
//%apply (double* IN_ARRAY1, int DIM1) {(double* ar, int len)};
//%apply (const double* IN_ARRAY1, int DIM1) {(const double* ar, int len)};
//%apply (double* IN_ARRAY1, int DIM1) {(double* v, int inLen)};
//%apply (double* IN_ARRAY1, int DIM1) {(double* v1, int len1)};
//%apply (double* IN_ARRAY1, int DIM1) {(double* v2, int len2)};
//
//%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* outVec, int len)};
//%apply (int* ARGOUT_ARRAY1, int DIM1) {(int* outVec, int outLen)};
//%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* outVec, int len)};
//%apply (double* ARGOUT_ARRAY1, int DIM1) {(double* outVec, int outLen)};

//%apply (double* IN_ARRAY1, size_t DIM1) {(double* vec_, size_t len_)}
//%apply (size_t DIM1, int* IN_ARRAY1) {(size_t len_, int* vec_)}

//%rename (to_vector_double1) my_to_vector_double1;
//%rename (to_vector_float1) my_to_vector_float1;
//%inline %{
//  std::vector<double> my_to_vector_double1(size_t len_, double* vec_) {
//      std::vector<double> v;
//      v.insert(v.end(), vec_, vec_ + len_);
//      return v;
//  }
//
//  std::vector<float> my_to_vector_float1(size_t len_, float* vec_) {
//    std::vector<float> v;
//    v.insert(v.end(), vec_, vec_ + len_);
//    return v;
//  }
//
//  std::vector<int> my_to_vector_int1(size_t len_, int* vec_) {
//    std::vector<int> v;
//    v.insert(v.end(), vec_, vec_ + len_);
//    return v;
//  }
//%}
//
//%apply (size_t DIM1, size_t DIM2, double* IN_ARRAY2) {(size_t len1_, size_t len2_, double* vec_)}
//%apply (size_t DIM1, size_t DIM2, int* IN_ARRAY2) {(size_t len1_, size_t len2_, int* vec_)}
//
////%rename (bar) my_bar;
//%inline %{
//  std::vector< std::vector<double> > to_vector_vector_double1(size_t len1_, size_t len2_, double* vec_) {
//      std::vector< std::vector<double> > v (len1_);
//      for (size_t i = 0; i < len1_; ++i) {
//          v[i].insert(v[i].end(), vec_ + i*len2_, vec_ + (i+1)*len2_);
//      }
//      return v;
//  }
//
//  std::vector< std::vector<int> > to_vector_vector_int1(size_t len1_, size_t len2_, int* vec_) {
//    std::vector< std::vector<int> > v (len1_);
//    for (size_t i = 0; i < len1_; ++i) {
//      v[i].insert(v[i].end(), vec_ + i*len2_, vec_ + (i+1)*len2_);
//    }
//    return v;
//  }
//%}

%feature("director") INotifications;

%include <WilsonSimulation/API.hpp>