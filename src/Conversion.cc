#include <iostream>
#include </usr/lib/python3/dist-packages/numpy/core/include/numpy/ndarraytypes.h>
#include </usr/local/opencv345/include/opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp> 
#include <Eigen/Core> 
#include <Eigen/Dense> 


#include <typeinfo>

#include "Common.h"
#include "Conversion.h"
#include <glog/logging.h>

using namespace cv;

namespace NumpyAPI
{

void Mat2Py_Array(const cv::Mat &img, uchar* &CArray) {
    int iChannels = img.channels();
    int iRows = img.rows;
    int iCols = img.cols * iChannels;
    if (img.isContinuous())
    {
        iCols *= iRows;
        iRows = 1;
    }
    const uchar* p;
    int id = -1;
    for (int i = 0; i < iRows; i++)
    {
        // get the pointer to the ith row
        p = img.ptr<uchar>(i);
        // operates on each pixel
        for (int j = 0; j < iCols; j++)
        {
            CArray[++id] = p[j];//连续空间
        }
    }
}

void Py_Array2Mat_uv(PyObject* &py_image, cv::Mat &out) {
    //获取矩阵维度
    // LOG(INFO) << "===Py_Array2Mat_uv====";
    npy_intp *Py_array_shape = PyArray_DIMS(py_image);
    int arrayrow = (int)Py_array_shape[0];
    int arraycol = (int)Py_array_shape[1];
    int channal = (int)Py_array_shape[2];
    // LOG(INFO) << channal;

    PyArrayObject *np = reinterpret_cast<PyArrayObject*>(py_image);
    float* ptr = reinterpret_cast<float*>(PyArray_DATA(np));
    // LOG(INFO) << *ptr;


    for (int k = 0; k < channal; k++) {
        for (int i = 0; i < arrayrow; i++) {
            for (int j = 0; j < arraycol; j++) {
                out.at<Vec2f>(i, j)[k] = *ptr;
                ptr++;
                
            }
        }
    }

}

void Py_Array2Mat(PyObject* &py_image, cv::Mat &out) {
    //获取矩阵维度
    // LOG(INFO) << "===Py_Array2Mat====";
    npy_intp *Py_array_shape = PyArray_DIMS(py_image);
    int arrayrow = (int)Py_array_shape[0];
    int arraycol = (int)Py_array_shape[1];
    int channal = (int)Py_array_shape[2];
    // LOG(INFO) << "channal: " << channal;

    PyArrayObject *np = reinterpret_cast<PyArrayObject*>(py_image);
    uchar* ptr = reinterpret_cast<uchar*>(PyArray_DATA(np));
    
    for (int i = 0; i < arrayrow; i++) {
        for (int j = 0; j < arraycol; j++) {
            for (int k = 0; k < channal; k++) {
                out.at<Vec3b>(i, j)[k] = *ptr;
                ptr++;
            }
        }
    }
}

void Py_Array2Mat_img(PyObject* &py_image, cv::Mat &out) {
    //获取矩阵维度
    // LOG(INFO) << "===Py_Array2Mat_img====";
    npy_intp *Py_array_shape = PyArray_DIMS(py_image);
    npy_intp arrayrow = Py_array_shape[0];
    npy_intp arraycol = Py_array_shape[1];

    cv::Mat img(arrayrow, arraycol, CV_8UC3, PyArray_DATA(py_image));
    // cv::Mat img(arrayrow, arraycol, CV_32FC2, PyArray_DATA(py_image));
    out = img;
}

}  // end namespace NumpyAPI