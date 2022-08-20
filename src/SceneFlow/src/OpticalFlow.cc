#include <chrono>
#include </usr/local/opencv345/include/opencv2/opencv.hpp>
#include </usr/local/opencv345/include/opencv2/imgcodecs.hpp>
#include </home/opencv345/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc.hpp>
#include </usr/local/opencv345/include/opencv2/core/core.hpp>
#include "OpticalFlow.h"
#include "Conversion.h"
#include <glog/logging.h>

using namespace std;
using namespace cv;
using namespace dyna;

OpticalFlow::OpticalFlow() {

    // ----初始化
    Py_Initialize();
    if (!Py_IsInitialized())
    {
        printf("初始化失败");
        PyErr_Print();
        std::exit(1);
    }
    PyRun_SimpleString("import sys");
    string dir = "sys.path.append('/mnt/SceneFlow/src/RAFT/core')";
    PyRun_SimpleString(dir.c_str());
    PyRun_SimpleString("import cv2");
	PyRun_SimpleString("import numpy as np");

    // -----导入模型
    LOG(INFO) << "------pModule";
    this->pModule = PyImport_ImportModule("OpticalFlow");
    if (this->pModule == nullptr)
    {                                                                                
        PyErr_Print();
        std::exit(1);
    }
    assert(pModule != NULL);

    // -----导入函数
    LOG(INFO) << "------pFunc";
    this->pFunc = PyObject_GetAttrString(this->pModule, "optical");
    if (this->pFunc == nullptr)
    {
        PyErr_Print();
        std::exit(1);
    }
    assert(pFunc != NULL);

}

OpticalFlow::~OpticalFlow() {

    // delete this->pModule;
    // delete this->pFunc;
    // delete this->py_optical;
    // Clean up
    // Py_XDECREF是很有必要的，为了避免内存泄漏
    Py_DECREF(pModule);
    Py_DECREF(pFunc);

    Py_Finalize();
}

void OpticalFlow::GetOpticalFlow(const cv::Mat& in_image1, const cv::Mat& in_image2, cv::Mat& optical_img, cv::Mat& optical_uv) {
    // -----循环执行任务
    LOG(INFO) << "start compute RAFT optical flow";
    // auto sz = in_image1.size();	// 获取图像的尺寸
    int x = 640;
    int y = 480;
    int z = in_image1.channels();

    if(in_image1.empty() || in_image2.empty())
    {
        cerr << endl << "Failed to load image" << endl;
    }
    // LOG(INFO) << "------CArrays";
    //------- CV::Mat 转 python numpy-------
    uchar *CArrays1 = new uchar[x*y*z];   //这一行申请的内存需要释放指针，否则存在内存泄漏的问题
    uchar *CArrays2 = new uchar[x*y*z];

    // LOG(INFO) << "------Mat2Py_Array";
    NumpyAPI::Mat2Py_Array(in_image1, CArrays1) ;
    NumpyAPI::Mat2Py_Array(in_image2, CArrays2) ;
    
    npy_intp Dims[3] = { y, x, z }; //注意这个维度数据顺序！
    import_array1();
    PyObject *pArray1 = PyArray_SimpleNewFromData(3, Dims, NPY_UINT8, CArrays1);
    PyObject *pArray2 = PyArray_SimpleNewFromData(3, Dims, NPY_UINT8, CArrays2);
    //------- 得到函数参数
    // LOG(INFO) << "------pArg";
    PyObject *pArg = PyTuple_New(2);
    PyTuple_SetItem(pArg, 0, pArray1);
    PyTuple_SetItem(pArg, 1, pArray2);
    if (pArg == nullptr)
    {
        PyErr_Print();
        std::exit(1);
    } 
    LOG(INFO) << "------py_optical";
    py_optical = PyObject_CallObject(this->pFunc, pArg);    //返回元组
    if (py_optical == nullptr)
    {
        PyErr_Print();
        std::exit(1);
    }
    assert(py_optical != nullptr);

    //转换从Python读取的数据
    PyObject *py_optical_img = PyTuple_GetItem(py_optical, 0);
    PyObject *py_optical_uv = PyTuple_GetItem(py_optical, 1);
    NumpyAPI::Py_Array2Mat(py_optical_img, optical_img);
    NumpyAPI::Py_Array2Mat_uv(py_optical_uv, optical_uv);
    // LOG(INFO) << "输出";
    // LOG(INFO) << "===========================================";
    // // cout << format(optical, Formatter::FMT_PYTHON) << endl << endl;
    // cout << optical_uv.at<Vec2f>(0,0) << endl;
    // cout << optical_uv.at<Vec2f>(89,89) << endl;
    // cout << optical_uv.at<Vec2f>(380,570)<< endl;
    // cout << optical_uv.at<Vec2f>(6,450) << endl;
    // cout << optical_uv.at<Vec2f>(100,100) << endl;

    // Py_XDECREF(PyArray);
    /*这里Py_XDECREF(ArgList); 和 Py_XDECREF(PyArray);不能同时使用，否则会引起内存访问冲突
        * 我的理解是：PyTuple_SetItem并不复制数据，只是引用的复制。因此对这两个对象中的任意一个使用
        * Py_XDECREF都可以回收对象。使用两次的话反而会导致冲突。
        */
    Py_XDECREF(pArg);
    delete[] CArrays1;		// 释放数组内存，最好在PyArray被使用完以后释放
    delete[] CArrays2;
    CArrays1 = nullptr;
    CArrays2 = nullptr;
    Py_XDECREF(py_optical);
    py_optical = nullptr;

    LOG(INFO) << "finish RAFT optical flow";
}

void OpticalFlow::SaveResults(const cv::Mat &in_image1, const cv::Mat &imD1, const cv::Mat &imDepth1, 
     const cv::Mat &optical, const cv::Mat &binary_Mask, const cv::Mat &error_img, const cv::Mat &projection_error, std::string name) {
    
    cv::imwrite("/mnt/SceneFlow/output/rgb/" + name, in_image1);
    cv::imwrite("/mnt/SceneFlow/output/raw_depth/" + name, imD1);
    cv::imwrite("/mnt/SceneFlow/output/depth/" + name, imDepth1);

    cv::imwrite("/mnt/SceneFlow/output/optical/" + name, optical);

    cv::imwrite("/mnt/SceneFlow/output/binary_Mask/" + name, binary_Mask);
    cv::imwrite("/mnt/SceneFlow/output/error_Mask/" + name, error_img);
    cv::imwrite("/mnt/SceneFlow/output/projection_error/" + name, projection_error);
}