#ifndef _SCENEFLOW_H
#define _SCENEFLOW_H

#include <python3.6/Python.h>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include </usr/local/opencv345/include/opencv2/opencv.hpp>
#include </usr/local/opencv345/include/opencv2/imgcodecs.hpp>
#include </usr/local/opencv345/include/opencv2/core/core.hpp>

class OpticalFlow {
private:
    PyObject* pModule;
    PyObject* pFunc;
    PyObject* py_optical;
    
public:
    OpticalFlow();
    ~OpticalFlow();
    
    void SaveResults(const cv::Mat &optical, const cv::Mat &mask, std::string name);
    void GetOpticalFlow(const cv::Mat& in_image1, const cv::Mat& in_image2, cv::Mat& optical);
};

#endif