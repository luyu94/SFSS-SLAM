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

namespace dyna
{

class OpticalFlow {
private:
    PyObject* pModule;
    PyObject* pFunc;
    PyObject* py_optical;
    
public:
    OpticalFlow();
    ~OpticalFlow();
    
    static void SaveResults(const cv::Mat &in_image1, const cv::Mat &imD1, const cv::Mat &imDepth1, 
               const cv::Mat &optical, const cv::Mat &binary_Mask, const cv::Mat &error_img,  
               const cv::Mat &projection_error, const cv::Mat &add, const cv::Mat &bMask, std::string name);
    void GetOpticalFlow(const std::vector<cv::Mat>& in_image1, const std::vector<cv::Mat>& in_image2, std::vector<cv::Mat>& optical_img, std::vector<cv::Mat>& optical_uv);
};

}

#endif