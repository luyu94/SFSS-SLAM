#ifndef _SCENEFLOW_H
#define _SCENEFLOW_H

#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <opencv2/opencv.hpp>
#include <python3.6/Python.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/core.hpp>


void LoadDataset(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenameRGB, 
                 std::vector<std::string> &vstrImageFilenameD);

void SaveResults(const cv::Mat &optical, const cv::Mat &mask, std::string name);

cv::Mat GetOpticalFlow(const cv::Mat& in_image1, const cv::Mat& in_image2);
void GetSceneFlowObj();


#endif