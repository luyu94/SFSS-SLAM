#include <assert.h>
#include <fstream>
#include <iostream>
#include <string>
#include <stdlib.h>
#include <vector>
#include <chrono>
#include </usr/local/opencv345/include/opencv2/opencv.hpp>
#include </usr/local/opencv345/include/opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include </home/opencv345/opencv_contrib/modules/ximgproc/include/opencv2/ximgproc.hpp>
#include </usr/local/opencv345/include/opencv2/core/core.hpp>
#include <glog/logging.h>

#include "OpticalFlow.h"
#include "SceneFlow.h"

using namespace std;
using namespace cv;
using namespace dyna;


// typedef struct _imgPair{
//     Mat *src;
//     Mat *dst;
//     void* Param;
//     char* winName;
//     int* ksize;
//     int* num;
// }ImgPair;


// void on_medianSigmaBar(int ksize, void *userdata)
// {
//     ImgPair* pImgPair = (ImgPair*)userdata;
//     for(int i=0; i<*(pImgPair->num); i++) {
//         medianBlur(*(pImgPair->src), *(pImgPair->dst), ksize/2*2+1);
//     }
//     imshow(pImgPair->winName, *(pImgPair->dst));
// }

// void on_medianMutiBar(int num, void *userdata)
// {
//     ImgPair* pImgPair = (ImgPair*)userdata;
//     for(int i=0; i<num; i++) {
//         medianBlur(*(pImgPair->src), *(pImgPair->dst), (*(pImgPair->ksize))/2*2+1);
//     }
//     imshow(pImgPair->winName, *(pImgPair->dst));

// }

void LoadDataset(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenameRGB, 
                 std::vector<std::string> &vstrImageFilenameD);

int main () {
    const double _alpha = 50;  // get H threshold hyperparameter  0.45
    const double _beta = 1.73;
    string strAssociationFilename = "/mnt/SceneFlow/associations/fr3_walking_rpy.txt";
    string ROOT = "/mnt/rgbd_dataset_freiburg3_walking_rpy/";

    // Deptmap values factor
    // float mDepthMapFactor = 1.0f / 5000.0;

    LOG(INFO) << ROOT;
    LOG(INFO) << strAssociationFilename;
    vector<string> vstrImageFilenameRGB;
    vector<string> vstrImageFilenameD;
    
    LoadDataset(strAssociationFilename, vstrImageFilenameRGB, vstrImageFilenameD);

    int nImages = vstrImageFilenameRGB.size();
    cout << "nImages: " << nImages << endl;
    cv::Mat in_image1, in_image2, imD;
    int num = nImages-1;


    dyna::OpticalFlow *oflow;
    oflow = new dyna::OpticalFlow();

    dyna::SceneFlow *sflow;
    sflow = new dyna::SceneFlow(_alpha, _beta);

    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    for(int ni=422; ni<423; ni++) //422
    {
        // Read image and depthmap from file
        LOG(INFO) << "read";
        
        cv::Mat in_image1 = cv::imread(ROOT + vstrImageFilenameRGB[ni], CV_LOAD_IMAGE_COLOR); //16 CV_8UC3
        cv::Mat in_image2 = cv::imread(ROOT + vstrImageFilenameRGB[ni+1], CV_LOAD_IMAGE_COLOR);
        cv::Mat imD1 = cv::imread(ROOT + vstrImageFilenameD[ni], CV_LOAD_IMAGE_UNCHANGED);
        cv::Mat imD2 = cv::imread(ROOT + vstrImageFilenameD[ni+1], CV_LOAD_IMAGE_UNCHANGED); //2 CV_16UC1
        // cout << imD1.type() << endl;
        // cout << in_image1.type() << endl;
        // cout << imD1 << endl;

        cv::Mat VelocityXYZ = cv::Mat::zeros(480, 640, CV_32FC3);

        // cv::Mat Depth = imD.clone();

        /*-------get optical flow-----------*/
        cv::Mat optical_img = cv::Mat::zeros(480, 640, CV_8UC3);    //获取光流显示图
        cv::Mat optical_uv = cv::Mat::zeros(480, 640, CV_32FC2);
        
        vector<cv::Mat> optical = {optical_img, optical_uv};
        oflow->OpticalFlow::GetOpticalFlow(in_image1, in_image2, optical_img, optical_uv); //获取光流速度
        
        // cout << optical_uv << endl;
        // cout << optical.type() << endl;
        
        /*-------medianBlur-----------*/
        cv::Mat MedianBlurImg1 = imD1.clone();
        cv::Mat MedianBlurImg2 = imD2.clone();
        int kernelSize = 15;
        // int num = 1;
        // ImgPair  medianPair = { &Depth, &MedianBlurImg, nullptr, "MedianBlurImg", &kernelSize, &num};
        cv::medianBlur(imD1, MedianBlurImg1, kernelSize);
        cv::medianBlur(imD2, MedianBlurImg2, kernelSize);
        // imshow("MedianBlurImg", MedianBlurImg);
        // createTrackbar("kernelsize", "MedianBlurImg", &(kernelSize), 30, on_medianMutiBar, &medianPair);
        // createTrackbar("num", "MedianBlurImg", &(num), 30, on_medianSigmaBar, &medianPair);
        // waitKey(0);

        /*-------get scene flow-----------*/
        cv::Mat imDepth1 = cv::Mat::zeros(480, 640, CV_32F);
        cv::Mat imDepth2 = cv::Mat::zeros(480, 640, CV_32F);
        float mDepthMapFactor = 1.0f / 5000.0;
        //深度图格式转换成float类型
        if((fabs(mDepthMapFactor-1.0f)>1e-5)) {
            MedianBlurImg1.convertTo(imDepth1, CV_32F, mDepthMapFactor);
        }
        if((fabs(mDepthMapFactor-1.0f)>1e-5)) {
            MedianBlurImg2.convertTo(imDepth2, CV_32F, mDepthMapFactor);
        }
        // cout << imDepth1 << endl;

        //------------test
        // VelocityXYZ = sflow->SceneFlow::getVelXYZ(imDepth1, imDepth2, optical);
        // sflow->SceneFlow::getHomography_RANSAC(imDepth1, imDepth2, optical_uv);

        cv::Mat binary_Mask(480, 640, CV_8UC1, Scalar(0));  // 255-white-mask   0-black-background
        cv::Mat optical_flow = cv::Mat::zeros(480, 640, CV_32FC2);
        cv::Mat optical_flow_img = cv::Mat::zeros(480, 640, CV_8UC1);
        
        // sflow->SceneFlow::GetCleanFlow(optical_uv, optical_flow);

        //-----getRt_RANSAC
        // Eigen::Matrix3d best_R = Eigen::Matrix3d::Zero();
        // Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
        // double best_th;
        // sflow->SceneFlow::getRt_RANSAC(imDepth1, imDepth2, optical_uv, best_R, best_t, best_th);

        //-----getHomography_RANSAC
        // double best_th;
        // sflow->SceneFlow::getHomography_RANSAC(imDepth1, imDepth2, optical_uv, best_th);


        //-----GetSceneFlow
        cv::Mat projection_error_Mask = cv::Mat(480, 640, CV_8UC4, Scalar(0, 0, 0, 0));
        cv::Mat projection_error_img = cv::Mat(480, 640, CV_16UC1, Scalar(0));
        std::chrono::steady_clock::time_point t_start_GetSceneFlow = std::chrono::steady_clock::now();
        sflow->SceneFlow::GetSceneFlow(imDepth1, imDepth2, optical_uv, binary_Mask, projection_error_Mask, projection_error_img);
        std::chrono::steady_clock::time_point t_end_GetSceneFlow = std::chrono::steady_clock::now();
        double t_GetSceneFlow = std::chrono::duration_cast<std::chrono::duration<double> >(t_end_GetSceneFlow - t_start_GetSceneFlow).count(); //单位秒
        cout << "t_GetSceneFlow : " << t_GetSceneFlow  << " s" << endl;

        cv::Mat error_Mask_img = cv::Mat(480, 640, CV_8UC4, Scalar(0, 0, 0, 0));
        cvtColor(in_image1, in_image1, COLOR_BGR2BGRA);
        cv::addWeighted(projection_error_Mask, 8, in_image1, 1, 0.0, error_Mask_img, 4);


        cv::imwrite("/mnt/SceneFlow/output/mask1.png", binary_Mask);
        cv::Mat struct2 = getStructuringElement(1, Size(11, 11));   // 0 rect 1 cross
        cv::dilate(binary_Mask, binary_Mask, struct2);

        string i = to_string(ni);
        LOG(INFO) << "process: " << ni+1;
        OpticalFlow::SaveResults(in_image1, imD1, MedianBlurImg1, optical_img, binary_Mask, error_Mask_img, projection_error_img, i + ".png");
    }
    // cout << "vec_sd: " <<endl;
    // for(int ni=0; ni<num ; ni++) {
    //     cout << sflow->SceneFlow::vec_sd[ni] << " ";
    //     if(ni == 10){
    //         cout << endl;
    //     }
    // }

    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t11= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count(); //单位秒

    cout << "t11: " << t11 << " s" << endl;
}

void LoadDataset(const std::string &strAssociationFilename, std::vector<std::string> &vstrImageFilenameRGB, 
                 std::vector<std::string> &vstrImageFilenameD) {
    
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation, s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            ss >> sRGB;
            vstrImageFilenameRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenameD.push_back(sD);
        }
    }
}