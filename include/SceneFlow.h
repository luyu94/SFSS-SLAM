#ifndef SCENEFLOW_H_
#define SCENEFLOW_H_


#include <string>
#include <iostream>
#include <vector>
#include <queue>

#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d.hpp> 
#include <opencv2/core/core.hpp> 

#include <Eigen/Core>

#include "KeyFrame.h"
#include "Tracking.h"
#include "SlamConfig.h"
#include "OpticalFlow.h"


using namespace std;
using namespace Eigen;
using namespace ORB_SLAM2;

namespace dyna
{

class SceneFlow {
public:

    SceneFlow();
    ~SceneFlow();
    void FinalStage();

    static SceneFlow* GetInstance();

    void getNormalized(vector<Vector4d> &total_vec_Points1, vector<Vector4d> &total_vec_Points2, Eigen::Matrix<double, 4, 4> &T1, Eigen::Matrix<double, 4, 4> &T2);
    void getRt_RANSAC(const cv::Mat& imDepth1, const cv::Mat& imDepth2, const cv::Mat& optical_uv, Eigen::Matrix3d& best_R, Eigen::Vector3d& best_t, double &best_th, double & th1, int &scale, double& best_avg_error);
    void CalculateICP(vector<Vector4d>& total_vec_Points1, vector<Vector4d>& total_vec_Points2, vector<int>& index, Eigen::Matrix3d& R, Eigen::Vector3d& t);

    // Evaluation
    std::vector<float> mvTimeUpdateMovingProbability;
    std::vector<float> mvTimeMaskGeneration;
    std::vector<float> mvTimeSceneFlowOptimization;


private:
    const int sample_n = 4;  // minimal size of sample pair points
    const double p = 0.99; // desired probability of success
    const double pw = 0.55;  // pw = number of inliers in data / number of points in data
    double n_iterations = 80;  // 公式计算 The number of iterations

    double alpha;  // get H threshold hyperparameter
    double beta;  // get dynamic pixels threshold hyperparameter




    const int W = 640;
    const int H = 480;
    const int w = 160; //160
    const int h = 160; //160
        
    const double camera_cx=320.1;
    const double camera_cy=247.6;  //247.6
    const double camera_fx=535.4;
    const double camera_fy=539.2;


    vector<double> vec_sd;
    void getDepthFlow(const cv::Mat& imD1, const cv::Mat& imD2, const cv::Mat& optical, cv::Mat& Vxyz_show);
    void GetSceneFlow(const std::vector<cv::Mat>& imDepth1s, const std::vector<cv::Mat>& imDepth2s, const std::vector<cv::Mat>& optical_uvs, std::vector<cv::Mat>& binary_Masks, std::vector<cv::Mat>& projection_error_Masks, std::vector<cv::Mat>& projection_error_imgs, std::vector<cv::Mat>& bMasks);
    

    Eigen::Matrix4d getHomography_RANSAC(const cv::Mat& imDepth1, const cv::Mat& imDepth2, const cv::Mat& optical_uv, double & best_th);
    vector<Vector4d> getunnormalizedPoints(int x1, int y1, double z1, cv::Mat &flag, const cv::Mat& imDepth2, const cv::Mat& optical);

    Eigen::Matrix<double, 4, 4> CalculateH(vector<Vector4d>& total_vec_Points1, vector<Vector4d>& total_vec_Points2, vector<int>& index);
    vector<Vector4d> get_vector_Points(vector<Vector4d>& total_vec_Points, vector<int>& index);
    void FloodFill(const cv::Mat& Depth, const cv::Mat& binaryMask, const cv::Mat& reprojection_error, double& th1, cv::Mat& bMask);
    void flood(int i, int j, float d, float depth, float avg_depth, float sum_depth, float total_avg_depth, float total_sum_depth, const cv::Mat& Depth, cv::Mat& mask, cv::Mat& flag, queue<Vector2i>& Q);
    void compare(int srci, int srcj, float d, float depth, float avg_depth, float sum_depth, float total_avg_depth, float total_sum_depth, const cv::Mat& Depth, cv::Mat& mask, cv::Mat& flag, queue<Vector2i>& Q);
    void fill(cv::Mat& mask, int srci, int srcj, int oi, int oj, cv::Mat& Depth);
    bool inArea(cv::Mat& image, int srci, int srcj);


    // latest keyframe that has SceneFlow
    size_t mnLatestSceneFlowKeyFrameID;
    // Total number of keyframes that has SceneFlow
    size_t mnTotalSceneFlowFrameNum;

    // disable or enable  moving probability
    bool mbIsUseSceneFlow;


    cv::Size winSize = cv::Size(160, 160);

    int mBatchSize;

    static SceneFlow* mInstance;


};  

}

#endif