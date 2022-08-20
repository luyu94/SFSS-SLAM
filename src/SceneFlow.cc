#include <iostream> 
#include <algorithm>
#include <vector>
#include <queue>
#include <random>  
#include <chrono> // time module 
#include <algorithm>

#include <stdio.h> 
#include <stdlib.h> 
#include <cmath> 
#include <ctime> 
#include <time.h>

#include <glog/logging.h>

#include <Eigen/Core> 
#include <Eigen/Dense> 
#include <Eigen/Eigenvalues> 
#include <Eigen/QR> 
#include <Eigen/SVD>


#include "SceneFlow.h"


using namespace std;
using namespace cv;
using namespace Eigen; 
using namespace ORB_SLAM2;

namespace dyna {

SceneFlow::SceneFlow()
{
    mnLatestSceneFlowKeyFrameID = 0;
    mnTotalSceneFlowFrameNum = 0;
    alpha = 2;
    beta = 1.732;

    mBatchSize = 2;
    // mthDynamicThreshold = 0.5;

    mvTimeMaskGeneration.reserve(1000);

    mbIsUseSceneFlow = true;
}

SceneFlow::~SceneFlow(){

}


void SceneFlow::GetSceneFlow(const std::vector<cv::Mat>& imDepth1s, const std::vector<cv::Mat>& imDepth2s, const std::vector<cv::Mat>& optical_uvs, std::vector<cv::Mat>& binary_Masks, std::vector<cv::Mat>& projection_error_Masks, std::vector<cv::Mat>& projection_error_imgs, std::vector<cv::Mat>& bMasks) 
{
    LOG(INFO) << "---GetSceneFlow";

    int batch_size = imDepth1s.size();
    if (batch_size <= 0) {
        LOG(ERROR) << "No image data";
        return;
    }

    for (size_t l = 0; l < batch_size; l++)
    {
        cv::Mat projection_error_Mask = cv::Mat(480, 640, CV_8UC4, Scalar(0, 0, 0, 0));
        cv::Mat projection_error_img = cv::Mat(480, 640, CV_16UC1, Scalar(0));  //cv::Mat::zeros(mImRGB.size(), CV_8UC1);
        cv::Mat binary_Mask(480, 640, CV_8UC1, Scalar(0));  // 255-white-mask   0-black-background
        cv::Mat bMask(480, 640, CV_8UC1, Scalar(0));

        cv::Mat flag = cv::Mat::ones(480, 640, CV_8U);

        // LOG(INFO) << "------getRt_RANSAC";
        int scale = 0;
        double sd = 0;
        double th1 = 0;
        double best_th = 0;
        Eigen::Matrix3d best_R = Eigen::Matrix3d::Zero();
        Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
        getRt_RANSAC(imDepth1s[l], imDepth2s[l], optical_uvs[l], best_R, best_t, best_th, th1, scale, sd);
        
        // for each pixel
        cv::Mat projection_error = cv::Mat(480, 640, CV_32FC1, Scalar(0));
        Vector2d error2;
        Vector3d error3;
        Vector3d t;
        vector<float> vec_error;
        for(int i=0; i<imDepth1s[l].rows; i++) 
        {
            for(int j=0; j<imDepth1s[l].cols; j++) 
            {
                if(imDepth1s[l].at<float>(i, j)>1e-2 && imDepth1s[l].at<float>(i, j)<6) {
                    int x1 = j;
                    int y1 = i;
                    double z1 = (double) imDepth1s[l].at<float>(i, j);
                    double mu = (1.0f / z1) * (1.0f / z1);

                    
                    float x2;
                    float y2;
                    int rx2 = 0;
                    int ry2 = 0;

                    float vx = 0.f;
                    float vy = 0.f;
                    float vz = 0.f;
                    //=======得到2的点，深度值是亚像素双线性插值得===========
                    rx2 = x1;
                    ry2 = y1;
                    // std::cout << "x1: "<< x1 << "  y1: "<< y1 << endl;
                    if(   !cvIsNaN(optical_uvs[l].at<Vec2f>(ry2, rx2)[1])      && !cvIsNaN(optical_uvs[l].at<Vec2f>(ry2, rx2)[0])
                    && !cvIsInf(optical_uvs[l].at<Vec2f>(ry2, rx2)[1])      && !cvIsInf(optical_uvs[l].at<Vec2f>(ry2, rx2)[0])
                    && abs(optical_uvs[l].at<Vec2f>(ry2, rx2)[1])<1.0e+2    && abs(optical_uvs[l].at<Vec2f>(ry2, rx2)[0])<1.0e+2
                    && abs(optical_uvs[l].at<Vec2f>(ry2, rx2)[1])>1.0e-5    && abs(optical_uvs[l].at<Vec2f>(ry2, rx2)[0])>1.0e-5) {
                        vy = optical_uvs[l].at<Vec2f>(ry2, rx2)[1];   // get velocity of pixel 1-vy-v-i-h 0-vx-u-j-w
                        vx = optical_uvs[l].at<Vec2f>(ry2, rx2)[0];
                    }

                    x2 = (float) rx2 + vx;   // get x, y of next pixel
                    y2 = (float) ry2 + vy;
                    // std::cout << "x2: "<< x2 << "  y2: "<< y2 << endl;
                    int x2_int = (int) x2;  // 向下取整
                    int y2_int = (int) y2;
                    
                    int dx0 = x2_int; 
                    int dx1 = x2_int+1;
                    int dy0 = y2_int;
                    int dy1 = y2_int+1;
                    
                    //// fraction part of x, y
                    float fx = x2 - (float)x2_int;
                    float fy = y2 - (float)y2_int;

                    double z2;
                    if(0<=dx0 && dx0<imDepth2s[l].cols && 0<=dx1 && dx1<imDepth2s[l].cols && 0<=dy0 && dy0<imDepth2s[l].rows && 0<=dy1 && dy1<imDepth2s[l].rows) {
                        if(imDepth2s[l].at<float>(dy0, dx0)>1e-2 && imDepth2s[l].at<float>(dy0, dx1)>1e-2
                        && imDepth2s[l].at<float>(dy1, dx0)>1e-2 && imDepth2s[l].at<float>(dy1, dx1)>1e-2
                        && imDepth2s[l].at<float>(dy0, dx0)< 6.0 && imDepth2s[l].at<float>(dy0, dx1)< 6.0
                        && imDepth2s[l].at<float>(dy1, dx0)< 6.0 && imDepth2s[l].at<float>(dy1, dx1)< 6.0 ) {
                            float d00 = imDepth2s[l].at<float>(dy0, dx0);
                            float d10 = imDepth2s[l].at<float>(dy0, dx1);
                            float d01 = imDepth2s[l].at<float>(dy1, dx0);
                            float d11 = imDepth2s[l].at<float>(dy1, dx1);

                            z2 = (double)( d00 * (1.0f - fx) * (1.0f - fx) 
                                + d10 * fx * (1.0f - fy) 
                                + d01 * (1.0f - fx) * fy 
                                + d11 * fx * fy);
                        }else {
                            z2 = 0;
                        }

                    }else { // Todo: 运动像素离开边界
                            // if(imDepth2s[l].at<float>(ry2, rx2)>1e-2 && cvIsNaN(imDepth2s[l].at<float>(ry2, rx2)) && !cvIsInf(imDepth2s[l].at<float>(ry2, rx2))) {
                            //     z2 = imDepth2s[l].at<float>(ry2, rx2);
                            // }else {
                            //     z2 = 0;
                            // }
                            z2 = z1;
                    }
                    
                    if(z2!= 0) {
                        double mz1 = (double) mu * z1;
                        double mz2 = (double) mu * z2;
 
                        vz = (float)(abs(mz2 - mz1));

                        double x1c = ((x1 - camera_cx) / camera_fx ) * z1;
                        double y1c = ((y1 - camera_cy) / camera_fy ) * z1;

                        //-----reprojection error
                        Vector3d p2c_e;
                        Vector3d p1c;
                        Vector3d p1c2;


                        error3 << best_R(0, 0) * (x1 - camera_cx) + best_R(0, 1) * (y1 - camera_cy) + best_R(0, 2) * mz1 + best_t(0, 0) - (x2 - camera_cx),
                                best_R(1, 0) * (x1 - camera_cx) + best_R(1, 1) * (y1 - camera_cy) + best_R(1, 2) * mz1 + best_t(1, 0) - (y2 - camera_cy),
                                best_R(2, 0) * (x1 - camera_cx) + best_R(2, 1) * (y1 - camera_cy) + best_R(2, 2) * mz1 + best_t(2, 0) - 1/z2;


                        float error = (float) error3.norm();
                        projection_error.at<float>(i, j) = error;
                        vec_error.push_back(error);
                        

                        
                    }    // end if(z2!= 0)

                }

            }
        }
        

        //-----projection_error_Mask and binary_Mask
        double Th1=0;
        if(sd>beta){
            Th1 = best_th + sd;
        }else{
            Th1 = best_th + beta;
        }
        // cout << "best_th: " << best_th << endl;
        for(int i=0; i<projection_error.rows; i++) {
            for(int j=0; j<projection_error.cols; j++) {
                if (projection_error.at<float>(i, j) > best_th + beta) {    //median + beta  (best_th * 0.2)   || (projection_error.at<float>(i, j) > 0 && projection_error.at<float>(i, j) < (best_th - beta))

                    binary_Mask.at<uchar>(i, j) = 255;    // mask-255-white
                    projection_error_Mask.at<Vec4b>(i, j)[0] = 0;    //B
                    projection_error_Mask.at<Vec4b>(i, j)[1] = 0;    //G
                    projection_error_Mask.at<Vec4b>(i, j)[2] = 255;   //R
                    projection_error_Mask.at<Vec4b>(i, j)[3] = 255;
                }
            }
        }
        // cout << "scale: " << scale << endl;
        projection_error.convertTo(projection_error_img, CV_16UC1, scale);     // 50 300 5000 

        FloodFill(imDepth1s[l], binary_Mask, projection_error, th1, bMask);

        for(int i=0; i<bMask.rows; i++) {
            for(int j=0; j<bMask.cols; j++) {
                if(bMask.at<uchar>(i, j) == 255 || binary_Mask.at<uchar>(i, j) == 255)
                {
                    bMask.at<uchar>(i, j) = 255;
                }
            }
        }

        cv::Mat struct3 = getStructuringElement(0, Size(21, 21));   // 0 rect 1 cross
        cv::dilate(bMask, bMask, struct3);
    
        binary_Masks.push_back(binary_Mask);
        projection_error_Masks.push_back(projection_error_Mask);
        projection_error_imgs.push_back(projection_error_img);
        bMasks.push_back(bMask);

    }

    LOG(INFO) << "---Finish GetSceneFlow";

}




/*
RANSAC
*/
void SceneFlow::getRt_RANSAC(const cv::Mat& imDepth1, const cv::Mat& imDepth2, const cv::Mat& optical_uv, Eigen::Matrix3d& best_R, Eigen::Vector3d& best_t, double & best_th, double & th1, int &scale, double& best_avg_error) {
    // LOG(INFO) << "---getRt";
    int iterations = 0;
    double avg_th_error = 15;
    double V;
    Eigen::Matrix3d good_R = Eigen::Matrix3d::Zero();
    Eigen::Vector3d good_t = Eigen::Vector3d::Zero();

    vector<int> best_inliers_index = {};
    vector<Vector4d> best_vec_Points1 = {};
    vector<Vector4d> best_vec_Points2 = {};

    vector<Vector4d> total_vec_Points1 = {};
    vector<Vector4d> total_vec_Points2 = {};
    
    // vector<Vector4d> total_plane_Points1 = {};
    // vector<Vector4d> total_plane_Points2 = {};

    cv::Mat flag = cv::Mat::ones(480, 640, CV_8U);    //不重复标志
    for (int i = 0; i < H; i+=h) {    //160*160窗口 稀疏取样
        for (int j = 0; j < W; j+=w) {
            //===================每个网格10个点，取稀疏120样本===================
            std::random_device rd;      // 产生随机数
            std::mt19937 gen(rd());
            std::uniform_int_distribution<int32_t> distribw(0, w);
            std::uniform_int_distribution<int32_t> distribh(0, h);

            Rect roi(Point(j, i), winSize);
            cv::Mat roiD1 = imDepth1(roi);

            for(int num=0; num<30; num++) {   //随机30个数，不重复
                int rx1 = distribw(gen);
                int ry1 = distribh(gen);

                int x1 = (j / w) * w + rx1;
                int y1 = (i / h) * h + ry1;

                if(imDepth1.at<float>(y1, x1)>1e-2 && imDepth1.at<float>(y1, x1)<6 && flag.at<uchar>(y1, x1)==1) {
                    double z1 = imDepth1.at<float>(y1, x1);
                    vector<Vector4d> Points = getunnormalizedPoints(x1, y1, z1, flag, imDepth2, optical_uv);
                    if(!Points.empty()) {
                        total_vec_Points1.push_back(Points[0]);
                        total_vec_Points2.push_back(Points[1]);

                        flag.at<uchar>(y1, x1) += 1;
                        // total_plane_Points1.push_back(Points[2]);
                        // total_plane_Points2.push_back(Points[3]);
                    }
                }

            }   //end for 
        }
    }   // end for 

    std::random_device rd_120;
    std::mt19937 gen_120(rd_120());
    std::uniform_int_distribution<int32_t> distribw_120(0, W);
    std::uniform_int_distribution<int32_t> distribh_120(0, H);
    while(total_vec_Points1.size() < 360) {
        int x1 = distribw_120(gen_120);
        int y1 = distribh_120(gen_120);
        if(imDepth1.at<float>(y1, x1)>1e-2 && imDepth1.at<float>(y1, x1)<6 && flag.at<uchar>(y1, x1)==1) {
            double z1 = imDepth1.at<float>(y1, x1);
            vector<Vector4d> Points = getunnormalizedPoints(x1, y1, z1, flag, imDepth2, optical_uv);
            if(!Points.empty()) {
                total_vec_Points1.push_back(Points[0]);
                total_vec_Points2.push_back(Points[1]);
                // total_plane_Points1.push_back(Points[2]);
                // total_plane_Points2.push_back(Points[3]);
                flag.at<uchar>(y1, x1) += 1;
            }
        }
    } // end while 120

    // ---RANSAC

    while(iterations < n_iterations) {
		// LOG(INFO) << "Current iteration: " << iterations;
        vector<int> good_inliers_index = {};
  
        // minimum sample of four points
        std::random_device rd1;
        std::mt19937 gen1(rd1());
        std::uniform_int_distribution<int32_t> distrib(0, 360);
        vector<int> vflag(360, 0);
        vector<int32_t> vector_sample_idx = {};
        while (vector_sample_idx.size() != 4) {       //
            // Generate a random index between [0, 120]
            int32_t random_idx = distrib(gen1);

            vflag[random_idx] += 1;
            // double Vx = abs(total_vec_Points2[random_idx](0, 0) - total_vec_Points1[random_idx](0, 0));
            // double Vy = abs(total_vec_Points2[random_idx](1, 0) - total_vec_Points1[random_idx](1, 0));
            // double Vz = abs(total_vec_Points2[random_idx](2, 0) - total_vec_Points1[random_idx](2, 0));
            // double v = sqrt(Vx * Vx + Vy * Vy + Vz * Vz);
            if (vflag[random_idx] ==1) {     //&& Vx<0.3 && Vy<0.3   && Vz<2   && v<V
                vector_sample_idx.push_back(random_idx);

                // cout << random_idx << endl;                                    // ---show vector_sample
                // cout << total_vec_Points1[random_idx].transpose() << endl;
                // cout << total_vec_Points2[random_idx].transpose() << endl;
                // cout << "=========================" << endl;
            }
            // else{
            //     cout << total_vec_Points1[random_idx].transpose() << endl;
            //     cout << total_vec_Points2[random_idx].transpose() << endl;
            //     cout << "++++++++++++++++++++++++" << endl;
            // }
        }

        // ---------four points caculate Rt
        // LOG(INFO) << "-----CalculateICP";
        Eigen::Matrix3d R21 = Eigen::Matrix3d::Zero();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        CalculateICP(total_vec_Points1, total_vec_Points2, vector_sample_idx, R21, t);
        // cout << "R21 = " << endl;
        // cout << R21 << endl;
        // cout << "t = " << t.transpose() << endl;
        if(t.norm() == NAN) {
            iterations++;
            break;
        }

        // -------compute avg_error
        Eigen::Vector3d error3;
        double sum_error = 0;
        
        vector<double> error_norm2;  // cout error norm2

        for(int i=0; i<total_vec_Points1.size(); i++) {    
            Vector3d p1c, p1;
            Vector3d p2c, p2;
            // p1 << total_plane_Points1[i](0, 0), total_plane_Points1[i](1, 0), total_plane_Points1[i](2, 0);
            // p2 << total_plane_Points2[i](0, 0), total_plane_Points2[i](1, 0), total_plane_Points2[i](2, 0);
            p1 << total_vec_Points1[i](0, 0), total_vec_Points1[i](1, 0), total_vec_Points1[i](2, 0);
            p2 << total_vec_Points2[i](0, 0), total_vec_Points2[i](1, 0), total_vec_Points2[i](2, 0);

            error3 = R21 * p1 + t - p2;

            double error = error3.norm();
            // cout << "error3_Rt: " << error3.transpose() << endl;
            // cout << "error: " << error << endl;
            // cout << "&&&&&&&&&&&&&&&&" << endl;  
            sum_error += error;

            error_norm2.push_back(error);
        }
        double current_avg_error = sum_error / total_vec_Points1.size();
        // cout << "**current_avg_error: " << current_avg_error << endl;
        vector<double> sort_error_norm2 = error_norm2;
        sort(sort_error_norm2.begin(), sort_error_norm2.end());


        // for(int i=0; i<sort_error_norm2.size(); i++) {    // ---show sort_error_norm2
        //     cout << "**sort_error_norm2: " << sort_error_norm2[i]<< endl;    
        // }
        // cout << "**current_avg_error: " << current_avg_error << endl;

        int num_e = sort_error_norm2.size();
        double median;
        median = (sort_error_norm2[num_e / 2] + sort_error_norm2[num_e / 2 - 1]) / 2;


        
        //-------count inliers
        double sd2;
        if(iterations == 0) {
            good_R = R21;
            good_t = t;
            best_avg_error = current_avg_error;
            double powAll2 = 0.0;
            for (int i = 0; i < num_e; i++)
            {
                powAll2 += pow((error_norm2[i] - median), 2);
            }
            sd2 = sqrt(powAll2 / (double)num_e);
            // cout << "**median: " << median << endl; 
            // cout << "**sd2: " << sd2 << endl;

            for(int i=0; i<error_norm2.size(); i++) {
                if(error_norm2[i] < median+sd2 && error_norm2[i] > median-sd2) {
                    good_inliers_index.push_back(i);
                }
            }
            best_inliers_index = good_inliers_index;
            // cout << "********************************" << endl;
            // cout << "good_inliers_index.size: " << good_inliers_index.size() << endl;
            // cout << "********************************" << endl;
        }
        if(current_avg_error < best_avg_error) {
            good_R = R21;
            good_t = t;
            best_avg_error = current_avg_error;

            double powAll2 = 0.0;
            for (int i = 0; i < num_e; i++)
            {
                powAll2 += pow((error_norm2[i] - median), 2);
            }
            sd2 = sqrt(powAll2 / (double)num_e);
            // cout << "**median: " << median << endl; 
            // cout << "**sd2: " << sd2 << endl;
            
            for(int i=0; i<error_norm2.size(); i++) {
                if(error_norm2[i] < median+sd2 && error_norm2[i] > median-sd2) {
                    good_inliers_index.push_back(i);
                }
            }
            best_inliers_index = good_inliers_index;
            // cout << "********************************" << endl;
            // cout << "good_inliers_index.size: " << good_inliers_index.size() << endl;
            // cout << "********************************" << endl;

        }

        // best_inliers_index = good_inliers_index;
        
        // <th break
        if(current_avg_error < avg_th_error) {
            // cout << "$$$$$$" << endl;
            break;
        }
        iterations++;

    }  // end while iteration

    cout << "end while iterations" << endl;
    // -----all inliers ICP
    // cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;
    // for(int i=0; i< best_inliers_index.size(); i++) {
    //         cout << best_inliers_index[i] << " | ";
    // }
    CalculateICP(total_vec_Points1, total_vec_Points2, best_inliers_index, good_R, good_t);  

    best_R = good_R;
    best_t = good_t;

    // compute best_th
    Eigen::Vector3d error2;
    double sum_error2;
    vector<double> error_norm;  // cout error norm2
    for(int i=0; i<total_vec_Points1.size(); i++) {    
        Vector3d p1c, p1;
        Vector3d p2c, p2;


        p1 << total_vec_Points1[i](0, 0), total_vec_Points1[i](1, 0), total_vec_Points1[i](2, 0);
        p2 << total_vec_Points2[i](0, 0), total_vec_Points2[i](1, 0), total_vec_Points2[i](2, 0);

        error2 = best_R * p1 + best_t - p2;

        double error = error2.norm();
        // cout << "error3_Rt.norm(): " << error << endl;
        // cout << "&&&&&&&&&&&&&&&&" << endl;  

        error_norm.push_back(error2.norm());
        sum_error2 += error2.norm();  
    }

    vector<double> sort_error_norm = error_norm;
    sort(sort_error_norm.begin(), sort_error_norm.end());

    //     // for(int i=0; i<total_vec_Points1.size(); i++) {    // ---show error_norm2
    //     //     cout << error_norm2[i]<< endl;
    //     // }
    // cout << "after caculate best_Rt sort_error_norm2: " << endl;
    // for(int i=0; i<sort_error_norm.size(); i++) {    // ---show sort_error_norm2
    //     cout << sort_error_norm[i]<< endl;    
    // }

    int num_e2 = sort_error_norm.size();
    double median2 = (sort_error_norm[num_e2 / 2] + sort_error_norm[num_e2 / 2 - 1]) / 2;
    double quart2 = sort_error_norm[num_e2 / 4];
    
    // count standard deviation σ
    double powAll2 = 0.0;
    for (int i = 0; i < num_e2; i++)
    {
        powAll2 += pow((error_norm[i] - quart2), 2);
    }
    double sd = sqrt(powAll2 / (double)num_e2);
    // cout << "sd: " << sd << endl;
    double avg_error2 = sum_error2 / sort_error_norm.size(); 

    // cout << "median2: " << median2 << endl;
    // cout << "avg_error2: " << avg_error2 << endl;

    best_th = median2;

    th1 = sort_error_norm[72];

    scale = 40000 / sort_error_norm[num_e2-1];

    // cout << "best_R = " << endl;
    // cout << best_R << endl;
    // cout << "best_t = " << best_t.transpose() << endl;
    // cout << "best_avg_error: " << avg_error2 << endl;

    // LOG(INFO) << "---finish getRt_RANSAC";
}

void SceneFlow::CalculateICP(vector<Vector4d>& total_vec_Points1, vector<Vector4d>& total_vec_Points2, vector<int>& index, Eigen::Matrix3d& R, Eigen::Vector3d& t) {

    // cout << "index_size: " << index.size() << endl;

    vector<Vector4d> vec_Points1 = get_vector_Points(total_vec_Points1, index);
    vector<Vector4d> vec_Points2 = get_vector_Points(total_vec_Points2, index);

    int N = index.size();

    vector<Vector3d> pts1(N);
    vector<Vector3d> pts2(N);
    
	for(int i=0; i<index.size(); i++)
	{
        Vector4d p1 = vec_Points1[i];
        Vector4d p2 = vec_Points2[i];

        pts1[i](0, 0) = p1(0, 0);
        pts1[i](1, 0) = p1(1, 0);
        pts1[i](2, 0) = p1(2, 0);

        pts2[i](0, 0) = p2(0, 0);
        pts2[i](1, 0) = p2(1, 0);
        pts2[i](2, 0) = p2(2, 0);

	}

    // 1 cordinate
    Vector4d mm1;    // center of mass
    Vector4d mm2;
    Vector3d m1;    // center of mass
    Vector3d m2;
    for(int i=0; i<N; i++) {
        // cout << "Points1: " << (T1 * vec_Points1[i]).transpose() << endl;

        mm1 += vec_Points1[i];
        mm2 += vec_Points2[i];
    }
    m1 << mm1(0) / N, mm1(1) / N, mm1(2) / N;
    m2 << mm2(0) / N, mm2(1) / N, mm2(2) / N;
    vector<Vector3d> q1(N), q2(N);  // remove the center
    for(int i=0; i<N; i++) {
        q1[i] = pts1[i] - m1;
        q2[i] = pts2[i] - m2;
        // cout << "q1: " << q1[i].transpose() << endl;
    }

    // 2 W=sum(q2*q1^T)
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i<N; i++)
	{
		W += Eigen::Vector3d(q2[i](0, 0), q2[i](1, 0), q2[i](2, 0)) * Eigen::Vector3d(q1[i](0, 0), q1[i](1, 0), q1[i](2, 0)).transpose();
	}
	// cout << "W = " << W << endl;

    // 3 SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 4 R=U*V^T
    Eigen::Matrix3d R_;
    Eigen::Vector3d t_;
    R_ = U * (V.transpose());
    if(R_.determinant() < 0) {
        R_ = -R_;
    }
    t_ = m2 - R * m1;

    // Eigen::Matrix4d T_, T;
    // T_ << R_(0, 0), R_(0, 0), R_(0, 0), t_(0),
    //       R_(1, 0), R_(1, 1), R_(1, 2), t_(1),
    //       R_(2, 0), R_(2, 1), R_(2, 2), t_(2),
    //       0       ,        0,        0,     1;
    // T = T2.inverse() * T_ * T1;

    // R << T(0, 0), T(0, 0), T(0, 0),
    //      T(1, 0), T(1, 1), T(1, 2),
    //      T(2, 0), T(2, 1), T(2, 2);

    // t << T(0, 3), T(1, 3), T(2, 3);

    R = R_;
    t = t_;

}

vector<Vector4d> SceneFlow::getunnormalizedPoints(int x1, int y1, double z1, cv::Mat &flag, const cv::Mat& imDepth2, 
                    const cv::Mat& optical) {
    // LOG(INFO) << "----start--getunnormalizedPoints";
    vector<Vector4d> Points  = {};

    double mu = (1.0f / z1) * (1.0f / z1);

    float x2;
    float y2;
    int rx2 = 0;
    int ry2 = 0;

    float vx = 0.f;
    float vy = 0.f;
    float vz = 0.f;

    //=======得到2的点，深度值是亚像素双线性插值得===========
    rx2 = x1;
    ry2 = y1;
    // std::cout << "x1: "<< x1 << "  y1: "<< y1 << endl;

    if(    !cvIsNaN(optical.at<Vec2f>(ry2, rx2)[1])      && !cvIsNaN(optical.at<Vec2f>(ry2, rx2)[0])
        && !cvIsInf(optical.at<Vec2f>(ry2, rx2)[1])      && !cvIsInf(optical.at<Vec2f>(ry2, rx2)[0])
        && abs(optical.at<Vec2f>(ry2, rx2)[1])<1.0e+2    && abs(optical.at<Vec2f>(ry2, rx2)[0])<1.0e+2
        && abs(optical.at<Vec2f>(ry2, rx2)[1])>1.0e-5    && abs(optical.at<Vec2f>(ry2, rx2)[0])>1.0e-5
        ) {
                   
        vy = optical.at<Vec2f>(ry2, rx2)[1];   // get velocity of pixel 1-vy-v-i-h 0-vx-u-j-w
        vx = optical.at<Vec2f>(ry2, rx2)[0];
    }

    x2 = (float) rx2 + vx;   // get x, y of next pixel
    y2 = (float) ry2 + vy;
    // std::cout << "x2: "<< x2 << "  y2: "<< y2 << endl;
    int x2_int = (int) x2;  // 向下取整
    int y2_int = (int) y2;
    
    int dx0 = x2_int; 
    int dx1 = x2_int+1;
    int dy0 = y2_int;
    int dy1 = y2_int+1;
    
    //// fraction part of x, y
    float fx = x2 - (float)x2_int;
    float fy = y2 - (float)y2_int;

    double z2;
    if(0<=dx0 && dx0<imDepth2.cols && 0<=dx1 && dx1<imDepth2.cols && 0<=dy0 && dy0<imDepth2.rows && 0<=dy1 && dy1<imDepth2.rows) {
        if(imDepth2.at<float>(dy0, dx0)>1e-2 && imDepth2.at<float>(dy0, dx1)>1e-2
        && imDepth2.at<float>(dy1, dx0)>1e-2 && imDepth2.at<float>(dy1, dx1)>1e-2
        && imDepth2.at<float>(dy0, dx0)< 6.0 && imDepth2.at<float>(dy0, dx1)< 6.0
        && imDepth2.at<float>(dy1, dx0)< 6.0 && imDepth2.at<float>(dy1, dx1)< 6.0
        && !cvIsNaN(imDepth2.at<float>(dy0, dx0)) && !cvIsNaN(imDepth2.at<float>(dy0, dx1)) 
        && !cvIsNaN(imDepth2.at<float>(dy1, dx0)) && !cvIsNaN(imDepth2.at<float>(dy1, dx1))
        && !cvIsInf(imDepth2.at<float>(dy1, dx0)) && !cvIsInf(imDepth2.at<float>(dy0, dx1))
        && !cvIsInf(imDepth2.at<float>(dy1, dx0)) && !cvIsInf(imDepth2.at<float>(dy1, dx1)) ) {
            float d00 = imDepth2.at<float>(dy0, dx0);
            float d10 = imDepth2.at<float>(dy0, dx1);
            float d01 = imDepth2.at<float>(dy1, dx0);
            float d11 = imDepth2.at<float>(dy1, dx1);

            z2 = (double)( d00 * (1.0f - fx) * (1.0f - fx) 
                + d10 * fx * (1.0f - fy) 
                + d01 * (1.0f - fx) * fy 
                + d11 * fx * fy);
        }else {
            z2 = 0;
        }

    }else { // Todo: 运动像素离开边界  && imDepth2.at<float>(ry2, rx2)< 6.0
            // if(imDepth2.at<float>(ry2, rx2)>1e-2        && imDepth2.at<float>(ry2, rx2)< 6.0
            //   && !cvIsNaN(imDepth2.at<float>(ry2, rx2)) && !cvIsInf(imDepth2.at<float>(ry2, rx2))) 
            //   {
            //     z2 = imDepth2.at<float>(ry2, rx2);
            //   }else {
            //     z2 = 0;
            //   }
            z2 = z1;
    }

    if(z2!= 0) {
        double mz1 = (double) mu * z1;
        double mz2 = (double) mu * z2;
        // double mz1 = (double) z1 * z1;
        // double mz2 = (double) d2 * d2;
        flag.at<uchar>(y1, x1) += 1;

        double x1c = ((x1 - camera_cx) / camera_fx ) * z1;
        double y1c = ((y1 - camera_cy) / camera_fy ) * z1;

        double x2c = ((x2 - camera_cx) / camera_fx ) * z2;
        double y2c = ((y2 - camera_cy) / camera_fy ) * z2;

        Points.push_back(Vector4d((double) (x1 - camera_cx), (double) (y1 - camera_cy), mz1, 1.0));
        Points.push_back(Vector4d((double) (x2 - camera_cx), (double) (y2 - camera_cy), 1/z2, 1.0));    // sample coordinate to plane

        Points.push_back(Vector4d((double) (x1c), (double) (y1c), mz1, 1.0));
        Points.push_back(Vector4d((double) (x2c), (double) (y2c), mz2, 1.0));    // sample coordinate to camera

    }
    // LOG(INFO) << "----finish--getunnormalizedPoints";
    return Points;
}


vector<Vector4d> SceneFlow::get_vector_Points(vector<Vector4d>& total_vec_Points, vector<int>& index) {
    
    vector<Vector4d> vec_Points;
    for(int i=0; i<index.size(); i++) {
        vec_Points.push_back(total_vec_Points[index[i]]);
    }

    return vec_Points;

}



void SceneFlow::FloodFill(const cv::Mat& Depth, const cv::Mat& binaryMask, const cv::Mat& reprojection_error, double& th1, cv::Mat& bMask) {

    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int x, y;
    float depth;

    Mat b_Mask(480, 640, CV_8UC1, Scalar(0));
    Mat structureElement1 = getStructuringElement(MORPH_CROSS, Size(17, 17));  // MORPH_CROSS   MORPH_RECT    MORPH_ELLIPSE
    Mat structureElement2 = getStructuringElement(MORPH_CROSS, Size(5, 5));  // MORPH_CROSS   MORPH_RECT    MORPH_ELLIPSE
	erode(binaryMask, b_Mask, structureElement1);
    erode(b_Mask, b_Mask, structureElement2);
    int num = cv::connectedComponentsWithStats(b_Mask, labels, stats, centroids, 4, CV_32S);
    // cout << "aeras num = " << num-1 << endl;

    // cv::imwrite("/mnt/SceneFlow1/output/erosion.png", b_Mask);

    cv::Mat mask(480, 640, CV_8UC1, Scalar(0));
    cv::Mat flag(480, 640, CV_8UC1, Scalar(0));
    // cout << "start FloodFill" << endl;
    
    for(int label=1; label<num; label++) {
        queue<Vector2i> Q = {};
        x = (int) centroids.at<double>(label, 0);
        y = (int) centroids.at<double>(label, 1);
        // cout << "x: " << x << " y: " << y << endl;
        depth = Depth.at<float>(y, x);
        // cout << "depth: " << depth << endl;

        float total_sum_depth =0;
        for(int i=0; i<b_Mask.rows; i++) {
            for(int j=0; j<b_Mask.cols; j++) {

                if(b_Mask.at<uchar>(i, j) == 255 && labels.at<int32_t>(i, j) == label) { 
                    // cout << "labels: " << labels.at<int32_t>(i, j) << endl;
                    Vector2i p(i, j);
                    Q.push(p);
                    mask.at<uchar>(i, j) = 255;
                    total_sum_depth += Depth.at<float>(i, j);
                }
            }
        }
        float total_avg_depth = total_sum_depth / Q.size();

        while(!Q.empty()) {

            Vector2i p = Q.front();
            float sum_depth = 0;
            if(    !cvIsNaN(Depth.at<float>(p(0), p(1))) && !cvIsInf(Depth.at<float>(p(0), p(1)))
                && Depth.at<float>(p(0), p(1))>1e-2) {
                // cout << "p: " << p(0) << ", " << p(1) << endl;
                // cout << "depth: " << Depth.at<float>(p(0), p(1)) << endl;
                float d = Depth.at<float>(p(0), p(1));
                float th = 0.07 / Depth.at<float>(p(0), p(1));

                int i = p(0);
                int j = p(1);
                int sum_num = 0;
                for(int a=-10; a<21; a++){
                    for(int b=-10; b<21; b++) {
                        if(    !cvIsNaN(Depth.at<float>(i+a, j+b)) && !cvIsInf(Depth.at<float>(i+a, j+b))
                            && Depth.at<float>(i+a, j+b)>1e-2     
                            && mask.at<uchar>(i, j) == 255 ) {  //&& labels.at<int32_t>(i+a, j+b) == label
                                sum_num += 1;
                                sum_depth += Depth.at<float>(i, j);
                        }
                    }
                }

                float avg_depth = sum_depth / sum_num;


                flood(i, j, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q);

            }
            // cout << "=======================================" << endl;

            Q.pop();   
        }


    }

    bMask = mask;
    // cv::Mat struct1 = getStructuringElement(0, Size(15, 15));   // 0 rect 1 cross
    // cv::dilate(bMask, bMask, struct1);

    cout << "finish FloodFill" << endl;

}

void SceneFlow::flood(int i, int j, float d, float depth, float avg_depth, float sum_depth, float total_avg_depth, float total_sum_depth, const cv::Mat& Depth, cv::Mat& mask, cv::Mat& flag, queue<Vector2i>& Q) {
    
    
    compare(i,   j-1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-1, j-1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

    compare(i-1,   j, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-1, j+1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

    compare(i,   j+1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+1, j+1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

    compare(i+1,   j, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+1, j-1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+1, j-2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

    compare(i,   j-2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-1, j-2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-2, j-2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-2, j-1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-2,   j, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-2, j+1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i-2, j+2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

    compare(i-1, j+2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i,   j+2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+1, j+2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+2, j+2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

    compare(i+2, j+1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+2,   j, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+2, j-1, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;
    compare(i+2, j-2, d, depth, avg_depth, sum_depth, total_avg_depth, total_sum_depth, Depth, mask, flag, Q); 
    // if(flag.at<uchar>(i, j) == -1) return;

}

void SceneFlow::compare(int srci, int srcj, float d, float depth, float avg_depth, float sum_depth, float total_avg_depth, float total_sum_depth, const cv::Mat& Depth, cv::Mat& mask, cv::Mat& flag, queue<Vector2i>& Q) {

    if(flag.at<uchar>(srci, srcj) == -1) return;
    if((mask.at<uchar>(srci, srcj)) == 0) {
        if(    !cvIsNaN(Depth.at<float>(srci, srcj))
            && !cvIsInf(Depth.at<float>(srci, srcj))
            && Depth.at<float>(srci, srcj)>1e-2
            && Depth.at<float>(srci, srcj)<6   
            ) { //&& Depth.at<float>(p(0)-1, p(1))<6         && Depth.at<float>(p(0), p(1))<6
                // cout << "depth differ: " << abs(Depth.at<float>(srci, srcj) - d) << endl;
                if(Depth.at<float>(srci, srcj)>=1) {
                    if(    ( abs(Depth.at<float>(srci, srcj) - d) < 0.007 * d && abs(Depth.at<float>(srci, srcj) - d) < 0.0125 * avg_depth && abs(Depth.at<float>(srci, srcj) - total_avg_depth) < 0.125* total_avg_depth && abs(Depth.at<float>(srci, srcj) - depth) < 0.25* depth )
                        // || ( abs(Depth.at<float>(srci, srcj) - total_avg_depth) < 0.125 * total_avg_depth && abs(Depth.at<float>(srci, srcj) - d) < 0.0125 * avg_depth)
                        // || abs(Depth.at<float>(srci, srcj) - d) < 0.125 * avg_depth
                      ) {  //&& reprojection_error.at<float>(p(0)-1, p(1)) > (th2+a)  
                        // cout << "abs(Depth.at<float>(srci, srcj) - d): " << abs(Depth.at<float>(srci, srcj)*Depth.at<float>(srci, srcj) - d * d) << endl;
                        // cout << "0.0069 * d: " << 0.007 * d<< endl;
                        mask.at<uchar>(srci, srcj) = 255;
                        Vector2i q(srci, srcj);
                        Q.push(q);

                        total_sum_depth += Depth.at<float>(srci, srcj);
                    }
                }



        }
    }

    total_avg_depth = total_sum_depth / Q.size();
    flag.at<uchar>(srci, srcj) == -1;

}

void SceneFlow::fill(cv::Mat& mask, int srci, int srcj, int oi, int oj, cv::Mat& Depth) {

    if (!inArea(mask, srci, srcj)) return;
    if(mask.at<uchar>(srci, srcj) == -1) {
        return;
    }

    double th = 0.01 * Depth.at<float>(srci, srcj);
    // cout << "depth th: " << th << endl;

    if(    !cvIsNaN(Depth.at<float>(srci, srcj)) && !cvIsNaN(Depth.at<float>(oi, oj))
        && !cvIsInf(Depth.at<float>(srci, srcj)) && !cvIsInf(Depth.at<float>(oi, oj))
        && Depth.at<float>(srci, srcj)>0         && Depth.at<float>(oi, oj)>0
         ) { //&& Depth.at<float>(srci, srcj)<6         && Depth.at<float>(oi, oj)<6

            if(abs(Depth.at<float>(srci, srcj) - Depth.at<float>(oi, oj)) < th) {
                // cout << "depth differ: " << abs(Depth.at<float>(srci, srcj) - Depth.at<float>(oi, oj)) << endl;
                mask.at<uchar>(srci, srcj) = -1;
            }

    }else {
        return;
    }


    mask.at<uchar>(oi, oj) = -1;
    fill(mask, srci-1, srcj,   srci, srcj, Depth);
    fill(mask, srci,   srcj-1, srci, srcj, Depth);
    fill(mask, srci,   srcj+1, srci, srcj, Depth);
    fill(mask, srci+1, srcj,   srci, srcj, Depth);

    mask.at<uchar>(oi, oj) = 255;
    mask.at<uchar>(srci, srcj) = 255;

}

bool SceneFlow::inArea(cv::Mat& image, int srci, int srcj) {
    return srci >= 0 && srci < image.rows
        && srcj >= 0 && srcj < image.cols;
}



void SceneFlow::getNormalized(vector<Vector4d> &vec_Points1, vector<Vector4d> &vec_Points2, Eigen::Matrix<double, 4, 4> &T1, Eigen::Matrix<double, 4, 4> &T2) {

    int num = vec_Points1.size();
    double sum_x1;
    double sum_y1;
    double sum_z1;
    double sum_x2;
    double sum_y2;
    double sum_z2;

    for(int i=0; i<num; i++)
	{
        Vector4d p1 = vec_Points1[i];
        Vector4d p2 = vec_Points2[i];
        // cout << "un p1" << p1.transpose() << endl;

        sum_x1 += p1(0, 0);
        sum_y1 += p1(1, 0);
        sum_z1 += p1(2, 0);

        sum_x2 += p2(0, 0);
        sum_y2 += p2(1, 0);
        sum_z2 += p2(2, 0);
	}

    double avg_x1 = sum_x1 / num;
    double avg_y1 = sum_y1 / num;
    double avg_z1 = sum_z1 / num;

    double avg_x2 = sum_x2 / num;
    double avg_y2 = sum_y2 / num;
    double avg_z2 = sum_z2 / num;

    double sum1;
    double sum2;

    for(int i=0; i<num; i++) {
        Vector4d p1 = vec_Points1[i];
        Vector4d p2 = vec_Points2[i];
        sum1 += sqrt(pow((p1(0, 0) - avg_x1), 2) + pow((p1(1, 0) - avg_y1), 2) + pow((p1(2, 0) - avg_z1), 2));
        sum2 += sqrt(pow((p2(0, 0) - avg_x2), 2) + pow((p2(1, 0) - avg_y2), 2) + pow((p2(2, 0) - avg_z2), 2));
    }

    double s1 = sqrt(3) * num / sum1;
    // cout << "==========================================================s1==" << s1 << endl;
    // cout << "======================================================avg_x1==" << avg_x1 << endl;
    double s2 = sqrt(3) * num / sum2;

    T1 << s1, 0, 0, -s1*avg_x1,
          0, s1, 0, -s1*avg_y1,
          0, 0, s1, -s1*avg_z1,
          0, 0,  0, 1;
    
    T2 << s2, 0, 0, -s2*avg_x2,
          0, s1, 0, -s2*avg_y2,
          0, 0, s2, -s2*avg_z2,
          0, 0,  0, 1;

    for(int i=0; i<num; i++) {
        vec_Points1[i] = T1 * vec_Points1[i];
        vec_Points2[i] = T2 * vec_Points2[i];

        // cout << "norm p1: " << vec_Points1[i].transpose() << endl;
    }


}


void SceneFlow::FinalStage()
{
    LOG(INFO) << "==========[SceneFlow] FinalStage==============";
      // Time evaluation of mask generation

    std::cout << "------Total DynaMask KeyFrame Nums: " << mnTotalSceneFlowFrameNum << std::endl;

    float nTotalTimeMaskGeneration = 0;
    for (size_t i = 0; i < mvTimeMaskGeneration.size(); i++) {
        nTotalTimeMaskGeneration += (float)mvTimeMaskGeneration[i];
    }
    //LOG(INFO) << "------Average time of mask generation: " << nTotalTimeMaskGeneration / mvTimeMaskGeneration.size() * 1000 << " ms";
    std::cout << "Average time of SceneFlow mask generation: " << nTotalTimeMaskGeneration / mvTimeMaskGeneration.size() * 1000 << " ms" << std::endl;
}

// 创建SceneFlow()对象
SceneFlow* SceneFlow::GetInstance()
{
    if (mInstance == nullptr) {
        // mInstance = std::make_shared<Semantic>();
        mInstance = new dyna::SceneFlow();
    }
    return mInstance;
}


SceneFlow* SceneFlow::mInstance = nullptr;  //???
}