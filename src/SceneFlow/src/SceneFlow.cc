#include <iostream> 
#include <algorithm>
#include <vector> 
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

#include "/mnt/SceneFlow/include/SceneFlow.h"

using namespace std;
using namespace cv;
using namespace Eigen; 
using namespace dyna;

SceneFlow::SceneFlow(const double _alpha, const double _beta): alpha(_alpha), beta(_beta)
{

}

SceneFlow::~SceneFlow(){

}

void SceneFlow::GetCleanFlow(const cv::Mat& optical_uv, cv::Mat& optical_flow_img) {

    cv::Mat optical = cv::Mat::zeros(480, 640, CV_32FC1);
    cv::Mat optical_img = cv::Mat::zeros(480, 640, CV_16UC1);
    vector<float> vec_flow;
    vector<float> sort_vec_flow;
    float sum = 0;
    float avg = 0;
    for(int i=0; i<optical_uv.rows; i++) {
        for(int j=0; j<optical_uv.cols; j++) {
            if(    !cvIsNaN(optical_uv.at<Vec2f>(i, j)[1])      && !cvIsNaN(optical_uv.at<Vec2f>(i, j)[0])
                && !cvIsInf(optical_uv.at<Vec2f>(i, j)[1])      && !cvIsInf(optical_uv.at<Vec2f>(i, j)[0])
                && abs(optical_uv.at<Vec2f>(i, j)[1])<1.0e+2    && abs(optical_uv.at<Vec2f>(i, j)[0])<1.0e+2
                && abs(optical_uv.at<Vec2f>(i, j)[1])>1.0e-5    && abs(optical_uv.at<Vec2f>(i, j)[0])>1.0e-5
            ) {
            
                float vy = optical_uv.at<Vec2f>(i, j)[1];   // get velocity of pixel 1-vy-v-i-h 0-vx-u-j-w
                float vx = optical_uv.at<Vec2f>(i, j)[0];
                float flow = sqrt(vx * vx + vy * vy);
                optical.at<float>(i, j) = flow;
                vec_flow.push_back(flow);
                sum += flow;
            
            }
        }
    }

    int num = vec_flow.size();
    float median = (vec_flow[num / 2] + vec_flow[num / 2 - 1]) / 2;

    sort_vec_flow = vec_flow;
    sort(sort_vec_flow.begin(), sort_vec_flow.end());
    float min = sort_vec_flow[0];
    for(int i=0; i<sort_vec_flow.size(); i++) {
        cout << sort_vec_flow[i] << endl;
    }

    avg = sum / (optical_uv.rows * optical_uv.cols);
    cout << "------------min-------------" << endl;
    cout << min << endl;
    cout << "-----------median-------------" << endl;
    cout << median << endl;
    cout << "------------avg--------------" << endl;
    cout << avg << endl;

    // for(int i=0; i<optical_uv.rows; i++) {
    //     for(int j=0; j<optical_uv.cols; j++) { 
    //             // float a = abs(optical.at<float>(i, j) - min);
    //             // if(a > min) {
    //                 float b = abs(optical.at<float>(i, j));
    //                 // if(optical.at<float>(i, j) < median) {
    //                 //     optical_flow.at<Vec2f>(i, j)[1] = optical_uv.at<Vec2f>(i, j)[1];
    //                 //     optical_flow.at<Vec2f>(i, j)[0] = optical_uv.at<Vec2f>(i, j)[0];
    //                     optical_flow_img.at<uchar>(i, j) = 1;
    //                     cout << "row: " << i << " " << "clo: " << j<< endl;
    //                     optical_img.at<uchar>(i, j) = b;
    //                 // }
    //             // }


    //     }
    // }

    optical.convertTo(optical_img, CV_16UC1, 3000.0);
    cv::imwrite("/mnt/SceneFlow/output/flow.png", optical_img);

}

void SceneFlow::GetSceneFlow(const cv::Mat& imDepth1, const cv::Mat& imDepth2, const cv::Mat& optical_uv, cv::Mat& binary_Mask, cv::Mat& projection_error_Mask, cv::Mat& projection_error_img) {
    LOG(INFO) << "---GetSceneFlow";
    assert(!imDepth1.empty());
    assert(imDepth1.channels() == 1);

    assert(!optical_uv.empty());
    assert(optical_uv.channels() == 2);

    double best_th = 0;
    cv::Mat flag = cv::Mat::ones(480, 640, CV_8U);


    // LOG(INFO) << "------getHomography_RANSAC";
    // Eigen::Matrix4d Homo = getHomography_RANSAC(imDepth1, imDepth2, optical_uv, best_th);   // get homo
    // cout << "best Homo: " << Homo<< endl;
    // Vector4d error4;

    LOG(INFO) << "------getRt_RANSAC";
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Zero();
    Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
    getRt_RANSAC(imDepth1, imDepth2, optical_uv, best_R, best_t, best_th);
    
    // for each pixel
    cv::Mat projection_error = cv::Mat(480, 640, CV_32FC1, Scalar(0));
    Vector2d error2;
    Vector3d error3;
    Vector3d t;
    vector<float> vec_error;
    for(int i=0; i<imDepth1.rows; i++) {
        for(int j=0; j<imDepth1.cols; j++) {
            if(imDepth1.at<float>(i, j)>1e-3 && imDepth1.at<float>(i, j)<6) {
                int x1 = j;
                int y1 = i;
                double z1 = imDepth1.at<float>(i, j);
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
                if(   !cvIsNaN(optical_uv.at<Vec2f>(ry2, rx2)[1])      && !cvIsNaN(optical_uv.at<Vec2f>(ry2, rx2)[0])
                   && !cvIsInf(optical_uv.at<Vec2f>(ry2, rx2)[1])      && !cvIsInf(optical_uv.at<Vec2f>(ry2, rx2)[0])
                   && abs(optical_uv.at<Vec2f>(ry2, rx2)[1])<1.0e+2    && abs(optical_uv.at<Vec2f>(ry2, rx2)[0])<1.0e+2
                   && abs(optical_uv.at<Vec2f>(ry2, rx2)[1])>1.0e-5    && abs(optical_uv.at<Vec2f>(ry2, rx2)[0])>1.0e-5) {
                    vy = optical_uv.at<Vec2f>(ry2, rx2)[1];   // get velocity of pixel 1-vy-v-i-h 0-vx-u-j-w
                    vx = optical_uv.at<Vec2f>(ry2, rx2)[0];
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
                    if(imDepth2.at<float>(dy0, dx0)>1e-3 && imDepth2.at<float>(dy0, dx1)>1e-3
                    && imDepth2.at<float>(dy1, dx0)>1e-3 && imDepth2.at<float>(dy1, dx1)>1e-3
                    && imDepth2.at<float>(dy0, dx0)< 6.0 && imDepth2.at<float>(dy0, dx1)< 6.0
                    && imDepth2.at<float>(dy1, dx0)< 6.0 && imDepth2.at<float>(dy1, dx1)< 6.0 ) {
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

                }else { // Todo: 运动像素离开边界
                        if(imDepth2.at<float>(ry2, rx2)>1e-3 && cvIsNaN(imDepth2.at<float>(ry2, rx2)) && !cvIsInf(imDepth2.at<float>(ry2, rx2))) {
                            z2 = imDepth2.at<float>(ry2, rx2);
                        }else {
                            z2 = 0;
                    }
                }
                
                if(z2!= 0) {
                    double mz1 = (double) mu * z1;
                    double mz2 = (double) mu * z2;
                    // double mz1 = (double) z1 * z1;
                    // double mz2 = (double) d2 * d2;

                    vz = (float)(abs(mz2 - mz1));

                    double x1c = ((x1 - camera_cx) / camera_fx ) * z1;
                    double y1c = ((y1 - camera_cy) / camera_fy ) * z1;
                    double x2c = ((x2 - camera_cx) / camera_fx ) * z2;
                    double y2c = ((y2 - camera_cy) / camera_fy ) * z2;

                    // double x1c = x1-camera_cx;
                    // double y1c = y1-camera_cy;
                    // double x2c = x2-camera_cx;
                    // double y2c = y2-camera_cy;

                    //-----reprojection error
                    Vector3d p2c_e;
                    Vector3d p1c;
                    // p1c << ((x1 - camera_cx) / camera_fx ) * z1, 
                    //        ((y1 - camera_cy) / camera_fy ) * z1, 
                    //          z1;
                    p1c <<   x1,
                             y1,
                             mz1;
                    best_t << best_t(0, 0) * fx / best_t(2, 0), best_t(1, 0) * fy / best_t(2, 0), 1 / best_t(2, 0);
                    p2c_e = best_R * p1c + best_t;
                    cout << "==========" << endl;
                    cout << "p2c_e: " << p2c_e.transpose() << endl;
                    cout << "camera_fx: " << camera_fx << endl;
                    cout << "camera_cx: " << camera_cx << endl;
                    cout << "z1: " << z1 << endl;
                    cout << "x2: " << x2 << endl;
                    cout << "y2: " << y2 << endl;
                    // cout << "p2c_e(0, 0) / z1 * camera_fx + camera_cx: " << (p2c_e(0, 0) / z1 * camera_fx + camera_cx) << endl;
                    // cout << "p2c_e(1, 0) / z1 * camera_fy + camera_cy: " << (p2c_e(1, 0) / z1 * camera_fy + camera_cy) << endl;
                 
                    // error3 << p2c_e(0, 0) / z1 * camera_fx + camera_cx - x2,
                    //           p2c_e(1, 0) / z1 * camera_fy + camera_cy - y2,
                    //           mu * p2c_e(2, 0) - mz2;

                    // error3 << p2c_e(0, 0) - x2,
                    //           p2c_e(1, 0) - y2,
                    //           p2c_e(2, 0) - mz2;

                    error3 << best_R(0, 0) * x1 + best_R(0, 1) * y1 + best_R(0, 2) * mz1 + best_t(0, 0) - x2,
                              best_R(1, 0) * x1 + best_R(1, 1) * y1 + best_R(1, 2) * mz1 + best_t(1, 0) - y2,
                              best_R(2, 0) * x1 + best_R(2, 1) * y1 + best_R(2, 2) * mz1 + best_t(2, 0) - mz2;

                              
                    // error3 << best_R(0, 0) * (x1-camera_cx) + best_R(0, 1) * (y1-camera_cy) + best_R(0, 2) * mz1 + best_t(0, 0) - (x2-camera_cx),
                    //           best_R(1, 0) * (x1-camera_cx) + best_R(1, 1) * (y1-camera_cy) + best_R(1, 2) * mz1 + best_t(1, 0) - (y2-camera_cy),
                    //           best_R(2, 0) * (x1-camera_cx) + best_R(2, 1) * (y1-camera_cy) + best_R(2, 2) * mz1 + best_t(2, 0) - mz2;
                    cout << "error3: " << error3.transpose() << endl;

                    // error4 << Homo(0, 0) * x1 + Homo(0, 1) * y1 + Homo(0, 2) * mz1 + Homo(0, 3) * 1 - x2,
                    //           Homo(1, 0) * x1 + Homo(1, 1) * y1 + Homo(1, 2) * mz1 + Homo(1, 3) * 1 - y2,
                    //           Homo(2, 0) * x1 + Homo(2, 1) * y1 + Homo(2, 2) * mz1 + Homo(2, 3) * 1 - mz2,
                    //           Homo(3, 0) * x1 + Homo(3, 1) * y1 + Homo(3, 2) * mz1 + Homo(3, 3) * 1 - 1,
                    // cout << "error4: " << error4.transpose() << endl;
                    // error3 << error4(0, 0) / error4(3, 0), error4(1, 0) / error4(3, 0), error4(2, 0) / error4(3, 0);

                    float error = (float)  error3.norm();
                    projection_error.at<float>(i, j) = error;
                    vec_error.push_back(error);
                    
                    // cout << "error3_Rt: " << error3.transpose() << endl;
                    cout << "error3_Rt.norm(): " << error << endl;
                    cout << "best_th: " << best_th << endl;
                    cout << "==========" << endl;
                    
                }    // end if(z2!= 0)

            }

        }
    } //end for

    vector<float> sort_vec_error = vec_error;
    sort(sort_vec_error.begin(), sort_vec_error.end());
    int num = sort_vec_error.size();
    float median;
    for(int i=0; i<sort_vec_error.size(); i++) {
        cout << "sort_vec_error: " << sort_vec_error[i] << endl;
    }
    if(num / 2 == 0) {
        median = (sort_vec_error[num / 2] + sort_vec_error[num / 2 - 1]) / 2;
    }else {
        median = sort_vec_error[(num-1) / 2];
    }
    cout << "*****median: " << median << endl;

    //-----projection_error_Mask and binary_Mask
    for(int i=0; i<projection_error.rows; i++) {
        for(int j=0; j<projection_error.cols; j++) {
            if (projection_error.at<float>(i, j) >  median + beta) {

                binary_Mask.at<uchar>(i, j) = 255;    // mask-255-white
                projection_error_Mask.at<Vec4b>(i, j)[0] = 0;    //B
                projection_error_Mask.at<Vec4b>(i, j)[1] = 0;    //G
                projection_error_Mask.at<Vec4b>(i, j)[2] = 255;   //R
                projection_error_Mask.at<Vec4b>(i, j)[3] = 255;
            }
        }
    }

    projection_error.convertTo(projection_error_img, CV_16UC1, 2500);     // 50 300 5000 
}

void SceneFlow::getRt_RANSAC(const cv::Mat& imDepth1, const cv::Mat& imDepth2, const cv::Mat& optical_uv, Eigen::Matrix3d& best_R, Eigen::Vector3d& best_t, double & best_th) {
    LOG(INFO) << "---getRt";
    int iterations = 0;
    vector<int> best_inliers_index = {};
    // ---RANSAC
    while(iterations < n_iterations) {
		LOG(INFO) << "Current iteration: " << iterations;
        vector<Vector4d> total_vec_Points1 = {};
        vector<Vector4d> total_vec_Points2 = {};
        vector<Vector4d> total_plane_Points1 = {};
        vector<Vector4d> total_plane_Points2 = {};

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

                for(int num=0; num<10; num++) {   //随机10个数，不重复
                    int rx1 = distribw(gen);
                    int ry1 = distribh(gen);

                    int x1 = (j / w) * w + rx1;
                    int y1 = (i / h) * h + ry1;

                    if(imDepth1.at<float>(y1, x1)>1e-3 && imDepth1.at<float>(y1, x1)<6 && flag.at<uchar>(y1, x1)==1) {
                        double z1 = imDepth1.at<float>(y1, x1);
                        vector<Vector4d> Points = getunnormalizedPoints(x1, y1, z1, flag, imDepth2, optical_uv);
                        if(!Points.empty()) {
                            total_vec_Points1.push_back(Points[0]);
                            total_vec_Points2.push_back(Points[1]);
                            total_plane_Points1.push_back(Points[2]);
                            total_plane_Points2.push_back(Points[3]);
                        }
                    }

                }   //end for 10
            }
        }   // end for 160*160

        std::random_device rd_120;
        std::mt19937 gen_120(rd_120());
        std::uniform_int_distribution<int32_t> distribw_120(0, W);
        std::uniform_int_distribution<int32_t> distribh_120(0, H);
        while(total_vec_Points1.size() < 120) {
            int x1 = distribw_120(gen_120);
            int y1 = distribh_120(gen_120);
            if(imDepth1.at<float>(y1, x1)>1e-3 && imDepth1.at<float>(y1, x1)<6 && flag.at<uchar>(y1, x1)==1) {
                double z1 = imDepth1.at<float>(y1, x1);
                vector<Vector4d> Points = getunnormalizedPoints(x1, y1, z1, flag, imDepth2, optical_uv);
                if(!Points.empty()) {
                    total_vec_Points1.push_back(Points[0]);
                    total_vec_Points2.push_back(Points[1]);
                    total_plane_Points1.push_back(Points[2]);
                    total_plane_Points2.push_back(Points[3]);
                    flag.at<uchar>(y1, x1) += 1;
                }
            }
        } // end while 120
        
        // minimum sample of five points
        std::random_device rd1;
        std::mt19937 gen1(rd1());
        std::uniform_int_distribution<int32_t> distrib(0, 120);
        vector<int> vflag(120, 0);
        vector<int32_t> vector_sample_idx = {};
        while (vector_sample_idx.size() != sample_n) {       //
            // Generate a random index between [0, 120]
            int32_t random_idx = distrib(gen1);

            vflag[random_idx] += 1;
            double Vx = abs(total_vec_Points2[random_idx](0, 0) - total_vec_Points1[random_idx](0, 0));
            double Vy = abs(total_vec_Points2[random_idx](1, 0) - total_vec_Points1[random_idx](1, 0));
            double Vz = abs(total_vec_Points2[random_idx](2, 0) - total_vec_Points1[random_idx](2, 0));
            if (vflag[random_idx] ==1 && Vx<5 && Vy<5) {     //&& Vx<0.3 && Vy<0.3   && Vz<2
                vector_sample_idx.push_back(random_idx);

                cout << random_idx << endl;                                    // ---show vector_sample
                cout << total_vec_Points1[random_idx].transpose() << endl;
                cout << total_vec_Points2[random_idx].transpose() << endl;
                cout << "=========================" << endl;
            }
            else{
                cout << total_vec_Points1[random_idx].transpose() << endl;
                cout << total_vec_Points2[random_idx].transpose() << endl;
                cout << "++++++++++++++++++++++++" << endl;
            }
        }

        // ---------five points caculate Rt
        LOG(INFO) << "-----CalculateICP";
        Eigen::Matrix3d R21 = Eigen::Matrix3d::Zero();
        Eigen::Vector3d t = Eigen::Vector3d::Zero();
        CalculateICP(total_vec_Points1, total_vec_Points2, vector_sample_idx, R21, t);
        cout << "R21 = " << endl;
        cout << R21 << endl;
        cout << "t = " << t.transpose() << endl;

        // --------get threshold
        Eigen::Vector3d error3;
        vector<double> error_norm2;  // cout error norm2

        for(int i=0; i<total_vec_Points1.size(); i++) {    
            Vector3d p1;
            Vector3d p2;
            // p1 << total_plane_Points1[i](0, 0), total_plane_Points1[i](1, 0), total_plane_Points1[i](2, 0);
            // p2 << total_plane_Points2[i](0, 0), total_plane_Points2[i](1, 0), total_plane_Points2[i](2, 0);

            // error3 << R21(0, 0) * p1(0, 0) + R21(0, 1) * p1(1, 0) + R21(0, 2) * p1(2, 0) + best_t(0, 0) * camera_fx * p1(2, 0) - p2(0, 0),
            //           R21(1, 0) * p1(0, 0) + R21(1, 1) * p1(1, 0) + R21(1, 2) * p1(2, 0) + best_t(1, 0) * camera_fy * p1(2, 0) - p2(1, 0),
            //           R21(2, 0) * p1(0, 0) + R21(2, 1) * p1(1, 0) + R21(2, 2) * p1(2, 0) + best_t(2, 0) * p1(2, 0) - p2(2, 0);
            
            p1 << total_vec_Points1[i](0, 0), total_vec_Points1[i](1, 0), total_vec_Points1[i](2, 0);
            p2 << total_vec_Points2[i](0, 0), total_vec_Points2[i](1, 0), total_vec_Points2[i](2, 0);
            error3 = R21 * p1 + t - p2;

            double error = error3.norm();
            // cout << "error3_Rt: " << error3.transpose() << endl;
            // cout << "error3_Rt.norm(): " << error << endl;
            // cout << "&&&&&&&&&&&&&&&&" << endl;  
            

            error_norm2.push_back(error);  
        }
        vector<double> sort_error_norm2 = error_norm2;
        sort(sort_error_norm2.begin(), sort_error_norm2.end());

        // for(int i=0; i<total_vec_Points1.size(); i++) {    // ---show error_norm2
        //     cout << error_norm2[i]<< endl;
        // }
        for(int i=0; i<sort_error_norm2.size(); i++) {    // ---show sort_error_norm2
            cout << "after caculate Rt sort_error_norm2: " << i << " " ;
            cout << sort_error_norm2[i]<< endl;    
        }

        int num_e = sort_error_norm2.size();
        double median = (sort_error_norm2[num_e / 2] + sort_error_norm2[num_e / 2 - 1]) / 2;
        double quart = sort_error_norm2[num_e / 4];
        
        // count standard deviation σ
        double powAll = 0.0;
        for (int i = 0; i < num_e; i++)
        {
            powAll += pow((error_norm2[i] - median), 2);
        }
        double sd = sqrt(powAll / (double)num_e);
        cout << "sd: " << sd << endl;
        
        double th = 0;
        if(sd >= alpha){    // alpha = 1.5
            th = sd;
        }else {
            th = alpha;
        }
        th = alpha;
        cout << "median: " << median << endl;
        cout << "th: " << th << endl;

        //-------count the number of inliers
        vector<int> current_inliers_index = {};
        for(int i=0; i<error_norm2.size(); i++) {
            if(error_norm2[i] < th) {
                current_inliers_index.push_back(i);
            }
        }
        cout << "********************************" << endl;
        cout << "current_inliers_index.size: " << current_inliers_index.size() << endl;
        cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;
        cout << "********************************" << endl;

        if(current_inliers_index.size() >= 5 && current_inliers_index.size() > best_inliers_index.size()) {  // if current_inliers_index.size() is bigger, update best
            best_inliers_index.swap(current_inliers_index);
            cout << "===============================" << endl;
            Eigen::Matrix3d good_R =  Eigen::Matrix3d::Zero();
            Eigen::Vector3d good_t = Eigen::Vector3d::Zero();
            CalculateICP(total_vec_Points1, total_vec_Points2, best_inliers_index, good_R, good_t);  
            best_th = th;
            best_R = good_R;
            best_t = good_t;

            cout << "good_R = " << endl;
            cout << best_R << endl;
            cout << "good_t = " << best_t.transpose() << endl;
            cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;
            cout << "===============================" << endl;
        }
        
        total_vec_Points1.clear();
        total_vec_Points2.clear();
        total_plane_Points1.clear();
        total_plane_Points2.clear();

        total_vec_Points1.resize(0);
        total_vec_Points2.resize(0);
        total_plane_Points1.resize(0);
        total_plane_Points2.resize(0);

        iterations++;

    }  // end while iteration

    
    for(int i=0; i< best_inliers_index.size(); i++) {
            cout << best_inliers_index[i] << " | ";
    }
    cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;

    best_inliers_index.clear();
    best_inliers_index.resize(0);

    cout << "best_R = " << endl;
    cout << best_R << endl;
    cout << "best_t = " << best_t.transpose() << endl;

    LOG(INFO) << "---finish getRt_RANSAC";
}

void SceneFlow::CalculateICP(vector<Vector4d>& total_vec_Points1, vector<Vector4d>& total_vec_Points2, vector<int>& index, Eigen::Matrix3d& R, Eigen::Vector3d& t) {

    cout << "index_size: " << index.size() << endl;;

    vector<Vector4d> vec_Points1 = get_vector_Points(total_vec_Points1, index);
    vector<Vector4d> vec_Points2 = get_vector_Points(total_vec_Points2, index);

    // for(int i=0; i<vec_Points1.size(); i++) {
    //     cout << "un Points1: " << vec_Points1[i].transpose() << endl;
    // }

    Eigen::Matrix<double, 4, 4> T1;
    Eigen::Matrix<double, 4, 4> T2;
    int N = index.size();
    double sum_x1;
    double sum_y1;
    double sum_z1;
    double sum_x2;
    double sum_y2;
    double sum_z2;
	
    vector<Vector3d> pts1(N);
    vector<Vector3d> pts2(N);
    //========get T
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

        sum_x1 += p1(0, 0);
        sum_y1 += p1(1, 0);
        sum_z1 += p1(2, 0);

        sum_x2 += p2(0, 0);
        sum_y2 += p2(1, 0);
        sum_z2 += p2(2, 0);
	}

    double avg_x1 = sum_x1 / N;
    double avg_y1 = sum_y1 / N;
    double avg_z1 = sum_z1 / N;

    double avg_x2 = sum_x2 / N;
    double avg_y2 = sum_y2 / N;
    double avg_z2 = sum_z2 / N;

    double sum1;
    double sum2;

    for(int i=0; i<N; i++) {
        Vector4d p1 = vec_Points1[i];
        Vector4d p2 = vec_Points2[i];
        sum1 = sqrt(pow((p1(0, 0) - avg_x1), 2) + pow((p1(1, 0) - avg_y1), 2) + pow((p1(2, 0) - avg_z1), 2));
        sum2 = sqrt(pow((p2(0, 0) - avg_x2), 2) + pow((p2(1, 0) - avg_y2), 2) + pow((p2(2, 0) - avg_z2), 2));
    }

    double s1 = sqrt(3) * N / sum1;
    double s2 = sqrt(3) * N / sum2;

    T1 << s1, 0, 0, -s1*avg_x1,
          0, s1, 0, -s1*avg_y1,
          0, 0, s1, -s1*avg_z1,
          0, 0,  0, 1;
    
    T2 << s2, 0, 0, -s2*avg_x2,
          0, s1, 0, -s2*avg_y2,
          0, 0, s2, -s2*avg_z2,
          0, 0,  0, 1;


    // 1 cordinate
    Vector3d m1;    // center of mass
    Vector3d m2;
    for(int i=0; i<N; i++) {
        // cout << "Points1: " << (T1 * vec_Points1[i]).transpose() << endl;

        m1 += pts1[i];      //T1 * 
        m2 += pts2[i];
    }
    m1 = m1 / N;
    m2 = m2 / N;
    vector<Vector3d> q1(N), q2(N);  // remove the center
    for(int i=0; i<N; i++) {
        q1[i] = pts1[i] - m1;
        q2[i] = pts2[i] - m2;
    }

    // 2 W=sum(q2*q1^T)
    Eigen::Matrix3d W = Eigen::Matrix3d::Zero();
    for (int i = 0; i<N; i++)
	{
		W += Eigen::Vector3d(q2[i](0, 0), q2[i](1, 0), q2[i](2, 0)) * Eigen::Vector3d(q2[i](0, 0), q2[i](1, 0), q2[i](2, 0)).transpose();
	}
	cout << "W = " << W << endl;

    // 3 SVD on W
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(W, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // 4 R=U*V^T
    Eigen::Matrix3d R_;
    Eigen::Vector3d t_;
    R_ = U * (V.transpose());
    t_ = m2 - R * m1;

    R = R_;
    t << t_(0, 0)-camera_cx, t_(1, 0)-camera_cy, t_(2, 0);

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
        if(imDepth2.at<float>(dy0, dx0)>1e-3 && imDepth2.at<float>(dy0, dx1)>1e-3
        && imDepth2.at<float>(dy1, dx0)>1e-3 && imDepth2.at<float>(dy1, dx1)>1e-3
        // && imDepth2.at<float>(dy0, dx0)< 6.0 && imDepth2.at<float>(dy0, dx1)< 6.0
        // && imDepth2.at<float>(dy1, dx0)< 6.0 && imDepth2.at<float>(dy1, dx1)< 6.0
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
            if(imDepth2.at<float>(ry2, rx2)>1e-3  && !cvIsNaN(imDepth2.at<float>(ry2, rx2)) && !cvIsInf(imDepth2.at<float>(ry2, rx2))) {
                z2 = imDepth2.at<float>(ry2, rx2);
            }else {
                z2 = 0;
        }
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

        Points.push_back(Vector4d((double) x1, (double) y1, mz1, 1.0));
        Points.push_back(Vector4d((double) x2, (double) y2, mz2, 1.0));    // sample coordinate to plane

        Points.push_back(Vector4d((double) (x1 / z1), (double) (y1/z1), mz1, 1.0));
        Points.push_back(Vector4d((double) (x2 / z2), (double) (y2/z2), mz2, 1.0));    // sample coordinate to camera



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

void SceneFlow::getNormalized(vector<Vector4d> &total_vec_Points1, vector<Vector4d> &total_vec_Points2, Eigen::Matrix<double, 4, 4> &T1, Eigen::Matrix<double, 4, 4> &T2) {

    int num = total_vec_Points1.size();
    double sum_x1;
    double sum_y1;
    double sum_z1;
    double sum_x2;
    double sum_y2;
    double sum_z2;

    for(int i=0; i<num; i++)
	{
        Vector4d p1 = total_vec_Points1[i];
        Vector4d p2 = total_vec_Points2[i];
        cout << "un p1" << p1.transpose() << endl;

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
        Vector4d p1 = total_vec_Points1[i];
        Vector4d p2 = total_vec_Points2[i];
        sum1 += sqrt(pow((p1(0, 0) - avg_x1), 2) + pow((p1(1, 0) - avg_y1), 2) + pow((p1(2, 0) - avg_z1), 2));
        sum2 += sqrt(pow((p2(0, 0) - avg_x2), 2) + pow((p2(1, 0) - avg_y2), 2) + pow((p2(2, 0) - avg_z2), 2));
    }

    double s1 = sqrt(3) * num / sum1;
    cout << "==========================================================s1==" << s1 << endl;
    cout << "======================================================avg_x1==" << avg_x1 << endl;
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
        total_vec_Points1[i] = T1 * total_vec_Points1[i];
        total_vec_Points2[i] = T2 * total_vec_Points2[i];

        cout << "norm p1: " << total_vec_Points1[i].transpose() << endl;
    }


}


//===========RANSAC计算Homo=============================
Eigen::Matrix4d SceneFlow::getHomography_RANSAC(const cv::Mat& imDepth1, const cv::Mat& imDepth2, const cv::Mat& optical_uv, double & best_th) {
    LOG(INFO) << "---getHomography_RANSAC";

    Matrix4d best_matrix_H = Matrix4d::Identity();
    
    int iterations = 0;
    vector<int> best_inliers_index = {};
    
    while(iterations < n_iterations) {
		LOG(INFO) << "Current iteration: " << iterations;
        vector<Vector4d> total_vec_Points1 = {};
        vector<Vector4d> total_vec_Points2 = {};
        vector<Vector4d> sample_vec_normal_Points1 = {};
        vector<Vector4d> sample_vec_normal_Points2 = {};
        vector<Vector4d> sample_vec_Points1 = {};
        vector<Vector4d> sample_vec_Points2 = {};

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

                for(int num=0; num<10; num++) {   //随机10个数，不重复
                    int rx1 = distribw(gen);
                    int ry1 = distribh(gen);

                    int x1 = (j / w) * w + rx1;
                    int y1 = (i / h) * h + ry1;

                    if(imDepth1.at<float>(y1, x1)>1e-3 && imDepth1.at<float>(y1, x1)<6 && flag.at<uchar>(y1, x1)==1) {
                        double z1 = imDepth1.at<float>(y1, x1);
                        vector<Vector4d> Points = getunnormalizedPoints(x1, y1, z1, flag, imDepth2, optical_uv);
                        if(!Points.empty()) {
                            total_vec_Points1.push_back(Points[2]);
                            total_vec_Points2.push_back(Points[3]);
                        }
                    }

                }   //end for 10
            }
        }   // end for 160*160

        std::random_device rd_120;
        std::mt19937 gen_120(rd_120());
        std::uniform_int_distribution<int32_t> distribw_120(0, W);
        std::uniform_int_distribution<int32_t> distribh_120(0, H);
        while(total_vec_Points1.size() < 120) {
            int x1 = distribw_120(gen_120);
            int y1 = distribh_120(gen_120);
            if(imDepth1.at<float>(y1, x1)>1e-3 && imDepth1.at<float>(y1, x1)<6 && flag.at<uchar>(y1, x1)==1) {
                double z1 = imDepth1.at<float>(y1, x1);
                vector<Vector4d> Points = getunnormalizedPoints(x1, y1, z1, flag, imDepth2, optical_uv);
                if(!Points.empty()) {
                    total_vec_Points1.push_back(Points[2]);
                    total_vec_Points2.push_back(Points[3]);
                    flag.at<uchar>(y1, x1) += 1;
                }
            }
        } // end while 120

        // Eigen::Matrix<double, 4, 4> T1, T2;
        // getNormalized(total_vec_Points1, total_vec_Points1, T1, T2);

        // LOG(INFO) << "show total_vec_Points";
        // for(int i=0; i<total_vec_Points1.size(); i++) {
        //     cout << i << ":" << endl;
        //     cout << total_vec_Points1[i].transpose() << endl;
        //     cout << total_vec_Points2[i].transpose() << endl;
        //     cout << "--------------------------" << endl;
        // }
        
        // minimum sample of five points
        std::random_device rd1;
        std::mt19937 gen1(rd1());
        std::uniform_int_distribution<int32_t> distrib(0, 120);
        vector<int> vflag(120, 0);
        vector<int32_t> vector_sample_idx = {};
        while (vector_sample_idx.size() != sample_n) {       //
            // Generate a random index between [0, 120]
            int32_t random_idx = distrib(gen1);

            Eigen::Vector4d xy1 = total_vec_Points1[random_idx];
            Eigen::Vector4d xy2 = total_vec_Points2[random_idx];

            Eigen::Vector2d vel;
            vel(0, 0) = abs(xy2(0, 0) - xy1(0, 0));
            vel(1, 0) = abs(xy2(1, 0) - xy1(1, 0));

            if (vel(0, 0) < 3 && vel(1, 0) < 3) { // optical flow velocity constraint
                vflag[random_idx] += 1;
                if (vflag[random_idx] ==1) {
                    vector_sample_idx.push_back(random_idx);

                    cout << random_idx << endl;             // ---show vector_sample
                    cout << total_vec_Points1[random_idx].transpose() << endl;
                    cout << total_vec_Points2[random_idx].transpose() << endl;
                    cout << "=========================" << endl;
                }
            } else {
                 cout << "+++++++++++++++++++" << endl;
                 cout << vel.transpose() << endl;
            }
    

        }

        // ---------five points caculate Homography
        LOG(INFO) << "-----CalculateHomoMatrix";
        Eigen::Matrix<double, 4, 4> matrix_H = CalculateH(total_vec_Points1, total_vec_Points2, vector_sample_idx);
        // Eigen::Matrix<double, 4, 4> Homo = T2.inverse() * matrix_H  * T1;
        

        // --------get threshold
        Eigen::Vector4d error4;
        Eigen::Vector3d error3;
        vector<double> error_norm2;  // cout error norm2
        for(int i=0; i<total_vec_Points1.size(); i++) {     
            Eigen::Vector4d p21 = matrix_H * total_vec_Points1[i];
            // p21 << p21(0, 0) / p21(3, 0), p21(1, 0) / p21(3, 0), p21(2, 0) / p21(3, 0), 1.0;
            p21 << p21(0, 0), p21(1, 0), p21(2, 0), p21(3, 0);
            error4 = p21 - total_vec_Points2[i];
            error3(0, 0) = error4(0, 0) / error4(3, 0);
            error3(1, 0) = error4(1, 0) / error4(3, 0);
            error3(2, 0) = error4(2, 0) / error4(3, 0);

            error_norm2.push_back(error3.norm());
            cout << "p21: " << p21.transpose() << endl;
            cout << "total_vec_Points1[i]: " << total_vec_Points1[i].transpose() << endl;
            cout << "error3: " << error3.transpose() << endl; 
            cout << "&&&&&&&&&&&&&&&&" << endl;   
        }
        vector<double> sort_error_norm2 = error_norm2;
        sort(sort_error_norm2.begin(), sort_error_norm2.end());

        // for(int i=0; i<total_vec_Points1.size(); i++) {    // ---show error_norm2
        //     cout << error_norm2[i]<< endl;
        // }
        for(int i=0; i<sort_error_norm2.size(); i++) {    // ---show sort_error_norm2
            cout << "五点计算H之后 sort_error_norm2: " << i << " " ;
            cout << sort_error_norm2[i]<< endl;    
        }

        int num_e = sort_error_norm2.size();
        double median = (sort_error_norm2[num_e / 2] + sort_error_norm2[num_e / 2 - 1]) / 2;
        
        // count standard deviation σ
        double powAll = 0.0;
        for (int i = 0; i < num_e; i++)
        {
            powAll += pow((error_norm2[i] - median), 2);
        }
        double sd = sqrt(powAll / (double)num_e);
        cout << "sd: " << sd << endl;
        
        double th =0;
        if(sd >= alpha){    // alpha = 1.5
            th = sd;
        }else {
            th = alpha;
        }
        // th = 300;
        cout << "median: " << median << endl;
        cout << "th: " << th << endl;

        // // -------count the number of inliers
        vector<int> current_inliers_index = {};
        for(int i=0; i<error_norm2.size(); i++) {
            if(error_norm2[i] < th) {
                current_inliers_index.push_back(i);
            }
        }
        cout << "********************************" << endl;
        cout << "current_inliers_index.size: " << current_inliers_index.size() << endl;
        cout << "********************************" << endl;
        cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;
        if(current_inliers_index.size() > best_inliers_index.size()) {  // if current_inliers_index.size() is bigger, update best
            best_inliers_index.swap(current_inliers_index);
            best_matrix_H = CalculateH(total_vec_Points1, total_vec_Points2, best_inliers_index);
            best_th = th;

            cout << "best_th: " << best_th << endl;
            cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;
            cout << "********************************" << endl;
        }
        
        iterations++;

    }  // end while iteration
    for(int i=0; i< best_inliers_index.size(); i++) {
            cout << best_inliers_index[i] << " | ";
    }
    cout << "best_inliers_index.size: " << best_inliers_index.size() << endl;
    cout << "best_matrix_H: " << best_matrix_H << endl;
    best_inliers_index.clear();
    best_inliers_index.resize(0);
    return best_matrix_H;
    LOG(INFO) << "---finish getHomography_RANSAC";
}


Eigen::Matrix<double, 4, 4> SceneFlow::CalculateH(vector<Vector4d>& total_vec_Points1, vector<Vector4d>& total_vec_Points2, vector<int>& index) {

    Eigen::Matrix<double, 4, 4> T1, T2;
    cout << "total_size: " << total_vec_Points1.size() << endl;;

    cout << endl;

    vector<Vector4d> vec_Points1 = get_vector_Points(total_vec_Points1, index);
    vector<Vector4d> vec_Points2 = get_vector_Points(total_vec_Points2, index);

    for(int i=0; i<vec_Points1.size(); i++) {
        cout << "Points1: " << vec_Points1[i].transpose() << endl;
    }

	Eigen::Matrix<double, 16, Dynamic> A;    // Ah=0
    int num = index.size();
    double sum_x1;
    double sum_y1;
    double sum_z1;
    double sum_x2;
    double sum_y2;
    double sum_z2;

    for(int i=0; i<num; i++){
        Vector4d p1 = vec_Points1[i];
        Vector4d p2 = vec_Points2[i];
        cout << "un p1" << p1.transpose() << endl;
        cout << "un p2" << p2.transpose() << endl;

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
    cout << "==========================================================s1==" << s1 << endl;
    cout << "======================================================avg_x1==" << avg_x1 << endl;
    double s2 = sqrt(3) * num / sum2;

    T1 << s1, 0, 0, -s1*avg_x1,
          0, s1, 0, -s1*avg_y1,
          0, 0, s1, -s1*avg_z1,
          0, 0,  0, 1;
    
    T2 << s2, 0, 0, -s2*avg_x2,
          0, s1, 0, -s2*avg_y2,
          0, 0, s2, -s2*avg_z2,
          0, 0,  0, 1;


    for(int i=0; i<index.size(); i++) {
        
        Eigen::Matrix<double, 16, 1> ax;
		Eigen::Matrix<double, 16, 1> ay;
        Eigen::Matrix<double, 16, 1> az;

        Vector4d p1 = T1 * vec_Points1[i];
        Vector4d p2 = T2 * vec_Points2[i];
        cout << "normal p1" << p1.transpose() << endl;
        cout << "normal p2" << p2.transpose() << endl;

        ax << p1(0, 0), p1(1, 0), p1(2, 0), 1., 0., 0., 0., 0.,                   0., 0., 0., 0.,                   -1*p1(0, 0)*p2(0, 0), -1*p1(1, 0)*p2(0, 0), -1*p1(2, 0)*p2(0, 0), -1*p2(0, 0);
		ay << 0., 0., 0., 0.,                   p1(0, 0), p1(1, 0), p1(2, 0), 1., 0., 0., 0., 0.,                   -1*p1(0, 0)*p2(1, 0), -1*p1(1, 0)*p2(1, 0), -1*p1(2, 0)*p2(1, 0), -1*p2(1, 0);
        az << 0., 0., 0., 0.,                   0., 0., 0., 0.,                   p1(0, 0), p1(1, 0), p1(2, 0), 1., -1*p1(0, 0)*p2(2, 0), -1*p1(1, 0)*p2(2, 0), -1*p1(2, 0)*p2(2, 0), -1*p2(2, 0);

        A.conservativeResize( A.rows(), A.cols() + 1);
        A.col( A.cols()-1 ) << ax; //col()是对矩阵的某一列进行操作，后面转置
        A.conservativeResize( A.rows(), A.cols() + 1);
        A.col( A.cols()-1 ) << ay;
        A.conservativeResize( A.rows(), A.cols() + 1);
        A.col( A.cols()-1 ) << az;

    }

    Eigen::Matrix<double, Dynamic, 16> B = A.transpose();

    cout << "B: " << B << endl;

	Eigen::JacobiSVD<Eigen::Matrix<double, Dynamic, 16>> svd(B, Eigen::ComputeFullU | Eigen::ComputeFullV);

	Eigen::Matrix<double, 16, 16> V = svd.matrixV();
	Eigen::Matrix<double, 16, 1> h = V.col(15);     // last column is h

	Eigen::Matrix<double, 4, 4> Homography;
    Eigen::Matrix<double, 4, 4> Homo;


    Homo <<  h(0, 0),  h(1, 0),  h(2, 0), h(3, 0),
             h(4, 0),  h(5, 0),  h(6, 0), h(7, 0),
             h(8, 0),  h(9, 0),  h(10,0), h(11,0),
             h(12, 0), h(13, 0), h(14,0), h(15,0);

    Homo = T2.inverse() * Homo * T1;

    
    // Homo << Homo(0, 0) / Homo(3, 3),  Homo(0, 1) / Homo(3, 3),  Homo(0, 2) / Homo(3, 3), Homo(0, 3) / Homo(3, 3),
    //               Homo(1, 0) / Homo(3, 3),  Homo(1, 1) / Homo(3, 3),  Homo(1, 2) / Homo(3, 3), Homo(1, 3) / Homo(3, 3),
    //               Homo(2, 0) / Homo(3, 3),  Homo(2, 1) / Homo(3, 3),  Homo(2, 2) / Homo(3, 3), Homo(2, 3) / Homo(3, 3),
    //               Homo(3, 0) / Homo(3, 3),  Homo(3, 1) / Homo(3, 3),  Homo(3, 2) / Homo(3, 3), Homo(3, 3) / Homo(3, 3);

    cout << "H: " << Homo << endl;

	return Homo;

}

//================getDepthFlow================
void SceneFlow::getDepthFlow(const cv::Mat& imD1, const cv::Mat& imD2, const cv::Mat& optical, cv::Mat& Vxyz_show)
{
    LOG(INFO) << "getDepthFlow";
    assert(!imD1.empty());
    assert(imD1.channels() == 1);

    assert(!optical.empty());
    assert(optical.channels() == 2);

    cv::Mat depthflow = cv::Mat::zeros(480, 640, CV_32F);
    cv::Mat depthflow_img = cv::Mat::zeros(480, 640, CV_16U);

    cv::Mat opticalflow_vx  = cv::Mat::zeros(480, 640, CV_32F);
    cv::Mat opticalflow_vy = cv::Mat::zeros(480, 640, CV_32F);
    cv::Mat optical_vx = cv::Mat::zeros(480, 640, CV_16U);
    cv::Mat optical_vy = cv::Mat::zeros(480, 640, CV_16U);

    cv::Mat optical_img = cv::Mat::zeros(480, 640, CV_32FC1);


    vector<float> vec_flow;
    vector<float> sort_vec_flow;
    float sum = 0;
    float avg = 0;

    Vector3f Vxyz;

    LOG(INFO) << "------getRt_RANSAC";
    Vector3d Vxyz_Rt;
    Eigen::Matrix3d best_R = Eigen::Matrix3d::Zero();
    Eigen::Vector3d best_t = Eigen::Vector3d::Zero();
    double Th = 0;
    getRt_RANSAC(imD1, imD2, optical, best_R, best_t, Th);

    Eigen::Matrix3f best_Rf;
    best_Rf << (float)best_R(0, 0), (float)best_R(0, 1), (float)best_R(0, 2), 
               (float)best_R(1, 0), (float)best_R(1, 1), (float)best_R(1, 2), 
               (float)best_R(2, 0), (float)best_R(2, 1), (float)best_R(2, 2);
    Eigen::Vector3f best_tf;
    best_tf << (float)best_t(0, 0), (float)best_t(1, 0), (float)best_t(2, 0);


    // LOG(INFO) << "------getHomography_RANSAC";
    // double best_th = 0;
    // Vector3d Vxyz_H;
    // Vector4d p1;
    // Vector4d p2;
    // Matrix4d H = getHomography_RANSAC(imD1, imD2, optical, best_th);


    for(int i=0; i<imD1.rows; i++) {
        for(int j=0; j<imD1.cols; j++) {
            if(imD1.at<float>(i, j)>1e-3 && imD1.at<float>(i, j)<6) {
                int x1 = j;
                int y1 = i;
                double z1 = (double) imD1.at<float>(i, j);
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
                if(   !cvIsNaN(optical.at<Vec2f>(ry2, rx2)[1])      && !cvIsNaN(optical.at<Vec2f>(ry2, rx2)[0])
                   && !cvIsInf(optical.at<Vec2f>(ry2, rx2)[1])      && !cvIsInf(optical.at<Vec2f>(ry2, rx2)[0])
                   && abs(optical.at<Vec2f>(ry2, rx2)[1])<1.0e+2    && abs(optical.at<Vec2f>(ry2, rx2)[0])<1.0e+2
                   && abs(optical.at<Vec2f>(ry2, rx2)[1])>1.0e-5    && abs(optical.at<Vec2f>(ry2, rx2)[0])>1.0e-5) {
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

                if(0<=dx0 && dx0<imD2.cols && 0<=dx1 && dx1<imD2.cols && 0<=dy0 && dy0<imD2.rows && 0<=dy1 && dy1<imD2.rows) {
                    if(imD2.at<float>(dy0, dx0)>1e-3 && imD2.at<float>(dy0, dx1)>1e-3
                    && imD2.at<float>(dy1, dx0)>1e-3 && imD2.at<float>(dy1, dx1)>1e-3
                    && imD2.at<float>(dy0, dx0)< 6.0 && imD2.at<float>(dy0, dx1)< 6.0
                    && imD2.at<float>(dy1, dx0)< 6.0 && imD2.at<float>(dy1, dx1)< 6.0 ) {
                        float d00 = imD2.at<float>(dy0, dx0);
                        float d10 = imD2.at<float>(dy0, dx1);
                        float d01 = imD2.at<float>(dy1, dx0);
                        float d11 = imD2.at<float>(dy1, dx1);

                        z2 = (double)( d00 * (1.0f - fx) * (1.0f - fx) 
                            + d10 * fx * (1.0f - fy) 
                            + d01 * (1.0f - fx) * fy 
                            + d11 * fx * fy);
                    }else {
                        z2 = 0;
                    }

                }else { // Todo: 运动像素离开边界
                        if(imD2.at<float>(ry2, rx2)>1e-3 && cvIsNaN(imD2.at<float>(ry2, rx2)) && !cvIsInf(imD2.at<float>(ry2, rx2))) {
                            z2 = imD2.at<float>(ry2, rx2);
                        }else {
                            z2 = 0;
                    }
                }

                if(z2!= 0) {
                    double mz1 = (double) mu * z1;
                    double mz2 = (double) mu * z2;
                    // double mz1 = (double) z1 * z1;
                    // double mz2 = (double) d2 * d2;

                    vz = (float)(abs(mz2 - mz1));
                  

                    double x1c = ((x1-camera_cx) / camera_fx ) * z1;
                    double y1c = ((y1-camera_cy) / camera_fy ) * z1;

                    double x2c = ((x2-camera_cx) / camera_fx ) * z2;
                    double y2c = ((y2-camera_cy) / camera_fy ) * z2;

                    // double x1 = (x1-camera_cx);
                    // double y1 = (y1-camera_cy);

                    // double x2 = (x2-camera_cx);
                    // double y2 = (y2-camera_cy);

                    Vxyz(0, 0) = vx;
                    Vxyz(1, 0) = vy;
                    Vxyz(2, 0) = vz;


                    Vxyz_Rt << best_R(0, 0) * x1 + best_R(0, 1) * y1 + best_R(0, 2) * mz1 + best_t(0, 0) * camera_fx / z1 - x2,
                               best_R(1, 0) * x1 + best_R(1, 1) * y1 + best_R(1, 2) * mz1 + best_t(1, 0) * camera_fy / z1 - y2,
                               best_R(2, 0) * x1 + best_R(2, 1) * y1 + best_R(2, 2) * mz1 + best_t(2, 0) / z1 - mz2;

                    // cout << "p1: "<< x1 << " "<< y2 << " "<< mz1 << endl;
                    // cout << "tx "<< (best_t(0, 0) * camera_fx / z1) << endl;
                    cout << "Vxyz_Rt: " << Vxyz_Rt.transpose() << endl;
                    cout << "Vxyz_Rt.norm(): " << Vxyz_Rt.norm() << endl;
                    optical_img.at<float>(i, j) = Vxyz_Rt.norm();

                    // Vxyz = best_Rf * Vxyz + best_tf;
                    // optical_img.at<float>(i, j) = Vxyz.norm();

                    // ------Homo
                    // p1 << x1, y1, mz1, 1.0;
                    // p2 << x2, y2, mz2, 1.0;
                    // Vector4d error;
                    // error = H * p1 - p2;
                    // Vxyz_H << error(0, 0), error(1, 0), error(2, 0);
                    // cout << "Vxyz_H: " << Vxyz_H.transpose() << endl;
                    // cout << "Vxyz_H.norm(): " << Vxyz_H.norm() << endl;
                    // optical_img.at<float>(i, j) = Vxyz_H.norm();
                    
                }    


                // if(vz>0.3) {
                    depthflow.at<float>(i, j) = vz;
                // }

                // if(vx>0.3) {
                    opticalflow_vx.at<float>(i, j) = abs(vx);
                // }
                // if(vy>0.3) {
                    opticalflow_vy.at<float>(i, j) = abs(vy);
                // }
                // cout << vx << endl;

                
      
                
            } // end if
        }
    } // end for       
    // cout << depthflow << endl;


    depthflow.convertTo(depthflow_img, CV_16UC1, 8000.0);
    opticalflow_vx.convertTo(optical_vx, CV_16UC1, 8000.0);
    opticalflow_vy.convertTo(optical_vy, CV_16UC1, 8000.0);
    optical_img.convertTo(Vxyz_show, CV_16UC1, 5000.0);     // 5000

    cv::imwrite("/mnt/SceneFlow/output/depthflow.png", depthflow_img);
    cv::imwrite("/mnt/SceneFlow/output/optical_vx.png", optical_vx);
    cv::imwrite("/mnt/SceneFlow/output/optical_vy.png", optical_vy);
    // cv::imwrite("/mnt/SceneFlow/output/optical_show.png", optical_show);

  
}


// 	Eigen::JacobiSVD<Eigen::Matrix<double, 8, 9>> svd(A, Eigen::ComputeFullU | Eigen::ComputeFullV);

// 	Eigen::Matrix<double, 9, 9> V = svd.matrixV();
// 	Eigen::Matrix<double, 9, 1> h = V.col(8);

// 	Eigen::Matrix<double, 3, 3> homography;
// 	homography << h(0,0), h(1,0), h(2,0),
// 				  h(3,0), h(4,0), h(5,0),
// 				  h(6,0), h(7,0), h(8,0);

// 	return homography;

// }


