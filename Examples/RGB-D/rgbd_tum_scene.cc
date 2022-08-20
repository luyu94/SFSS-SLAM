/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>

#include </usr/local/opencv345/include/opencv2/core/core.hpp>
#include <opencv2/core/core.hpp>

#include <System.h>
#include "Common.h"
#include "SlamConfig.h"


#include <glog/logging.h>


using namespace cv;
using namespace std;
using namespace ORB_SLAM2;


void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);

int main(int argc, char **argv)
{
    if(argc < 5)
    {
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // ./Examples/RGB-D/rgbd_tum_scene Vocabulary/ORBvoc.txt Examples/RGB-D/TUM3.yaml /mnt/rgbd_dataset_freiburg3_walking_xyz /mnt/SFSS-SLAM/Examples/RGB-D/associations/fr3_walking_xyz.txt /mnt/SFSS-SLAM/a_output/output_fw3_xyz &> log

    std::cout << "===========================" << std::endl;
    std::cout << "argv[1] path_to_vocabulary: " << argv[1] << std::endl;
    std::cout << "argv[2] path_to_settings: " << argv[2] << std::endl;
    std::cout << "argv[3] path_to_sequence: " << argv[3] << std::endl;
    std::cout << "argv[4] path_to_association: " << argv[4] << std::endl;
    std::cout << "argv[5] result path: " << argv[5] << std::endl;
    std::cout << "===========================" << std::endl;

    if (argc == 6) {
        Config::GetInstance()->IsSaveResult(true);
        Config::GetInstance()->createSavePath(std::string(argv[5]));
    } else {
        Config::GetInstance()->IsSaveResult(false);
    }

    // Retrieve paths to images
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);
    Config::GetInstance()->LoadTUMDataset(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);

    // Check consistency in the number of images and depthmaps
    int nImages = vstrImageFilenamesRGB.size();
    if(vstrImageFilenamesRGB.empty())
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,false);

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(nImages);

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
    cv::Mat imMask;
    cv::Mat imSceneMask;
    for(int ni=0; ni<nImages-1; ni++)
    {
        // Read image and depthmap from file
        imRGB =       cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD =         cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);
        imMask =      cv::imread(string(argv[3])+"/results/"+ to_string(vTimestamps[ni]) + ".png", CV_LOAD_IMAGE_UNCHANGED);  //CV_8U   0
        imSceneMask = cv::imread(string(argv[3])+"/Floodfill/"+ to_string(vTimestamps[ni]) + ".png", CV_LOAD_IMAGE_UNCHANGED); //CV_8U  0

        cv::Mat struct1 = getStructuringElement(0, Size(13, 13));   // 0 rect 1 cross
        cv::dilate(imMask, imMask, struct1);
        double tframe = vTimestamps[ni];

        cv::Mat struct2 = getStructuringElement(0, Size(13, 13));   // 0 rect 1 cross
        cv::dilate(imSceneMask, imSceneMask, struct2);

        // LOG(INFO) << "imMask: " << imMask.type();
        // LOG(INFO) << "imSceneMask: " << imSceneMask.type();


        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();

        // Pass the image to the SLAM system
        LOG(INFO) << "Start TrackRGBD. ";
        SLAM.TrackRGBD(imRGB, imD, imMask, imSceneMask, tframe);

        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();


        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

        vTimesTrack[ni]=ttrack;

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];

        if(ttrack<T)
            usleep((T-ttrack)*1e6);
    }

    LOG(INFO) << "===============Tracking Finished============";

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);

        }
    }
}
