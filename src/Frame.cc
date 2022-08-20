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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

#include "SlamConfig.h"
#include <opencv2/core/core.hpp> 
#include <opencv2/imgproc/imgproc.hpp> 
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/xfeatures2d.hpp> 

#include <glog/logging.h>

using namespace cv;
namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
    , mImRGB(frame.mImRGB)  // add
    , mImSceneMask(frame.mImSceneMask)  // add
    , mImMask(frame.mImMask) 
    , mvbOutliers(frame.mvbOutliers)  //
    , mvbSceneFlowOutliers(frame.mvbSceneFlowOutliers)  // ++++
    , mvbSemanticOutliers(frame.mvbSemanticOutliers)  // +++
    , mvProbability(frame.mvProbability) //
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvProbability = vector<float>(N, 0.5);
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

//=============RGBD
Frame::Frame(const cv::Mat& imRGB, const cv::Mat &imGray, const cv::Mat &imDepth, const cv::Mat &imMask, const cv::Mat &imSceneMask, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
    , mImMask(imMask)
    , mImSceneMask(imSceneMask)
{
    // Frame ID
    LOG(INFO) << "Frame Build";
    mImRGB = imRGB;
    mImRGB = imRGB.clone();

    mImMask = imMask.clone();
    mImSceneMask = imSceneMask.clone();

    mProbability = 0.5;   //========+++++++初始地图点的动态概率 修改

    mnId=nNextId++;

    cv::Mat Mask = cv::Mat(480, 640, CV_8UC3, Scalar(0, 0, 0));
    cv::Mat add = cv::Mat(480, 640, CV_8UC3, Scalar(0, 0, 0));
    for(int i=0; i<mImMask.rows; i++) {
        for(int j=0; j<mImMask.cols; j++) {
            if(mImMask.at<uchar>(i, j) == 255)
            {
                Mask.at<Vec3b>(i, j)[0] = 255;
                Mask.at<Vec3b>(i, j)[1] = 255;
                Mask.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    cv::addWeighted(Mask, 0.5, mImRGB, 0.5, 0.0, add, 4);
    Config::GetInstance()->saveImage(add, "a_Mask", to_string(mnId) + ".png");

    cv::Mat Mask2 = cv::Mat(480, 640, CV_8UC3, Scalar(0, 0, 0));
    cv::Mat add2 = cv::Mat(480, 640, CV_8UC3, Scalar(0, 0, 0));
    for(int i=0; i<mImSceneMask.rows; i++) {
        for(int j=0; j<mImSceneMask.cols; j++) {
            if(mImSceneMask.at<uchar>(i, j) == 255)
            {
                Mask2.at<Vec3b>(i, j)[0] = 255;
                Mask2.at<Vec3b>(i, j)[1] = 255;
                Mask2.at<Vec3b>(i, j)[2] = 255;
            }
        }
    }
    cv::addWeighted(Mask2, 0.5, mImRGB, 0.5, 0.0, add2, 4);
    Config::GetInstance()->saveImage(add2, "a_SceneMask", to_string(mnId) + ".png");

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    // if(mvKeys.empty())
    //     return;
    // std::vector<cv::KeyPoint> _mvKeys;
    // cv::Mat _mDescriptors;
    // for (size_t i(0); i < mvKeys.size(); ++i)
    // {
    //     int val = (int)Mask_dil.at<uchar>(mvKeys[i].pt.y,mvKeys[i].pt.x);
    //     if (val == 1)
    //     {
    //         _mvKeys.push_back(mvKeys[i]);
    //         _mDescriptors.push_back(mDescriptors.row(i));
    //     }
    // }
    // mvKeys = _mvKeys;
    // mDescriptors = _mDescriptors;
    

    N = mvKeys.size();

    if(mvKeys.empty())
        return;


    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvProbability = vector<float>(N, 0.5);
    mvbOutlier = vector<bool>(N,false);

    // ==================Semantic=== indicate features whether are outliers===
    mvbOutliers = vector<bool>(N, false);    //存放外点信息，默认Frame的点都不是外点
    // +++++++++++++++++SceneFlow++ features whether are outliers+++
    mvbSceneFlowOutliers = vector<bool>(N, false); //默认都不是场景流外点
    mvbSemanticOutliers = vector<bool>(N, false); //默认都不是语义外点
    // ============update
    std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    UpdatePrioriMovingProbabilitySF(mImSceneMask, mImMask);
    std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    double t_p= std::chrono::duration_cast<std::chrono::duration<double> >(t2- t1).count(); //单位秒
    LOG(INFO) << "t_p: " << t_p;

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvProbability = vector<float>(N, 0.5);
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

void Frame::UpdatePrioriMovingProbabilitySF(const Mat& imSceneMask, const Mat& imMask)
{
    cv::Mat showFeature = mImRGB.clone();
    cv::Mat showProbability = mImRGB.clone();

    float p_zd_md = 0.98;    // 实际动态, 观测动态
    float p_zs_md = 0.02;    // 实际动态, 观测静态
    float p_zs_ms = 0.98;
    float p_zd_ms = 0.02;
    
    float p_zd_md1 = 0.9;    // 实际动态, 观测动态
    float p_zs_md1 = 0.1;    // 实际动态, 观测静态
    float p_zs_ms1 = 0.9;
    float p_zd_ms1 = 0.1;

    float p_zd_md2 = 0.9;    // 实际动态, 观测动态
    float p_zs_md2 = 0.1;    // 实际动态, 观测静态
    float p_zs_ms2 = 0.85;
    float p_zd_ms2 = 0.15;

    // auto start = std::chrono::steady_clock::now();

    bool bIsMapPointExists = false;

    for (int i = 0; i < this->N; i++) {
        // mark dynamic features
        cv::KeyPoint kp = this->mvKeys[i];
        if (kp.pt.x <= 0 || kp.pt.x >= this->mImMask.cols)
            continue;
        if (kp.pt.y <= 0 || kp.pt.y >= this->mImMask.rows)
            continue;

        MapPoint* pMP = this->mvpMapPoints[i];

        bIsMapPointExists = false;
        if (pMP) {
            if (!pMP->isBad())
                bIsMapPointExists = true;
        }


        //mask二值图， 255白色
        // if (this->mImSceneMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x)== 255 || this->mImMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) == 255) {  //  || this->mImSemanticMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) == 255
        //     this->mnDynamicPoints++;
        // }

        // cout << "row: " << (int)(kp.pt.y) << endl;
        // cout << "col: " << (int)(kp.pt.x) << endl;
        // LOG(INFO) << "kp.pt: " << kp.pt;
        // cout << "mImMask: " << mImMask << endl;
        // cout << "kp: " << mImMask.at<uchar>((int)(kp.pt.y), (int)(kp.pt.x)) << endl;
        

        // if (mImMask.at<uchar>((int)(kp.pt.y), (int)(kp.pt.x)) == 255) {  //  || this->mImMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) == 255
        //     mnDynamicPoints++;
        //     mvbOutliers[i] = true;
            
        // }


        // if (mImSceneMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) == 255) {
        //     mbIsHasDynamicObject = true;
        //     // dynamic object exists
        //     // visualization
        //     // cv::circle(showFeature, kp.pt, 3, cv::Scalar(0, 0, 255), -1); //场景流检测的动态点是红色
        //     mvbSceneFlowOutliers[i] = true;

        // } else {
        //     // visualization
        //     // cv::circle(showFeature, kp.pt, 3, cv::Scalar(0, 255, 0), -1);  //静态点 绿色
        //     mvbSceneFlowOutliers[i] = false;

        // }
        
        if (this->mImMask.at<uchar>((int)kp.pt.y, (int)kp.pt.x) == 255) {
            // LOG(INFO) << "000";
            this->mnDynamicPoints++;
            this->mbIsHasDynamicObject = true;
            this->mvbSemanticOutliers[i] = true;
            this->mvbOutliers[i] = true;
            // cv::circle(showFeature, kp.pt, 4, cv::Scalar(0, 0, 255), -1);

        } else {
            // visualization
            // LOG(INFO) << "111";
            this->mvbSemanticOutliers[i] = false;
            this->mvbOutliers[i] = false;
            // cv::circle(showFeature, kp.pt, 4, cv::Scalar(0, 255, 0), -1);

        }

        if (bIsMapPointExists) {   //+++++++++++++++贝叶斯   地图点存在就保存概率图
            // ==========update moving probability
            float p_old_d = pMP->GetMovingProbability();   //从地图点获得该点当前移动可能性
            float p_old_s = 1 - p_old_d;

            if (this->mvbSemanticOutliers[i] && this->mvbSceneFlowOutliers[i]) {   //是两种动态点
                float p_d = p_zd_md * p_old_d;
                float p_s = p_zd_ms * p_old_s;
                float eta = 1 / (p_d + p_s);
                pMP->SetMovingProbability(eta * p_d);
                this->mvProbability[i] = eta * p_d;

            } else if (!this->mvbSemanticOutliers[i] && !this->mvbSceneFlowOutliers[i]){ //两种都不是
                float p_d = p_zs_md * p_old_d;
                float p_s = p_zs_ms * p_old_s;
                float eta = 1 / (p_d + p_s);
                pMP->SetMovingProbability(eta * p_d);
                this->mvProbability[i] = eta * p_d;

            } else if (this->mvbSemanticOutliers[i] && !this->mvbSceneFlowOutliers[i] ) {   //只是Semantic动态点
                float p_d1 = p_zd_md1 * p_old_d;
                float p_s1 = p_zd_ms1 * p_old_s;
                float eta1 = 1 / (p_d1 + p_s1);
                pMP->SetMovingProbability(eta1 * p_d1);
                this->mvProbability[i] = eta1 * p_d1;

            } else if (this->mvbSceneFlowOutliers[i] && !this->mvbSemanticOutliers[i]) {   //是SceneFlow动态点
                float p_d2 = p_zd_md2 * p_old_d;
                float p_s2 = p_zd_ms2 * p_old_s;
                float eta2 = 1 / (p_d2 + p_s2);
                pMP->SetMovingProbability(eta2 * p_d2);
                this->mvProbability[i] = eta2 * p_d2;

            }

            // if (this->mvbSemanticOutliers[i]) {   //只是Semantic动态点
            //     float p_d1 = p_zd_md1 * p_old_d;
            //     float p_s1 = p_zd_ms1 * p_old_s;
            //     float eta1 = 1 / (p_d1 + p_s1);
            //     pMP->SetMovingProbability(eta1 * p_d1);
            //     this->mvProbability[i] = eta1 * p_d1;
                
            //     // LOG(INFO) << "P: " << eta1 * p_d1;
            // } else{
            //     float p_d1 = p_zs_md1 * p_old_d;
            //     float p_s1 = p_zs_ms1 * p_old_s;
            //     float eta1 = 1 / (p_d1 + p_s1);
            //     pMP->SetMovingProbability(eta1 * p_d1);
            //     this->mvProbability[i] = eta1 * p_d1;
            //     // LOG(INFO) << "P: " << eta1 * p_d1;
            // }

            // if (mvbSceneFlowOutliers[i]) {       //只是Scene动态点
            //     float p_d2 = p_zd_md2 * p_old_d;
            //     float p_s2 = p_zd_ms2 * p_old_s;
            //     float eta2 = 1 / (p_d2 + p_s2);
            //     pMP->SetMovingProbability(eta2 * p_d2);
            // } else{
            //     float p_d2 = p_zs_md2 * p_old_d;
            //     float p_s2 = p_zs_ms2 * p_old_s;
            //     float eta2 = 1 / (p_d2 + p_s2);
            //     pMP->SetMovingProbability(eta2 * p_d2);
            // }

            // if (pMP->GetMovingProbability() >= 0.6)
            // {
            //     cv::circle(showProbability, kp.pt, 4, cv::Scalar(0, 0, 255), -1);   //red
            // } 
            // if (pMP->GetMovingProbability() <= 0.4 && pMP->GetMovingProbability() >= 0) 
            // {
            //     cv::circle(showProbability, kp.pt, 4, cv::Scalar(0, 255, 0), -1);   //green
            // } 
            // if (pMP->GetMovingProbability() > 0.4 && pMP->GetMovingProbability() < 0.6) 
            // {
            //     cv::circle(showProbability, kp.pt, 4, cv::Scalar(0, 215, 255), -1); //yellow
            // }

        } else {
            float p_old_d = 0.5;
            float p_old_s = 0.5;
            if (this->mvbSemanticOutliers[i]) {   //只是Semantic动态点
                float p_d1 = p_zd_md1 * p_old_d;
                float p_s1 = p_zd_ms1 * p_old_s;
                float eta1 = 1 / (p_d1 + p_s1);
                this->mvProbability[i] = eta1 * p_d1;

            } else{
                float p_d1 = p_zs_md1 * p_old_d;
                float p_s1 = p_zs_ms1 * p_old_s;
                float eta1 = 1 / (p_d1 + p_s1);
                this->mvProbability[i] = eta1 * p_d1;
            }

            // if (this->mvProbability[i] >= 0.6)
            // {
            //     cv::circle(showProbability, kp.pt, 4, cv::Scalar(0, 0, 255), -1);
            // }
            // if (this->mvProbability[i] <= 0.45 && this->mvProbability[i] >= 0) 
            // {
            //     cv::circle(showProbability, kp.pt, 4, cv::Scalar(0, 255, 0), -1);
            // }
            // if (this->mvProbability[i] > 0.45 && this->mvProbability[i] < 0.6) 
            // {
            //     cv::circle(showProbability, kp.pt, 4, cv::Scalar(0, 215, 255), -1);
            // }
            

        }



    } //end for

    // auto end = std::chrono::steady_clock::now();
    // std::chrono::duration<double> diff = end - start;
    // LOG(INFO) << "Time to update moving probability:  " << std::setw(3) << diff.count() * 1000 << " ms";

    // dyna::SceneFlow::GetInstance()->mvTimeUpdateMovingProbability.emplace_back(diff.count());
    // Config::GetInstance()->saveImage(showFeature, "feature_SceneFlow", "SceneFlow_" + std::to_string(this->mnId) + "_" + std::to_string(this->mnFrameId) + ".png");
    // Config::GetInstance()->saveImage(showProbability, "Probability_SceneFlow", "Probability_" + std::to_string(mnId) + ".png");
    // Config::GetInstance()->saveImage(showFeature, "feature", "Feature_" + std::to_string(mnId) + ".png");
}

} //namespace ORB_SLAM
