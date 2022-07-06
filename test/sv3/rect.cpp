#include <iostream>
#include <sstream>
#include <ctime>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <unistd.h>
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
int main(void)
{
    //캘리브레이션으로 얻음
    cv::Mat cameraMatrix = (cv::Mat_<double>(3,3) << 3.1813200908026570e+03, 0., 2.0233377901022097e+03, 0.,3.1843767308721222e+03, 1.5346573163013811e+03, 0., 0., 1.);

    cv::Mat distCoeffs = (cv::Mat_<double>(1,5) <<1.6737836355186261e-01, -6.8606752819948158e-01,-2.0210343627090945e-04, 2.7600488250725125e-04,7.4523867629312845e-01);

    cv::Mat img1, img2;


    std::string saveimage = "result.jpg";

    std::string leftName = "i1.jpg";
    cv::imread(leftName).copyTo(img1);

    std::string rightName = "i2.jpg";
    cv::imread(rightName).copyTo(img2);

    if (img1.empty() | img2.empty())
    {
        std::cout << "no image" << std::endl;
        return 0;
    }

    //cv::Mat img1, img2;
    //cv::undistort(imgs1, img1, cameraMatrix, distCoeffs);
    //cv::undistort(imgs2, img2, cameraMatrix, distCoeffs);
    	//-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    	//surf일 때 괄호 안은 임계점. 높으면 정확한 매칭을 하지만 특징점이 적어진다
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(2000);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;

    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);//특징점과 디스크립>터를 구한다
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    	//cv::drawKeypoints(img1, keypoints1, img_keypoints1);
    	//cv::imwrite("img1_key.jpg", img_keypoints1);

    	//cv::drawKeypoints(img2, keypoints2, img_keypoints2);
    	//cv::imwrite("img2_key.jpg", img_keypoints2);

    	//-- Step 2: Matching descriptor vectors
    	// Since SURF is a floating-point descriptor NORM_L2 is used//디스크립터 거리측정 방식
    cv::BFMatcher matcher;//디스크립터의 유사도를 비교하여 매칭하는 역할:부르트 포스. matcher 객체 생성
    std::vector<cv::DMatch> matches;//Dmatch:디스크립터 매치 결과를 저장
    matcher.match(descriptors1, descriptors2, matches);//가장 좋은 매칭결과 반
    	//-- Filter matches using the Lowe's ratio test
    std::vector<cv::Point2f> pt1;
    std::vector<cv::Point2f> pt2;
    std::vector<cv::DMatch> good_matches;

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 0.78 * matches[i+1].distance)
        {
            	good_matches.push_back(matches[i]);
        }
    }

    std::vector<int> points1;
    std::vector<int> points2;
    for (auto iter = good_matches.begin(); iter != good_matches.end(); ++iter)
    {
        //matched keypoints 저장
        points1.push_back(iter->queryIdx);
        points2.push_back(iter->trainIdx);
    }


    // Convert keypoints into Point2f
    std::vector<cv::Point2f> selPoints1, selPoints2;
    cv::KeyPoint::convert(keypoints1, selPoints1, points1);
    cv::KeyPoint::convert(keypoints2, selPoints2, points2);

    //std::vector<cv::Point2f> upt1;
    //std::vector<cv::Point2f> upt2;

    //비왜곡 좌표
    //cv::undistortPoints(selPoints1, upt1, cameraMatrix, distCoeffs);
    //cv::undistortPoints(selPoints2, upt2, cameraMatrix, distCoeffs);

    cv::Mat E,ur,ut,maskm;
    E=cv::findEssentialMat(selPoints1, selPoints2, cameraMatrix,
	cv::RANSAC,0.99,3.0,maskm);//essentialMat

    std::cout << "E-Matrix= \n" << E << std::endl;

    std::cout << "cameraMatrix= \n" << cameraMatrix << std::endl;
    std::cout << "distcoeffs= \n" << distCoeffs << std::endl;
    cv::recoverPose(E,selPoints1,selPoints2,cameraMatrix,ur,ut,maskm);//E로  R/T 벡터 구하기

    std::cout << "r from E= \n" << ur << std::endl;
    std::cout << "t from E= \n" << ut << std::endl;

    cv::Mat F = cv::findFundamentalMat(cv::Mat(selPoints1),cv::Mat(selPoints2),cv::FM_RANSAC,3,0.99);//fundamentalMat구하기

    std::cout << "F-Matrix = \n" << F << std::endl;




    //epipolar line 그리기
    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1,  F, lines1);
    for (auto iter = lines1.begin(); iter != lines1.end(); ++iter){
        cv::line(cv::Mat(img2), cv::Point(0, -(*iter)[2] / (*iter)[1]),
                 cv::Point(img1.cols, -((*iter)[2] + (*iter)[0] * img1.cols) / (*iter)[1]),
                 cv::Scalar(255, 255, 255));
    }
    std::vector<cv::Vec3f> lines2;
    cv::computeCorrespondEpilines(cv::Mat(selPoints2), 2, F, lines2);
    for (auto iter = lines2.begin(); iter != lines2.end(); ++iter){

        cv::line(cv::Mat(img1), cv::Point(0, -(*iter)[2] / (*iter)[1]),
                 cv::Point(img2.cols, -((*iter)[2] + (*iter)[0] * img2.cols) / (*iter)[1]),
                 cv::Scalar(255, 255, 255));
    }
    cv::imwrite("Left Image.jpg", img1);
    cv::imwrite("Right Image.jpg", img2);

    cv::Mat R1;
    cv::Mat R2;
    cv::Mat P1;
    cv::Mat P2;
    cv::Mat Q;


    cv::Rect RL;
    cv::Rect RR;
    cv::Size imgsize(4032,3024);

    //F로 rectificaion
    cv::Mat H1(4,4, img1.type());
    cv::Mat H2(4,4, img2.type());
    cv::stereoRectifyUncalibrated(selPoints1, selPoints2, F, img1.size(), H1, H2);

    	/*
    	cv::Mat rectified1(img1.size(), img1.type());
    	cv::warpPerspective(img1, rectified1, H1, img1.size());
    	cv::imwrite("rectified1.jpg", rectified1);

    	cv::Mat rectified2(img2.size(), img2.type());
    	cv::warpPerspective(img2, rectified2, H2, img1.size());
    	cv::imwrite("rectified2.jpg", rectified2);
	*/


    //E에서 구한 r/t 로 rectification
    cv::stereoRectify( cameraMatrix,distCoeffs,cameraMatrix, distCoeffs, imgsize, ur, ut, R1, R2, P1, P2, Q, 0, 1.0, imgsize,  &RL, &RR );

    std::cout << "q= \n" << Q << std::endl; 
    R1 = cameraMatrix.inv()*H1*cameraMatrix;//stereoRectify()에서 나온건 init~에 적용 안되서 보정
    R2 = cameraMatrix.inv()*H2*cameraMatrix;

    cv::Mat view1, view2, map11, map12, map21, map22;

    initUndistortRectifyMap(cameraMatrix,distCoeffs,R1,cameraMatrix,imgsize, CV_16SC2 , map11, map12);
    initUndistortRectifyMap(cameraMatrix,distCoeffs,R2,cameraMatrix,imgsize, CV_16SC2 , map21, map22);


    remap(img1,view1, map11,map12, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    remap(img2,view2, map21,map22, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    cv::imwrite("View1.jpg", view1);
    cv::imwrite("View2.jpg", view2);
    cv::Mat concat;
    cv::hconcat(view1,view2,concat);
    cv::imwrite("concat.jpg",concat);

    cv::Mat gray1, gray2, disp, falsemap;
    cv::cvtColor(view1, gray1, cv::COLOR_RGB2GRAY);
    cv::cvtColor(view2, gray2, cv::COLOR_RGB2GRAY);
    
    double minval,maxval;
    cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0,80,7,
	0,0,0,16,0,0,0,cv::StereoSGBM::MODE_SGBM);
    cv::Ptr<cv::StereoBM> sbm=cv::StereoBM::create(80,7);
    cv::Mat disparity;
   // sbm->compute(gray1,gray2,disparity);

    sgbm->compute(gray1,gray2,disparity);

    minMaxLoc(disparity, &minval, &maxval);
    printf("min: %f  max: %f \n",minval,maxval);
    cv::imwrite("dis.jpg", disparity);

    return 0;
}
