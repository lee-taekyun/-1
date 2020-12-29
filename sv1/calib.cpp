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
class Settings
{
public:
    Settings() : goodimg(false) {}//변수를 0으로 설정 
 
/*
    void write(cv::FileStorage& fs) //cv::file~~에 오버로드 될것임.
    {
	    fs << "{"
            << "BoardSize_Width" << boardSize.width
            << "BoardSize_Height" << boardSize.height
            << "Square_Size" << squareSize
            << "Calibrate_Pattern" << patternToUse
            << "Calibrate_NrOfFrameToUse" << imgnumber
            << "Calibrate_FixAspectRatio" << aspectRatio
	    << "Write_DetectedFeaturePoints" << writePoints
            << "Write_extrinsicParameters" << writeExtrinsics
	    << "Write_outputFileName" << outputFileName
            << "Calibrate_AssumeZeroTangentialDistortion" << calibZeroTangentDist
            << "Calibrate_FixPrincipalPointAtTheCenter" << calibFixPrincipalPoint
            << "Show_UndistortedImage" << showUndistorsed
            << "Input_Delay" << delay
            << "Input" << input
            << "}";
    }
*/

    void read(const cv::FileNode& node) //read로 열면 Filenode가 생성된다.
    {
        node["BoardSize_Width"] >> boardSize.width;
        node["BoardSize_Height"] >> boardSize.height;
	node["Square_Size"] >> squareSize;
        node["Calibrate_Pattern"] >> patternToUse;    
        node["Calibrate_NrOfFrameToUse"] >> imgnumber;
        node["Calibrate_FixAspectRatio"] >> aspectRatio;
        node["Write_DetectedFeaturePoints"] >> writePoints;
        node["Write_extrinsicParameters"] >> writeExtrinsics;
        node["Write_outputFileName"] >> outputFileName;
        node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
        node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
        node["Show_UndistortedImage"] >> showUndistorsed;
	node["Input_Delay"] >> delay;
        node["Input"] >> input;
        validate();
    }

    void validate()//스크립트에 적힌 사진정보가 정상적인지
    {
        goodimg = true; //이미지에 문제 없음
        if (boardSize.width <= 0 || boardSize.height <= 0)
        {
	    std::cerr << "iamge size error" << std::endl;
            goodimg = false;
        }
        if (squareSize <= 0)
        {
	    std::cerr << "sqare size error " << squareSize << std::endl;
            goodimg = false;
        }
        if (imgnumber <= 0)
        {

	    std::cerr << "image number error " << imgnumber << std::endl;
            goodimg = false;
        }

        if (input.empty())      // Check for valid input
            inputType = INVALID;
        else
        {
            if (input[0] >= '0' && input[0] <= '9')
            {
		std::stringstream ss(input);
                ss >> cameraID;//<<, >>등을 이용해서 스트링 구성가능. 파일에 읽고 쓰기 모두 가능
                inputType = CAMERA;
            }
            else
            {
                if (readStringList(input, imageList))
                {
                    inputType = IMAGE_LIST;
                    imgnumber = (imgnumber < imageList.size()) ? imgnumber : imageList.size();
			//설정했던 갯수로 할지, 유저 사진갯수로 할지
                }
                else
                    inputType = VIDEO_FILE;
            }
            if (inputType == CAMERA)
                inputCapture.open(cameraID);
            if (inputType == VIDEO_FILE)
                inputCapture.open(input);
            if (inputType != IMAGE_LIST && !inputCapture.isOpened())
                inputType = INVALID;
        }
        if (inputType == INVALID)
        {
	    std::cerr << " Input does not exist: " << input;
            goodimg = false;
        }

        flag = 0;
        if (calibFixPrincipalPoint) flag |= cv::CALIB_FIX_PRINCIPAL_POINT;
        if (calibZeroTangentDist)   flag |= cv::CALIB_ZERO_TANGENT_DIST;
        if (aspectRatio)            flag |= cv::CALIB_FIX_ASPECT_RATIO;


       /* calibrationPattern = NOT_EXISTING;*/
        if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
        atImageList = 0;

    }
    cv::Mat nextImage()
    {
	cv::Mat result;
        /*if (inputCapture.isOpened())//
        {
            cv::Mat view0;
            inputCapture >> view0;
            view0.copyTo(result);
        }
	*/
        if(atImageList < (int)imageList.size())//유저 사진 갯수보다 적게 반복됐다면
            	result = cv::imread(imageList[atImageList++], cv::IMREAD_COLOR);//읽어서 result에 저장
        return result;
    }

    static bool readStringList(const std::string& filename, std::vector<std::string>& l)
    {
        l.clear();//배열 비우기
	cv::FileStorage fs(filename, cv::FileStorage::READ);
        if (!fs.isOpened())
            return false;
	cv::FileNode n = fs.getFirstTopLevelNode();//첫번째 요소 반환  <core.hpp>
	cv::FileNodeIterator it = n.begin(), it_end = n.end();//반복자. 요소 처음부터 끝까지 반복함<core.hpp>
        for (; it != it_end; ++it)
            l.push_back((std::string)*it);//l에 저장
        return true;
    }
    enum Pattern { NOT_EXISTING, CHESSBOARD/*, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID */};
    enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };
public:
    cv::Size boardSize;              // The size of the board -> Number of items by width and height
    Pattern calibrationPattern;  // One of the Chessboard, circles, or asymmetric circle pattern
    float squareSize;            // The size of a square in your defined unit (point, millimeter,etc).
    int imgnumber;                // The number of frames to use from the input for calibration
    float aspectRatio;           // The aspect ratio
    int delay;                   // In case of a video input
    bool writePoints;            // Write detected feature points
    bool writeExtrinsics;        // Write extrinsic parameters
    bool calibZeroTangentDist;   // Assume zero tangential distortion
    bool calibFixPrincipalPoint; // Fix the principal point at the center
    std::string outputFileName;       // The name of the file where to write
    bool showUndistorsed;        // Show undistorted images after calibration
    std::string input;                // The input ->


    int cameraID;
    std::vector<std::string> imageList; //벡터는 배열과 비슷하나 사이즈가 동적이다
    int atImageList;
    cv::VideoCapture inputCapture;
    InputType inputType;
    bool goodimg;
    int flag;

private:
    std::string patternToUse;

};


enum { DETECTION = 0, CAPTURING = 1, CALIBRATED = 2 };

//bool runCalibrationAndSave(Settings& s, cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
    //std::vector<std::vector<cv::Point2f> > imagePoints);

static inline void read(const cv::FileNode& node, Settings& x, const Settings& default_value = Settings())
{
    if (node.empty())
        x = default_value;
    else
        x.read(node);
}

//! [board_corners]
static void calcBoardCornerPositions(cv::Size boardSize, float squareSize, std::vector<cv::Point3f>& corners, Settings::Pattern patternType /*= Settings::CHESSBOARD*/)//corners에 좌표 저
{
    corners.clear();
    for (int i = 0; i < boardSize.height; ++i)
        for (int j = 0; j < boardSize.width; ++j)
            corners.push_back(cv::Point3f(float(j * squareSize), (float)i * squareSize, 0));
}
//! [board_corners]
static bool runCalibration(Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
    std::vector<std::vector<cv::Point2f> > imagePoints, std::vector<cv::Mat>& rvecs, std::vector<cv::Mat>& tvecs)
{
    //! [fixed_aspect]
    cameraMatrix = cv::Mat::eye(3, 3, CV_64F);//3행 3열의 64float타입 단위(대각이 1인)행렬 반환
    if (s.flag & cv::CALIB_FIX_ASPECT_RATIO)
        cameraMatrix.at<double>(0, 0) = 1.0f;
    //! [fixed_aspect]

    distCoeffs = cv::Mat::zeros(8, 1, CV_64F);//8행 1열의 64f타입의 영행렬 반환

    std::vector<std::vector<cv::Point3f> > objectPoints(1);
    calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

    objectPoints.resize(imagePoints.size(), objectPoints[0]);

    //Find intrinsic and extrinsic camera parameters
    double rms = cv::calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
        distCoeffs, rvecs, tvecs,cv::CALIB_FIX_K5);
	//op:실세계에서의 3d지점(단순화하기 위해 z=0).ip:2d상의 이미지 지점
	//**캘리브레이션 함수** 내부,외곡계수,외부파라미터 저장<calib3d.hpp>

    std::cout << "distortion degree: \n " << rms << std::endl;//0에 가까울 수록 왜곡 덜 된것.
    bool ok = cv::checkRange(cameraMatrix) && cv::checkRange(distCoeffs);//NaN,무한이 있는지 확인<core.hpp>

    return ok;
}

// Print camera parameters to the output file
static void saveCameraParams(Settings& s, cv::Size& imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,
    const std::vector<cv::Mat>& rvecs, const std::vector<cv::Mat>& tvecs/*, const std::vector<std::vector<cv::Point2f> >& imagePoints*/)
{
    cv::FileStorage fs(s.outputFileName, cv::FileStorage::WRITE);

    time_t t;
    time(&t);
    struct tm* t2 = localtime(&t);
    char buf[1024];
    strftime(buf, sizeof(buf) - 1, "%c", t2);

    fs << "calibration_time" << buf;

    fs << "image_width" << imageSize.width;
    fs << "image_height" << imageSize.height;
    fs << "board_width" << s.boardSize.width;
    fs << "board_height" << s.boardSize.height;
    fs << "square_size" << s.squareSize;
    fs << "camera_matrix" << cameraMatrix;
    fs << "distortion_coefficients" << distCoeffs;
    
    if (!rvecs.empty() && !tvecs.empty())
    {
        CV_Assert(rvecs[0].type() == tvecs[0].type());//Assert:참이면 정상수행, 거짓이면 에러 발생
	cv::Mat bigmat((int)rvecs.size(), 6, rvecs[0].type());

        for (int i = 0; i < (int)rvecs.size(); i++)//??????????????
        {
            cv::Mat r = bigmat(cv::Range(i, i + 1), cv::Range(0, 3));
	    cv::Mat t = bigmat(cv::Range(i, i + 1), cv::Range(3, 6));

            CV_Assert(rvecs[i].rows == 3 && rvecs[i].cols == 1);
            CV_Assert(tvecs[i].rows == 3 && tvecs[i].cols == 1);
            r = rvecs[i].t();
            t = tvecs[i].t();

        }
        fs.writeComment("a set of 6-tuples (rotation vector + translation vector) for each view");
        fs << "extrinsic_parameters" << bigmat;
	fs << "rvecs" << rvecs;
	fs << "tvecs" << tvecs;
    }

}

//! [run_and_save]
/*
bool runCalibrationAndSave(Settings& s,cv::Size imageSize, cv::Mat& cameraMatrix, cv::Mat& distCoeffs,std::vector<std::vector<cv::Point2f> > imagePoints)
{
    std::vector<cv::Mat> rvecs;//회전행렬(외부 파라미터)
    std::vector<cv::Mat> tvecs;//이동행렬(외부 파라미터)
    //std::vector<float> reprojErrs;

    bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs);	//모든 행렬을 정상적으로 구했으면i
    std::cout << (ok ? "succeeded\n" : "failed\n");//파일에 기록

    if (ok)
        saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
    return ok;
}*/
//! [run_and_save]

///////////////////////////////
////////////////////////////////
//////////////////////////////////
///////////////////////////////
int main(int argc, char* argv[])
{
    Settings s;//클래스 객체 선언
    const std::string inputSettingsFile = argc > 1 ? argv[1] : "default.xml";
    cv::FileStorage fs(inputSettingsFile, cv::FileStorage::READ); // Read the settings
    if (!fs.isOpened())
    {
	    std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
        return -1;
    }
    fs["Settings"] >> s;
    fs.release();                                         // close Settings file

    if (!s.goodimg)
    {
	    std::cout << "Invalid input detected. Application stopping. " << std::endl;
        return -1;
    }

    std::vector<std::vector<cv::Point2f> > imagePoints;
    cv::Mat cameraMatrix, distCoeffs;
    std::vector<cv::Mat> rvecs;//회전행렬(외부 파라미터)
    std::vector<cv::Mat> tvecs;//이동행렬(외부 파라미터)
    cv::Size imageSize;
    int mode = s.inputType == Settings::IMAGE_LIST ? CAPTURING : DETECTION;
    const cv::Scalar RED(0,0,255), GREEN(0,255,0);
    clock_t prevTimestamp = 0;
    const char ESC_KEY = 27;

    //! [get_input]
    for (int i=0;;i++)
    {
	cv::Mat view;
        view = s.nextImage();
        if (view.empty())// If there are no more images stop the loop
        {
    		bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs);
    		std::cout << (ok ? "succeeded\n" : "failed\n");
    		if (ok)
        		saveCameraParams(s, imageSize, cameraMatrix, distCoeffs, rvecs, tvecs);
            	break;
        }
        imageSize = view.size(); 
        //! [find_pattern]
        std::vector<cv::Point2f> pointBuf;
        bool found;
	found = cv::findChessboardCorners(view, s.boardSize, pointBuf,0);

	//체스판 내부 모서리 위치 저장<calib3d.hpp>
	//![find_pattern]
	//! [pattern_found]
        if (found)                // If done with success,
        {
            // improve the found corners' coordinate accuracy for chessboard
            if (s.calibrationPattern == Settings::CHESSBOARD)
            {
		cv::Mat viewGray;
		cv::cvtColor(view, viewGray, cv::COLOR_BGR2GRAY);
		cv::cornerSubPix(viewGray, pointBuf, cv::Size(11,11),//내부모서리 위치 다듬는 함수<imgprog.hpp>
                    cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
            }

            if (mode == CAPTURING &&  // For camera only take new samples after delay time
                (!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay * 1e-3 * CLOCKS_PER_SEC))
            {
                imagePoints.push_back(pointBuf);
                prevTimestamp = clock();
                blinkOutput = s.inputCapture.isOpened();
            }

            // Draw the corners.
	    cv::drawChessboardCorners(view, s.boardSize, cv::Mat(pointBuf), found);//찾은 보드모서리에 그림표시<calib3d.hpp>
        }

        //! [pattern_found]
        //----------------------------- Output Text ------------------------------------------------
        //! [output_text]
/*
	std::string msg = (mode == CAPTURING) ? "100/100" :
            mode == CALIBRATED ? "Calibrated" : "Press 'g' to start";
	int baseLine = 0;
        cv::Size textSize = cv::getTextSize(msg, 1, 1, 1, &baseLine);
        cv::Point textOrigin(view.cols - 2*textSize.width - 10, view.rows - 2*baseLine - 10);

        if( mode == CAPTURING )
        {
            if(s.showUndistorsed)
                msg = cv::format( "%d/%d Undist", (int)imagePoints.size(), s.imgnumber );
            else
                msg = cv::format( "%d/%d", (int)imagePoints.size(), s.imgnumber );
        }

        putText( view, msg, textOrigin, 1, 1, mode == CALIBRATED ?  GREEN : RED);
        if (blinkOutput)
            cv::bitwise_not(view, view);
*/
        //! [output_text]
        //------------------------- Video capture  output  undistorted ------------------------------
        //! [output_undistorted]
        if (mode == CALIBRATED && s.showUndistorsed)
        {
		
            cv::Mat temp = view.clone();
			cv::undistort(temp, view, cameraMatrix, distCoeffs);
        }
        //! [output_undistorted]
        //------------------------------ Show image and check for input commands -------------------
        //! [await_input]
	sleep(1);
	cv::imshow("Image View", view);
        char key = cv::waitKey(s.inputCapture.isOpened() ? 50 : s.delay);

        if (key == ESC_KEY)
            break;

        if (key == 'u' && mode == CALIBRATED)
            s.showUndistorsed = !s.showUndistorsed;

        if (s.inputCapture.isOpened() && key == 'g')
        {
            mode = CAPTURING;
            imagePoints.clear();
        }

        //! [await_input]
    }

    // -----------------------Show the undistorted image for the image list ------------------------
    //! [show_results]
    if (s.inputType == Settings::IMAGE_LIST && s.showUndistorsed)
    {

	cv::Mat view, rview, map1, map2;
        initUndistortRectifyMap(cameraMatrix, distCoeffs, cv::Mat(),cameraMatrix,
            imageSize, CV_16SC2, map1, map2);
        for (int i = 0; i < (int)s.imageList.size(); i++)
        {
            view = cv::imread(s.imageList[i], 1);
            if (view.empty())
                continue;
	    cv::remap(view, rview, map1, map2,1);
            cv::imshow("Image View", rview);
            char c = cv::waitKey();
            if (c == ESC_KEY || c == 'q' || c == 'Q')
                break;
        }
    }
    //! [show_results]

/*    
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
*/
    //cv::Mat img1, img2;
    //cv::undistort(imgs1, img1, cameraMatrix, distCoeffs);
    //cv::undistort(imgs2, img2, cameraMatrix, distCoeffs);
    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    //surf일 때 괄호 안은 임계점. 높으면 정확한 매칭을 하지만 특징점이 적어진다
/*
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(2000);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    
    //cv::Mat img_keypoints1;
    //cv::Mat img_keypoints2;

    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);//특징점과 디스크립>터를 구한다
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);
*/
    //cv::drawKeypoints(img1, keypoints1, img_keypoints1);
    //cv::imwrite("img1_key.jpg", img_keypoints1);

    //cv::drawKeypoints(img2, keypoints2, img_keypoints2);
    //cv::imwrite("img2_key.jpg", img_keypoints2);

    //-- Step 2: Matching descriptor vectors
    // Since SURF is a floating-point descriptor NORM_L2 is used//디스크립터 거리측정 방식
/*    cv::BFMatcher matcher;//디스크립터의 유사도를 비교하여 매칭하는 역할:부르트 포스. matcher 객체 생성
    std::vector<cv::DMatch> matches;//Dmatch:디스크립터 매치 결과를 저장
    matcher.match(descriptors1, descriptors2, matches);//가장 좋은 매칭결과 반
    //-- Filter matches using the Lowe's ratio test
    std::vector<cv::Point2f> pt1;
    std::vector<cv::Point2f> pt2;
    std::vector<cv::DMatch> good_matches;

    for (size_t i = 0; i < matches.size(); i++)
    {
        if (matches[i].distance < 0.7 * matches[i+1].distance)
        {
            good_matches.push_back(matches[i]);
            //pt1.push_back(keypoints1[matches[i].queryIdx].pt);
            //pt2.push_back(keypoints2[matches[i].trainIdx].pt);

        }
    }

    std::vector<int> points1;
    std::vector<int> points2;
    for (auto iter = good_matches.begin(); iter != good_matches.end(); ++iter)
    {
        // Get the indexes of the selected matched keypoints
        points1.push_back(iter->queryIdx);
        points2.push_back(iter->trainIdx);
    }

    // Convert keypoints into Point2f
    std::vector<cv::Point2f> selPoints1, selPoints2;
    cv::KeyPoint::convert(keypoints1, selPoints1, points1);
   cv::KeyPoint::convert(keypoints2, selPoints2, points2);
*/    
    /*
    //-- Draw matches
    cv::Mat img_matches;
    cv::drawMatches(img1, keypoints1, img2, keypoints2, good_matches, img_matches, cv::Scalar::all(-1),
    cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //-- Show detected matches
    std::cout << "left image's keypoints: \n"<< keypoints1.size() << std::endl;
    std::cout << "right image's keypoints: \n"<< keypoints2.size() << std::endl;
    std::cout << "matched count: "<< good_matches.size() << std::endl;

    cv::imshow("result", img_matches);
    cv::imwrite(saveimage, img_matches);
    cv::waitKey();
    */

/*    
    std::vector<cv::Point2f> upt1;
    std::vector<cv::Point2f> upt2;

    //cv::undistortPoints(selPoints1, upt1, cameraMatrix, distCoeffs);
    //cv::undistortPoints(selPoints2, upt2, cameraMatrix, distCoeffs);

    cv::Mat E,ur,ut,maskm;
    E=cv::findEssentialMat(selPoints1, selPoints2, cameraMatrix,cv::RANSAC,0.99,1.0,maskm);//essentialMat 구하기

    std::cout << "E-Matrix" << E << std::endl;

    std::cout << "cameraMatrix" << cameraMatrix << std::endl;
    std::cout << "distcoeffs" << distCoeffs << std::endl;
    cv::recoverPose(E,selPoints1,selPoints2,cameraMatrix,ur,ut,maskm);//E로  R/T 벡터 구하기
*/
    /*
    std::cout << "ur" << ur << std::endl;
    std::cout << "ut" << ut << std::endl;

    cv::Mat F = cv::findFundamentalMat(cv::Mat(selPoints1),cv::Mat(selPoints2),cv::FM_RANSAC,3,0.99);//fundamentalMat구하기

    std::cout << "F-Matrix = \n" << F << std::endl;
   


    std::vector<cv::Vec3f> lines1;
    cv::computeCorrespondEpilines(cv::Mat(selPoints1), 1,  F, lines1);
    for (auto iter = lines1.begin(); iter != lines1.end(); ++iter){
        cv::line(cv::Mat(img2), cv::Point(0, -(*iter)[2] / (*iter)[1]),
                 cv::Point(img1.cols, -((*iter)[2] + (*iter)[0] * img1.cols) / (*iter)[1]),
                 cv::Scalar(255, 255, 255));
    }

    // Draw the RIGHT points corresponding epipolar lines in left image
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
    */

    /*
    cv::Mat rr = cv::Mat::zeros(3,3,CV_64F);

    
    for(int i=0; i<(int)rvecs.size();i++)
    {
	cv::Rodrigues(rvecs[i], rr,cv::noArray());
    }
    
    cv::Mat tt(3,1,CV_64F);
    for( const auto& t: tvecs)
    {
      tt += t;
    }
    // avoid division by zero
    if (!tvecs.empty())
      tt /= tvecs.size();
    */
    
    //F로 rectificaion
/*    
    cv::Mat H1(4,4, img1.type());
    cv::Mat H2(4,4, img2.type());
*/
    //cv::stereoRectifyUncalibrated(selPoints1, selPoints2, F, img1.size(), H1, H2);

    /*
    cv::Mat rectified1(img1.size(), img1.type());
    cv::warpPerspective(img1, rectified1, H1, img1.size());
    cv::imwrite("rectified1.jpg", rectified1);

    cv::Mat rectified2(img2.size(), img2.type());
    cv::warpPerspective(img2, rectified2, H2, img1.size());
    cv::imwrite("rectified2.jpg", rectified2);
    */


    //E에서 구한 r/t 로 rectification
    
    //cv::stereoRectify( cameraMatrix,distCoeffs,cameraMatrix, distCoeffs, imgsize, ur, ut, R1, R2, P1, P2, Q, 0, 1.0, imgsize,  &RL, &RR );
    
    /*
    std::cout << "rr = " << rr << std::endl;
    std::cout << "tt = " << tt << std::endl;
    std::cout << "cameramat = " << cameraMatrix << std::endl;
    std::cout << "dist = " << distCoeffs << std::endl;


    std::cout << "R1 = " << R1 << std::endl;
    std::cout << "R2 = " << R1 << std::endl;

    std::cout << "P1 = " << P1 << std::endl;
    std::cout << "P2 = " << P1 << std::endl;

    std::cout << " Q = " << Q << std::endl;
    std::cout << "RL = " << RL << std::endl;
    std::cout << "RR = " << RR << std::endl;
    */
    /*
    R1 = cameraMatrix.inv()*H1*cameraMatrix;//stereoRectify()에서 나온건 init~에 적용 안되서 보정 
    R2 = cameraMatrix.inv()*H2*cameraMatrix;


    cv::Mat view1, view2, map11, map12, map21, map22;

    initUndistortRectifyMap(cameraMatrix,distCoeffs,R1,cameraMatrix,imgsize, CV_16SC2 , map11, map12);
    initUndistortRectifyMap(cameraMatrix,distCoeffs,R2,cameraMatrix,imgsize, CV_16SC2 , map21, map22);

    remap(img1,view1, map11,map12, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());
    remap(img2,view2, map21,map22, cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar());

    cv::imwrite("View1.jpg", view1);
    cv::imwrite("View2.jpg", view2);
    */
    return 0;
}

