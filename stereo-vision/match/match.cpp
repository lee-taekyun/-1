#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
int main(int argc, char* argv[])
{
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

    //-- Step 1: Detect the keypoints using SURF Detector, compute the descriptors
    //surf일 때 괄호 안은 임계점. 높으면 정확한 매칭을 하지만 특징점이 적어진다
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(4500);

    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    //디스크립터:특징점에 해당하는 정보, 특징점과 같은 개수로 생성, 실제 유사도를 판별하기 위한 데이터
    //특징점 주변의 일정한 영역 내에 이웃하고 있는 픽셀의 밝기 변화(방향을 갖는 벡터값이고 8방향으로 나누어 수치화한다)
    cv::Mat img_keypoints1;
    cv::Mat img_keypoints2;
    
    detector->detectAndCompute(img1, cv::noArray(), keypoints1, descriptors1);//특징점과 디스크립터를 구한다
    detector->detectAndCompute(img2, cv::noArray(), keypoints2, descriptors2);

    cv::drawKeypoints(img1, keypoints1, img_keypoints1);
    cv::imwrite("img1_key.jpg", img_keypoints1);

    cv::drawKeypoints(img2, keypoints2, img_keypoints2);
    cv::imwrite("img2_key.jpg", img_keypoints2);

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
        if (matches[i].distance < 0.35 * matches[i+1].distance)
        {
            good_matches.push_back(matches[i]);
	    pt1.push_back(keypoints1[matches[i].queryIdx].pt);
	    pt2.push_back(keypoints2[matches[i].trainIdx].pt);

        }
    }
	

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
    return 0;
    //색이 확 변하는 곳에서 미분을 하면 밝기값에 따라 방향을 갖는 기울기가 발생한다. 그 값들을 다 더한게 디스크립터인데
    //디스크립터는 하나의 값으로 나타내서, 양 쪽 사진에서 디스크립터가 같거나 거의 비슷하면 같은 특징점으로 인식하고
    //매칭한다.
}
