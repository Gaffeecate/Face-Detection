#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 177.0, 123.0);

int main() {
    // 딥러닝 모델 로드
    cv::dnn::Net net = cv::dnn::readNetFromCaffe(
        "deploy.prototxt",
        "res10_300x300_ssd_iter_140000_fp16.caffemodel");



    // 웹캠 캡처 시작
    cv::VideoCapture capture(0);
    if (!capture.isOpened()) {
        std::cerr << "Error opening video capture. Exiting!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (capture.read(frame)) {
        if (frame.empty()) {
            std::cerr << "No captured frame. Breaking!" << std::endl;
            break;
        }

        // 프레임을 블롭으로 변환
        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor,
            cv::Size(inWidth, inHeight), meanVal, false, false);

        // 네트워크에 블롭 입력
        net.setInput(inputBlob, "data");

        // 얼굴 검출 실행
        cv::Mat detection = net.forward("detection_out");
        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        // 검출된 얼굴에 박스 그리기
        for (int i = 0; i < detectionMat.rows; i++) {
            float confidence = detectionMat.at<float>(i, 2);

            if (confidence > confidenceThreshold) {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frame.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frame.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frame.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frame.rows);

                cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2);
                std::string label = cv::format("Face: %.2f", confidence);
                int baseLine;
                cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                cv::putText(frame, label, cv::Point(x1, y1 - labelSize.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0));
            }
        }

        // 결과 화면에 표시
        cv::imshow("Face Detection", frame);

        // 'q' 키를 누르면 종료
        if (cv::waitKey(10) == 'q') {
            break;
        }
    }

    capture.release();
    cv::destroyAllWindows();
    return 0;
}