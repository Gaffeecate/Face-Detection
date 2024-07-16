#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int main() {
    // Haar Cascade 파일 경로
    std::string face_cascade_name = "C:\\opencv\\sources\\data\\haarcascade_frontalface_default.xml";
    cv::CascadeClassifier face_cascade;

    // Haar Cascade 파일 로드
    if (!face_cascade.load(face_cascade_name)) {
        std::cerr << "Error loading face cascade file. Exiting!" << std::endl;
        return -1;
    }

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

        // 회색조 이미지로 변환
        cv::Mat frame_gray;
        cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(frame_gray, frame_gray);

        // 얼굴 검출
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(frame_gray, faces);

        // 얼굴 주위에 네모 박스 그리기
        for (size_t i = 0; i < faces.size(); i++) {
            cv::rectangle(frame, faces[i], cv::Scalar(255, 0, 0), 2);
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
