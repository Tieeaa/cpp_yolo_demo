#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "yolov8.h"

using namespace cv;

int main(int argc, char *argv[]) {
  YOLOV8 yolo{};
  cv::Mat raw_img = cv::imread(LENA_FILE_PATH, cv::IMREAD_COLOR);

  cv::Mat res_img = yolo.inference(raw_img);
  cv::imshow("raw img", raw_img);
  cv::imshow("res img", res_img);
  cv::waitKey(0);

  return 0;
}