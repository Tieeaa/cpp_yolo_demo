#pragma once

#include <onnxruntime_cxx_api.h>

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;

const std::vector<std::string> class_names = {
    "person",        "bicycle",      "car",
    "motorcycle",    "airplane",     "bus",
    "train",         "truck",        "boat",
    "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",        "bird",
    "cat",           "dog",          "horse",
    "sheep",         "cow",          "elephant",
    "bear",          "zebra",        "giraffe",
    "backpack",      "umbrella",     "handbag",
    "tie",           "suitcase",     "frisbee",
    "skis",          "snowboard",    "sports ball",
    "kite",          "baseball bat", "baseball glove",
    "skateboard",    "surfboard",    "tennis racket",
    "bottle",        "wine glass",   "cup",
    "fork",          "knife",        "spoon",
    "bowl",          "banana",       "apple",
    "sandwich",      "orange",       "broccoli",
    "carrot",        "hot dog",      "pizza",
    "donut",         "cake",         "chair",
    "couch",         "potted plant", "bed",
    "dining table",  "toilet",       "tv",
    "laptop",        "mouse",        "remote",
    "keyboard",      "cell phone",   "microwave",
    "oven",          "toaster",      "sink",
    "refrigerator",  "book",         "clock",
    "vase",          "scissors",     "teddy bear",
    "hair drier",    "toothbrush"};

struct Detection {
  cv::Rect box;
  float confidence{};
  int class_id{};
};

vector<float> preprocess(const cv::Mat &image) {
  cv::Mat resized;
  cv::resize(image, resized, cv::Size(640, 640));
  resized.convertTo(resized, CV_32F, 1 / 255.0);
  cv::cvtColor(resized, resized, cv::COLOR_BGR2RGB);

  vector<cv::Mat> channels;
  cv::split(resized, channels);

  vector<float> chw;
  for (auto &channel : channels) {
    chw.insert(chw.end(), channel.begin<float>(), channel.end<float>());
  }
  return std::move(chw);
}

vector<Detection> postprocess(const float *output_data, float conf_threshold,
                              float nms_threshold, int img_width,
                              int img_height) {
  const int num_classes = 80;
  const int num_boxes = 8400;

  std::vector<cv::Rect> all_boxes;
  all_boxes.reserve(num_boxes);

  for (int i = 0; i < num_boxes; i++) {
    float x = output_data[0 * num_boxes + i];
    float y = output_data[1 * num_boxes + i];
    float w = output_data[2 * num_boxes + i];
    float h = output_data[3 * num_boxes + i];

    int left = static_cast<int>((x - w / 2) / 640.0 * img_width);
    int top = static_cast<int>((y - h / 2) / 640.0 * img_height);
    int width = static_cast<int>(w / 640.0 * img_width);
    int height = static_cast<int>(h / 640.0 * img_height);

    left = std::max(0, std::min(left, img_width - 1));
    top = std::max(0, std::min(top, img_height - 1));
    width = std::max(1, std::min(width, img_width - left));
    height = std::max(1, std::min(height, img_height - top));

    all_boxes.emplace_back(left, top, width, height);
  }

  std::vector<Detection> final_detections;

  for (int class_id = 0; class_id < num_classes; class_id++) {
    std::vector<cv::Rect> class_boxes;
    std::vector<float> class_confidences;
    std::vector<int> class_indices;

    for (int i = 0; i < num_boxes; i++) {
      float confidence = output_data[(4 + class_id) * num_boxes + i];
      if (confidence >= conf_threshold) {
        class_boxes.push_back(all_boxes[i]);
        class_confidences.push_back(confidence);
        class_indices.push_back(i);
      }
    }
    std::vector<int> nms_indices;
    if (!class_boxes.empty()) {
      cv::dnn::NMSBoxes(class_boxes, class_confidences, conf_threshold,
                        nms_threshold, nms_indices);
    }

    for (int idx : nms_indices) {
      Detection det;
      det.box = class_boxes[idx];
      det.confidence = class_confidences[idx];
      det.class_id = class_id;
      final_detections.push_back(det);
    }
  }

  return std::move(final_detections);
}

class YOLOV8 {
 public:
  explicit YOLOV8() {
    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8");
    Ort::SessionOptions session_options{};

    std::string model_path{YOLO_FILE_PATH};
    session = Ort::Session(
        env, std::wstring(model_path.begin(), model_path.end()).c_str(),
        session_options);

    Ort::AllocatorWithDefaultOptions allocator{};
    auto input_cnt = session.GetInputCount();
    for (auto idx = 0; idx < input_cnt; ++idx) {
      cout << "input " << idx
           << " name : " << session.GetInputNameAllocated(idx, allocator)
           << endl;
    }

    auto output_cnt = session.GetOutputCount();
    for (auto idx = 0; idx < output_cnt; ++idx) {
      cout << "output " << idx
           << " name : " << session.GetOutputNameAllocated(idx, allocator)
           << endl;
    }

    auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session.GetOutputNameAllocated(0, allocator);

    input_name_str = input_name_ptr.get();
    output_name_str = output_name_ptr.get();
  }

  [[nodiscard]] cv::Mat inference(const cv::Mat &image) {
    if (image.empty()) {
      std::cerr << "failed to read image" << std::endl;
      return image;
    }

    auto input_data = preprocess(image);

    std::vector<int64_t> input_shape = {1, 3, 640, 640};

    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(),
        input_shape.size());

    std::vector<const char *> input_names = {input_name_str.c_str()};
    std::vector<const char *> output_names = {output_name_str.c_str()};

    auto output_tensors = session.Run(
        Ort::RunOptions{nullptr}, input_names.data(), &input_tensor,
        input_names.size(), output_names.data(), output_names.size());

    auto *output_data = output_tensors.front().GetTensorMutableData<float>();

    std::vector<Detection> detections =
        postprocess(output_data, 0.3, 0.6, image.cols, image.rows);

    cv::Mat res_img = image.clone();
    std::cout << "检测到 " << detections.size() << " 个对象:" << std::endl;
    for (size_t i = 0; i < detections.size(); ++i) {
      const auto &det = detections[i];
      std::string class_name = (det.class_id < class_names.size())
                                   ? class_names[det.class_id]
                                   : "未知类别";

      std::cout << i + 1 << ". 类别: " << class_name
                << ", 置信度: " << std::fixed << std::setprecision(2)
                << det.confidence * 100 << "%"
                << ", 位置: [" << det.box.x << ", " << det.box.y << ", "
                << det.box.width << ", " << det.box.height << "]" << std::endl;

      cv::rectangle(res_img, det.box, cv::Scalar(0, 255, 0), 2);

      std::string label =
          class_name + " " + std::to_string(int(det.confidence * 100)) + "%";
      int baseline = 0;
      cv::Size label_size =
          cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
      cv::rectangle(res_img,
                    cv::Point(det.box.x, det.box.y - label_size.height - 5),
                    cv::Point(det.box.x + label_size.width, det.box.y),
                    cv::Scalar(0, 255, 0), -1);
      cv::putText(res_img, label, cv::Point(det.box.x, det.box.y - 5),
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    return res_img;
  }

 private:
  Ort::Env env{};
  Ort::Session session{nullptr};
  string input_name_str{};
  string output_name_str{};
};
