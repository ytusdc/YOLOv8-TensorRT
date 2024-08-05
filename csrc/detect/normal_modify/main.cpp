#include "opencv2/opencv.hpp"
#include "yolov8.hpp"
#include <chrono>
#include "common.hpp"


int main(int argc, char** argv)
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s [engine_path] [image_path/image_dir/video_path]\n", argv[0]);
        return -1;
    }

    // cuda:0
    cudaSetDevice(0);

    const std::string engine_file_path{argv[1]};
    const std::string    path{argv[2]};

    std::vector<std::string> imagePathList;
    bool                     isVideo{false};

    auto yolov8 = new YOLOv8();

    bool re = yolov8->initConfig(engine_file_path);

    if (!re) {
        std::cout << "init model failed.";
        return 0;
    }

    if (IsFile(path)) {
        std::string suffix = path.substr(path.find_last_of('.') + 1);
        if (suffix == "jpg" || suffix == "jpeg" || suffix == "png") {
            imagePathList.push_back(path);
        }
        else if (suffix == "mp4" || suffix == "avi" || suffix == "m4v" || suffix == "mpeg" || suffix == "mov"
                 || suffix == "mkv") {
            isVideo = true;
        }
        else {
            printf("suffix %s is wrong !!!\n", suffix.c_str());
            std::abort();
        }
    }
    else if (IsFolder(path)) {
        cv::glob(path + "/*.jpg", imagePathList);
    }


    cv::Mat             image;
    cv::Size            size = cv::Size{640, 640};
    int      num_labels  = 3;
    int      topk        = 100;
    float    score_thres = 0.25f;
    float    iou_thres   = 0.65f;

    std::vector<Box> boxes_vec;

    cv::namedWindow("result", cv::WINDOW_AUTOSIZE);

    if (isVideo) {
        cv::VideoCapture cap(path);

        if (!cap.isOpened()) {
            printf("can not open %s\n", path.c_str());
            return -1;
        }
        while (cap.read(image)) {
            boxes_vec.clear();
            yolov8->detect(image, boxes_vec);
            yolov8->draw_objects(image, boxes_vec);
            if (cv::waitKey(10) == 'q') {
                break;
            }
        }
    }
    else {
        for (auto& path : imagePathList) {
            boxes_vec.clear();
            image = cv::imread(path);
            yolov8->detect(image, boxes_vec);
            yolov8->draw_objects(image, boxes_vec);
            cv::imshow("result", image);
            cv::waitKey(0);
        }
    }
    cv::destroyAllWindows();
    delete yolov8;
    return 0;
}
