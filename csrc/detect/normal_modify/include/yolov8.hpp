//
// Created by ubuntu on 1/20/23.
//
#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include "utils.hpp"


using namespace det;

class YOLOv8 {
public:
    explicit YOLOv8(const std::string& engine_file_path);
    ~YOLOv8();

    void                 make_pipe(bool warmup = true);
    void                 copy_from_Mat(const cv::Mat& image);
    void                 copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void                 letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size);
    void                 infer();
    void                 postprocess(std::vector<Object>& objs,
                                     float                score_thres = 0.25f,
                                     float                iou_thres   = 0.65f,
                                     int                  topk        = 100,
                                     int                  num_labels  = 80);
    static void          draw_objects(const cv::Mat&                                image,
                                      cv::Mat&                                      res,
                                      const std::vector<Object>&                    objs,
                                      const std::vector<std::string>&               CLASS_NAMES,
                                      const std::vector<std::vector<unsigned int>>& COLORS);
    int                  m_num_bindings;
    int                  m_num_inputs  = 0;
    int                  m_num_outputs = 0;
    std::vector<Binding> m_input_bindings;
    std::vector<Binding> m_output_bindings;
    // std::vector<void*>   m_host_ptrs;
    // std::vector<void*>   m_device_ptrs;

    std::vector<float*>   m_host_ptrs;
    std::vector<float*>   m_device_ptrs;

    PreParam m_pparam;

private:
    nvinfer1::ICudaEngine*       m_engine  = nullptr;
    nvinfer1::IRuntime*          m_runtime = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    cudaStream_t                 m_cudaStream  = nullptr;
    Logger                       m_gLogger{nvinfer1::ILogger::Severity::kERROR};

    AffineMatrix m_affine;
};
#endif  // DETECT_NORMAL_YOLOV8_HPP
