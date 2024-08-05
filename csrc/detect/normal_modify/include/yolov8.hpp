#ifndef DETECT_NORMAL_YOLOV8_HPP
#define DETECT_NORMAL_YOLOV8_HPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include "utils.hpp"


using namespace det;

class YOLOv8 {
public:
    explicit YOLOv8(){};
    ~YOLOv8();
    bool    initConfig(const std::string& engine_file_path, 
                                    float score_thres = 0.25f, 
                                    float iou_thres   = 0.65f,
                                    int   topk        = 100);
    // void    make_pipe(bool warmup = true);
    // void    copy_from_Mat(const cv::Mat& image, cv::Size& size);
    void    infer(cv::Mat& image);
    void    postprocess(std::vector<Box>& boxes_vec);
    void    detect(cv::Mat& image,std::vector<Box>& boxes);
    static void     draw_objects(const cv::Mat&                               image,
                                const std::vector<Box>&                       boxes_vec);


    cv::Size             m_input_size   = cv::Size{640, 640};   // cv::Size{width, height}
    float                m_score_thres    = 0.25f; 
    float                m_iou_thres      = 0.65f;
    int                  m_topk           = 100;

    int                  m_output_numbox;
    int                  m_output_numprob;
    int                  m_output_numclass;

    int                  m_num_inputs  = 0;
    int                  m_num_outputs = 0;
    int                  m_num_bindings;
    std::vector<Binding> m_input_bindings;
    std::vector<Binding> m_output_bindings;
    // std::vector<void*>   m_host_ptrs;
    // std::vector<void*>   m_device_ptrs;

    std::vector<float*>   m_host_ptrs;
    std::vector<float*>   m_device_ptrs;


private:
    nvinfer1::ICudaEngine*       m_engine  = nullptr;
    nvinfer1::IRuntime*          m_runtime = nullptr;
    nvinfer1::IExecutionContext* m_context = nullptr;
    cudaStream_t                 m_cudaStream  = nullptr;
    Logger                       m_gLogger{nvinfer1::ILogger::Severity::kERROR};
    AffineMatrix m_affine;
};
#endif  // DETECT_NORMAL_YOLOV8_HPP
