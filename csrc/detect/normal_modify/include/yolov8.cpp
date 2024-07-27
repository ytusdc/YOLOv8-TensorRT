//
// Created by ubuntu on 1/20/23.
//
#ifndef DETECT_NORMAL_YOLOV8_CPP
#define DETECT_NORMAL_YOLOV8_CPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include "yolov8.hpp"

using namespace det;
YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    std::ifstream file(engine_file_path, std::ios::binary);
    assert(file.good());
    file.seekg(0, std::ios::end);
    auto size = file.tellg();
    file.seekg(0, std::ios::beg);
    char* trtModelStream = new char[size];
    assert(trtModelStream);
    file.read(trtModelStream, size);
    file.close();
    initLibNvInferPlugins(&this->m_gLogger, "");
    this->m_runtime = nvinfer1::createInferRuntime(this->m_gLogger);
    assert(this->m_runtime != nullptr);

    this->m_engine = this->m_runtime->deserializeCudaEngine(trtModelStream, size);
    assert(this->m_engine != nullptr);
    delete[] trtModelStream;
    this->m_context = this->m_engine->createExecutionContext();

    assert(this->m_context != nullptr);
    cudaStreamCreate(&this->m_cudaStream);

#ifdef TRT_10
    this->m_num_bindings = this->m_engine->getNbIOTensors();
#else
    this->m_num_bindings = this->m_num_bindings = this->m_engine->getNbBindings();
#endif

    for (int i = 0; i < this->m_num_bindings; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;

#ifdef TRT_10
        std::string        name  = this->m_engine->getIOTensorName(i);
        nvinfer1::DataType dtype = this->m_engine->getTensorDataType(name.c_str());
#else
        nvinfer1::DataType dtype = this->m_engine->getBindingDataType(i);
        std::string        name  = this->m_engine->getBindingName(i);
#endif
        binding.name  = name;
        binding.dsize = type_to_size(dtype);
#ifdef TRT_10
        bool IsInput = m_engine->getTensorIOMode(name.c_str()) == nvinfer1::TensorIOMode::kINPUT;
#else
        bool IsInput = m_engine->bindingIsInput(i);
#endif
        if (IsInput) {
            this->m_num_inputs += 1;
#ifdef TRT_10
            dims = this->m_engine->getProfileShape(name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->m_context->setInputShape(name.c_str(), dims);
#else
            dims = this->m_engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->m_context->setBindingDimensions(i, dims);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->m_input_bindings.push_back(binding);
        }
        else {
#ifdef TRT_10
            dims = this->m_context->getTensorShape(name.c_str());
#else
            dims = this->m_context->getBindingDimensions(i);
#endif
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->m_output_bindings.push_back(binding);
            this->m_num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
#ifdef TRT_10
    delete this->m_context;
    delete this->m_engine;
    delete this->m_runtime;
#else
    this->m_context->destroy();
    this->m_engine->destroy();
    this->m_runtime->destroy();
#endif
    cudaStreamDestroy(this->m_cudaStream);
    for (auto& ptr : this->m_device_ptrs) {
        CHECK(cudaFree(ptr));
    }

    for (auto& ptr : this->m_host_ptrs) {
        CHECK(cudaFreeHost(ptr));
    }
}
void YOLOv8::make_pipe(bool warmup)
{

    for (auto& bindings : this->m_input_bindings) {
        void* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->m_cudaStream));
        this->m_device_ptrs.push_back(d_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->m_context->setInputShape(name, bindings.dims);
        this->context->setTensorAddress(name, d_ptr);
#endif
    }

    for (auto& bindings : this->m_output_bindings) {
        void * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->m_cudaStream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->m_device_ptrs.push_back(d_ptr);
        this->m_host_ptrs.push_back(h_ptr);

#ifdef TRT_10
        auto name = bindings.name.c_str();
        this->m_context->setTensorAddress(name, d_ptr);
#endif
    }

    if (warmup) {
        for (int i = 0; i < 10; i++) {
            for (auto& bindings : this->m_input_bindings) {
                size_t size  = bindings.size * bindings.dsize;
                void*  h_ptr = malloc(size);
                memset(h_ptr, 0, size);
                CHECK(cudaMemcpyAsync(this->m_device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->m_cudaStream));
                free(h_ptr);
            }
            this->infer();
        }
        printf("model warmup 10 times\n");
    }
}

void YOLOv8::letterbox(const cv::Mat& image, cv::Mat& out, cv::Size& size)
{
    const float inp_h  = size.height;
    const float inp_w  = size.width;
    float       height = image.rows;
    float       width  = image.cols;

    float r    = std::min(inp_h / height, inp_w / width);
    int   padw = std::round(width * r);
    int   padh = std::round(height * r);

    cv::Mat tmp;
    if ((int)width != padw || (int)height != padh) {
        cv::resize(image, tmp, cv::Size(padw, padh));
    }
    else {
        tmp = image.clone();
    }

    float dw = inp_w - padw;
    float dh = inp_h - padh;

    dw /= 2.0f;
    dh /= 2.0f;
    int top    = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left   = int(std::round(dw - 0.1f));
    int right  = int(std::round(dw + 0.1f));

    cv::copyMakeBorder(tmp, tmp, top, bottom, left, right, cv::BORDER_CONSTANT, {114, 114, 114});

    out.create({1, 3, (int)inp_h, (int)inp_w}, CV_32F);

    std::vector<cv::Mat> channels;
    cv::split(tmp, channels);

    cv::Mat c0((int)inp_h, (int)inp_w, CV_32F, (float*)out.data);
    cv::Mat c1((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w);
    cv::Mat c2((int)inp_h, (int)inp_w, CV_32F, (float*)out.data + (int)inp_h * (int)inp_w * 2);

    channels[0].convertTo(c2, CV_32F, 1 / 255.f);
    channels[1].convertTo(c1, CV_32F, 1 / 255.f);
    channels[2].convertTo(c0, CV_32F, 1 / 255.f);

    this->m_pparam.ratio  = 1 / r;
    this->m_pparam.dw     = dw;
    this->m_pparam.dh     = dh;
    this->m_pparam.height = height;
    this->m_pparam.width  = width;
    ;
}

void YOLOv8::copy_from_Mat(const cv::Mat& image)
{
    cv::Mat  nchw;
    auto&    in_binding = this->m_input_bindings[0];
    int      width      = in_binding.dims.d[3];
    int      height     = in_binding.dims.d[2];
    cv::Size size{width, height};
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->m_device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->m_cudaStream));

#ifdef TRT_10
    auto name = this->m_input_bindings[0].name.c_str();
    this->m_context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->m_context->setTensorAddress(name, this->m_device_ptrs[0]);
#else
    this->m_context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
#endif
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);

    CHECK(cudaMemcpyAsync(
        this->m_device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->m_cudaStream));

#ifdef TRT_10
    auto name = this->m_input_bindings[0].name.c_str();
    this->m_context->setInputShape(name, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    this->m_context->setTensorAddress(name, this->m_device_ptrs[0]);
#else
    this->m_context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
#endif
}

void YOLOv8::infer()
{
#ifdef TRT_10
    this->m_context->enqueueV3(this->m_cudaStream);
#else
    this->m_context->enqueueV2(this->m_device_ptrs.data(), this->m_cudaStream, nullptr);
#endif
    for (int i = 0; i < this->m_num_outputs; i++) {
        size_t osize = this->m_output_bindings[i].size * this->m_output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->m_host_ptrs[i], this->m_device_ptrs[i + this->m_num_inputs], osize, cudaMemcpyDeviceToHost, this->m_cudaStream));
    }
    cudaStreamSynchronize(this->m_cudaStream);
}

void YOLOv8::postprocess(std::vector<Object>& objs, float score_thres, float iou_thres, int topk, int num_labels)
{
    objs.clear();
    int num_channels = this->m_output_bindings[0].dims.d[1];
    int num_anchors  = this->m_output_bindings[0].dims.d[2];

    auto& dw     = this->m_pparam.dw;
    auto& dh     = this->m_pparam.dh;
    auto& width  = this->m_pparam.width;
    auto& height = this->m_pparam.height;
    auto& ratio  = this->m_pparam.ratio;

    std::vector<cv::Rect> bboxes;
    std::vector<float>    scores;
    std::vector<int>      labels;
    std::vector<int>      indices;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->m_host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;
        if (score > score_thres) {
            float x = *bboxes_ptr++ - dw;
            float y = *bboxes_ptr++ - dh;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clamp((x - 0.5f * w) * ratio, 0.f, width);
            float y0 = clamp((y - 0.5f * h) * ratio, 0.f, height);
            float x1 = clamp((x + 0.5f * w) * ratio, 0.f, width);
            float y1 = clamp((y + 0.5f * h) * ratio, 0.f, height);

            int              label = max_s_ptr - scores_ptr;
            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

#ifdef BATCHED_NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, score_thres, iou_thres, indices);
#else
    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);
#endif

    int cnt = 0;
    for (auto& i : indices) {
        if (cnt >= topk) {
            break;
        }
        Object obj;
        obj.rect  = bboxes[i];
        obj.prob  = scores[i];
        obj.label = labels[i];
        objs.push_back(obj);
        cnt += 1;
    }
}

void YOLOv8::draw_objects(const cv::Mat&                                image,
                          cv::Mat&                                      res,
                          const std::vector<Object>&                    objs,
                          const std::vector<std::string>&               CLASS_NAMES,
                          const std::vector<std::vector<unsigned int>>& COLORS)
{
    res = image.clone();
    for (auto& obj : objs) {
        cv::Scalar color = cv::Scalar(COLORS[obj.label][0], COLORS[obj.label][1], COLORS[obj.label][2]);
        cv::rectangle(res, obj.rect, color, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", CLASS_NAMES[obj.label].c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows) {
            y = res.rows;
        }
        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
#endif  // DETECT_NORMAL_YOLOV8_HPP
