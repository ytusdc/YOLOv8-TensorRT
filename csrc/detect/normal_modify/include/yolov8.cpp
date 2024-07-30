//
// Created by ubuntu on 1/20/23.
//
#ifndef DETECT_NORMAL_YOLOV8_CPP
#define DETECT_NORMAL_YOLOV8_CPP
#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include "yolov8.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>


void warp_affine_bilinear(uint8_t* src, int src_line_size, int src_width, int src_height,
	float* dst,int dst_width, int dst_height, uint8_t fill_value, AffineMatrix matrix, cudaStream_t stream);

using namespace det;
YOLOv8::YOLOv8(const std::string& engine_file_path)
{
    // size_t size{0};
    // char *trtModelStream{nullptr};

    // std::ifstream file(engine_file_path, std::ios::binary);
    // assert(file.good());
    // if (file.good()) {
    //     file.seekg(0, std::ios::end);
    //     auto size = file.tellg();
    //     file.seekg(0, std::ios::beg);
    //     trtModelStream = new char[size];
    //     assert(trtModelStream);
    //     file.read(trtModelStream, size);
    //     file.close();
    //     std::cout << "engine init finished" << std::endl;
    // }

    // initLibNvInferPlugins(&this->m_gLogger, "");
    // this->m_runtime = nvinfer1::createInferRuntime(this->m_gLogger);
    // assert(this->m_runtime != nullptr);

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
    this->m_num_bindings = this->m_engine->getNbBindings();

    for (int i = 0; i < this->m_num_bindings; ++i) {
        Binding        binding;
        nvinfer1::Dims dims;
        nvinfer1::DataType dtype = this->m_engine->getBindingDataType(i);
        std::string        name  = this->m_engine->getBindingName(i);
        binding.name  = name;
        binding.dsize = type_to_size(dtype);
        bool IsInput = this->m_engine->bindingIsInput(i);
        if (IsInput) {
            this->m_num_inputs += 1;
            dims = this->m_engine->getProfileDimensions(i, 0, nvinfer1::OptProfileSelector::kMAX);
            // set max opt shape
            this->m_context->setBindingDimensions(i, dims);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->m_input_bindings.push_back(binding);
        }
        else
        {
            dims = this->m_context->getBindingDimensions(i);
            binding.size = get_size_by_dims(dims);
            binding.dims = dims;
            this->m_output_bindings.push_back(binding);
            this->m_num_outputs += 1;
        }
    }
}

YOLOv8::~YOLOv8()
{
    this->m_context->destroy();
    this->m_engine->destroy();
    this->m_runtime->destroy();
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
        // void* d_ptr;
        float* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->m_cudaStream));
        this->m_device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->m_output_bindings) {
        // void * d_ptr, *h_ptr;
        float * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->m_cudaStream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->m_device_ptrs.push_back(d_ptr);
        this->m_host_ptrs.push_back(h_ptr);
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


    this->m_context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, height, width}});
}

void YOLOv8::copy_from_Mat(const cv::Mat& image, cv::Size& size)
{
    cv::Mat nchw;
    this->letterbox(image, nchw, size);



	int width = image.cols;
	int height = image.rows;
	int channels = image.channels();
	int src_size = width * height * channels;
	uint8_t* psrc_device = nullptr;
    // void* psrc_device = nullptr;

	CHECK(cudaMalloc(&psrc_device, src_size));
	CHECK(cudaMemcpyAsync(psrc_device, image.data, src_size, cudaMemcpyHostToDevice, this->m_cudaStream));
    
    int input_width = 640;
    int input_height = 640;
    float* input_data_device = nullptr;

	AffineMatrix affine;
	affine.compute(width, height, input_width, input_height);
    m_affine = affine;

            std::cout << "affine" << std::endl;
        std::cout<< "d2i[0] = " << affine.d2i[0] << std::endl;
        std::cout<< "d2i[2] = " << affine.d2i[2] << std::endl;
        std::cout<< "d2i[5] = " << affine.d2i[5] << std::endl;

	warp_affine_bilinear(psrc_device, width * 3, width, height, this->m_device_ptrs[0], input_width, input_height, 114, affine, this->m_cudaStream);

    // CHECK(cudaMemcpyAsync(
    //     this->m_device_ptrs[0], nchw.ptr<float>(), nchw.total() * nchw.elemSize(), cudaMemcpyHostToDevice, this->m_cudaStream));


    this->m_context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, size.height, size.width}});
    
}

void YOLOv8::infer()
{

    // this->m_context->enqueueV2(this->m_device_ptrs.data(), this->m_cudaStream, nullptr);

    this->m_context->enqueueV2((void**)this->m_device_ptrs.data(), this->m_cudaStream, nullptr);
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


    if (false) {

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


    cv::dnn::NMSBoxes(bboxes, scores, score_thres, iou_thres, indices);

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
    else {


    AffineMatrix affine;

    affine = m_affine;
    std::vector<cv::Rect> bboxes;
	std::vector<int> classIds;
	std::vector<float> scores;

    float confThreshold_ =0.5;
    float nmsThreshold_ =0.5;
    // yolo v8 postpress
	auto output_dims = this->m_engine->getBindingDimensions(1);
    // output_numbox = output_dims.d[1];
    // output_numprob = output_dims.d[2];
    // num_classes = output_numprob - 5;

    // v8
    int output_numbox = output_dims.d[2];
    int output_numprob = output_dims.d[1];

    // num_channels
    // num_anchors

	num_labels = num_channels - 4;

    cv::Mat output = cv::Mat(num_channels, num_anchors, CV_32F, static_cast<float*>(this->m_host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < num_anchors; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float score      = *max_s_ptr;

		float objness = row_ptr[4];

        if (score > confThreshold_) {

             int label = max_s_ptr - scores_ptr;
            // cv::Rect_<float> bbox;
            // bbox.x      = x0;
            // bbox.y      = y0;
            // bbox.width  = x1 - x0;
            // bbox.height = y1 - y0;


			float cx = row_ptr[0];
			float cy = row_ptr[1];
			float w1 = row_ptr[2];
			float h1 = row_ptr[3];
			float x1 = cx - w1 / 2;
			float y1 = cy - h1 / 2;

			cv::Rect box;
			box.x = x1;
			box.y = y1;
			box.width = w1;
			box.height = h1;

            bboxes.push_back(box);
            classIds.push_back(label);
            scores.push_back(score);

            std::cout << "score = " << score << std::endl;
        }
    }

	if (bboxes.size() == 0) {
		
		return ;
	}

	std::vector<int> indexes;
	cv::dnn::NMSBoxes(bboxes, scores, confThreshold_, nmsThreshold_, indexes);
	for (size_t i = 0; i < indexes.size(); i++) {
		int idx = indexes[i];
		auto& ibox = bboxes[idx];
		float x = ibox.x;
		float y = ibox.y;
		float w = ibox.width;
		float h = ibox.height;

        // std::cout << "m_affine" << std::endl;
        // std::cout<< "d2i[0] = " << affine.d2i[0] << std::endl;
        // std::cout<< "d2i[2] = " << affine.d2i[2] << std::endl;
        // std::cout<< "d2i[5] = " << affine.d2i[5] << std::endl;


		int image_base_x = static_cast<int>(affine.d2i[0] * x + affine.d2i[2]);
		int image_base_y = static_cast<int>(affine.d2i[0] * y + affine.d2i[5]);
		int image_base_width = static_cast<int>(affine.d2i[0] * w);
		int image_base_height = static_cast<int>(affine.d2i[0] * h);
		if ((image_base_width <= 0) || (image_base_height <= 0))
			continue;

		// struct Box box;
		// box.x = image_base_x;
		// box.y = image_base_y;
		// box.width = image_base_width;
		// box.height = image_base_height;
		// box.score = scores[idx];
		// box.class_id = classIds[idx];
		// objs.push_back(box);


        Object obj;
        obj.rect  = cv::Rect(image_base_x, image_base_y, image_base_width, image_base_height);
        obj.prob  = scores[idx];
        obj.label = classIds[idx];
        objs.push_back(obj);

        std::cout <<  obj.prob  << std::endl;

	    }
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
#endif  // DETECT_NORMAL_YOLOV8_CPP
