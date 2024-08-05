#include "NvInferPlugin.h"
#include "common.hpp"
#include <fstream>
#include "yolov8.hpp"
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
// #include<windows.h>
#include <unistd.h>


void warp_affine_bilinear(uint8_t* src, int src_width, int src_height,
	                    float* dst,int dst_width, int dst_height, 
                        uint8_t fill_value, AffineMatrix matrix, cudaStream_t stream);

using namespace det;


bool YOLOv8::initConfig(const std::string& engine_file_path, 
                                     float score_thres, 
                                     float iou_thres,
                                     int   topk)
{
    size_t size{0};
    char *trtModelStream{nullptr};
    std::ifstream file(engine_file_path, std::ios::binary);
    if (file.good()) {
        file.seekg(0, std::ios::end);
        size = file.tellg();
        file.seekg(0, std::ios::beg);
        trtModelStream = new char[size];
        assert(trtModelStream);
        file.read(trtModelStream, size);
        file.close();
        std::cout << "engine init finished" << std::endl;
    }
    else 
    {
        return false;
    }

    // std::ifstream file(engine_file_path, std::ios::binary);
    // assert(file.good());
    // file.seekg(0, std::ios::end);
    // auto size = file.tellg();
    // file.seekg(0, std::ios::beg);
    // char* trtModelStream = new char[size];
    // assert(trtModelStream);
    // file.read(trtModelStream, size);
    // file.close();


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

    for (auto& bindings : this->m_input_bindings) {
        float* d_ptr;
        CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->m_cudaStream));
        this->m_device_ptrs.push_back(d_ptr);
    }

    for (auto& bindings : this->m_output_bindings) {
        float * d_ptr, *h_ptr;
        size_t size = bindings.size * bindings.dsize;
        CHECK(cudaMallocAsync(&d_ptr, size, this->m_cudaStream));
        CHECK(cudaHostAlloc(&h_ptr, size, 0));
        this->m_device_ptrs.push_back(d_ptr);
        this->m_host_ptrs.push_back(h_ptr);
    }
    
    // v8 detect 是单输入单输出
    m_output_numprob = this->m_output_bindings[0].dims.d[1];
    m_output_numbox  = this->m_output_bindings[0].dims.d[2];
    // v8
    m_output_numclass = m_output_numprob - 4;

    // yolo v8 postpress
	// auto output_dims = this->m_engine->getBindingDimensions(1);
    // int m_output_numbox = output_dims.d[2];
    // int m_output_numprob = output_dims.d[1];
	// m_output_numclass = num_channels - 4;

    m_score_thres = score_thres;
    m_iou_thres   = iou_thres;
    m_topk        = topk;

    return true;
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

// void YOLOv8::make_pipe(bool warmup)
// {
//     for (auto& bindings : this->m_input_bindings) {
//         float* d_ptr;
//         CHECK(cudaMallocAsync(&d_ptr, bindings.size * bindings.dsize, this->m_cudaStream));
//         this->m_device_ptrs.push_back(d_ptr);
//     }

//     for (auto& bindings : this->m_output_bindings) {
//         float * d_ptr, *h_ptr;
//         size_t size = bindings.size * bindings.dsize;
//         CHECK(cudaMallocAsync(&d_ptr, size, this->m_cudaStream));
//         CHECK(cudaHostAlloc(&h_ptr, size, 0));
//         this->m_device_ptrs.push_back(d_ptr);
//         this->m_host_ptrs.push_back(h_ptr);
//     }

//     if (warmup) {
//         for (int i = 0; i < 10; i++) {
//             for (auto& bindings : this->m_input_bindings) {
//                 size_t size  = bindings.size * bindings.dsize;
//                 void*  h_ptr = malloc(size);
//                 memset(h_ptr, 0, size);
//                 CHECK(cudaMemcpyAsync(this->m_device_ptrs[0], h_ptr, size, cudaMemcpyHostToDevice, this->m_cudaStream));
//                 free(h_ptr);
//             }
//             // this->infer(this->m_device_ptrs[0]);
//         }
//         printf("model warmup 10 times\n");
//     }
// }

void YOLOv8::detect(cv::Mat& image,std::vector<Box>& boxes) 
{
  auto start_infer = std::chrono::system_clock::now();
  this->infer(image);
  auto end_infer = std::chrono::system_clock::now();
  this->postprocess(boxes);
  auto end_postprocess = std::chrono::system_clock::now();

  auto tc_infer = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_infer - start_infer).count() / 1000.;
  auto tc_postprocess = (double)std::chrono::duration_cast<std::chrono::microseconds>(end_postprocess - end_infer).count() / 1000.;
  printf("infer cost %2.4lf ms\n", tc_infer);
  printf("postprocess cost %2.4lf ms\n", tc_postprocess);

}

void YOLOv8::infer(cv::Mat& image)
{
    int width = image.cols;
	int height = image.rows;
	int channels = image.channels();
	int src_size = width * height * channels;
	uint8_t* psrc_device = nullptr;

	CHECK(cudaMallocAsync(&psrc_device, src_size, this->m_cudaStream));
	CHECK(cudaMemcpyAsync(psrc_device, image.data, src_size, cudaMemcpyHostToDevice, this->m_cudaStream));
    
	AffineMatrix affine;
	affine.compute(width, height, this->m_input_size.width, this->m_input_size.height);
    this->m_affine = affine;

	warp_affine_bilinear(psrc_device, width, height, 
                        this->m_device_ptrs[0], this->m_input_size.width, this->m_input_size.height, 
                        114, affine, this->m_cudaStream);

    this->m_context->setBindingDimensions(0, nvinfer1::Dims{4, {1, 3, this->m_input_size.height, this->m_input_size.width}});

    this->m_context->enqueueV2((void**)this->m_device_ptrs.data(), this->m_cudaStream, nullptr);
    for (int i = 0; i < this->m_num_outputs; i++) {
        size_t osize = this->m_output_bindings[i].size * this->m_output_bindings[i].dsize;
        CHECK(cudaMemcpyAsync(
            this->m_host_ptrs[i], this->m_device_ptrs[i + this->m_num_inputs], osize, cudaMemcpyDeviceToHost, this->m_cudaStream));
    }
    cudaStreamSynchronize(this->m_cudaStream);
}

void YOLOv8::postprocess(std::vector<Box>& boxes_vec)
{   
    AffineMatrix  affine = this->m_affine;
    boxes_vec.clear();

    std::vector<cv::Rect> bboxes;
	std::vector<int> classIds;
	std::vector<float> scores;

    cv::Mat output = cv::Mat(this->m_output_numprob, this->m_output_numbox, CV_32F, static_cast<float*>(this->m_host_ptrs[0]));
    output         = output.t();
    for (int i = 0; i < this->m_output_numbox; i++) {
        auto  row_ptr    = output.row(i).ptr<float>();
        auto  bboxes_ptr = row_ptr;
        auto  scores_ptr = row_ptr + 4;
        auto  max_score_ptr  = std::max_element(scores_ptr, scores_ptr + this->m_output_numclass);
        float score      = *max_score_ptr;

		// float objness = row_ptr[4];

        if (score > m_score_thres) {

            int label = max_score_ptr - scores_ptr;
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
        }
    }

	if (bboxes.size() == 0) {
		
		return ;
	}

	std::vector<int> indexes_vec;
	cv::dnn::NMSBoxes(bboxes, scores, this->m_score_thres, this->m_iou_thres, indexes_vec);

    for (auto idx :indexes_vec ) {
		auto& ibox = bboxes[idx];
		float x = ibox.x;
		float y = ibox.y;
		float w = ibox.width;
		float h = ibox.height;

		int image_base_x = static_cast<int>(affine.d2i[0] * x + affine.d2i[2]);
		int image_base_y = static_cast<int>(affine.d2i[0] * y + affine.d2i[5]);
		int image_base_width = static_cast<int>(affine.d2i[0] * w);
		int image_base_height = static_cast<int>(affine.d2i[0] * h);
		if ((image_base_width <= 0) || (image_base_height <= 0))
			continue;

        Box box;
        box.rect  = cv::Rect(image_base_x, image_base_y, image_base_width, image_base_height);
        box.score  = scores[idx];
        box.class_id = classIds[idx];
        boxes_vec.push_back(box);
    }
}

void YOLOv8::draw_objects(const cv::Mat&                                image,
                          const std::vector<Box>&                       boxes_vec)
{
    // res = image.clone();
    for (auto& box : boxes_vec) {
        cv::Scalar color = cv::Scalar(COLORS[box.class_id][0], COLORS[box.class_id][1], COLORS[box.class_id][2]);
        cv::rectangle(image, box.rect, color, 2);

        char text[256];
        
        if (CLASS_NAMES.size() >  0) {
            sprintf(text, "%s %.1f%%", CLASS_NAMES[box.class_id].c_str(), box.score * 100);
        } else {
            sprintf(text, "%d %.1f%%", box.class_id, box.score * 100);
        }

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.4, 1, &baseLine);

        int x = (int)box.rect.x;
        int y = (int)box.rect.y + 1;

        if (y > image.rows) {
            y = image.rows;
        }
        cv::rectangle(image, cv::Rect(x, y, label_size.width, label_size.height + baseLine), {0, 0, 255}, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.4, {255, 255, 255}, 1);
    }
}
