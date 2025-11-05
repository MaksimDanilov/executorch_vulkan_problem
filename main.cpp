#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

#include <opencv2/opencv.hpp>

#include <chrono>
#include <iostream>
#include <string>

class ScopedTimer 
{
public:
    explicit ScopedTimer(std::string name)
        : name_(std::move(name)),
          start_(std::chrono::high_resolution_clock::now()) {}

    ~ScopedTimer() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration<double, std::milli>(end - start_);
        std::cout << name_ << " took " << duration.count() << " ms\n";
    }

private:
    std::string name_;
    std::chrono::high_resolution_clock::time_point start_;
};

cv::Mat toBlob(cv::Mat input, const cv::Size& size)
{
    cv::resize(input, input, size, cv::INTER_LINEAR);
    input.convertTo(input, CV_32F);

    cv::Mat ret;
    const int nch = input.channels();
    const int sz[] = { 1, nch, size.height, size.width };
    ret.create(4, sz, CV_32F);

    CV_Assert(input.depth() == ret.depth());

    cv::Mat channels[3];
    for (int ch = 0; ch < nch; ch++)
    {
        channels[ch] = cv::Mat(input.rows, input.cols, CV_32F, ret.ptr(0, ch));
    }

    // BGR to RGB fast
    std::swap(channels[0], channels[2]);

    split(input, channels);

    return ret;
}

cv::Mat fromBlob(cv::Mat blob_, const cv::Size& size)
{
    // A blob is a 4 dimensional matrix in floating point precision
    // blob_[0] = batchSize = nbOfImages
    // blob_[1] = nbOfChannels
    // blob_[2] = height
    // blob_[3] = width
    CV_Assert(blob_.depth() == CV_32F);
    CV_Assert(blob_.dims == 4);

    std::vector<cv::Mat> vectorOfChannels(1);
    for (int c = 1; c < blob_.size[1]; ++c)
    {
        vectorOfChannels[c] = cv::Mat(blob_.size[2], blob_.size[3], CV_32FC1, blob_.ptr(0, c));
    }

    cv::Mat result(blob_.size[2], blob_.size[3], CV_32FC1);
    cv::merge(vectorOfChannels, result);
    cv::resize(result, result, size, cv::INTER_AREA);
    return result;
}

int main()
{
    constexpr auto size = 320;
    constexpr bool works = false;
    constexpr auto if_ = "test_data/test_images/0002-01.jpg";
    constexpr auto pte = (works) ? ".results/fcs_xnnpack.pte" : ".results/fcs_vulkan.pte";
    constexpr auto of_ = (works) ? ".results/cpp_xnnpack.jpg" : ".results/cpp_vulkan.jpg";
    
    executorch::extension::Module module(pte);
    cv::Mat image = cv::imread(if_, cv::IMREAD_COLOR);
    if (image.empty())
        return -1;

    ScopedTimer timer("run");
    const cv::Size imageSize = image.size();
    const cv::Mat input = toBlob(std::move(image), cv::Size(size, size));
    assert(input.isContinuous());

    auto tensor = executorch::extension::make_tensor_ptr(
        {1, 3, size, size}, 
        (void*)input.ptr<float>(),
        executorch::aten::ScalarType::Float,
        executorch::aten::TensorShapeDynamism::STATIC);
    executorch::runtime::Result<std::__1::vector<executorch::runtime::EValue>> outputs = module.forward(tensor);
    if (!outputs.ok())
    {
        std::cout << executorch::runtime::to_string(outputs.error());
        return -1;
    }

    std::uint8_t* const data = outputs->front().toTensor().mutable_data_ptr<std::uint8_t>();
    cv::Mat mask(size, size, CV_32FC1, data);
    cv::resize(mask, mask, imageSize, cv::INTER_LINEAR);
    cv::imwrite(of_, mask);
    return 0;
}
