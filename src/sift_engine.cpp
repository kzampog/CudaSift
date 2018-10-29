#include <cuda_sift/sift_engine.hpp>

SiftEngine::SiftEngine(size_t width, size_t height) {
    cuda_image_.Allocate((int)width, (int)height, iAlignUp((int)width, 128), false, NULL, NULL);
}

SiftEngine& SiftEngine::uploadImageData(float *data_ptr) {
    cuda_image_.h_data = data_ptr;
    cuda_image_.Download();
    return *this;
}

SiftEngine& SiftEngine::extractFeatures(SiftData &sift_data, size_t num_octaves, double init_blur, float thresh, float lowest_scale, bool scale_up) {
    ExtractSift(sift_data, cuda_image_, (int)num_octaves, init_blur, thresh, lowest_scale, scale_up);
    return *this;
}

void SiftEngine::matchFeatures(SiftData &sift_data1, SiftData &sift_data2) {
    MatchSiftData(sift_data1, sift_data2);
}
