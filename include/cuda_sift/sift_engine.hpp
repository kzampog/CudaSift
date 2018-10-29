#pragma once

#include <vector>
#include <cuda_sift/cudaSift.h>

class SiftEngine {
public:
    SiftEngine(size_t width, size_t height);
    SiftEngine& uploadImageData(float * data_ptr);
    SiftEngine& extractFeatures(SiftData &sift_data, size_t num_octaves = 5, double init_blur = 1.0, float thresh = 3.5f, float lowest_scale = 0.0f, bool scale_up = false);

    static void matchFeatures(SiftData &sift_data1, SiftData &sift_data2);
private:
    CudaImage cuda_image_;
};
