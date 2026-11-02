#include "DRRGenerator.h"
#include <iostream>
#include <algorithm>
#include "../libs/stb_image_write.h"

const bool INVERT_INTENSITY = false;

void DRRGenerator::generate() {
    drrImage.resize(camera.rows * camera.cols, 0.0f);
    
    std::cout << "Generating DRR image via ray casting..." << std::endl;
    
    for (int r = 0; r < camera.rows; r++) {
        for (int c = 0; c < camera.cols; c++) {
            Vector3 pixelWorld = camera.getPixelWorldPos(r, c);
            Vector3 rayDir = (pixelWorld - camera.sourcePos).normalize();
            float integral = marcher.marchRay(camera.sourcePos, rayDir, ct, boxMin, boxMax);
            drrImage[r * camera.cols + c] = integral;
        }
    }
    
    std::cout << "Ray casting complete." << std::endl;
}

void DRRGenerator::postProcess() {
    if (drrImage.empty()) return;
    
    float minVal = drrImage[0], maxVal = drrImage[0];
    for (float val : drrImage) {
        minVal = std::min(minVal, val);
        maxVal = std::max(maxVal, val);
    }
    
    float range = maxVal - minVal + 1e-8f;
    for (size_t i = 0; i < drrImage.size(); i++) {
        float normalized = (drrImage[i] - minVal) / range;
        drrImage[i] = INVERT_INTENSITY ? (1.0f - normalized) : normalized;
    }
}

void DRRGenerator::saveAsPNG(const std::string& filename) const {
    std::vector<unsigned char> rgbData(camera.rows * camera.cols * 3);
    
    for (int i = 0; i < camera.rows * camera.cols; i++) {
        unsigned char gray = static_cast<unsigned char>(
            std::max(0.0f, std::min(255.0f, drrImage[i] * 255.0f))
        );
        rgbData[i*3] = rgbData[i*3+1] = rgbData[i*3+2] = gray;
    }
    
    if (stbi_write_png(filename.c_str(), camera.cols, camera.rows, 3, rgbData.data(), camera.cols * 3)) {
        std::cout << "Saved: " << filename << std::endl;
    } else {
        std::cerr << "ERROR: Failed to write PNG: " << filename << std::endl;
    }
}
