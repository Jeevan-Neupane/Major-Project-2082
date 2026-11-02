#ifndef DRR_GENERATOR_H
#define DRR_GENERATOR_H

#include <vector>
#include <string>
#include "CTVolume.h"
#include "XRayCamera.h"
#include "RayMarcher.h"

class DRRGenerator {
public:
    CTVolume ct;
    XRayCamera camera;
    RayMarcher marcher;
    std::vector<float> drrImage;
    Vector3 boxMin, boxMax;
    
    void generate();
    void postProcess();
    void saveAsPNG(const std::string& filename) const;
};

#endif // DRR_GENERATOR_H
