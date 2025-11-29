#ifndef XRAY_CAMERA_H
#define XRAY_CAMERA_H

#include <vector>
#include <string>
#include "MathUtils.h"
#include "CTVolume.h"

class XRayCamera {
public:
    Vector3 sourcePos, detectorCenter, uDir, vDir, lookDir;
    int rows, cols;
    float pixelSize;
    std::vector<float> uCoords, vCoords;
    std::string sourceAxis;
    
    void setup(const CTVolume& ct, const std::string& axis);
    Vector3 getPixelWorldPos(int row, int col) const;
    std::string getDetectorAxis() const;
    void printSetupInfo(const Vector3& ctCenter, const Vector3& boxMin, const Vector3& boxMax, const CTVolume& ct) const;
};

#endif // XRAY_CAMERA_H
