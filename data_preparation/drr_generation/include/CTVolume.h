#ifndef CT_VOLUME_H
#define CT_VOLUME_H

#include <vector>
#include <string>
#include "MathUtils.h"

class CTVolume {
public:
    std::vector<float> data;
    int nx, ny, nz;
    Vector3 spacing;
    Vector3 origin;
    Matrix3x3 direction;
    Matrix3x3 invDirection;

    CTVolume();

    // Existing MHD loader
    bool loadMHD(const std::string& mhdPath);

    // New: generic dispatcher
    bool loadVolume(const std::string& path);
    bool loadNIfTI(const std::string& niiPath);

    bool loadRawData(const std::string& rawPath, const std::string& elemType);
    float interpolate(float x, float y, float z) const;
    Vector3 indexToWorld(const Vector3& idx) const;
    Vector3 worldToIndex(const Vector3& world) const;
    void getWorldBounds(Vector3& wmin, Vector3& wmax) const;
    void printAllCorners() const;
    Vector3 getCenter() const;
};

#endif // CT_VOLUME_H
