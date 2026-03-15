#include "RayMarcher.h"
#include <algorithm>

const float RAY_STEP_SIZE = 0.0f;
const float MIN_HU_THRESHOLD = 200.0f;
const float MU_WATER = 0.022f;

RayMarcher::RayMarcher() : stepSize(RAY_STEP_SIZE) {}

bool RayMarcher::rayBoxIntersect(const Vector3& rayOrigin, const Vector3& rayDir,
                     const Vector3& boxMin, const Vector3& boxMax,
                     float& t0, float& t1) const {
    Vector3 invDir(1.0f/(rayDir.x + 1e-12f), 1.0f/(rayDir.y + 1e-12f), 1.0f/(rayDir.z + 1e-12f));
    
    Vector3 t1v = (boxMin - rayOrigin);
    t1v.x *= invDir.x; t1v.y *= invDir.y; t1v.z *= invDir.z;
    
    Vector3 t2v = (boxMax - rayOrigin);
    t2v.x *= invDir.x; t2v.y *= invDir.y; t2v.z *= invDir.z;
    
    float tmin = std::max({std::min(t1v.x, t2v.x), std::min(t1v.y, t2v.y), std::min(t1v.z, t2v.z)});
    float tmax = std::min({std::max(t1v.x, t2v.x), std::max(t1v.y, t2v.y), std::max(t1v.z, t2v.z)});
    
    if (tmax < 0 || tmin > tmax) return false;
    
    t0 = std::max(tmin, 0.0f);
    t1 = tmax;
    return true;
}

float RayMarcher::huToMu(float hu) const {
    return (hu <= MIN_HU_THRESHOLD) ? 0.0f : (hu / 1000.0f) * MU_WATER + MU_WATER;
}

float RayMarcher::marchRay(const Vector3& rayOrigin, const Vector3& rayDir,
               const CTVolume& ct, const Vector3& boxMin, const Vector3& boxMax) const {
    float t0, t1;
    if (!rayBoxIntersect(rayOrigin, rayDir, boxMin, boxMax, t0, t1)) return 0.0f;
    
    float sum = 0.0f;
    int count = 0;
    
    for (float t = t0; t <= t1; t += stepSize) {
        Vector3 worldPos = rayOrigin + rayDir * t;
        Vector3 idx = ct.worldToIndex(worldPos);
        float hu = ct.interpolate(idx.x, idx.y, idx.z);
        sum += huToMu(hu);
        count++;
    }
    
    return (count > 0) ? (sum / count) : 0.0f;
}
