#ifndef RAY_MARCHER_H
#define RAY_MARCHER_H

#include "MathUtils.h"
#include "CTVolume.h"

class RayMarcher {
public:
    float stepSize;
    
    RayMarcher();
    
    bool rayBoxIntersect(const Vector3& rayOrigin, const Vector3& rayDir,
                         const Vector3& boxMin, const Vector3& boxMax,
                         float& t0, float& t1) const;
    
    float huToMu(float hu) const;
    
    float marchRay(const Vector3& rayOrigin, const Vector3& rayDir,
                   const CTVolume& ct, const Vector3& boxMin, const Vector3& boxMax) const;
};

#endif // RAY_MARCHER_H
