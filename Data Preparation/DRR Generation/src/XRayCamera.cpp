#include "XRayCamera.h"
#include <iostream>

// Configuration Parameters
const float SOURCE_DISTANCE_FROM_CT = 4000.0f;
const float DETECTOR_DISTANCE_FROM_CT = 2000.0f;
const int DETECTOR_ROWS = 1024;
const int DETECTOR_COLS = 1024;
const float DETECTOR_PIXEL_SIZE = 1.0f;

void XRayCamera::setup(const CTVolume& ct, const std::string& axis) {
    rows = DETECTOR_ROWS;
    cols = DETECTOR_COLS;
    pixelSize = DETECTOR_PIXEL_SIZE;
    sourceAxis = axis;
    
    Vector3 ctCenter = ct.getCenter();
    
    Vector3 sourceDirection(0, 1, 0);
    
    if (axis == "-X") sourceDirection = Vector3(-1, 0, 0);
    else if (axis == "+X") sourceDirection = Vector3(1, 0, 0);
    else if (axis == "-Y") sourceDirection = Vector3(0, -1, 0);
    else if (axis == "+Y") sourceDirection = Vector3(0, 1, 0);
    else if (axis == "-Z") sourceDirection = Vector3(0, 0, -1);
    else if (axis == "+Z") sourceDirection = Vector3(0, 0, 1);
    
    sourcePos = ctCenter + sourceDirection * SOURCE_DISTANCE_FROM_CT;
    detectorCenter = ctCenter - sourceDirection * DETECTOR_DISTANCE_FROM_CT;
    lookDir = (detectorCenter - sourcePos).normalize();
    
   if (axis == "-X") {
            uDir = Vector3(0, -1, 0);   
            vDir = Vector3(0, 0, -1);  
        } else if (axis == "+X") {
            uDir = Vector3(0, 1, 0);  
            vDir = Vector3(0, 0, -1); 
        } else if (axis == "-Y") {
            uDir = Vector3(1, 0, 0);  
            vDir = Vector3(0, 0, -1);  
        } else if (axis == "+Y") {
            uDir = Vector3(-1, 0, 0);   
            vDir = Vector3(0, 0, -1); 
        } else if (axis == "-Z") {
            uDir = Vector3(1, 0, 0);   
            vDir = Vector3(0, 1, 0);  
        } else if (axis == "+Z") {
            uDir = Vector3(-1, 0, 0); 
            vDir = Vector3(0, 1, 0); 
        }
    
    uDir = uDir.normalize();
    vDir = vDir.normalize();
    
    
    uCoords.resize(cols);
    vCoords.resize(rows);
    for (int c = 0; c < cols; c++) uCoords[c] = (c - (cols-1)/2.0f) * pixelSize;
    for (int r = 0; r < rows; r++) vCoords[r] = (r - (rows-1)/2.0f) * pixelSize;
}

Vector3 XRayCamera::getPixelWorldPos(int row, int col) const {
    return detectorCenter + uDir * uCoords[col] + vDir * vCoords[row];
}

std::string XRayCamera::getDetectorAxis() const {
    if (sourceAxis == "+X") return "-X";
    if (sourceAxis == "-X") return "+X";
    if (sourceAxis == "+Y") return "-Y";
    if (sourceAxis == "-Y") return "+Y";
    if (sourceAxis == "+Z") return "-Z";
    if (sourceAxis == "-Z") return "+Z";
    return "unknown";
}

void XRayCamera::printSetupInfo(const Vector3& ctCenter, const Vector3& boxMin, const Vector3& boxMax, const CTVolume& ct) const {
    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "                 X-RAY CAMERA SETUP" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
    
    std::cout << "\n[AXES CONFIGURATION]" << std::endl;
    std::cout << "  Source Axis:   " << sourceAxis << std::endl;
    std::cout << "  Detector Axis: " << getDetectorAxis() << std::endl;
    
    std::cout << "\n[KEY POSITIONS (World Coordinates in mm)]" << std::endl;
    std::cout << "  CT Center:       (" << ctCenter.x << ", " << ctCenter.y << ", " << ctCenter.z << ")" << std::endl;
    std::cout << "  Source Position: (" << sourcePos.x << ", " << sourcePos.y << ", " << sourcePos.z << ")" << std::endl;
    std::cout << "  Detector Center: (" << detectorCenter.x << ", " << detectorCenter.y << ", " << detectorCenter.z << ")" << std::endl;
    
    std::cout << "\n[CT SCAN AABB BOX (World Coordinates in mm)]" << std::endl;
    std::cout << "  Min Corner: (" << boxMin.x << ", " << boxMin.y << ", " << boxMin.z << ")" << std::endl;
    std::cout << "  Max Corner: (" << boxMax.x << ", " << boxMax.y << ", " << boxMax.z << ")" << std::endl;
    std::cout << "  Size: " << (boxMax.x - boxMin.x) << " x " 
              << (boxMax.y - boxMin.y) << " x " << (boxMax.z - boxMin.z) << " mm" << std::endl;
    
    // Print all 8 CT corners
    ct.printAllCorners();
    
    Vector3 topLeft = getPixelWorldPos(0, 0);
    Vector3 topRight = getPixelWorldPos(0, cols-1);
    Vector3 bottomLeft = getPixelWorldPos(rows-1, 0);
    Vector3 bottomRight = getPixelWorldPos(rows-1, cols-1);
    
    std::cout << "\n[DETECTOR CORNERS (World Coordinates in mm)]" << std::endl;
    std::cout << "  Top-Left:     (" << topLeft.x << ", " << topLeft.y << ", " << topLeft.z << ")" << std::endl;
    std::cout << "  Top-Right:    (" << topRight.x << ", " << topRight.y << ", " << topRight.z << ")" << std::endl;
    std::cout << "  Bottom-Left:  (" << bottomLeft.x << ", " << bottomLeft.y << ", " << bottomLeft.z << ")" << std::endl;
    std::cout << "  Bottom-Right: (" << bottomRight.x << ", " << bottomRight.y << ", " << bottomRight.z << ")" << std::endl;
    
    std::cout << "\n[PNG IMAGE MAPPING]" << std::endl;
    std::cout << "  Image dimensions: " << cols << " x " << rows << " pixels" << std::endl;
    std::cout << "  Pixel size: " << pixelSize << " mm" << std::endl;
    std::cout << "  Physical size: " << (cols * pixelSize) << " x " << (rows * pixelSize) << " mm" << std::endl;
    std::cout << "\n  First pixel in PNG (Top-Left corner):" << std::endl;
    std::cout << "    Image coordinate: [row=0, col=0]" << std::endl;
    std::cout << "    World position: (" << topLeft.x << ", " << topLeft.y << ", " << topLeft.z << ") mm" << std::endl;
    
    std::cout << "\n  PNG corners map to world as:" << std::endl;
    std::cout << "    PNG[0,0] (Top-Left)           -> World(" << topLeft.x << ", " << topLeft.y << ", " << topLeft.z << ")" << std::endl;
    std::cout << "    PNG[0," << (cols-1) << "] (Top-Right)      -> World(" << topRight.x << ", " << topRight.y << ", " << topRight.z << ")" << std::endl;
    std::cout << "    PNG[" << (rows-1) << ",0] (Bottom-Left)   -> World(" << bottomLeft.x << ", " << bottomLeft.y << ", " << bottomLeft.z << ")" << std::endl;
    std::cout << "    PNG[" << (rows-1) << "," << (cols-1) << "] (Bottom-Right) -> World(" << bottomRight.x << ", " << bottomRight.y << ", " << bottomRight.z << ")" << std::endl;
    
    std::cout << "\n" << std::string(60, '=') << "\n" << std::endl;
}
