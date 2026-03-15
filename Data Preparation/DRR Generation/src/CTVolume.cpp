#include "CTVolume.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cmath>
#include <cstring>
#include <zlib.h>            // added for .nii.gz
#include <cctype>            // for std::tolower
#include <cstdint>           // FIX: uint16_t, uint32_t, int8_t, etc.

// Configuration Parameters (needed for interpolate)
// Ideally these should be passed in or part of the class, but for now keeping as constants or members?
// The original code had them as global constants.
// I will keep the logic inside the class but the constants were used in interpolate?
// Wait, interpolate used hardcoded -1000.0f.
// The constants were:
// const bool USE_METADATA_OFFSET = false;

// I'll define the constant here locally or make it a member if needed.
// For now, I'll follow the original logic.
const bool USE_METADATA_OFFSET = false;

CTVolume::CTVolume() : nx(0), ny(0), nz(0) {}

bool CTVolume::loadMHD(const std::string& mhdPath) {
    std::ifstream file(mhdPath);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open file: " << mhdPath << std::endl;
        return false;
    }
    
    std::map<std::string, std::string> metadata;
    std::string line;
    
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        size_t pos = line.find('=');
        if (pos != std::string::npos) {
            std::string key = line.substr(0, pos);
            std::string value = line.substr(pos + 1);
            key.erase(0, key.find_first_not_of(" \t\r\n"));
            key.erase(key.find_last_not_of(" \t\r\n") + 1);
            value.erase(0, value.find_first_not_of(" \t\r\n"));
            value.erase(value.find_last_not_of(" \t\r\n") + 1);
            metadata[key] = value;
        }
    }
    file.close();
    
    if (metadata.count("DimSize")) {
        std::istringstream iss(metadata["DimSize"]);
        iss >> nx >> ny >> nz;
    } else {
        std::cerr << "ERROR: Missing DimSize in MHD file" << std::endl;
        return false;
    }
    
    if (metadata.count("ElementSpacing")) {
        std::istringstream iss(metadata["ElementSpacing"]);
        iss >> spacing.x >> spacing.y >> spacing.z;
    } else spacing = Vector3(1.0f, 1.0f, 1.0f);
    
    Vector3 metadataOrigin(0.0f, 0.0f, 0.0f);
    if (metadata.count("Offset") || metadata.count("Origin")) {
        std::string originKey = metadata.count("Offset") ? "Offset" : "Origin";
        std::istringstream iss(metadata[originKey]);
        iss >> metadataOrigin.x >> metadataOrigin.y >> metadataOrigin.z;
    }
    
    origin = USE_METADATA_OFFSET ? metadataOrigin : Vector3(0.0f, 0.0f, 0.0f);
    
    if (metadata.count("TransformMatrix") || metadata.count("Orientation")) {
        std::string dirKey = metadata.count("TransformMatrix") ? "TransformMatrix" : "Orientation";
        std::istringstream iss(metadata[dirKey]);
        for (int i = 0; i < 9; i++) iss >> direction.m[i];
        invDirection = direction.inverse();
    } else {
        direction = Matrix3x3::identity();
        invDirection = Matrix3x3::identity();
    }
    
    std::string rawFile;
    if (metadata.count("ElementDataFile")) {
        rawFile = metadata["ElementDataFile"];
        if (rawFile.find('/') != 0 && rawFile.find(':') == std::string::npos) {
            size_t lastSlash = mhdPath.find_last_of("/\\");
            if (lastSlash != std::string::npos) {
                rawFile = mhdPath.substr(0, lastSlash + 1) + rawFile;
            }
        }
    } else {
        std::cerr << "ERROR: Missing ElementDataFile in MHD file" << std::endl;
        return false;
    }
    
    std::string elemType = metadata.count("ElementType") ? metadata["ElementType"] : "MET_SHORT";
    std::cout << "Loading data type: " << elemType << std::endl;
    
    return loadRawData(rawFile, elemType);
}

bool CTVolume::loadRawData(const std::string& rawPath, const std::string& elemType) {
    std::ifstream file(rawPath, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open raw file: " << rawPath << std::endl;
        return false;
    }
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    size_t totalVoxels = static_cast<size_t>(nx) * ny * nz;
    data.resize(totalVoxels);
    
    size_t bytesPerElement;
    if (elemType == "MET_SHORT" || elemType == "MET_INT16") {
        bytesPerElement = sizeof(short);
    } else if (elemType == "MET_USHORT" || elemType == "MET_UINT16") {
        bytesPerElement = sizeof(unsigned short);
    } else if (elemType == "MET_FLOAT") {
        bytesPerElement = sizeof(float);
    } else if (elemType == "MET_DOUBLE") {
        bytesPerElement = sizeof(double);
    } else {
        std::cerr << "ERROR: Unsupported element type: " << elemType << std::endl;
        return false;
    }
    
    size_t expectedBytes = totalVoxels * bytesPerElement;
    
    if (fileSize < expectedBytes) {
        std::cerr << "ERROR: File too small. Expected " << expectedBytes 
                  << " bytes, got " << fileSize << std::endl;
        return false;
    }
    
    if (elemType == "MET_SHORT" || elemType == "MET_INT16") {
        std::vector<short> buffer(totalVoxels);
        file.read(reinterpret_cast<char*>(buffer.data()), totalVoxels * sizeof(short));
        for (size_t i = 0; i < totalVoxels; i++) data[i] = static_cast<float>(buffer[i]);
    } else if (elemType == "MET_USHORT" || elemType == "MET_UINT16") {
        std::vector<unsigned short> buffer(totalVoxels);
        file.read(reinterpret_cast<char*>(buffer.data()), totalVoxels * sizeof(unsigned short));
        for (size_t i = 0; i < totalVoxels; i++) data[i] = static_cast<float>(buffer[i]);
    } else if (elemType == "MET_FLOAT") {
        file.read(reinterpret_cast<char*>(data.data()), totalVoxels * sizeof(float));
    } else if (elemType == "MET_DOUBLE") {
        std::vector<double> buffer(totalVoxels);
        file.read(reinterpret_cast<char*>(buffer.data()), totalVoxels * sizeof(double));
        for (size_t i = 0; i < totalVoxels; i++) data[i] = static_cast<float>(buffer[i]);
    }
    
    file.close();
    return true;
}

float CTVolume::interpolate(float x, float y, float z) const {
    if (x < 0 || y < 0 || z < 0 || x > nx-2 || y > ny-2 || z > nz-2) return -1000.0f;
    
    int x0 = static_cast<int>(std::floor(x));
    int y0 = static_cast<int>(std::floor(y));
    int z0 = static_cast<int>(std::floor(z));
    
    float xd = x - x0, yd = y - y0, zd = z - z0;
    
    auto getVoxel = [&](int xi, int yi, int zi) -> float {
        return data[zi * ny * nx + yi * nx + xi];
    };
    
    float c000 = getVoxel(x0, y0, z0);
    float c100 = getVoxel(x0+1, y0, z0);
    float c010 = getVoxel(x0, y0+1, z0);
    float c110 = getVoxel(x0+1, y0+1, z0);
    float c001 = getVoxel(x0, y0, z0+1);
    float c101 = getVoxel(x0+1, y0, z0+1);
    float c011 = getVoxel(x0, y0+1, z0+1);
    float c111 = getVoxel(x0+1, y0+1, z0+1);
    
    float c00 = c000*(1-xd) + c100*xd;
    float c10 = c010*(1-xd) + c110*xd;
    float c01 = c001*(1-xd) + c101*xd;
    float c11 = c011*(1-xd) + c111*xd;
    float c0 = c00*(1-yd) + c10*yd;
    float c1 = c01*(1-yd) + c11*yd;
    
    return c0*(1-zd) + c1*zd;
}

Vector3 CTVolume::indexToWorld(const Vector3& idx) const {
    Vector3 scaled(idx.x * spacing.x, idx.y * spacing.y, idx.z * spacing.z);
    return direction * scaled + origin;
}

Vector3 CTVolume::worldToIndex(const Vector3& world) const {
    Vector3 diff = world - origin;
    Vector3 rotated = invDirection * diff;
    return Vector3(rotated.x / spacing.x, rotated.y / spacing.y, rotated.z / spacing.z);
}

void CTVolume::getWorldBounds(Vector3& wmin, Vector3& wmax) const {
    Vector3 corners[8] = {
        indexToWorld(Vector3(0, 0, 0)),
        indexToWorld(Vector3(nx-1, 0, 0)),
        indexToWorld(Vector3(0, ny-1, 0)),
        indexToWorld(Vector3(0, 0, nz-1)),
        indexToWorld(Vector3(nx-1, ny-1, 0)),
        indexToWorld(Vector3(nx-1, 0, nz-1)),
        indexToWorld(Vector3(0, ny-1, nz-1)),
        indexToWorld(Vector3(nx-1, ny-1, nz-1))
    };
    
    wmin = wmax = corners[0];
    for (int i = 1; i < 8; i++) {
        wmin.x = std::min(wmin.x, corners[i].x);
        wmin.y = std::min(wmin.y, corners[i].y);
        wmin.z = std::min(wmin.z, corners[i].z);
        wmax.x = std::max(wmax.x, corners[i].x);
        wmax.y = std::max(wmax.y, corners[i].y);
        wmax.z = std::max(wmax.z, corners[i].z);
    }
}

void CTVolume::printAllCorners() const {
    std::cout << "\n[CT SCAN 8 CORNER COORDINATES (World Space in mm)]" << std::endl;
    std::cout << "  Corner 0 [0, 0, 0]:             (" 
              << indexToWorld(Vector3(0, 0, 0)).x << ", "
              << indexToWorld(Vector3(0, 0, 0)).y << ", "
              << indexToWorld(Vector3(0, 0, 0)).z << ")" << std::endl;
    
    std::cout << "  Corner 1 [nx-1, 0, 0]:          (" 
              << indexToWorld(Vector3(nx-1, 0, 0)).x << ", "
              << indexToWorld(Vector3(nx-1, 0, 0)).y << ", "
              << indexToWorld(Vector3(nx-1, 0, 0)).z << ")" << std::endl;
    
    std::cout << "  Corner 2 [0, ny-1, 0]:          (" 
              << indexToWorld(Vector3(0, ny-1, 0)).x << ", "
              << indexToWorld(Vector3(0, ny-1, 0)).y << ", "
              << indexToWorld(Vector3(0, ny-1, 0)).z << ")" << std::endl;
    
    std::cout << "  Corner 3 [0, 0, nz-1]:          (" 
              << indexToWorld(Vector3(0, 0, nz-1)).x << ", "
              << indexToWorld(Vector3(0, 0, nz-1)).y << ", "
              << indexToWorld(Vector3(0, 0, nz-1)).z << ")" << std::endl;
    
    std::cout << "  Corner 4 [nx-1, ny-1, 0]:       (" 
              << indexToWorld(Vector3(nx-1, ny-1, 0)).x << ", "
              << indexToWorld(Vector3(nx-1, ny-1, 0)).y << ", "
              << indexToWorld(Vector3(nx-1, ny-1, 0)).z << ")" << std::endl;
    
    std::cout << "  Corner 5 [nx-1, 0, nz-1]:       (" 
              << indexToWorld(Vector3(nx-1, 0, nz-1)).x << ", "
              << indexToWorld(Vector3(nx-1, 0, nz-1)).y << ", "
              << indexToWorld(Vector3(nx-1, 0, nz-1)).z << ")" << std::endl;
    
    std::cout << "  Corner 6 [0, ny-1, nz-1]:       (" 
              << indexToWorld(Vector3(0, ny-1, nz-1)).x << ", "
              << indexToWorld(Vector3(0, ny-1, nz-1)).y << ", "
              << indexToWorld(Vector3(0, ny-1, nz-1)).z << ")" << std::endl;
    
    std::cout << "  Corner 7 [nx-1, ny-1, nz-1]:    (" 
              << indexToWorld(Vector3(nx-1, ny-1, nz-1)).x << ", "
              << indexToWorld(Vector3(nx-1, ny-1, nz-1)).y << ", "
              << indexToWorld(Vector3(nx-1, ny-1, nz-1)).z << ")" << std::endl;
}

Vector3 CTVolume::getCenter() const {
    return indexToWorld(Vector3(nx/2.0f, ny/2.0f, nz/2.0f));
}

bool CTVolume::loadVolume(const std::string& path) {
    auto endsWith = [&](const std::string& ext){
        if (path.size() < ext.size()) return false;
        return std::equal(ext.rbegin(), ext.rend(), path.rbegin(),
                          [](char a,char b){ return std::tolower(a)==std::tolower(b); });
    };
    if (endsWith(".mhd")) return loadMHD(path);
    if (endsWith(".nii") || endsWith(".nii.gz")) return loadNIfTI(path);
    std::cerr << "ERROR: Unsupported volume extension: " << path << std::endl;
    return false;
}

// Minimal NIfTI-1 header (first 348 bytes)
#pragma pack(push,1)
struct nifti_1_header {
    int32_t sizeof_hdr;
    char    data_type[10];
    char    db_name[18];
    int32_t extents;
    int16_t session_error;
    char    regular;
    char    dim_info;
    int16_t dim[8];
    float   intent_p1;
    float   intent_p2;
    float   intent_p3;
    int16_t intent_code;
    int16_t datatype;
    int16_t bitpix;
    int16_t slice_start;
    float   pixdim[8];
    float   vox_offset;
    float   scl_slope;
    float   scl_inter;
    float   slice_end;
    char    slice_code;
    char    xyzt_units;
    float   cal_max;
    float   cal_min;
    float   slice_duration;
    float   toffset;
    int32_t glmax;
    int32_t glmin;
    char    descrip[80];
    char    aux_file[24];
    int16_t qform_code;
    int16_t sform_code;
    float   quatern_b;
    float   quatern_c;
    float   quatern_d;
    float   qoffset_x;
    float   qoffset_y;
    float   qoffset_z;
    float   srow_x[4];
    float   srow_y[4];
    float   srow_z[4];
    char    intent_name[16];
    char    magic[4];
};
#pragma pack(pop)

// Map nifti datatype -> bytes and conversion lambda
static bool niftiDatatypeInfo(int16_t code, int &bytes, bool &isFloat, bool &isSigned) {
    switch(code) {
        case 2:  bytes=1; isFloat=false; isSigned=false; return true; // DT_UNSIGNED_CHAR
        case 256: bytes=1; isFloat=false; isSigned=true; return true; // DT_INT8
        case 4:  bytes=2; isFloat=false; isSigned=true; return true; // DT_INT16
        case 512: bytes=2; isFloat=false; isSigned=false; return true; // DT_UINT16
        case 8:  bytes=4; isFloat=false; isSigned=true; return true; // DT_INT32
        case 768: bytes=4; isFloat=false; isSigned=false; return true; // DT_UINT32
        case 16: bytes=4; isFloat=true; isSigned=true; return true; // DT_FLOAT32
        case 64: bytes=8; isFloat=true; isSigned=true; return true; // DT_FLOAT64
        default: return false;
    }
}

static uint16_t bswap16(uint16_t v){ return (v>>8) | (v<<8); }
static uint32_t bswap32(uint32_t v){ return (v>>24) | ((v>>8)&0xFF00) | ((v<<8)&0xFF0000) | (v<<24); }
static float bswapFloat(float f){ uint32_t u=*reinterpret_cast<uint32_t*>(&f); u=bswap32(u); return *reinterpret_cast<float*>(&u); }

bool CTVolume::loadNIfTI(const std::string& niiPath) {
    bool gz = (niiPath.size() >= 7 &&
               (niiPath.substr(niiPath.size()-7) == ".nii.gz" ||
                niiPath.substr(niiPath.size()-7) == ".NII.GZ"));

    auto readHeader = [&](unsigned char* buf, size_t want, gzFile gzF, std::ifstream* fs)->bool{
        if (gzF) {
            return gzread(gzF, buf, (int)want) == (int)want;
        } else {
            fs->read(reinterpret_cast<char*>(buf), want);
            return fs->gcount() == (std::streamsize)want;
        }
    };

    unsigned char header[540] = {0};
    gzFile gzF = nullptr;
    std::ifstream fs;

    if (gz) {
        gzF = gzopen(niiPath.c_str(),"rb");
        if (!gzF) { std::cerr << "ERROR: Cannot open gz NIfTI: " << niiPath << std::endl; return false; }
        if (!readHeader(header, 348, gzF, nullptr)) { std::cerr << "ERROR: Failed reading first 348 bytes\n"; gzclose(gzF); return false; }
    } else {
        fs.open(niiPath, std::ios::binary);
        if (!fs.is_open()) { std::cerr << "ERROR: Cannot open NIfTI file: " << niiPath << std::endl; return false; }
        if (!readHeader(header, 348, nullptr, &fs)) { std::cerr << "ERROR: Failed reading first 348 bytes\n"; return false; }
    }

    int32_t sizeof_hdr = *reinterpret_cast<int32_t*>(header);
    bool needSwap = false;
    if (sizeof_hdr != 348 && sizeof_hdr != 540) {
        int32_t swapped = (int32_t)bswap32(sizeof_hdr);
        if (swapped == 348 || swapped == 540) {
            needSwap = true;
            sizeof_hdr = swapped;
        }
    }

    bool isNifti2 = (sizeof_hdr == 540);
    if (isNifti2) {
        // Need remainder of header (we already read 348, read 192 more to reach 540)
        size_t already = 348;
        size_t remain = 540 - already;
        if (!gz ? !readHeader(header+already, remain, nullptr, &fs)
                : !readHeader(header+already, remain, gzF, nullptr)) {
            std::cerr << "ERROR: Failed reading full 540-byte NIfTI-2 header\n";
            if (gzF) gzclose(gzF);
            return false;
        }
    }

    // Extract magic
    std::string magic;
    if (!isNifti2) {
        magic.assign(reinterpret_cast<char*>(header + 344), 4);
    } else {
        magic.assign(reinterpret_cast<char*>(header + 4), 4);
    }

    auto magicOK = [&](const std::string& m){
        return (m.rfind("n+1",0)==0 || m.rfind("ni1",0)==0 ||
                m.rfind("n+2",0)==0 || m.rfind("ni2",0)==0);
    };

    if (!magicOK(magic)) {
        std::cerr << "ERROR: Invalid NIfTI magic '" << magic << "' (sizeof_hdr=" << sizeof_hdr << ")\n";
        if (gzF) gzclose(gzF);
        return false;
    }

    // Dimensions & spacing
    if (!isNifti2) {
        int16_t* dim = reinterpret_cast<int16_t*>(header + 40);
        float* pixdim = reinterpret_cast<float*>(header + 76);
        if (needSwap) {
            for (int i=0;i<8;i++) dim[i] = (int16_t)bswap16(dim[i]);
            for (int i=0;i<8;i++) pixdim[i] = bswapFloat(pixdim[i]);
        }
        nx = dim[1]; ny = dim[2]; nz = dim[3];
        spacing = Vector3(pixdim[1], pixdim[2], pixdim[3]);

        int16_t datatype = *(int16_t*)(header + 70);
        if (needSwap) datatype = (int16_t)bswap16(datatype);
        float vox_offset = *(float*)(header + 108);
        float scl_slope = *(float*)(header + 112);
        float scl_inter = *(float*)(header + 116);
        if (needSwap) { vox_offset=bswapFloat(vox_offset); scl_slope=bswapFloat(scl_slope); scl_inter=bswapFloat(scl_inter); }

        int16_t sform_code = *(int16_t*)(header + 254);
        if (needSwap) sform_code = (int16_t)bswap16(sform_code);
        float* srow_x = reinterpret_cast<float*>(header + 280);
        float* srow_y = reinterpret_cast<float*>(header + 296);
        float* srow_z = reinterpret_cast<float*>(header + 312);
        if (needSwap) {
            for(int i=0;i<4;i++){ srow_x[i]=bswapFloat(srow_x[i]); srow_y[i]=bswapFloat(srow_y[i]); srow_z[i]=bswapFloat(srow_z[i]); }
        }

        if (sform_code > 0) {
            direction.m[0]=srow_x[0]; direction.m[1]=srow_y[0]; direction.m[2]=srow_z[0];
            direction.m[3]=srow_x[1]; direction.m[4]=srow_y[1]; direction.m[5]=srow_z[1];
            direction.m[6]=srow_x[2]; direction.m[7]=srow_y[2]; direction.m[8]=srow_z[2];
            origin = Vector3(srow_x[3], srow_y[3], srow_z[3]);
        } else {
            direction = Matrix3x3::identity();
            origin = Vector3(0,0,0);
        }
        invDirection = direction.inverse();

        int bytesPer; bool isFloat, isSigned;
        if (!niftiDatatypeInfo(datatype, bytesPer, isFloat, isSigned)) {
            std::cerr << "ERROR: Unsupported NIfTI datatype code: " << datatype << std::endl;
            if (gzF) gzclose(gzF);
            return false;
        }

        long long voxelCount = (long long)nx * ny * nz;
        data.resize(voxelCount);

        // Read voxel data
        auto readData = [&](unsigned char* dst, size_t n)->bool{
            if (gzF) {
                // Need to skip to vox_offset by reading/discarding if > header size
                if (vox_offset > 348) {
                    size_t skip = (size_t)vox_offset - 348;
                    std::vector<unsigned char> junk(skip);
                    if (gzread(gzF, junk.data(), (int)skip) != (int)skip) return false;
                }
                return gzread(gzF, dst, (int)n) == (int)n;
            } else {
                fs.seekg((std::streamoff)vox_offset, std::ios::beg);
                fs.read(reinterpret_cast<char*>(dst), n);
                return fs.gcount() == (std::streamsize)n;
            }
        };

        std::vector<unsigned char> buffer(voxelCount * bytesPer);
        if (!readData(buffer.data(), buffer.size())) {
            std::cerr << "ERROR: Failed reading voxel data\n";
            if (gzF) gzclose(gzF);
            return false;
        }
        if (gzF) gzclose(gzF);

        const unsigned char* p = buffer.data();
        float slope = (scl_slope == 0.0f ? 1.0f : scl_slope);
        float inter = scl_inter;
        for (long long i=0;i<voxelCount;i++) {
            float v=0.0f;
            if (datatype == 2) v = (float)p[i];
            else if (datatype == 256) v = (float)reinterpret_cast<const int8_t*>(p)[i];
            else if (datatype == 4) v = (float)reinterpret_cast<const int16_t*>(p)[i];
            else if (datatype == 512) v = (float)reinterpret_cast<const uint16_t*>(p)[i];
            else if (datatype == 8) v = (float)reinterpret_cast<const int32_t*>(p)[i];
            else if (datatype == 768) v = (float)reinterpret_cast<const uint32_t*>(p)[i];
            else if (datatype == 16) v = reinterpret_cast<const float*>(p)[i];
            else if (datatype == 64) v = (float)reinterpret_cast<const double*>(p)[i];
            v = v * slope + inter;
            data[i] = v;
        }
        return true;
    } else {
        std::cerr << "ERROR: NIfTI-2 not fully supported in this minimal loader (sizeof_hdr=540)\n";
        if (gzF) gzclose(gzF);
        return false;
    }
}
