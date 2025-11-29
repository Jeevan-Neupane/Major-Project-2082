#include <iostream>
#include <string>
#include <algorithm>
#include <sys/stat.h>
#include <filesystem>
#include "DRRGenerator.h"

const float RAY_STEP_SIZE = 0.0f;
const bool OUTPUT_PNG = true;

// Helper to create directory
void ensureDirectoryExists(const std::string& path) {
    std::error_code ec;
    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path, ec);
        if (ec) {
            std::cerr << "Failed to create directory '" << path << "': " << ec.message() << std::endl;
        }
    }
}

// Helper to extract filename stem
std::string getFilenameStem(const std::string& path) {
    size_t lastSlash = path.find_last_of("/\\");
    std::string filename = (lastSlash == std::string::npos) ? path : path.substr(lastSlash + 1);
    size_t lastDot = filename.find_last_of('.');
    if (lastDot != std::string::npos) {
        return filename.substr(0, lastDot);
    }
    return filename;
}

void generateView(DRRGenerator& generator, const std::string& axis, const std::string& outputDir, const std::string& viewName) {
    std::cout << "\nGenerating " << viewName << " view (Source Axis: " << axis << ")..." << std::endl;
    
    Vector3 ctCenter = generator.ct.getCenter();
    generator.camera.setup(generator.ct, axis);
    generator.camera.printSetupInfo(ctCenter, generator.boxMin, generator.boxMax, generator.ct);
    
    generator.generate();
    generator.postProcess();
    
    if (OUTPUT_PNG) {
        std::string filename = outputDir + "/" + viewName + ".png";
        generator.saveAsPNG(filename);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <input.(mhd|nii|nii.gz)> [source_axis]" << std::endl;
        std::cerr << "  source_axis (optional): +X, -X, +Y, -Y, +Z, or -Z" << std::endl;
        std::cerr << "  If source_axis is NOT provided, generates AP (-Y) and Lat (+X) views." << std::endl;
        return 1;
    }
    
    std::string volumePath = argv[1];
    std::string specificAxis = "";
    bool generateAll = true;
    
    if (argc >= 3) {
        specificAxis = argv[2];
        generateAll = false;
        
        // Validate source axis
        if (specificAxis != "+X" && specificAxis != "-X" && 
            specificAxis != "+Y" && specificAxis != "-Y" && 
            specificAxis != "+Z" && specificAxis != "-Z") {
            std::cerr << "ERROR: Invalid source axis '" << specificAxis << "'. Use: +X, -X, +Y, -Y, +Z, or -Z" << std::endl;
            return 1;
        }
    }
    
    DRRGenerator generator;
    
    if (!generator.ct.loadVolume(volumePath)) {
        std::cerr << "Failed to load volume file: " << volumePath << std::endl;
        return 1;
    }
    
    std::cout << "\n=== CT Volume Information ===" << std::endl;
    std::cout << "Dimensions: " << generator.ct.nx << " x " << generator.ct.ny << " x " << generator.ct.nz << std::endl;
    std::cout << "Spacing: (" << generator.ct.spacing.x << ", " << generator.ct.spacing.y << ", " << generator.ct.spacing.z << ") mm" << std::endl;
    std::cout << "Origin: (" << generator.ct.origin.x << ", " << generator.ct.origin.y << ", " << generator.ct.origin.z << ") mm" << std::endl;
    std::cout << "============================\n" << std::endl;
    
    generator.ct.getWorldBounds(generator.boxMin, generator.boxMax);
    
    if (RAY_STEP_SIZE <= 0.0f) {
        generator.marcher.stepSize = std::min({
            generator.ct.spacing.x, generator.ct.spacing.y, generator.ct.spacing.z
        });
    }
    std::cout << "Ray Step Size: " << generator.marcher.stepSize << " mm\n" << std::endl;

    // Prepare output directory
    std::string stem = getFilenameStem(volumePath);
    ensureDirectoryExists("outputs");
    std::string outputDir = "outputs/" + stem;
    ensureDirectoryExists(outputDir);
    
    if (generateAll) {
        // Choose axes based on file extension (.nii / .nii.gz use reversed conventions)
        auto endsWithCI = [&](const std::string& s, const std::string& ext){
            if (s.size() < ext.size()) return false;
            for (size_t i=0;i<ext.size();++i){
                char a = std::tolower(s[s.size()-ext.size()+i]);
                char b = std::tolower(ext[i]);
                if (a!=b) return false;
            }
            return true;
        };
        bool isNifti = endsWithCI(volumePath, ".nii") || endsWithCI(volumePath, ".nii.gz");

        std::string apAxis  = isNifti ? "+Y" : "-Y";  // AP view axis
        std::string latAxis = isNifti ? "-X" : "+X";  // Lateral view axis

        generateView(generator, apAxis, outputDir, "AP");
        generateView(generator, latAxis, outputDir, "Lat");
    } else {
        // Generate specific axis
        // Sanitize axis name for filename (though + and - are usually fine)
        std::string viewName = specificAxis; 
        generateView(generator, specificAxis, outputDir, viewName);
    }
    
    std::cout << "\nGeneration complete! Outputs saved to: " << outputDir << std::endl;
    
    return 0;
}
