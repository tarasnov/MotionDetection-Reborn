#include <gtest/gtest.h>
#include <filesystem>
#include <fstream>

#include "model.h"

namespace {

bool FileExists(const std::string file_path) {
    std::ifstream file(file_path);
    return !!file;
}

}  // namespace

namespace fs = std::filesystem;

// A simple test case
TEST(SampleTest, Load) {
    fs::path model_path = fs::current_path() / "../data/model-kpoint-input-1x15x640x960_simplified_compressed.hef";
    EXPECT_TRUE(FileExists(model_path));
    
    Model model("dummy_path");
}
