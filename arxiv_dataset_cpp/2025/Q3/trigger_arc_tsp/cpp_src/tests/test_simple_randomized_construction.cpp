#include <gtest/gtest.h>

#include <filesystem>

#include "instance.hpp"
#include "simple_randomized_construction.hpp"

namespace fs = std::filesystem;

class SimpleRandomizedConstructionTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Standard logic to find test instances from the project root
        fs::path currentPath = fs::current_path();
        while (currentPath.has_parent_path() && !fs::exists(currentPath / "cpp_src")) {
            currentPath = currentPath.parent_path();
        }
        cppSourceDir_ = currentPath / "cpp_src";
        examplePath_ = cppSourceDir_ / "tests" / "instances" / "example_1.txt";
    }

    fs::path cppSourceDir_;
    fs::path examplePath_;
};

TEST_F(SimpleRandomizedConstructionTest, SolvesExampleInstance) {
    auto inst = Instance::loadInstanceFromFile(examplePath_.string());
    ASSERT_NE(inst, nullptr);

    SimpleRandomizedConstruction heuristic(*inst);
    heuristic.run();
    auto tour = heuristic.getSolution();

    if (!tour.empty()) {
        ASSERT_FALSE(tour.empty());
        EXPECT_EQ(tour.size(), inst->getN());
        EXPECT_TRUE(inst->checkSolutionCorrectness(tour));
    }
}