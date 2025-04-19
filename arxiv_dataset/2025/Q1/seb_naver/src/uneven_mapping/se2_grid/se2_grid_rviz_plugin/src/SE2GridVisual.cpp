#include "se2_grid_rviz_plugin/SE2GridVisual.hpp"

#include <OGRE/OgreManualObject.h>
#include <OGRE/OgreMaterialManager.h>
#include <OGRE/OgreSceneManager.h>
#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreTechnique.h>
#include <OGRE/OgreTextureManager.h>
#include <OGRE/OgreVector3.h>
#include <rviz/ogre_helpers/billboard_line.h>
#include <rviz/uniform_string_stream.h>
#include <chrono>

#include "se2_grid_rviz_plugin/SE2GridColorMaps.hpp"

namespace se2_grid_rviz_plugin
{

SE2GridVisual::SE2GridVisual(Ogre::SceneManager* sceneManager, Ogre::SceneNode* parentNode) : manualObject_(0), haveMap_(false) {
  sceneManager_ = sceneManager;
  frameNode_ = parentNode->createChildSceneNode();

  // Create BillboardLine object.
  meshLines_.reset(new rviz::BillboardLine(sceneManager_, frameNode_));
}

SE2GridVisual::~SE2GridVisual()
{
  // Destroy the ManualObject if it was created.
  if (manualObject_) {
    sceneManager_->destroyManualObject(manualObject_);
    material_->unload();
    Ogre::MaterialManager::getSingleton().remove(material_->getName());
  }

  // Destroy the frame node.
  sceneManager_->destroySceneNode(frameNode_);
}

void SE2GridVisual::setMessage(const se2_grid_msgs::SE2Grid::ConstPtr& msg)
{
  // Convert se2 grid message.
  se2_grid::SE2GridRosConverter::fromMessage(*msg, map_);
  
  haveMap_ = true;
}

void SE2GridVisual::computeVisualization(float alpha, bool showGridLines, bool flatTerrain, std::string heightLayer, bool flatColor,
                                         bool noColor, Ogre::ColourValue meshColor, bool mapLayerColor, std::string colorLayer,
                                         std::string colorMap, bool useColorMap, bool invertColorMap, Ogre::ColourValue minColor,
                                         Ogre::ColourValue maxColor, bool autocomputeIntensity, float minIntensity, float maxIntensity, float gridLineThickness,
                                         float so2_value) {
  const auto startTime = std::chrono::high_resolution_clock::now();
  if (!haveMap_)
  {
    // std::cout << "\033[31m Unable to visualize se2 grid, no map data. Use setMessage() first! \033[0m" <<std::endl;
    return;
  }

  // Get list of layers and check if the requested ones are present.
  const std::vector<std::string> layerNames = map_.getLayers();
  if (layerNames.size() < 1)
  {
    // std::cout << "\033[31m Unable to visualize se2 grid, map must contain at least one layer." <<std::endl;
    return;
  }
  if ((!flatTerrain && !map_.exists(heightLayer)) || (!noColor && !flatColor && !map_.exists(colorLayer)))
  {
    // std::cout << "\033[31m Unable to visualize se2 grid, requested layer(s) not available." <<std::endl;
    return;
  }

  // std::cout<<"plugin: convertToDefaultStartIndex"<<std::endl;
  // Convert to simple format, makes things easier.
  map_.convertToDefaultStartIndex();
  // std::cout<<"convertToDefaultStartIndex done."<<std::endl;

  // Basic se2 grid data.
  const size_t rows = map_.getSizePos()(0);
  const size_t cols = map_.getSizePos()(1);
  if (rows < 2 || cols < 2) 
  {
    // std::cout << "\033[31m SE2Grid has not enough cells." <<std::endl;
    return;
  }
  const double resolution = map_.getResolutionPos();
  int yaw_idx = 0;
  map_.yaw2Index((double)so2_value, yaw_idx);

  int yaw_idx_h = 0;
  int yaw_idx_c = 0;
  std::string layer_choose_h = flatTerrain ? layerNames[0] : heightLayer;
  std::string layer_choose_c = flatTerrain ? layerNames[0] : colorLayer;
  if (map_.hasSO2(layer_choose_h))
    yaw_idx_h = yaw_idx;
  if (map_.hasSO2(layer_choose_c))
    yaw_idx_c = yaw_idx;
  const Eigen::MatrixXf &heightData = map_[layer_choose_h][yaw_idx_h];
  const Eigen::MatrixXf &colorData = map_[layer_choose_c][yaw_idx_c];

  // Reset and begin the manualObject (mesh).
  // For more information: https://www.ogre3d.org/docs/api/1.7/class_ogre_1_1_manual_object.html#details
  const size_t nVertices = cols * rows;
  initializeAndBeginManualObject(nVertices);

  // Reset the mesh lines.
  meshLines_->clear();
  if (showGridLines)
  {
    initializeMeshLines(cols, rows, resolution, alpha, gridLineThickness);
  }

  // Compute a mask of valid cells.
  MaskArray isValid = MaskArray::Ones(map_.getSizePos()(0), map_.getSizePos()(1));

  isValid = isValid && map_[layer_choose_h][yaw_idx_h].array().unaryExpr([](float v) { return std::isfinite(v); });
  isValid = isValid && map_[layer_choose_c][yaw_idx_c].array().unaryExpr([](float v) { return std::isfinite(v); });

  // Compute the display heights for each cell.
  Eigen::ArrayXXf heightOrFlatData;
  if (flatTerrain)
  {
    heightOrFlatData = Eigen::ArrayXXf::Zero(heightData.rows(), heightData.cols());
  } else
  {
    heightOrFlatData = heightData.array();
  }

  // Compute the color data for each cell.
  ColoringMethod coloringMethod;
  if (flatColor || noColor) {
    coloringMethod = ColoringMethod::FLAT;
  } else if(mapLayerColor) {
    coloringMethod = ColoringMethod::COLOR_LAYER;
  } else if (!useColorMap) {
    coloringMethod = ColoringMethod::INTENSITY_LAYER_MANUAL;
  } else if (!invertColorMap) {
    coloringMethod = ColoringMethod::INTENSITY_LAYER_COLORMAP;
  } else {
    coloringMethod = ColoringMethod::INTENSITY_LAYER_INVERTED_COLORMAP;
  }

  const auto colorValues = computeColorValues(heightData, colorData, coloringMethod, colorMap, meshColor,
                                              minIntensity, maxIntensity, autocomputeIntensity, minColor, maxColor);

  // Initialize loop constants.
  Eigen::Vector3d topLeft;
  map_.index2Pos(Eigen::Array3i(0, 0, 0), topLeft);

  Eigen::ArrayXXi indexToOgreIndex;
  indexToOgreIndex.setConstant(rows, cols, -1);

  int ogreIndex = 0;

  // Add vertices for mesh.
  for (size_t i = 0; i < rows; ++i) {
    for (size_t j = 0; j < cols; ++j) {
      if(!noColor) {
        std::vector<int> vertices;
        std::vector<Ogre::ColourValue> colors;

        // Add the vertex to the scene
        Eigen::Array2i index(i, j);
        if (!isValid(index(0), index(1)))
        {
          continue;
        }
        Eigen::Vector2d position = topLeft.head(2).array() - index.cast<double>() * resolution;
        manualObject_->position(position(0), position(1), heightOrFlatData(index(0), index(1)));

        const Ogre::ColourValue& color = colorValues(index(0), index(1));
        manualObject_->colour(color.r, color.g, color.b, alpha);

        indexToOgreIndex(index(0), index(1)) = ogreIndex;
        ogreIndex++;

        // We can only add triangles to the top left side of the current vertex if we have data.
        if (i == 0 || j == 0) {
          continue;
        }

        // Add triangles and grid to scene.
        std::vector<int> vertexIndices;
        for (size_t k = 0; k < 2; k++) {
          for (size_t l = 0; l < 2; l++) {
            Eigen::Array2i index(i - k, j - l);
            if (!isValid(index(0), index(1))) {
              continue;
            }
            vertexIndices.emplace_back(indexToOgreIndex(index(0), index(1)));
          }
        }

        // Plot triangles if we have enough vertices.
        if (vertexIndices.size() > 2) {
          // Create one or two triangles from the vertices depending on how many vertices we have.
          if (vertexIndices.size() == 3) {
            manualObject_->triangle(vertexIndices[0], vertexIndices[1], vertexIndices[2]);
          } else {
            manualObject_->quad(vertexIndices[0], vertexIndices[2], vertexIndices[3], vertexIndices[1]);
          }
        }
      }

      // compute grid lines vertices
      const bool isLastRow{i == rows - 1};
      const bool isLastCol{j == cols - 1};
      const bool isDrawMeshLines{true};

      if (!showGridLines || !isDrawMeshLines)
      {
        continue;
      }
      std::vector<Ogre::Vector3> meshLineVertices = computeMeshLineVertices(i, j, isLastRow, isLastCol, 
                                                                            resolution, topLeft.head(2), 
                                                                            heightOrFlatData, isValid);

      // plot grid lines if we have enough points
      if (meshLineVertices.size() > 2) {
        meshLines_->addPoint(meshLineVertices[0]);
        meshLines_->addPoint(meshLineVertices[1]);
        meshLines_->newLine();

        if (meshLineVertices.size() == 3) {
          meshLines_->addPoint(meshLineVertices[1]);
          meshLines_->addPoint(meshLineVertices[2]);
          meshLines_->newLine();
        } else {
          meshLines_->addPoint(meshLineVertices[1]);
          meshLines_->addPoint(meshLineVertices[3]);
          meshLines_->newLine();

          meshLines_->addPoint(meshLineVertices[3]);
          meshLines_->addPoint(meshLineVertices[2]);
          meshLines_->newLine();
        }

        meshLines_->addPoint(meshLineVertices[2]);
        meshLines_->addPoint(meshLineVertices[0]);
        meshLines_->newLine();
      }

    }  // end for loop cols
  }    // end for loop rows

  manualObject_->end();
  material_->getTechnique(0)->setLightingEnabled(false);

  if (alpha < 0.9998)
  {
    material_->getTechnique(0)->setSceneBlending(Ogre::SBT_TRANSPARENT_ALPHA);
    material_->getTechnique(0)->setDepthWriteEnabled(false);
  } else 
  {
    material_->getTechnique(0)->setSceneBlending(Ogre::SBT_REPLACE);
    material_->getTechnique(0)->setDepthWriteEnabled(true);
  }

  const auto stopTime = std::chrono::high_resolution_clock::now();
  const auto elapsedTime = std::chrono::duration_cast<std::chrono::milliseconds>(stopTime - startTime);
//   ROS_DEBUG_STREAM("Visualization of se2_grid took: " << elapsedTime.count() << " ms.");
}

std::vector<Ogre::Vector3> SE2GridVisual::computeMeshLineVertices(int i, int j, bool /*isLastRow*/, bool /*isLastCol*/, 
                                                                    double resolution, const Eigen::Vector2d& topLeft,
                                                                    const Eigen::ArrayXXf& heightOrFlatData, const MaskArray& isValid) const 
{
  std::vector<Ogre::Vector3> meshLineVertices;
  meshLineVertices.reserve(4);
  for (int k = 0; k < 2; ++k) {
    for (int l = 0; l < 2; ++l) {
      const int x = i - k;
      const int y = j - l;
      Eigen::Array2i index(x > 0 ? x : 0, y > 0 ? y : 0);
      if (!isValid(index(0), index(1))) {
        continue;
      }
      const Eigen::Vector2d position = topLeft.array() - index.cast<double>() * resolution;
      meshLineVertices.emplace_back(position(0), position(1), heightOrFlatData(index(0), index(1)));
    }
  }
  return meshLineVertices;
}

void SE2GridVisual::initializeAndBeginManualObject(size_t nVertices) {
  if (!manualObject_)
  {
    static uint32_t count = 0;
    rviz::UniformStringStream ss;
    ss << "Mesh" << count++;
    manualObject_ = sceneManager_->createManualObject(ss.str());
    manualObject_->setDynamic(true);
    frameNode_->attachObject(manualObject_);

    ss << "Material";
    materialName_ = ss.str();
    material_ = Ogre::MaterialManager::getSingleton().create(materialName_, Ogre::ResourceGroupManager::DEFAULT_RESOURCE_GROUP_NAME);
    material_->setReceiveShadows(false);
    material_->getTechnique(0)->setLightingEnabled(true);
    material_->setCullingMode(Ogre::CULL_NONE);
  }

  manualObject_->clear();
  manualObject_->estimateVertexCount(nVertices);
  manualObject_->begin(materialName_, Ogre::RenderOperation::OT_TRIANGLE_LIST);
}

SE2GridVisual::ColorArray SE2GridVisual::computeColorValues(SE2GridVisual::MatrixConstRef heightData,
                                                            SE2GridVisual::MatrixConstRef colorData,
                                                            SE2GridVisual::ColoringMethod coloringMethod, std::string cmap,
                                                            Ogre::ColourValue flatColor, double minIntensity, double maxIntensity,
                                                            bool autocomputeIntensity, Ogre::ColourValue minColor, Ogre::ColourValue maxColor) {
  // Determine max and min intensity.
  bool isIntensityColoringMethod = coloringMethod == ColoringMethod::INTENSITY_LAYER_INVERTED_COLORMAP ||
                                   coloringMethod == ColoringMethod::INTENSITY_LAYER_COLORMAP ||
                                   coloringMethod == ColoringMethod::INTENSITY_LAYER_MANUAL;
  if (autocomputeIntensity && isIntensityColoringMethod)
  {
    minIntensity = colorData.minCoeffOfFinites();
    maxIntensity = minIntensity + std::max(colorData.maxCoeffOfFinites() - minIntensity, 0.2);  // regularize the intensity range.
  }

  const std::vector<std::vector<float>>& ctable = colorMap.at(cmap);

  switch (coloringMethod) {
    case ColoringMethod::FLAT:
      return ColorArray::Constant(heightData.rows(), heightData.cols(), flatColor);
    case ColoringMethod::COLOR_LAYER:
      return colorData.unaryExpr([](float color)
      {
        Eigen::Vector3f colorVectorRGB;
        // se2_grid::colorValueToVector(color, colorVectorRGB);
        const unsigned long tempColorValue = *reinterpret_cast<const unsigned long*>(&color);
        Eigen::Vector3i tempColorVectorI;
        tempColorVectorI(0) = (tempColorValue >> 16) & 0x0000ff;
        tempColorVectorI(1) = (tempColorValue >> 8) & 0x0000ff;
        tempColorVectorI(2) =  tempColorValue & 0x0000ff;
        colorVectorRGB = ((tempColorVectorI.cast<float>()).array() / 255.0).matrix();
        return Ogre::ColourValue(colorVectorRGB(0), colorVectorRGB(1), colorVectorRGB(2));
      });
    case ColoringMethod::INTENSITY_LAYER_MANUAL:
      return colorData.unaryExpr([&](float color) {
        normalizeIntensity(color, minIntensity, maxIntensity);
        return getInterpolatedColor(color, minColor, maxColor);
      });
    case ColoringMethod::INTENSITY_LAYER_COLORMAP:
      return colorData.unaryExpr([&](float color) {
        normalizeIntensity(color, minIntensity, maxIntensity);
        return getColorMap(color, ctable);
      });
    case ColoringMethod::INTENSITY_LAYER_INVERTED_COLORMAP:
      return colorData.unaryExpr([&](float color) {
        normalizeIntensity(color, minIntensity, maxIntensity);
        return getColorMap(1.f - color, ctable);
      });
    default:
      throw std::invalid_argument(std::string("An unknown coloring method was provided: ") +
                                  std::to_string(static_cast<int>(coloringMethod)));
  }
}

void SE2GridVisual::initializeMeshLines(size_t cols, size_t rows, double resolution, double alpha, double lineWidth)
{
  meshLines_->setColor(0.0, 0.0, 0.0, alpha);
  meshLines_->setLineWidth(resolution * lineWidth);
  meshLines_->setMaxPointsPerLine(2);
  // In the algorithm below, we have to account for max. 4 lines per cell.
  const size_t nLines = 2 * (rows * (cols - 1) + cols * (rows - 1));
  meshLines_->setNumLines(nLines);
}

void SE2GridVisual::setFramePosition(const Ogre::Vector3& position)
{
  frameNode_->setPosition(position);
}

void SE2GridVisual::setFrameOrientation(const Ogre::Quaternion& orientation)
{
  frameNode_->setOrientation(orientation);
}

std::vector<std::string> SE2GridVisual::getLayerNames()
{
  return map_.getLayers();
}

void SE2GridVisual::normalizeIntensity(float& intensity, float min_intensity, float max_intensity)
{
  intensity = std::min(intensity, max_intensity);
  intensity = std::max(intensity, min_intensity);
  intensity = (intensity - min_intensity) / (max_intensity - min_intensity);
}

Ogre::ColourValue SE2GridVisual::getRainbowColor(float intensity)
{
  intensity = std::min(intensity, 1.0f);
  intensity = std::max(intensity, 0.0f);

  float h = intensity * 5.0f + 1.0f;
  int i = floor(h);
  float f = h - i;
  if (!(i & 1)) f = 1 - f;  // if i is even
  float n = 1 - f;

  Ogre::ColourValue color;
  if (i <= 1)
    color[0] = n, color[1] = 0, color[2] = 1;
  else if (i == 2)
    color[0] = 0, color[1] = n, color[2] = 1;
  else if (i == 3)
    color[0] = 0, color[1] = 1, color[2] = n;
  else if (i == 4)
    color[0] = n, color[1] = 1, color[2] = 0;
  else if (i >= 5)
    color[0] = 1, color[1] = n, color[2] = 0;

  return color;
}

// Get interpolated color value.
Ogre::ColourValue SE2GridVisual::getInterpolatedColor(float intensity, Ogre::ColourValue min_color, Ogre::ColourValue max_color)
{
  intensity = std::min(intensity, 1.0f);
  intensity = std::max(intensity, 0.0f);

  Ogre::ColourValue color;
  color.r = intensity * (max_color.r - min_color.r) + min_color.r;
  color.g = intensity * (max_color.g - min_color.g) + min_color.g;
  color.b = intensity * (max_color.b - min_color.b) + min_color.b;

  return color;
}

}  // namespace se2_grid_rviz_plugin