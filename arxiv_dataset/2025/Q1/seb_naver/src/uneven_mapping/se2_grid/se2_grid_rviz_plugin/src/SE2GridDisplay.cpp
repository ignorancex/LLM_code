#include "se2_grid_rviz_plugin/SE2GridVisual.hpp"
#include "se2_grid_rviz_plugin/SE2GridDisplay.hpp"
#include "se2_grid_rviz_plugin/SE2GridColorMaps.hpp"

#include "se2_grid_rviz_plugin/modified/frame_manager.h"

#include <OGRE/OgreSceneNode.h>
#include <OGRE/OgreSceneManager.h>

#include <tf/transform_listener.h>

#include <rviz/visualization_manager.h>
#include <rviz/properties/bool_property.h>
#include <rviz/properties/color_property.h>
#include <rviz/properties/float_property.h>
#include <rviz/properties/int_property.h>
#include <rviz/properties/enum_property.h>
#include <rviz/properties/editable_enum_property.h>

namespace se2_grid_rviz_plugin {

SE2GridDisplay::SE2GridDisplay()
{
  qRegisterMetaType<se2_grid_msgs::SE2Grid::ConstPtr>("se2_grid_msgs::SE2Grid::ConstPtr");

  alphaProperty_ = new rviz::FloatProperty("Alpha", 1.0,
                                           "0 is fully transparent, 1.0 is fully opaque.", this,
                                           SLOT(updateVisualization()));

  historyLengthProperty_ = new rviz::IntProperty("History Length", 1,
                                                 "Number of prior grid maps to display.", this,
                                                 SLOT(updateHistoryLength()));
  
  so2ValueProperty_ = new rviz::FloatProperty("SO(2) Value", 0.0,
                                              "SO(2) value of the map [rad].", this,
                                              SLOT(updateVisualization()));

  showGridLinesProperty_ = new rviz::BoolProperty("Show Grid Lines", true, "Whether to show the lines connecting the grid cells.", this,
                                                  SLOT(updateGridLines()));

  gridLinesThicknessProperty_ =
      new rviz::FloatProperty("Grid Line Thickness", 0.1, "Set thickness for the grid lines.", this, SLOT(updateVisualization()));

  heightModeProperty_ = new rviz::EnumProperty("Height Transformer", "GridMapLayer",
                                               "Select the transformer to use to set the height.",
                                               this, SLOT(updateHeightMode()));
  heightModeProperty_->addOption("Layer", 0);
  heightModeProperty_->addOption("Flat", 1);

  heightTransformerProperty_ = new rviz::EditableEnumProperty(
      "Height Layer", "elevation", "Select the grid map layer to compute the height.", this,
      SLOT(updateVisualization()));

  colorModeProperty_ = new rviz::EnumProperty("Color Transformer", "GridMapLayer",
                                              "Select the transformer to use to set the color.",
                                              this, SLOT(updateColorMode()));
  colorModeProperty_->addOption("IntensityLayer", 0);
  colorModeProperty_->addOption("ColorLayer", 1);
  colorModeProperty_->addOption("FlatColor", 2);
  colorModeProperty_->addOption("None", 3);

  colorTransformerProperty_ = new rviz::EditableEnumProperty(
      "Color Layer", "elevation", "Select the grid map layer to compute the color.", this,
      SLOT(updateVisualization()));

  colorMapProperty_ = new rviz::EditableEnumProperty(
      "ColorMap", "default", "Select the colormap to be used.", this,
      SLOT(updateVisualization()));

  colorProperty_ = new rviz::ColorProperty("Color", QColor(200, 200, 200),
                                           "Color to draw the mesh.", this,
                                           SLOT(updateVisualization()));
  colorProperty_->hide();

  useColorMapProperty_ = new rviz::BoolProperty(
      "Use ColorMap", true,
      "Whether to use a colormap or to interpolate between two colors.", this,
      SLOT(updateUseColorMap()));
  
  invertColorMapProperty_ = new rviz::BoolProperty(
      "Invert ColorMap", false,
      "Whether to invert the colormap colors.", this,
      SLOT(updateVisualization()));

  minColorProperty_ = new rviz::ColorProperty(
      "Min Color", QColor(0, 0, 0), "Color to assign to cells with the minimum intensity.  "
      "Actual color is interpolated between this and Max Color.",
      this, SLOT(updateVisualization()));
  minColorProperty_->hide();

  maxColorProperty_ = new rviz::ColorProperty(
      "Max Color", QColor(255, 255, 255), "Color to assign to cells with the maximum intensity.  "
      "Actual color is interpolated between Min Color and this.",
      this, SLOT(updateVisualization()));
  maxColorProperty_->hide();

  autocomputeIntensityBoundsProperty_ = new BoolProperty(
      "Autocompute Intensity Bounds", true,
      "Whether to automatically compute the intensity min/max values.", this,
      SLOT(updateAutocomputeIntensityBounds()));

  minIntensityProperty_ = new rviz::FloatProperty(
      "Min Intensity", 0.0,
      "Minimum possible intensity value, used to interpolate from Min Color to Max Color.", this,
      SLOT(updateVisualization()));
  minIntensityProperty_->hide();

  maxIntensityProperty_ = new rviz::FloatProperty(
      "Max Intensity", 10.0,
      "Maximum possible intensity value, used to interpolate from Min Color to Max Color.", this,
      SLOT(updateVisualization()));
  maxIntensityProperty_->hide();

  historyLengthProperty_->setMin(1);
  historyLengthProperty_->setMax(100);
}

SE2GridDisplay::~SE2GridDisplay()
{
}

void SE2GridDisplay::onInitialize()
{
  MFDClass::onInitialize();	 //  "MFDClass" = typedef of "MessageFilterDisplay<message type>"
  updateHistoryLength();
  updateColorMapList();
}

void SE2GridDisplay::onEnable()
{
  isReset_ = false;
  connect(this, &SE2GridDisplay::process, this, &SE2GridDisplay::onProcessMessage);
  MessageFilterDisplay<se2_grid_msgs::SE2Grid>::onEnable();
}

void SE2GridDisplay::reset()
{
  isReset_ = true;
  disconnect(this, &SE2GridDisplay::process, this, &SE2GridDisplay::onProcessMessage);
  MFDClass::reset();
  visuals_.clear();
}

void SE2GridDisplay::updateHistoryLength()
{
  visuals_.rset_capacity(historyLengthProperty_->getInt());
}

void SE2GridDisplay::updateHeightMode()
{
  updateVisualization();
  heightTransformerProperty_->setHidden(heightModeProperty_->getOptionInt() == 1);
}

void SE2GridDisplay::updateColorMode()
{
  updateVisualization();
  
  bool intensityColor = colorModeProperty_->getOptionInt() == 0;
  bool flatColor = colorModeProperty_->getOptionInt() == 2;
  bool none = colorModeProperty_->getOptionInt() == 3;
  colorProperty_->setHidden(!flatColor);
  colorTransformerProperty_->setHidden(flatColor || none);
  useColorMapProperty_->setHidden(!intensityColor);
  invertColorMapProperty_->setHidden(!intensityColor);
  autocomputeIntensityBoundsProperty_->setHidden(!intensityColor);
  bool useColorMap = useColorMapProperty_->getBool();
  minColorProperty_->setHidden(!intensityColor || useColorMap);
  maxColorProperty_->setHidden(!intensityColor || useColorMap);
  bool autocomputeIntensity = autocomputeIntensityBoundsProperty_->getBool();
  minIntensityProperty_->setHidden(!intensityColor || autocomputeIntensity);
  minIntensityProperty_->setHidden(!intensityColor || autocomputeIntensity);
}

void SE2GridDisplay::updateUseColorMap()
{
  updateVisualization();
  bool useColorMap = useColorMapProperty_->getBool();
  minColorProperty_->setHidden(useColorMap);
  maxColorProperty_->setHidden(useColorMap);
  invertColorMapProperty_->setHidden(!useColorMap);
}

void SE2GridDisplay::updateGridLines()
{
  updateVisualization();
  const bool isShowGridLines = showGridLinesProperty_->getBool();
  gridLinesThicknessProperty_->setHidden(!isShowGridLines);
}

void SE2GridDisplay::updateAutocomputeIntensityBounds()
{
  updateVisualization();
  minIntensityProperty_->setHidden(autocomputeIntensityBoundsProperty_->getBool());
  maxIntensityProperty_->setHidden(autocomputeIntensityBoundsProperty_->getBool());
}

void SE2GridDisplay::updateVisualization()
{
  float alpha = alphaProperty_->getFloat();
  bool showGridLines = showGridLinesProperty_->getBool();
  bool flatTerrain = heightModeProperty_->getOptionInt() == 1;
  std::string heightLayer = heightTransformerProperty_->getStdString();
  bool mapLayerColor = colorModeProperty_->getOptionInt() == 1;
  bool flatColor = colorModeProperty_->getOptionInt() == 2;
  bool noColor = colorModeProperty_->getOptionInt() == 3;
  Ogre::ColourValue meshColor = colorProperty_->getOgreColor();
  std::string colorLayer = colorTransformerProperty_->getStdString();
  std::string colorMap = colorMapProperty_->getStdString();
  bool useColorMap = useColorMapProperty_->getBool();
  bool invertColorMap = invertColorMapProperty_->getBool();
  Ogre::ColourValue minColor = minColorProperty_->getOgreColor();
  Ogre::ColourValue maxColor = maxColorProperty_->getOgreColor();
  bool autocomputeIntensity = autocomputeIntensityBoundsProperty_->getBool();
  float minIntensity = minIntensityProperty_->getFloat();
  float maxIntensity = maxIntensityProperty_->getFloat();
  const float gridLineThickness = gridLinesThicknessProperty_->getFloat();
  const float so2_value = so2ValueProperty_->getFloat();

  for (size_t i = 0; i < visuals_.size(); i++) {
    visuals_[i]->computeVisualization(alpha, showGridLines, flatTerrain, heightLayer, flatColor, noColor, meshColor, mapLayerColor,
                                      colorLayer, colorMap, useColorMap, invertColorMap, minColor, maxColor, autocomputeIntensity, minIntensity,
                                      maxIntensity, gridLineThickness, so2_value);
  }
}

void SE2GridDisplay::processMessage(const se2_grid_msgs::SE2Grid::ConstPtr& msg)
{
  process(msg);
}

void SE2GridDisplay::onProcessMessage(const se2_grid_msgs::SE2Grid::ConstPtr& msg)
{
  // Check if the display was already reset.
  if (isReset_) {
    return;
  }

  // Check if transform between the message's frame and the fixed frame exists.
  Ogre::Quaternion orientation;
  Ogre::Vector3 position;
  if (!context_->getFrameManager()->getTransform(msg->info.header.frame_id, msg->info.header.stamp,
                                                 position, orientation)) {
    ROS_DEBUG("Error transforming from frame '%s' to frame '%s'", msg->info.header.frame_id.c_str(),
              qPrintable(fixed_frame_));
    return;
  }

  boost::shared_ptr<SE2GridVisual> visual;
  if (visuals_.full()) {
    visual = visuals_.front();
  } else {
    visual.reset(new SE2GridVisual(context_->getSceneManager(), scene_node_));
  }

  visual->setMessage(msg);
  visual->setFramePosition(position);
  visual->setFrameOrientation(orientation);

  visual->computeVisualization(alphaProperty_->getFloat(), showGridLinesProperty_->getBool(),
                               heightModeProperty_->getOptionInt() == 1, heightTransformerProperty_->getStdString(),
                               colorModeProperty_->getOptionInt() == 2, colorModeProperty_->getOptionInt() == 3,
                               colorProperty_->getOgreColor(), colorModeProperty_->getOptionInt() == 1,
                               colorTransformerProperty_->getStdString(), colorMapProperty_->getStdString(),
                               useColorMapProperty_->getBool(), invertColorMapProperty_->getBool(),
                               minColorProperty_->getOgreColor(), maxColorProperty_->getOgreColor(),
                               autocomputeIntensityBoundsProperty_->getBool(), minIntensityProperty_->getFloat(),
                               maxIntensityProperty_->getFloat(),
                               gridLinesThicknessProperty_->getFloat(), so2ValueProperty_->getFloat());

  std::vector<std::string> layer_names = visual->getLayerNames();
  heightTransformerProperty_->clearOptions();
  colorTransformerProperty_->clearOptions();
  for (size_t i = 0; i < layer_names.size(); i++) {
    heightTransformerProperty_->addOptionStd(layer_names[i]);
    colorTransformerProperty_->addOptionStd(layer_names[i]);
  }

  visuals_.push_back(visual);
}

void SE2GridDisplay::updateColorMapList()
{
  updateVisualization();
  for (auto cmap : getColorMapNames()) {
    colorMapProperty_->addOptionStd(cmap);
  }
}


}  // end namespace se2_grid_rviz_plugin

#include <pluginlib/class_list_macros.h>
PLUGINLIB_EXPORT_CLASS(se2_grid_rviz_plugin::SE2GridDisplay, rviz::Display)
