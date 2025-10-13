/*               _
 _ __ ___   ___ | | __ _
| '_ ` _ \ / _ \| |/ _` | Modular Optimization framework for
| | | | | | (_) | | (_| | Localization and mApping (MOLA)
|_| |_| |_|\___/|_|\__,_| https://github.com/MOLAorg/mola

 Copyright (C) 2018-2025 Jose Luis Blanco, University of Almeria,
                         and individual contributors.
 SPDX-License-Identifier: GPL-3.0
 See LICENSE for full license information.
*/

/**
 * @file   FactorStereoProjectionPose.cpp
 * @brief
 * @author Jose Luis Blanco Claraco
 * @date   Jan 08, 2019
 */

#include <mola_kernel/factors/FactorStereoProjectionPose.h>
#include <mola_kernel/interfaces/BackEndBase.h>
#include <mrpt/serialization/CArchive.h>

using namespace mola;

// arguments: classname, parent class, namespace
IMPLEMENTS_SERIALIZABLE(FactorStereoProjectionPose, FactorBase, mola);

std::size_t FactorStereoProjectionPose::edge_count() const { return 1; }
mola::id_t  FactorStereoProjectionPose::edge_indices(const std::size_t i) const
{
  ASSERT_EQUAL_(i, 0UL);
  return observing_kf_;
}

// Implementation of the CSerializable virtual interface:
uint8_t FactorStereoProjectionPose::serializeGetVersion() const { return 0; }
void    FactorStereoProjectionPose::serializeTo(mrpt::serialization::CArchive& out) const
{
  baseSerializeTo(out);
  out << sigma_xleft_ << sigma_xright_ << sigma_y_ << observation_.x_left << observation_.x_right
      << observation_.y << observing_kf_ << observed_landmark_ << camera_params_id_
      << cameraPoseOnRobot_;
}
void FactorStereoProjectionPose::serializeFrom(mrpt::serialization::CArchive& in, uint8_t version)
{
  baseSerializeFrom(in);

  switch (version)
  {
    case 0:
    {
      in >> sigma_xleft_ >> sigma_xright_ >> sigma_y_ >> observation_.x_left >>
          observation_.x_right >> observation_.y >> observing_kf_ >> observed_landmark_ >>
          camera_params_id_ >> cameraPoseOnRobot_;
    }
    break;
    default:
      MRPT_THROW_UNKNOWN_SERIALIZATION_VERSION(version);
  };
}
