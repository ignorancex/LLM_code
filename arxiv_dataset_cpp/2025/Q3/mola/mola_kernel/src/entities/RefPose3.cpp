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
 * @file   RefPose3.cpp
 * @brief
 * @author Jose Luis Blanco Claraco
 * @date   Jan 08, 2019
 */

#include <mola_kernel/entities/RefPose3.h>
#include <mrpt/serialization/CArchive.h>

using namespace mola;

// arguments: classname, parent class, namespace
IMPLEMENTS_SERIALIZABLE(RefPose3, EntityBase, mola);

// Implementation of the CSerializable virtual interface:
uint8_t RefPose3::serializeGetVersion() const { return 0; }
void    RefPose3::serializeTo(mrpt::serialization::CArchive& out) const { baseSerializeTo(out); }
void    RefPose3::serializeFrom(mrpt::serialization::CArchive& in, uint8_t version)
{
  baseSerializeFrom(in);

  switch (version)
  {
    case 0:
    {
    }
    break;
    default:
      MRPT_THROW_UNKNOWN_SERIALIZATION_VERSION(version);
  };
}
