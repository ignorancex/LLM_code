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
 * @file   FactorBase.cpp
 * @brief  Base class for all "factors" in the world model
 * @author Jose Luis Blanco Claraco
 * @date   Jan 08, 2019
 */

#include <mola_kernel/factors/FactorBase.h>
#include <mrpt/serialization/CArchive.h>

using namespace mola;

// arguments: class, parent class, namespace
IMPLEMENTS_VIRTUAL_SERIALIZABLE(FactorBase, mrpt::serialization::CSerializable, mola)

FactorBase::~FactorBase() = default;

void FactorBase::baseSerializeTo(mrpt::serialization::CArchive& out) const
{
  out << my_id_ << robust_param_;
  out.WriteAs<uint8_t>(robust_type_);
}
void FactorBase::baseSerializeFrom(mrpt::serialization::CArchive& in)
{
  in >> my_id_ >> robust_param_;
  robust_type_ = static_cast<mola::Robust>(in.ReadAs<uint8_t>());
}
