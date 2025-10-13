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
 * @file   BackEndBase.cpp
 * @brief  Virtual interface for SLAM back-ends
 * @author Jose Luis Blanco Claraco
 * @date   Dec 21, 2018
 */

#include <mola_kernel/WorldModel.h>
#include <mola_kernel/interfaces/BackEndBase.h>
#include <mola_kernel/interfaces/RawDataSourceBase.h>
#include <mrpt/containers/yaml.h>

using namespace mola;

// arguments: class_name, parent_class, class namespace
IMPLEMENTS_VIRTUAL_MRPT_OBJECT(BackEndBase, ExecutableBase, mola)

BackEndBase::BackEndBase() = default;

void BackEndBase::initialize(const Yaml& cfg)
{
  MRPT_TRY_START

  // attach to world model:
  auto wms = findService<WorldModel>();
  ASSERTMSG_(!wms.empty(), "No WorldModel found in the system!");
  ASSERTMSG_(wms.size() == 1, "Only one WorldModel can coexist in the system!");

  worldmodel_ = std::dynamic_pointer_cast<WorldModel>(wms[0]);
  ASSERT_(worldmodel_);
  MRPT_LOG_INFO_FMT(
      "Attached to WorldModel module `%s`", worldmodel_->getModuleInstanceName().c_str());

  // children config:
  this->initialize_backend(cfg);

  MRPT_TRY_END
}
