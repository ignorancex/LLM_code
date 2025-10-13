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
 * @file   Factor.cpp
 * @brief  Utilities for factor variant types
 * @author Jose Luis Blanco Claraco
 * @date   Jan 28, 2019
 */

#include <mola_kernel/Factor.h>
#include <mola_kernel/variant_helper.h>

using namespace mola;

FactorBase& mola::factor_get_base(Factor& f)
{
  FactorBase* ret = nullptr;
  std::visit(
      overloaded{
          [&ret](FactorBase& b) { ret = &b; },
          [&ret](FactorOther& o)
          {
            ASSERT_(o);
            ret = o.get();
          },
          [](std::monostate) {}},
      f);

  if (!ret) THROW_EXCEPTION("factor_get_base(): Empty variant.");

  return *ret;
}

const FactorBase& mola::factor_get_base(const Factor& f)
{
  const FactorBase* ret = nullptr;
  std::visit(
      overloaded{
          [&ret](const FactorBase& b) { ret = &b; },
          [&ret](const FactorOther& o)
          {
            ASSERT_(o);
            ret = o.get();
          },
          [](std::monostate) {}},
      f);

  if (!ret) THROW_EXCEPTION("factor_get_base(): Empty variant.");
  return *ret;
}
