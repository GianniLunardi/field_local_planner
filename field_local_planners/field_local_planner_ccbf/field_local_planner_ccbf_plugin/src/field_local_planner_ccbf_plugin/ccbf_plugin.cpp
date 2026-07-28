//----------------------------------------
// This file is part of field_local_planner
//
// Copyright (C) 2020-2025 Matías Mattamala, University of Oxford.
//
// field_local_planner is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
// License as published by the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// field_local_planner is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
// the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License along with field_local_planner.
// If not, see <http://www.gnu.org/licenses/>.
//
//----------------------------------------
#include <pluginlib/class_list_macros.h>
#include <field_local_planner_ccbf_plugin/ccbf_plugin.hpp>

PLUGINLIB_EXPORT_CLASS(field_local_planner::CcbfPlugin, field_local_planner::BasePlugin)

namespace field_local_planner {

CcbfPlugin::CcbfPlugin() : BasePlugin() {
  local_planner_ = std::make_shared<Ccbf>();
}

void CcbfPlugin::loadParameters(ros::NodeHandle& nh) {
  Ccbf::Parameters p;
  p.angular_gain_p = utils::getParameterDefault(nh, "angular_gain_p", 1.0);
  p.linear_gain_p = utils::getParameterDefault(nh, "linear_gain_p", 1.0);

  std::dynamic_pointer_cast<Ccbf>(local_planner_)->setParameters(p);
}

void CcbfPlugin::setupRos(ros::NodeHandle& nh) {
  dynamic_reconfigure_server_ = std::make_shared<DynParamServer>(nh);
  dynamic_reconfigure_callback_ = boost::bind(&CcbfPlugin::dynamicReconfigureCallback, this, _1, _2);
  dynamic_reconfigure_server_->setCallback(dynamic_reconfigure_callback_);
}

void CcbfPlugin::dynamicReconfigureCallback(CcbfConfig& config, uint32_t level) {
  Ccbf::Parameters p = std::dynamic_pointer_cast<Ccbf>(local_planner_)->getParameters();

  // Ccbf parameters
  UPDATE_COMMON_PARAMS(angular_gain_p)
  UPDATE_COMMON_PARAMS(linear_gain_p)

  std::dynamic_pointer_cast<Ccbf>(local_planner_)->setParameters(p);
}

void CcbfPlugin::publishVisualizations() {
  // TODO implement
}

}  // namespace field_local_planner
