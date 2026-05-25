#pragma once

#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "brain_config.h"
#include "brain_data.h"
#include "brain_tree.h"
#include "locator.h"

/**
 * Field / robot 坐标变换与定位可信度。
 * 原 posProjector.h 内的 PosPredictor 已迁至 robot_frame_predictor.h。
 */
class PosProjector {
public:
    void init(std::shared_ptr<Locator> locator,
              std::shared_ptr<BrainData> data,
              std::shared_ptr<BrainTree> tree,
              std::shared_ptr<BrainConfig> config,
              rclcpp::Clock::SharedPtr clock);

    bool isLocalizationTrusted() const;

    void robotToField(double rx, double ry, double &fx, double &fy) const;
    void fieldToRobot(double fx, double fy, double &rx, double &ry) const;

private:
    std::shared_ptr<Locator> locator_;
    std::shared_ptr<BrainData> data_;
    std::shared_ptr<BrainTree> tree_;
    std::shared_ptr<BrainConfig> config_;
    rclcpp::Clock::SharedPtr clock_;

    double localizationCovarianceTrace() const;
};
