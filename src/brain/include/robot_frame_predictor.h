#pragma once

#include <rclcpp/rclcpp.hpp>

#include "brain_config.h"

struct RobotFrameBallPrediction {
    float pred100[2] = {0.f, 0.f};
    float pos[2] = {0.f, 0.f};
    float vel[2] = {0.f, 0.f};
    float confidence = 0.f;
    bool valid = false;
};

/**
 * Robot-frame short-horizon ball prediction when field localization is untrusted.
 */
class RobotFramePredictor {
public:
    void configure(rclcpp::Clock::SharedPtr clock, const BrainConfig &config);

    void reset();
    void propagate(double dt_sec);
    void add(double robot_x, double robot_y, double confidence, double range_m);
    void handleOccluded();

    RobotFrameBallPrediction getPrediction() const;

private:
    rclcpp::Clock::SharedPtr clock_;
    BrainConfig config_{};

    bool initialized_ = false;
    bool valid_ = false;
    float confidence_ = 0.f;
    int frames_since_detection_ = 0;

    double x_ = 0.0;
    double y_ = 0.0;
    double vx_ = 0.0;
    double vy_ = 0.0;
    rclcpp::Time last_update_time_;
};
