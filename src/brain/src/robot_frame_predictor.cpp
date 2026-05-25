#include "robot_frame_predictor.h"

#include <algorithm>
#include <cmath>

void RobotFramePredictor::configure(rclcpp::Clock::SharedPtr clock, const BrainConfig &config)
{
    clock_ = std::move(clock);
    config_ = config;
    reset();
}

void RobotFramePredictor::reset()
{
    initialized_ = false;
    valid_ = false;
    confidence_ = 0.f;
    frames_since_detection_ = 0;
    x_ = y_ = vx_ = vy_ = 0.0;
    last_update_time_ = rclcpp::Time(0, 0, RCL_ROS_TIME);
}

void RobotFramePredictor::propagate(double dt_sec)
{
    if (!initialized_ || dt_sec <= 0.0) {
        return;
    }
    x_ += vx_ * dt_sec;
    y_ += vy_ * dt_sec;
}

void RobotFramePredictor::add(double robot_x, double robot_y, double confidence, double /*range_m*/)
{
    frames_since_detection_ = 0;
    confidence_ = static_cast<float>(std::clamp(confidence, 0.0, 1.0));

    const auto now = clock_->now();
    if (!initialized_) {
        x_ = robot_x;
        y_ = robot_y;
        vx_ = vy_ = 0.0;
        initialized_ = true;
        valid_ = true;
        last_update_time_ = now;
        return;
    }

    const double dt = std::max(1e-3, (now - last_update_time_).seconds());
  const double alpha = 0.35;
    const double inst_vx = (robot_x - x_) / dt;
    const double inst_vy = (robot_y - y_) / dt;
    vx_ = (1.0 - alpha) * vx_ + alpha * inst_vx;
    vy_ = (1.0 - alpha) * vy_ + alpha * inst_vy;
    x_ = robot_x;
    y_ = robot_y;
    last_update_time_ = now;
    valid_ = true;
}

void RobotFramePredictor::handleOccluded()
{
    if (!initialized_) {
        return;
    }
    ++frames_since_detection_;
    confidence_ *= static_cast<float>(config_.immBallConfidenceDecayRate);
    if (frames_since_detection_ > config_.immMaxOccludedFrames) {
        valid_ = false;
    }
}

RobotFrameBallPrediction RobotFramePredictor::getPrediction() const
{
    RobotFrameBallPrediction out;
    if (!initialized_ || !valid_) {
        return out;
    }

    constexpr double kPredictSec = 0.1;
    const double acc = config_.immFrictionDecayHz > 0 ? -config_.immFrictionDecayHz : -0.3;
    auto predict_axis = [&](double p, double v) {
        const double t = kPredictSec;
        const double v_end = v + acc * t;
        const double dist = v_end > 0 ? (v + v_end) * 0.5 * t : (v * v) / (2.0 * std::fabs(acc) + 1e-6);
        return p + dist;
    };

    out.pos[0] = static_cast<float>(x_);
    out.pos[1] = static_cast<float>(y_);
    out.vel[0] = static_cast<float>(vx_);
    out.vel[1] = static_cast<float>(vy_);
    out.pred100[0] = static_cast<float>(predict_axis(x_, vx_));
    out.pred100[1] = static_cast<float>(predict_axis(y_, vy_));
    out.confidence = confidence_;
    out.valid = true;
    return out;
}
