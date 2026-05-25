#pragma once

#include <rclcpp/rclcpp.hpp>

class Brain;
class BrainData;

/**
 * 50 Hz head tracking / search, independent of BehaviorTree.
 */
class HeadController {
public:
    explicit HeadController(Brain *brain);

    void init();
    void update();

    /** 0=TRACK, 1=SEARCH_DIRECTED, 2=SEARCH_SWEEP — for debug / rerun */
    int lastModeIndex() const { return static_cast<int>(lastMode_); }

private:
    enum class Mode { TRACK, SEARCH_DIRECTED, SEARCH_SWEEP };

    Mode selectMode() const;
    bool resolvePred100Angles(double &pitch, double &yaw) const;
    bool resolveMemoryBallAngles(double &pitch, double &yaw) const;

    void trackBall();
    void searchDirected();
    void searchSweep();
    void applySmoothedHeadCommand(double targetPitch, double targetYaw);

    void cacheLastBallVelocity();

    Brain *brain_;

    int framesSinceDetection_ = 0;
    float lastBallVelRobot_[2] = {0.f, 0.f};
    rclcpp::Time sweepOrigin_;
    Mode lastMode_ = Mode::SEARCH_SWEEP;
};
