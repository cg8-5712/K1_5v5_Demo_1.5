#include "head_controller.h"

#include <algorithm>
#include <cmath>

#include "brain.h"
#include "brain_config.h"
#include "brain_data.h"
#include "brain_tree.h"
#include "pos_predictor.h"
#include "robot_frame_predictor.h"
#include "utils/math.h"

namespace {

constexpr double kPi = 3.14159265358979323846;

double norm2d(double x, double y)
{
    return std::sqrt(x * x + y * y);
}

} // namespace

HeadController::HeadController(Brain *brain) : brain_(brain) {}

void HeadController::init()
{
    if (!brain_ || !brain_->get_clock()) {
        return;
    }
    sweepOrigin_ = brain_->get_clock()->now();
    framesSinceDetection_ = 0;
    lastBallVelRobot_[0] = 0.f;
    lastBallVelRobot_[1] = 0.f;
}

void HeadController::cacheLastBallVelocity()
{
    if (!brain_->tree->getEntry<bool>("ball_prediction_valid")) {
        return;
    }

    auto *data = brain_->data.get();
    if (data->usingFieldFrame && brain_->ballImmPredictor_->hasEstimate()) {
        const BallPrediction pred = brain_->ballImmPredictor_->getPrediction();
        const double theta = data->robotPoseToField.theta;
        const double c = std::cos(theta);
        const double s = std::sin(theta);
        const double vx_f = pred.vel[0];
        const double vy_f = pred.vel[1];
        lastBallVelRobot_[0] = static_cast<float>(c * vx_f + s * vy_f);
        lastBallVelRobot_[1] = static_cast<float>(-s * vx_f + c * vy_f);
        return;
    }

    const RobotFrameBallPrediction pred = brain_->robotFramePredictor_->getPrediction();
    if (pred.valid) {
        lastBallVelRobot_[0] = pred.vel[0];
        lastBallVelRobot_[1] = pred.vel[1];
    }
}

void HeadController::update()
{
    if (!brain_ || !brain_->config || !brain_->data || !brain_->client) {
        return;
    }
    if (!brain_->config->headControllerEnabled) {
        return;
    }

    const bool predictionValid = brain_->tree->getEntry<bool>("ball_prediction_valid");
    if (predictionValid) {
        framesSinceDetection_ = 0;
        cacheLastBallVelocity();
    } else {
        ++framesSinceDetection_;
    }

    const Mode mode = selectMode();
    lastMode_ = mode;

    switch (mode) {
    case Mode::TRACK:
        trackBall();
        break;
    case Mode::SEARCH_DIRECTED:
        searchDirected();
        break;
    case Mode::SEARCH_SWEEP:
        searchSweep();
        break;
    }
}

HeadController::Mode HeadController::selectMode() const
{
    if (brain_->tree->getEntry<bool>("ball_prediction_valid")) {
        return Mode::TRACK;
    }

    const auto &cfg = *brain_->config;
    const double vel = norm2d(lastBallVelRobot_[0], lastBallVelRobot_[1]);
    if (framesSinceDetection_ < cfg.headDirectedMaxFrames && vel > cfg.headDirectedVelMin) {
        return Mode::SEARCH_DIRECTED;
    }

    return Mode::SEARCH_SWEEP;
}

bool HeadController::resolvePred100Angles(double &pitch, double &yaw) const
{
    auto *data = brain_->data.get();
    double rx = 0.0;
    double ry = 0.0;

    if (data->usingFieldFrame && data->pred100Field.valid) {
        const Pose2D pr = data->field2robot(Pose2D{data->pred100Field.x, data->pred100Field.y, 0.0});
        rx = pr.x;
        ry = pr.y;
    } else if (data->pred100Robot.valid) {
        rx = data->pred100Robot.x;
        ry = data->pred100Robot.y;
    } else {
        return false;
    }

    const double range = std::max(0.1, norm2d(rx, ry));
    yaw = std::atan2(ry, rx);
    pitch = std::atan2(brain_->config->robotHeight, range);
    return true;
}

bool HeadController::resolveMemoryBallAngles(double &pitch, double &yaw) const
{
    auto *data = brain_->data.get();
    const bool iKnowBall = brain_->tree->getEntry<bool>("ball_location_known");
    const bool tmReliable = brain_->tree->getEntry<bool>("tm_ball_pos_reliable");

    if (data->ballDetected && iKnowBall) {
        pitch = data->ball.pitchToRobot;
        yaw = data->ball.yawToRobot;
        return true;
    }
    if (tmReliable) {
        pitch = data->tmBall.pitchToRobot;
        yaw = data->tmBall.yawToRobot;
        return true;
    }
    if (iKnowBall) {
        pitch = data->ball.pitchToRobot;
        yaw = data->ball.yawToRobot;
        return true;
    }
    return false;
}

void HeadController::applySmoothedHeadCommand(double targetPitch, double targetYaw)
{
    const double smoother = std::max(1.0, brain_->config->headTrackSmoother);
    const double pitch = brain_->data->headPitch + (targetPitch - brain_->data->headPitch) / smoother;
    const double yaw = brain_->data->headYaw + (targetYaw - brain_->data->headYaw) / smoother;
    brain_->client->moveHead(pitch, yaw);
}

void HeadController::trackBall()
{
    double pitch = 0.0;
    double yaw = 0.0;

    if (brain_->tree->getEntry<bool>("ball_prediction_valid")) {
        if (resolvePred100Angles(pitch, yaw)) {
            applySmoothedHeadCommand(pitch, yaw);
            return;
        }
    }

    if (resolveMemoryBallAngles(pitch, yaw)) {
        applySmoothedHeadCommand(pitch, yaw);
        return;
    }

    searchSweep();
}

void HeadController::searchDirected()
{
    const auto &cfg = *brain_->config;
    const double baseYaw = std::atan2(lastBallVelRobot_[1], lastBallVelRobot_[0]);
    const double t = (brain_->get_clock()->now() - sweepOrigin_).seconds();
    const double phase = 2.0 * kPi * cfg.headDirectedScanHz * t;
    const double yaw = baseYaw + cfg.headDirectedScanRad * std::sin(phase);
    const double pitch = (cfg.headSweepPitchLow + cfg.headSweepPitchHigh) * 0.5;

    brain_->client->moveHead(pitch, yaw);
}

void HeadController::searchSweep()
{
    const auto &cfg = *brain_->config;
    const double t = (brain_->get_clock()->now() - sweepOrigin_).seconds();
    const double phase = 2.0 * kPi * cfg.headSweepHz * t;
    const double u = 0.5 * (1.0 + std::sin(phase));

    const double yaw =
        cfg.headYawLimitRight + (cfg.headYawLimitLeft - cfg.headYawLimitRight) * u;
    const double pitch =
        cfg.headSweepPitchHigh + (cfg.headSweepPitchLow - cfg.headSweepPitchHigh) * (0.5 + 0.5 * std::sin(phase * 0.5));

    brain_->client->moveHead(pitch, yaw);
}
