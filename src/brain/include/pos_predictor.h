#pragma once

#include <array>

#include <Eigen/Dense>

#include "brain_config.h"

struct BallPrediction {
    float pos[2] = {0.f, 0.f};
    float vel[2] = {0.f, 0.f};
    float pred100[2] = {0.f, 0.f};
    float pred300[2] = {0.f, 0.f};
    float confidence = 0.f;
    float mode_prob[2] = {0.5f, 0.5f};
    bool valid = false;
    bool pred300_valid = false;
};

/**
 * Field-frame IMM-EKF ball predictor.
 */
class BallImmPredictor {
public:
    void configure(const BrainConfig &config);
    void reset();

    void propagate(double dt_sec);
    void add(double field_x, double field_y, double confidence, double range_m);
    void handleOccluded();

    BallPrediction getPrediction() const;

    bool hasEstimate() const { return initialized_; }
    void getPositionEstimate(double &px, double &py, Eigen::Matrix2d &pos_cov) const;
    /** Squared Mahalanobis distance in field frame; -1 if gating inactive. */
    double gateMahalanobisSq(double field_x, double field_y, double range_m) const;

private:
    struct ModelState {
        Eigen::Vector4d x = Eigen::Vector4d::Zero();
        Eigen::Matrix4d P = Eigen::Matrix4d::Identity();
        Eigen::Matrix4d Q_base = Eigen::Matrix4d::Identity() * 1e-3;
        double friction_mu = 0.3;
    };

    static Eigen::Matrix4d makeTransition(double dt, double mu);
    static double measurementNoise(double range_m);
    static void predictPosition(const Eigen::Vector4d &x, double mu, double dt, double out[2]);

    void predictModel(ModelState &model, double dt);
    double updateModel(ModelState &model, double zx, double zy, double range_m);

    ModelState stationary_;
    ModelState rolling_;
    std::array<double, 2> mode_prob_{{0.5, 0.5}};

    BrainConfig config_{};
    bool initialized_ = false;
    bool valid_ = false;
    float confidence_ = 0.f;
    int frames_since_detection_ = 0;
    double last_dt_ = 0.02;
    double occlusion_q_scale_ = 1.0;
};
