#include "pos_predictor.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr int kStationary = 0;
constexpr int kRolling = 1;

Eigen::Matrix<double, 2, 4> observationMatrix()
{
    Eigen::Matrix<double, 2, 4> H;
    H.setZero();
    H(0, 0) = 1.0;
    H(1, 1) = 1.0;
    return H;
}

} // namespace

void BallImmPredictor::configure(const BrainConfig &config)
{
    config_ = config;

    stationary_.friction_mu = 5.0;
    rolling_.friction_mu = config.immFrictionDecayHz;

    stationary_.Q_base.setIdentity();
    rolling_.Q_base.setIdentity();
    stationary_.Q_base.diagonal() << 1e-4, 1e-4, 1e-3, 1e-3;
    rolling_.Q_base.diagonal() << 1e-3, 1e-3, 5e-2, 5e-2;

    reset();
}

void BallImmPredictor::reset()
{
    stationary_.x.setZero();
    rolling_.x.setZero();
    stationary_.P.setIdentity();
    rolling_.P.setIdentity();
    stationary_.P *= 10.0;
    rolling_.P *= 10.0;
    mode_prob_ = {0.5, 0.5};
    initialized_ = false;
    valid_ = false;
    confidence_ = 0.f;
    frames_since_detection_ = 0;
    occlusion_q_scale_ = 1.0;
}

Eigen::Matrix4d BallImmPredictor::makeTransition(double dt, double mu)
{
    Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
    const double v_decay = std::max(0.0, 1.0 - mu * dt);
    F(0, 2) = dt;
    F(1, 3) = dt;
    F(2, 2) = v_decay;
    F(3, 3) = v_decay;
    return F;
}

double BallImmPredictor::measurementNoise(double range_m)
{
    const double d = std::max(0.1, range_m);
    const double sigma = 0.124 * d + 0.149;
    return sigma * sigma;
}

void BallImmPredictor::predictPosition(const Eigen::Vector4d &x, double mu, double dt, double out[2])
{
    const double t = dt;
    const double px = x(0) + x(2) * t - 0.5 * mu * x(2) * t * t;
    const double py = x(1) + x(3) * t - 0.5 * mu * x(3) * t * t;
    out[0] = px;
    out[1] = py;
}

void BallImmPredictor::predictModel(ModelState &model, double dt)
{
    const Eigen::Matrix4d F = makeTransition(dt, model.friction_mu);
    model.P = F * model.P * F.transpose() + model.Q_base * occlusion_q_scale_;
    model.x = F * model.x;
}

double BallImmPredictor::updateModel(ModelState &model, double zx, double zy, double range_m)
{
    const Eigen::Matrix<double, 2, 4> H = observationMatrix();
    const double R = measurementNoise(range_m);
    Eigen::Matrix2d S = H * model.P * H.transpose();
    S(0, 0) += R;
    S(1, 1) += R;

    Eigen::Vector2d z(zx, zy);
    Eigen::Vector2d y = z - H * model.x;
    Eigen::Matrix<double, 4, 2> K = model.P * H.transpose() * S.inverse();
    model.x = model.x + K * y;
    Eigen::Matrix4d I = Eigen::Matrix4d::Identity();
    model.P = (I - K * H) * model.P;

    const double det = std::max(S.determinant(), 1e-12);
    const double expo = -0.5 * y.transpose() * S.inverse() * y;
    return std::exp(expo) / std::sqrt(2.0 * M_PI * det);
}

void BallImmPredictor::propagate(double dt_sec)
{
    if (dt_sec <= 0.0 || dt_sec > 0.5) {
        dt_sec = 0.02;
    }
    last_dt_ = dt_sec;

    if (!initialized_) {
        return;
    }

    predictModel(stationary_, dt_sec);
    predictModel(rolling_, dt_sec);
}

void BallImmPredictor::add(double field_x, double field_y, double confidence, double range_m)
{
    frames_since_detection_ = 0;
    confidence_ = static_cast<float>(std::clamp(confidence, 0.0, 1.0));

    if (!initialized_) {
        stationary_.x << field_x, field_y, 0.0, 0.0;
        rolling_.x << field_x, field_y, 0.0, 0.0;
        stationary_.P.setIdentity();
        rolling_.P.setIdentity();
        stationary_.P *= 1.0;
        rolling_.P *= 1.0;
        initialized_ = true;
        valid_ = true;
        mode_prob_ = {0.5, 0.5};
        occlusion_q_scale_ = 1.0;
        return;
    }

    occlusion_q_scale_ = 1.0;

    const double like_s = updateModel(stationary_, field_x, field_y, range_m);
    const double like_r = updateModel(rolling_, field_x, field_y, range_m);

    const double cbar_s = 0.05 * mode_prob_[kStationary] + 0.95 * mode_prob_[kRolling];
    const double cbar_r = 0.95 * mode_prob_[kStationary] + 0.05 * mode_prob_[kRolling];
    const double denom = like_s * cbar_s + like_r * cbar_r + 1e-12;
    mode_prob_[kStationary] = (like_s * cbar_s) / denom;
    mode_prob_[kRolling] = (like_r * cbar_r) / denom;

    valid_ = true;
}

void BallImmPredictor::handleOccluded()
{
    if (!initialized_) {
        return;
    }

    ++frames_since_detection_;
    occlusion_q_scale_ *= config_.immOcclusionNoiseGrowth;

    confidence_ *= static_cast<float>(config_.immBallConfidenceDecayRate);

    if (frames_since_detection_ > config_.immMaxOccludedFrames) {
        valid_ = false;
    }
}

BallPrediction BallImmPredictor::getPrediction() const
{
    BallPrediction out;
    if (!initialized_ || !valid_) {
        return out;
    }

    Eigen::Vector4d x_mix = mode_prob_[kStationary] * stationary_.x + mode_prob_[kRolling] * rolling_.x;

    out.pos[0] = static_cast<float>(x_mix(0));
    out.pos[1] = static_cast<float>(x_mix(1));
    out.vel[0] = static_cast<float>(x_mix(2));
    out.vel[1] = static_cast<float>(x_mix(3));
    out.confidence = confidence_;
    out.mode_prob[0] = static_cast<float>(mode_prob_[kStationary]);
    out.mode_prob[1] = static_cast<float>(mode_prob_[kRolling]);
    out.valid = true;
    out.pred300_valid = true;

    const double mu_mix =
        mode_prob_[kStationary] * stationary_.friction_mu + mode_prob_[kRolling] * rolling_.friction_mu;

    double p100[2] = {0.0, 0.0};
    double p300[2] = {0.0, 0.0};
    const double mu_use = (mode_prob_[kStationary] > 0.65) ? stationary_.friction_mu : mu_mix;
    predictPosition(x_mix, mu_use, 0.1, p100);
    predictPosition(x_mix, mu_use, 0.3, p300);
    out.pred100[0] = static_cast<float>(p100[0]);
    out.pred100[1] = static_cast<float>(p100[1]);
    out.pred300[0] = static_cast<float>(p300[0]);
    out.pred300[1] = static_cast<float>(p300[1]);

    return out;
}

void BallImmPredictor::getPositionEstimate(double &px, double &py, Eigen::Matrix2d &pos_cov) const
{
    px = 0.0;
    py = 0.0;
    pos_cov.setIdentity();

    if (!initialized_) {
        return;
    }

    Eigen::Vector4d x_mix = mode_prob_[kStationary] * stationary_.x + mode_prob_[kRolling] * rolling_.x;
    px = x_mix(0);
    py = x_mix(1);

    Eigen::Matrix2d P_s = stationary_.P.block<2, 2>(0, 0);
    Eigen::Matrix2d P_r = rolling_.P.block<2, 2>(0, 0);
    pos_cov = mode_prob_[kStationary] * P_s + mode_prob_[kRolling] * P_r;
}

double BallImmPredictor::gateMahalanobisSq(double field_x, double field_y, double range_m) const
{
    if (!initialized_) {
        return -1.0;
    }

    double px = 0.0;
    double py = 0.0;
    Eigen::Matrix2d pos_cov;
    getPositionEstimate(px, py, pos_cov);

    const double R = measurementNoise(range_m);
    Eigen::Matrix2d S = pos_cov;
    S(0, 0) += R;
    S(1, 1) += R;

    Eigen::Vector2d y(field_x - px, field_y - py);
    const double det = std::max(S.determinant(), 1e-12);
    if (det < 1e-9) {
        return y.squaredNorm();
    }
    return y.transpose() * S.inverse() * y;
}
