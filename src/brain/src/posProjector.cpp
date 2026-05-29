#include "posProjector.h"

#include <cmath>

#include "utils/math.h"

void PosProjector::init(std::shared_ptr<Locator> locator,
                        std::shared_ptr<BrainData> data,
                        std::shared_ptr<BrainTree> tree,
                        std::shared_ptr<BrainConfig> config,
                        rclcpp::Clock::SharedPtr clock)
{
    locator_ = std::move(locator);
    data_ = std::move(data);
    tree_ = std::move(tree);
    config_ = std::move(config);
    clock_ = std::move(clock);
}

double PosProjector::localizationCovarianceTrace() const
{
    if (!locator_ || locator_->hypos.rows() == 0) {
        return 1e6;
    }

    double mean_x = 0.0;
    double mean_y = 0.0;
    double weight_sum = 0.0;
    const int n = static_cast<int>(locator_->hypos.rows());

    for (int i = 0; i < n; ++i) {
        const double w = locator_->hypos(i, 4);
        if (w < 1e-12) {
            continue;
        }
        mean_x += locator_->hypos(i, 0) * w;
        mean_y += locator_->hypos(i, 1) * w;
        weight_sum += w;
    }

    if (weight_sum < 1e-12) {
        return 1e6;
    }

    mean_x /= weight_sum;
    mean_y /= weight_sum;

    double var_x = 0.0;
    double var_y = 0.0;
    for (int i = 0; i < n; ++i) {
        const double w = locator_->hypos(i, 4) / weight_sum;
        if (w < 1e-12) {
            continue;
        }
        const double dx = locator_->hypos(i, 0) - mean_x;
        const double dy = locator_->hypos(i, 1) - mean_y;
        var_x += w * dx * dx;
        var_y += w * dy * dy;
    }

    return var_x + var_y;
}

bool PosProjector::isLocalizationTrusted() const
{
    if (!data_ || !tree_ || !config_) {
        return false;
    }
    if (!tree_->getEntry<bool>("odom_calibrated")) {
        return false;
    }

    const double localize_age_ms =
        (clock_ && data_->lastSuccessfulLocalizeTime.nanoseconds() > 0)
            ? (clock_->now() - data_->lastSuccessfulLocalizeTime).seconds() * 1000.0
            : 1e9;
    if (localize_age_ms > 3000.0) {
        return false;
    }

    return localizationCovarianceTrace() < config_->immLocalizationTrustCovMax;
}

void PosProjector::robotToField(double rx, double ry, double &fx, double &fy) const
{
    if (!data_) {
        fx = rx;
        fy = ry;
        return;
    }
    Pose2D pr{rx, ry, 0.0};
    Pose2D pf = data_->robot2field(pr);
    fx = pf.x;
    fy = pf.y;
}

void PosProjector::fieldToRobot(double fx, double fy, double &rx, double &ry) const
{
    if (!data_) {
        rx = fx;
        ry = fy;
        return;
    }
    Pose2D pf{fx, fy, 0.0};
    Pose2D pr = data_->field2robot(pf);
    rx = pr.x;
    ry = pr.y;
}
