#include "booster_vision/base/vision_profiler.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace booster_vision {

namespace {

double percentile(std::deque<double> values, double p)
{
    if (values.empty()) {
        return 0.0;
    }
    std::sort(values.begin(), values.end());
    const double rank = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(rank));
    const size_t hi = static_cast<size_t>(std::ceil(rank));
    if (lo == hi) {
        return values[lo];
    }
    const double frac = rank - static_cast<double>(lo);
    return values[lo] * (1.0 - frac) + values[hi] * frac;
}

} // namespace

void VisionProfiler::StageStats::add(double ms)
{
    samples.push_back(ms);
    constexpr size_t kMaxSamples = 512;
    while (samples.size() > kMaxSamples) {
        samples.pop_front();
    }
}

std::pair<double, double> VisionProfiler::StageStats::percentile50_95() const
{
    if (samples.empty()) {
        return {0.0, 0.0};
    }
    auto copy50 = samples;
    auto copy95 = samples;
    return {percentile(copy50, 0.50), percentile(copy95, 0.95)};
}

void VisionProfiler::markReceive()
{
    const auto now = std::chrono::steady_clock::now();
    if (has_last_receive_) {
        const double gap_ms = std::chrono::duration<double, std::milli>(now - last_receive_time_).count();
        const double expected_ms = 1000.0 / std::max(config_.alert_fps_min, 1.0);
        if (gap_ms > 2.5 * expected_ms) {
            ++report_window_dropped_;
        }
    }
    last_receive_time_ = now;
    has_last_receive_ = true;
    e2e_start_ = now;
    e2e_active_ = true;
}

void VisionProfiler::record(const std::string &stage, double duration_ms)
{
    if (stage == "preprocess") {
        preprocess_.add(duration_ms);
    } else if (stage == "inference") {
        inference_.add(duration_ms);
    } else if (stage == "postprocess") {
        postprocess_.add(duration_ms);
    } else if (stage == "pose_estimate") {
        pose_estimate_.add(duration_ms);
    } else if (stage == "publish") {
        publish_.add(duration_ms);
    }
}

void VisionProfiler::markPublished()
{
    if (e2e_active_) {
        const double e2e_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - e2e_start_)
                                  .count();
        e2e_.add(e2e_ms);
        e2e_active_ = false;
    }
    ++total_published_;
    ++report_window_frames_;
    maybeReport();
}

double VisionProfiler::windowFps() const
{
    if (report_window_frames_ <= 0 || e2e_.samples.empty()) {
        return 0.0;
    }
    double sum_ms = 0.0;
    for (double v : e2e_.samples) {
        sum_ms += v;
    }
    const double avg_ms = sum_ms / static_cast<double>(e2e_.samples.size());
    return avg_ms > 1e-6 ? 1000.0 / avg_ms : 0.0;
}

double VisionProfiler::windowDropRatePercent() const
{
    const int denom = report_window_frames_ + report_window_dropped_;
    if (denom <= 0) {
        return 0.0;
    }
    return 100.0 * static_cast<double>(report_window_dropped_) / static_cast<double>(denom);
}

void VisionProfiler::maybeReport()
{
    if (report_window_frames_ < config_.report_every_n_frames) {
        return;
    }

    const auto p_preprocess = preprocess_.percentile50_95();
    const auto p_inference = inference_.percentile50_95();
    const auto p_postprocess = postprocess_.percentile50_95();
    const auto p_pose = pose_estimate_.percentile50_95();
    const auto p_publish = publish_.percentile50_95();
    const auto p_e2e = e2e_.percentile50_95();
    const double jitter = std::max(0.0, p_e2e.second - p_e2e.first);
    const double fps = windowFps();
    const double drop = windowDropRatePercent();

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(1);
    oss << "[VisionProfiler] fps=" << fps << "  drop=" << drop << "%\n"
        << "  preprocess  p50=" << std::setw(5) << p_preprocess.first << "ms  p95=" << std::setw(5)
        << p_preprocess.second << "ms\n"
        << "  inference   p50=" << std::setw(5) << p_inference.first << "ms  p95=" << std::setw(5)
        << p_inference.second << "ms\n"
        << "  postprocess p50=" << std::setw(5) << p_postprocess.first << "ms  p95=" << std::setw(5)
        << p_postprocess.second << "ms\n"
        << "  pose_est    p50=" << std::setw(5) << p_pose.first << "ms  p95=" << std::setw(5) << p_pose.second
        << "ms\n"
        << "  publish     p50=" << std::setw(5) << p_publish.first << "ms  p95=" << std::setw(5) << p_publish.second
        << "ms\n"
        << "  e2e         p50=" << std::setw(5) << p_e2e.first << "ms  p95=" << std::setw(5) << p_e2e.second
        << "ms  jitter_p95-p50=" << jitter << "ms";

    const bool alert_fps = fps > 0.0 && fps < config_.alert_fps_min;
    const bool alert_drop = drop > config_.alert_drop_rate_max;
    const bool alert_e2e = p_e2e.second > config_.alert_e2e_p95_ms_max;
    const bool alert_jitter = jitter > config_.alert_jitter_ms_max;
    if (alert_fps || alert_drop || alert_e2e || alert_jitter) {
        oss << "\n  [ALERT]";
        if (alert_fps) {
            oss << " fps<" << config_.alert_fps_min;
        }
        if (alert_drop) {
            oss << " drop>" << config_.alert_drop_rate_max << "%";
        }
        if (alert_e2e) {
            oss << " e2e_p95>" << config_.alert_e2e_p95_ms_max << "ms";
        }
        if (alert_jitter) {
            oss << " jitter>" << config_.alert_jitter_ms_max << "ms";
        }
    }

    std::cout << oss.str() << std::endl;

    report_window_frames_ = 0;
    report_window_dropped_ = 0;
}

} // namespace booster_vision
