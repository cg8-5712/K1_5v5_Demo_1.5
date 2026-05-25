#pragma once

#include <chrono>
#include <cstddef>
#include <deque>
#include <iostream>
#include <string>
#include <utility>

namespace booster_vision {

class VisionProfiler {
public:
    struct Config {
        double alert_fps_min = 40.0;
        double alert_drop_rate_max = 2.0;
        double alert_e2e_p95_ms_max = 20.0;
        double alert_jitter_ms_max = 5.0;
        int report_every_n_frames = 100;
    };

    void setConfig(const Config &cfg) { config_ = cfg; }

    /** Called when a color frame enters the pipeline (start e2e). */
    void markReceive();

    /** Record stage duration in milliseconds. */
    void record(const std::string &stage, double duration_ms);

    /** Called after detection topic publish; may emit periodic report. */
    void markPublished();

private:
    struct StageStats {
        std::deque<double> samples;
        void add(double ms);
        std::pair<double, double> percentile50_95() const;
    };

    void maybeReport();
    double windowFps() const;
    double windowDropRatePercent() const;

    Config config_;
    int report_window_frames_ = 0;
    int report_window_dropped_ = 0;
    int total_published_ = 0;

    bool e2e_active_ = false;
    std::chrono::steady_clock::time_point e2e_start_;
    std::chrono::steady_clock::time_point last_receive_time_;
    bool has_last_receive_ = false;

    StageStats preprocess_;
    StageStats inference_;
    StageStats postprocess_;
    StageStats pose_estimate_;
    StageStats publish_;
    StageStats e2e_;
};

} // namespace booster_vision
