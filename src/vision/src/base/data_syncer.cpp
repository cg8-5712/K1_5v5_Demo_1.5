#include "booster_vision/base/data_syncer.hpp"

#include <algorithm>
#include <cfloat>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <regex>

namespace booster_vision {

void DataSyncer::LoadData(const std::string &data_dir) {
    if (!std::filesystem::is_directory(data_dir)) {
        throw std::runtime_error("data directory does not exist: " + data_dir);
    }
    data_dir_ = data_dir;
    // list all files in the directory
    std::vector<std::string> files;
    for (const auto &entry : std::filesystem::directory_iterator(data_dir)) {
        files.push_back(entry.path().filename());
    }
    // regex for matching file names color_xx.xx.jpg, parse xx.xx as timestamp
    std::regex color_file_regex("color_([0-9]+\\.[0-9]+)\\.jpg");
    std::smatch m;
    for (const std::string &file : files) {
        if (std::regex_match(file, m, color_file_regex)) {
            double timestamp = std::stod(m[1].str());
            // std::cout << std::fixed << timestamp << std::endl;
            std::cout << "found timestamp : " << std::fixed << timestamp << std::endl;
            time_stamp_list_.push_back(timestamp);
        }
    }
    std::sort(time_stamp_list_.begin(), time_stamp_list_.end());
    data_index_ = 0;
    std::cout << "loaded " << time_stamp_list_.size() << " data" << std::endl;
}

void DataSyncer::AddDepth(const DepthDataBlock &depth_data) {
    if (!enable_depth_) return;
    std::lock_guard<std::mutex> lock(depth_buffer_mutex_);
    depth_buffer_.push_back(depth_data);
    static uint64_t depth_add_count = 0;
    depth_add_count++;
    if (depth_add_count <= 20 || depth_add_count % 100 == 0) {
        std::cout << "[DataSyncer][add depth] count=" << depth_add_count
                  << " ts=" << depth_data.timestamp
                  << " buffer_size=" << depth_buffer_.size() << std::endl;
    }
}

void DataSyncer::AddPose(const PoseDataBlock &pose_data) {
    std::lock_guard<std::mutex> lock(pose_buffer_mutex_);
    pose_buffer_.push_back(pose_data);
    static uint64_t pose_add_count = 0;
    pose_add_count++;
    if (pose_add_count <= 20 || pose_add_count % 100 == 0) {
        std::cout << "[DataSyncer][add pose] count=" << pose_add_count
                  << " ts=" << pose_data.timestamp
                  << " buffer_size=" << pose_buffer_.size() << std::endl;
    }
}

SyncedDataBlock DataSyncer::getSyncedDataBlock() {
    std::string color_file_path = data_dir_ + "/color_" + std::to_string(time_stamp_list_[data_index_]) + ".jpg";
    std::string depth_file_path = data_dir_ + "/depth_" + std::to_string(time_stamp_list_[data_index_]) + ".png";
    std::string pose_file_path = data_dir_ + "/pose_" + std::to_string(time_stamp_list_[data_index_]) + ".yaml";
    double timestamp = time_stamp_list_[data_index_];
    SyncedDataBlock synced_data;

    if (std::filesystem::exists(color_file_path)) {
        ColorDataBlock color_data;
        color_data.data = cv::imread(color_file_path);
        color_data.timestamp = timestamp;
        synced_data.color_data = color_data;
    } else {
        std::cout << "color file not found: " << color_file_path << std::endl;
    }

    if (enable_depth_) {
        if (std::filesystem::exists(depth_file_path)) {
            DepthDataBlock depth_data;
            depth_data.data = cv::imread(depth_file_path, cv::IMREAD_ANYDEPTH);
            depth_data.timestamp = timestamp;
            synced_data.depth_data = depth_data;
        } else {
            std::cout << "depth file not found: " << depth_file_path << std::endl;
        }
    }

    if (std::filesystem::exists(pose_file_path)) {
        PoseDataBlock pose_data;
        YAML::Node pose_node = YAML::LoadFile(pose_file_path);
        if (pose_node["pose"]) {
            pose_data.data = pose_node["pose"].as<Pose>();
        } else {
            pose_data.data = pose_node.as<Pose>();
        }
        pose_data.timestamp = timestamp;
        synced_data.pose_data = pose_data;
    } else {
        std::cout << "pose file not found: " << pose_file_path << std::endl;
    }

    data_index_ = (data_index_ + 1) % time_stamp_list_.size();
    if (data_index_ == 0) {
        std::cout << "looped all data, reset index to 0" << std::endl;
    }
    return synced_data;
}

SyncedDataBlock DataSyncer::getSyncedDataBlock(const ColorDataBlock &color_data) {
    SyncedDataBlock synced_data;
    synced_data.color_data = color_data;

    DepthBuffer depth_buffer_cp;
    {
        std::lock_guard<std::mutex> lock(depth_buffer_mutex_);
        depth_buffer_cp = depth_buffer_;
    }
    PoseBuffer pose_buffer_cp;
    {
        std::lock_guard<std::mutex> lock(pose_buffer_mutex_);
        pose_buffer_cp = pose_buffer_;
    }

    bool depth_selected = false;
    bool pose_selected = false;
    double depth_best_diff = DBL_MAX;
    double pose_best_diff = DBL_MAX;

    double color_timestamp = color_data.timestamp;
    if (enable_depth_) {
        if (depth_buffer_cp.empty()) {
            static int depth_empty_log_count = 0;
            if (depth_empty_log_count < 20) {
                std::cerr << "[DataSyncer] depth buffer empty when syncing color ts=" << color_timestamp << std::endl;
                depth_empty_log_count++;
            }
        } else {
            double smallest_depth_timestamp_diff = DBL_MAX;
            size_t depth_count = std::min(depth_buffer_cp.size(), static_cast<size_t>(kDepthBufferLength));
            for (size_t i = 0; i < depth_count; ++i) {
                auto it = depth_buffer_cp.rbegin() + i;
                double diff = std::abs(it->timestamp - color_timestamp);
                if (diff < smallest_depth_timestamp_diff) {
                    smallest_depth_timestamp_diff = diff;
                    synced_data.depth_data = *it;
                    synced_data.depth_data.timestamp = it->timestamp;
                    it->data.copyTo(synced_data.depth_data.data);
                    depth_selected = true;
                    depth_best_diff = diff;
                } else {
                    break;
                }
            }
        }
    }

    if (pose_buffer_cp.empty()) {
        static int pose_empty_log_count = 0;
        if (pose_empty_log_count < 20) {
            std::cerr << "[DataSyncer] pose buffer empty when syncing color ts=" << color_timestamp << std::endl;
            pose_empty_log_count++;
        }
    } else {
        double smallest_pose_timestamp_diff = DBL_MAX;
        size_t pose_count = std::min(pose_buffer_cp.size(), static_cast<size_t>(kPoseBufferLength));
        for (size_t i = 0; i < pose_count; ++i) {
            auto it = pose_buffer_cp.rbegin() + i;
            double diff = std::abs(it->timestamp - color_timestamp);
            if (diff < smallest_pose_timestamp_diff) {
                smallest_pose_timestamp_diff = diff;
                synced_data.pose_data = *it;
                pose_selected = true;
                pose_best_diff = diff;
            } else {
                break;
            }
        }
    }

    static uint64_t sync_count = 0;
    sync_count++;
    if (sync_count <= 20 || sync_count % 50 == 0) {
        std::cout << "[DataSyncer][sync] count=" << sync_count
                  << " color_ts=" << color_timestamp
                  << " depth_buf=" << depth_buffer_cp.size()
                  << " pose_buf=" << pose_buffer_cp.size();
        if (enable_depth_) {
            std::cout << " depth_selected=" << depth_selected
                      << " depth_dt_ms=" << (depth_selected ? depth_best_diff * 1000.0 : -1.0);
        }
        std::cout << " pose_selected=" << pose_selected
                  << " pose_dt_ms=" << (pose_selected ? pose_best_diff * 1000.0 : -1.0)
                  << std::endl;
    }

    if (pose_selected && synced_data.pose_data.timestamp == 0) {
        static int pose_zero_ts_log_count = 0;
        if (pose_zero_ts_log_count < 20) {
            std::cerr << "[DataSyncer][warn] selected pose timestamp is 0, likely startup warmup state" << std::endl;
            pose_zero_ts_log_count++;
        }
    }
    if (enable_depth_ && depth_selected && synced_data.depth_data.timestamp == 0) {
        static int depth_zero_ts_log_count = 0;
        if (depth_zero_ts_log_count < 20) {
            std::cerr << "[DataSyncer][warn] selected depth timestamp is 0, likely startup warmup state" << std::endl;
            depth_zero_ts_log_count++;
        }
    }

    return synced_data;
}

} // namespace booster_vision
