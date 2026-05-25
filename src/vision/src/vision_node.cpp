#include "booster_vision/vision_node.h"

#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <functional>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <fstream>

#include <yaml-cpp/yaml.h>
#include "ament_index_cpp/get_package_share_directory.hpp"

#include "vision_interface/msg/detected_object.hpp"
#include "vision_interface/msg/detections.hpp"
#include "vision_interface/msg/cal_param.hpp"

#include "booster_vision/base/data_syncer.hpp"
#include "booster_vision/base/data_logger.hpp"
#include "booster_vision/base/misc_utils.hpp"
#include "booster_vision/model//detector.h"
#include "booster_vision/model//segmentor.h"
#include "booster_vision/pose_estimator/pose_estimator.h"
#include "booster_vision/img_bridge.h"

namespace booster_vision {

VisionNode::VisionNode(const std::string &node_name) :
    rclcpp::Node(node_name) {
    this->declare_parameter<bool>("offline_mode", false);
    this->declare_parameter<bool>("show_det", false);
    this->declare_parameter<bool>("show_seg", false);
    this->declare_parameter<bool>("save_data", true);
    this->declare_parameter<bool>("save_depth", true);
    this->declare_parameter<int>("save_fps", 3);
    this->declare_parameter<std::string>("detection_model_path", "");
    this->declare_parameter<std::string>("segmentation_model_path", "");
    this->declare_parameter<std::string>("camera_type", "");
}

// TODO(GW): oneline offline
void VisionNode::Init(const std::string &cfg_template_path, const std::string &cfg_path) {
    if (!std::filesystem::exists(cfg_template_path)) {
        // TODO(SS): throw exception here
        std::cerr << "Error: Configuration template file '" << cfg_template_path << "' does not exist." << std::endl;
        return;
    }

    YAML::Node node = YAML::LoadFile(cfg_template_path);
    if (!std::filesystem::exists(cfg_path)) {
        std::cout << "Warning: Configuration file empty!" << std::endl;
    } else {
        YAML::Node cfg_node = YAML::LoadFile(cfg_path);
        // merge input cfg to template cfg
        MergeYAML(node, cfg_node);
    }

    std::cout << "loaded file: " << std::endl
              << node << std::endl;

    this->get_parameter<bool>("show_det", show_det_);
    this->get_parameter<bool>("show_seg", show_seg_);
    this->get_parameter<bool>("save_data", save_data_);
    this->get_parameter<bool>("save_depth", save_depth_);
    this->get_parameter<bool>("offline_mode", offline_mode_);
    this->get_parameter<std::string>("camera_type", camera_type_);
    this->get_parameter<std::string>("detection_model_path", detection_model_path);
    std::cout << "detection_model_path origin: " << detection_model_path << std::endl;

    if(!detection_model_path.empty()){
        if(detection_model_path[0] == '/') {
            // absolute path, do nothing
        } else {
            std::string package_path = ament_index_cpp::get_package_share_directory("vision");
            detection_model_path = std::filesystem::path(package_path) / detection_model_path;
        }
    }


    this->get_parameter<std::string>("segmentation_model_path", segmentation_model_path);
    std::cout << "segmentation_model_path origin: " << segmentation_model_path << std::endl;

    if(!segmentation_model_path.empty()){
        if(segmentation_model_path[0] == '/') {
            // absolute path, do nothing
        } else {
            std::string package_path = ament_index_cpp::get_package_share_directory("vision");
            segmentation_model_path = std::filesystem::path(package_path) / segmentation_model_path;
        }
    }
    
    int save_fps = 0;
    this->get_parameter<int>("save_fps", save_fps);
    save_depth_ = save_depth_ && save_data_;
    std::cout << "offline_mode: " << offline_mode_ << std::endl;
    std::cout << "show_det: " << show_det_ << std::endl;
    std::cout << "show_seg: " << show_seg_ << std::endl;
    std::cout << "save_data: " << save_data_ << std::endl;
    std::cout << "save_depth: " << save_depth_ << std::endl;
    std::cout << "save_fps: " << save_fps << std::endl;
    std::cout << "camera_type: " << camera_type_ << std::endl;
    std::cout << "detection_model_path: " << detection_model_path << std::endl;
    std::cout << "segmentation_model_path: " << segmentation_model_path << std::endl;
    save_every_n_frame_ = std::max(1, save_fps > 0 ? 30 / save_fps : 1);
    std::cout << "save_every_n_frame: " << save_every_n_frame_ << std::endl;

    // read camera param
    if (!node["camera"]) {
        // TODO(SS): throw exception here
        std::cerr << "no camera param found here" << std::endl;
        return;
    } else {
        if(camera_type_.empty())
        {
            std::cout << "camera type not overridden by launch file, using default: " << node["camera"]["type"].as<std::string>() << std::endl;
            camera_type_ = node["camera"]["type"].as<std::string>();
        }
        intr_ = Intrinsics(node["camera"]["intrin"]);
        p_eye2head_ = as_or<Pose>(node["camera"]["extrin"], Pose());

        float pitch_comp = as_or<float>(node["camera"]["pitch_compensation"], 0.0);
        float yaw_comp = as_or<float>(node["camera"]["yaw_compensation"], 0.0);
        float z_comp = as_or<float>(node["camera"]["z_compensation"], 0.0);

        p_headprime2head_ = Pose(0, 0, z_comp, 0, pitch_comp * M_PI / 180, yaw_comp * M_PI / 180);
    }

    // init detector
    if (!node["detection_model"]) {
        std::cerr << "no detection model param here" << std::endl;
        return;
    } else {
        detector_ = YoloV8Detector::CreateYoloV8Detector(node["detection_model"], detection_model_path);
        classnames_ = node["detection_model"]["classnames"].as<std::vector<std::string>>();
        // detector post processing
        float default_threshold = as_or<float>(node["detection_model"]["confidence_threshold"], 0.2);
        if (node["detection_model"]["post_process"]) {
            enable_post_process_ = true;
            single_ball_assumption_ = as_or<bool>(node["detection_model"]["post_process"]["single_ball_assumption"], false);
            if (node["detection_model"]["post_process"]["confidence_thresholds"]) {
                for (const auto &item : node["detection_model"]["post_process"]["confidence_thresholds"]) {
                    confidence_map_[item.first.as<std::string>()] = item.second.as<float>();
                }
                // set default confidence for other classes
                for (const auto &classname : classnames_) {
                    if (confidence_map_.find(classname) == confidence_map_.end()) {
                        confidence_map_[classname] = default_threshold;
                    }
                }
            } else {
                std::cout << "all class apply same default threshold: " << default_threshold << std::endl;
            }
        }
    }

    if (!node["segmentation_model"]) {
        std::cerr << "no segmentation model param found" << std::endl;
    } else {
        segmentor_ = YoloV8Segmentor::CreateYoloV8Segmentor(node["segmentation_model"], segmentation_model_path);
    }

    // add detector_ warmup

    // init data_syncer
    use_depth_ = as_or<bool>(node["use_depth"], false);
    data_syncer_ = std::make_shared<DataSyncer>(use_depth_);
    bool save_data_nonstationary = as_or<bool>(node["misc"]["save_data_nonstationary"], true);
    std::string log_root = std::string(std::getenv("HOME")) + "/Workspace/vision_log/" + getTimeString();
    data_logger_ = save_data_ ? std::make_shared<DataLogger>(log_root, save_data_nonstationary) : nullptr;
    data_logger_->LogYAML(node, "vision_local.yaml");
    seg_data_syncer_ = std::make_shared<DataSyncer>(false);

    // init robot color classifier
    if (node["robot_color_classifier"]) {
        color_classifier_ = std::make_shared<ColorClassifier>();
        color_classifier_->Init(node["robot_color_classifier"]);
    }

    // init pose estimator
    pose_estimator_ = std::make_shared<PoseEstimator>(intr_);
    pose_estimator_->Init(YAML::Node());
    pose_estimator_map_["default"] = pose_estimator_;

    if (node["ball_pose_estimator"]) {
        pose_estimator_map_["ball"] = std::make_shared<BallPoseEstimator>(intr_);
        pose_estimator_map_["ball"]->Init(node["ball_pose_estimator"]);
    }

    if (node["human_like_pose_estimator"]) {
        pose_estimator_map_["human_like"] = std::make_shared<HumanLikePoseEstimator>(intr_);
        pose_estimator_map_["human_like"]->Init(node["human_like_pose_estimator"]);
    }

    if (node["field_marker_pose_estimator"]) {
        pose_estimator_map_["field_marker"] = std::make_shared<FieldMarkerPoseEstimator>(intr_);
        pose_estimator_map_["field_marker"]->Init(node["field_marker_pose_estimator"]);

        line_segment_area_threshold_ = as_or<int>(node["field_marker_pose_estimator"]["line_segment_area_threshold"], 75);
    }

    if (node["vision_profiler"]) {
        VisionProfiler::Config profiler_cfg;
        profiler_cfg.alert_fps_min = as_or<double>(node["vision_profiler"]["alert_fps_min"], 40.0);
        profiler_cfg.alert_drop_rate_max = as_or<double>(node["vision_profiler"]["alert_drop_rate_max"], 2.0);
        profiler_cfg.alert_e2e_p95_ms_max = as_or<double>(node["vision_profiler"]["alert_e2e_p95_ms_max"], 20.0);
        profiler_cfg.alert_jitter_ms_max = as_or<double>(node["vision_profiler"]["alert_jitter_ms_max"], 5.0);
        profiler_cfg.report_every_n_frames = as_or<int>(node["vision_profiler"]["report_every_n_frames"], 100);
        vision_profiler_.setConfig(profiler_cfg);
        std::cout << "vision_profiler enabled, report every " << profiler_cfg.report_every_n_frames << " frames"
                  << std::endl;
    }

    if (node["ball_prefilter"]) {
        ball_prefilter_conf_min_ = as_or<float>(node["ball_prefilter"]["conf_min"], 0.10f);
    }
    std::cout << "ball_prefilter conf_min: " << ball_prefilter_conf_min_ << std::endl;

    // init ros related

    std::cout << "current camera_type : " << camera_type_ << std::endl;
    std::string color_topic;
    std::string depth_topic;
    if (camera_type_.find("zed") != std::string::npos) {
        color_topic = "/zed/zed_node/left/image_rect_color";
        depth_topic = "/zed/zed_node/depth/depth_registered";
    } else if (camera_type_ == "d-robotics") {
        color_topic = "/image_left_raw";
        depth_topic = "/image_left_raw/camera_info";
    } else if (camera_type_ == "orbbec") {
        color_topic = "/camera/color/image_raw";
        depth_topic = "/camera/depth/image_raw";
    } else {
        // realsense
        color_topic = "/camera/camera/color/image_raw";
        depth_topic = "/camera/camera/aligned_depth_to_color/image_raw";
    }

    callback_group_sub_1_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_2_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_3_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_sub_4_ = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    auto sub_opt_1 = rclcpp::SubscriptionOptions();
    sub_opt_1.callback_group = callback_group_sub_1_;
    auto sub_opt_2 = rclcpp::SubscriptionOptions();
    sub_opt_2.callback_group = callback_group_sub_2_;
    auto sub_opt_3 = rclcpp::SubscriptionOptions();
    sub_opt_3.callback_group = callback_group_sub_3_;
    auto sub_opt_4 = rclcpp::SubscriptionOptions();
    sub_opt_4.callback_group = callback_group_sub_4_;

    it_ = std::make_shared<image_transport::ImageTransport>(shared_from_this());
    image_transport::TransportHints hints(this, "compressed");
    // Subscribe to both raw and compressed image topics for color
    if (camera_type_.find("compressed") != std::string::npos) {
        color_sub_ = it_->subscribe(color_topic, 1, &VisionNode::ColorCallback, this, &hints, sub_opt_1);
    } else {
        color_sub_ = it_->subscribe(color_topic, 1, &VisionNode::ColorCallback, this, nullptr, sub_opt_1);
    } 
    if (use_depth_) {
        depth_sub_ = it_->subscribe(depth_topic, 1, &VisionNode::DepthCallback, this, nullptr, sub_opt_3);
    }
    if (camera_type_.find("compressed") != std::string::npos) {
        color_sub_ = it_->subscribe(color_topic, 2, &VisionNode::ColorCallback, this, &hints, sub_opt_1);
    } else {
        color_sub_ = it_->subscribe(color_topic, 2, &VisionNode::ColorCallback, this, nullptr, sub_opt_1);
    } 
    if (use_depth_) {
        depth_sub_ = it_->subscribe(depth_topic, 2, &VisionNode::DepthCallback, this, nullptr, sub_opt_3);
    }
    std::cout << "[VisionNode] detection color_sub topic=" << color_topic
              << " depth=" << (use_depth_ ? depth_topic : "disabled") << std::endl;

    // auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1))
    // auto qos_profile = rclcpp::QoS(rclcpp::KeepLast(1))
    //     .reliability(rclcpp::ReliabilityPolicy::BestEffort)  // Use best effort for real-time performance
    //     .durability(rclcpp::DurabilityPolicy::Volatile);

    // color_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    //     color_topic,
    //     qos_profile,
    //     std::bind(&VisionNode::ColorCallback, this, std::placeholders::_1),
    //     sub_opt_1);

    // depth_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
    //     depth_topic,
    //     qos_profile,
    //     std::bind(&VisionNode::DepthCallback, this, std::placeholders::_1),
    //     sub_opt_3);

    detection_pub_ = this->create_publisher<vision_interface::msg::Detections>("/booster_vision/detection", rclcpp::QoS(1));

    if (node["segmentation_model"]) {
        std::cout << "[VisionNode] create sub for segmentor on " << color_topic << std::endl;
        std::cout.flush();
        if (camera_type_.find("compressed") != std::string::npos) {
            color_seg_sub_ = it_->subscribe(color_topic, 1, &VisionNode::SegmentationCallback, this, &hints, sub_opt_2);
        } else {
            color_seg_sub_ = it_->subscribe(color_topic, 1, &VisionNode::SegmentationCallback, this, nullptr, sub_opt_2);
        }
        std::cout << "[VisionNode] segmentor subscription ok" << std::endl;
        std::cout.flush();
        field_line_pub_ = this->create_publisher<vision_interface::msg::LineSegments>("/booster_vision/line_segments", rclcpp::QoS(1));
    }
    ball_pub_ = this->create_publisher<vision_interface::msg::Ball>("/booster_vision/ball", rclcpp::QoS(1));

    if (offline_mode_) {
        pose_tf_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>("/booster_vision/t_head2base", 10, std::bind(&VisionNode::PoseTFCallBack, this, std::placeholders::_1));
    } else {
        pose_sub_ = this->create_subscription<geometry_msgs::msg::Pose>("/head_pose", 10, std::bind(&VisionNode::PoseCallBack, this, std::placeholders::_1), sub_opt_4);
        calParam_sub_ = this->create_subscription<vision_interface::msg::CalParam>("/booster_vision/cal_param", 10, std::bind(&VisionNode::CalParamCallback, this, std::placeholders::_1));
        pose_tf_pub_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/booster_vision/t_head2base", rclcpp::QoS(10));
    }

    std::cout << "[VisionNode] Init done"
              << " offline=" << offline_mode_
              << " use_depth=" << use_depth_
              << " save_data=" << save_data_
              << " segmentor=" << (segmentor_ != nullptr)
              << " color=" << color_topic
              << " pose=" << (offline_mode_ ? "/booster_vision/t_head2base" : "/head_pose")
              << std::endl;
    std::cout.flush();
}

void VisionNode::ProcessData(SyncedDataBlock &synced_data, vision_interface::msg::Detections &detection_msg) {
    using SteadyClock = std::chrono::steady_clock;
    auto stage_start = SteadyClock::now();
    auto elapsed_ms = [&](const SteadyClock::time_point &start) {
        return std::chrono::duration<double, std::milli>(SteadyClock::now() - start).count();
    };

    double timestamp = synced_data.color_data.timestamp;
    double depth_time_diff = (timestamp - synced_data.depth_data.timestamp) * 1000;
    double pose_time_diff = (timestamp - synced_data.pose_data.timestamp) * 1000;
    if (use_depth_ && depth_time_diff > 40) {
        std::cerr << "color depth time diff: " << depth_time_diff << "ms" << std::endl;
    }
    if (pose_time_diff > 40) {
        std::cerr << "color pose time diff: " << pose_time_diff << " ms" << std::endl;
    }
    cv::Mat color = synced_data.color_data.data;
    cv::Mat depth = synced_data.depth_data.data;

    cv::Mat depth_float;
    if (!depth.empty() && depth.depth() == CV_16U) {
        depth.convertTo(depth_float, CV_32F, 0.001, 0);
    } else {
        depth_float = depth;
    }

    Pose p_head2base = synced_data.pose_data.data;
    Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;
    std::cout << "det: p_eye2base: \n"
              << p_eye2base.toCVMat() << std::endl;

    vision_profiler_.record("preprocess", elapsed_ms(stage_start));
    stage_start = SteadyClock::now();

    static uint64_t det_infer_log_count = 0;
    const bool log_det_infer = (++det_infer_log_count <= 5);
    if (log_det_infer) {
        std::cout << "[VisionNode][det] TRT inference start #" << det_infer_log_count << std::endl;
        std::cout.flush();
    }
    auto detections = detector_->Inference(color);
    if (log_det_infer) {
        std::cout << "[VisionNode][det] TRT inference done #" << det_infer_log_count
                  << " ms=" << elapsed_ms(stage_start) << std::endl;
        std::cout.flush();
    }
    vision_profiler_.record("inference", elapsed_ms(stage_start));
    stage_start = SteadyClock::now();
    std::cout << detections.size() << " objects detected." << std::endl;

    auto get_estimator = [&](const std::string &class_name) {
        if (class_name == "Ball") {
            return pose_estimator_map_.find("ball") != pose_estimator_map_.end() ? pose_estimator_map_["ball"] : pose_estimator_map_["default"];
        } else if (class_name == "Person" || class_name == "Opponent" || class_name == "Goalpost") {
            return pose_estimator_map_.find("human_like") != pose_estimator_map_.end() ? pose_estimator_map_["human_like"] : pose_estimator_map_["default"];
        } else if (class_name.find("Cross") != std::string::npos || class_name == "PenaltyPoint") {
            return pose_estimator_map_.find("field_marker") != pose_estimator_map_.end() ? pose_estimator_map_["field_marker"] : pose_estimator_map_["default"];
        } else {
            return pose_estimator_map_["default"];
        }
    };

    std::vector<booster_vision::DetectionRes> filtered_detections;
    if (enable_post_process_ && !detections.empty()) {
        auto passes_confidence = [&](const booster_vision::DetectionRes &detection) {
            const auto classname = classnames_[detection.class_id];
            if (!confidence_map_.empty()) {
                const auto it = confidence_map_.find(classname);
                if (it != confidence_map_.end() && detection.confidence < it->second) {
                    return false;
                }
            }
            if (classname == "Ball" && detection.confidence < ball_prefilter_conf_min_) {
                return false;
            }
            return true;
        };

        for (auto &detection : detections) {
            if (passes_confidence(detection)) {
                filtered_detections.push_back(detection);
            }
        }

        // Legacy: optional single-ball mode
        if (single_ball_assumption_) {
            std::vector<booster_vision::DetectionRes> ball_detections;
            std::vector<booster_vision::DetectionRes> filtered_detections_bk = filtered_detections;
            filtered_detections.clear();

            for (const auto &detection : filtered_detections_bk) {
                if (classnames_[detection.class_id] == "Ball") {
                    ball_detections.push_back(detection);
                } else {
                    filtered_detections.push_back(detection);
                }
            }

            if (ball_detections.size() > 1) {
                std::cout << "Multiple ball detections found, keeping the one with highest confidence." << std::endl;
                auto max_ball_detection = *std::max_element(ball_detections.begin(), ball_detections.end(),
                                                            [](const booster_vision::DetectionRes &a, const booster_vision::DetectionRes &b) {
                                                                return a.confidence < b.confidence;
                                                            });
                filtered_detections.push_back(max_ball_detection);
            } else {
                filtered_detections.insert(filtered_detections.end(), ball_detections.begin(), ball_detections.end());
            }
        }
    } else {
        for (auto &detection : detections) {
            const auto classname = classnames_[detection.class_id];
            if (classname == "Ball" && detection.confidence < ball_prefilter_conf_min_) {
                continue;
            }
            filtered_detections.push_back(detection);
        }
    }

    {
        int ball_cnt = 0;
        for (const auto &det : filtered_detections) {
            if (classnames_[det.class_id] == "Ball") {
                ++ball_cnt;
            }
        }
        static uint64_t post_log_count = 0;
        if (++post_log_count <= 5 || post_log_count % 100 == 0) {
            std::cout << "[vision] ball candidates after prefilter: " << ball_cnt << std::endl;
        }
    }

    vision_profiler_.record("postprocess", elapsed_ms(stage_start));
    stage_start = SteadyClock::now();

    std::vector<booster_vision::DetectionRes> detections_for_display;
    for (auto &detection : filtered_detections) {
        vision_interface::msg::DetectedObject detection_obj;

        detection.class_name = detector_->kClassLabels[detection.class_id];

        auto pose_estimator = get_estimator(detection.class_name);
        Pose pose_obj_by_color = pose_estimator->EstimateByColor(p_eye2base, detection, color);
        Pose pose_obj_by_depth = pose_estimator->EstimateByDepth(p_eye2base, detection, color, depth_float);

        // filter out incorrect ball detection
        if (pose_estimator->use_depth_ && detection.class_name == "Ball" && pose_obj_by_depth == Pose()) {
            std::cout << "filtered out ball detection by depth" << std::endl;
            continue;
        }
        detection_obj.position_projection = pose_obj_by_color.getTranslationVec();
        detection_obj.position = pose_obj_by_depth.getTranslationVec();

        auto xyz = p_head2base.getTranslationVec();
        auto rpy = p_head2base.getEulerAnglesVec();
        detection_obj.received_pos = {xyz[0], xyz[1], xyz[2],
                                      static_cast<float>(rpy[0] / CV_PI * 180), static_cast<float>(rpy[1] / CV_PI * 180), static_cast<float>(rpy[2] / CV_PI * 180)};

        detection_obj.confidence = detection.confidence * 100;
        detection_obj.xmin = detection.bbox.x;
        detection_obj.ymin = detection.bbox.y;
        detection_obj.xmax = detection.bbox.x + detection.bbox.width;
        detection_obj.ymax = detection.bbox.y + detection.bbox.height;
        detection_obj.label = detection.class_name;

        if ((color_classifier_ != nullptr) && (detection.class_name == "Opponent")) {
            // get a crop of the image given detection.bbox
            cv::Mat crop = color(detection.bbox);
            std::string robot_color_str = color_classifier_->Classify(crop);
            // add robot color to detection_obj
            detection_obj.color = robot_color_str;
        }

        // publish detection
        detection_msg.detected_objects.push_back(detection_obj);
        detections_for_display.push_back(detection);
    }

    vision_profiler_.record("pose_estimate", elapsed_ms(stage_start));
    stage_start = SteadyClock::now();

    // compute corner points positision
    std::vector<cv::Point2f> corner_uvs = {cv::Point2f(0, 0), cv::Point2f(color.cols - 1, 0),
                                           cv::Point2f(color.cols - 1, color.rows - 1), cv::Point2f(0, color.rows - 1),
                                           cv::Point2f(color.cols / 2.0, color.rows / 2.0)};
    for (auto &uv : corner_uvs) {
        auto corner_pos = CalculatePositionByIntersection(p_eye2base, uv, intr_);
        detection_msg.corner_pos.push_back(corner_pos.x);
        detection_msg.corner_pos.push_back(corner_pos.y);
    }

    vision_interface::msg::Ball ball_msg;
    ball_msg.header = detection_msg.header;
    ball_msg.confidence = 0;
    for (auto &detection : filtered_detections) {
        if (detection.class_name == "Ball" && detection.confidence > ball_msg.confidence) {
            if (detection.confidence <= ball_msg.confidence) {
               continue;
            }

            auto pose_estimator = get_estimator(detection.class_name);
            Pose pose_obj_by_color = pose_estimator->EstimateByColor(p_eye2base, detection, color);
            auto value = pose_obj_by_color.getTranslationVec();
            if (value[0] < -2 || value[0] > 10 || value[1] < -5 || value[1] > 5) {
                continue;
            }
            ball_msg.x = value[0];
            ball_msg.y = value[1];
            ball_msg.confidence = detection.confidence;
            break;
        }
    }

    detection_pub_->publish(detection_msg);
    ball_pub_->publish(ball_msg);
    vision_profiler_.record("publish", elapsed_ms(stage_start));
    vision_profiler_.markPublished();
    std::cout << std::endl;

    {
        static double last_pub_time = -1.0;
        static uint64_t count = 0;
        double pub_ts = detection_msg.header.stamp.sec +
                        static_cast<double>(detection_msg.header.stamp.nanosec) * 1e-9;
        if (last_pub_time >= 0.0) {
            double diff_ms = (pub_ts - last_pub_time) * 1000.0;
            std::cout << "[Detections Pub Interval] #" << (count) 
                      << " -> #" << (count + 1) << ": " << diff_ms << " ms" << std::endl;
        }
        last_pub_time = pub_ts;
        count++;
    }

    // show vision results
    if (show_det_) {
        cv::Mat color_rgb;
        cv::cvtColor(color, color_rgb, cv::COLOR_BGR2RGB);
        cv::Mat img_out = YoloV8Detector::DrawDetection(color_rgb, detections_for_display);
        cv::imshow("Detection", img_out);

        // color jet depth_float and show
        // if (!depth_float.empty()) {
        //     cv::Mat depth_colormap;
        //     cv::normalize(depth_float, depth_float, 0, 255, cv::NORM_MINMAX);
        //     depth_float.convertTo(depth_float, CV_8U);
        //     cv::applyColorMap(depth_float, depth_colormap, cv::COLORMAP_JET);
        //     cv::imshow("Depth", depth_colormap);
        // }

        cv::waitKey(1);
    }

    if (save_data_) {
        save_cnt_++;
        if (save_cnt_ % save_every_n_frame_ != 0) {
            return;
        } else {
            save_cnt_ = 0;
        }
        data_logger_->LogDataBlock(synced_data);
    }
}

void VisionNode::ColorCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    static uint64_t color_cb_count = 0;
    color_cb_count++;
    if (color_cb_count <= 20 || color_cb_count % 50 == 0) {
        std::cout << "[VisionNode][det cb] count=" << color_cb_count << std::endl;
    }
    auto start = std::chrono::system_clock::now();
    if (!msg) {
        std::cerr << "empty image message." << std::endl;
        return;
    }

    vision_profiler_.markReceive();

    // cv_bridge::CvImagePtr cv_ptr;
    cv::Mat img;
    try {
        // cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        img = toCVMat(*msg);
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        return;
    }

    if (camera_type_ == "realsense") {
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    }

    vision_interface::msg::Detections detection_msg;
    detection_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // get synced data
    SyncedDataBlock synced_data = data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    if ((color_cb_count <= 20 || color_cb_count % 50 == 0)) {
        double pose_dt_ms = (timestamp - synced_data.pose_data.timestamp) * 1000;
        bool depth_ready = !use_depth_ || !synced_data.depth_data.data.empty();
        std::cout << "[VisionNode][det sync] ts=" << timestamp
                  << " pose_dt_ms=" << pose_dt_ms
                  << " depth_ready=" << depth_ready
                  << " use_depth=" << use_depth_ << std::endl;
    }

    ProcessData(synced_data, detection_msg);
    auto end = std::chrono::system_clock::now();
    std::cout << "color callback takes: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;
}

void VisionNode::ProcessSegmentationData(SyncedDataBlock &synced_data, vision_interface::msg::LineSegments &field_line_segs_msg) {
    double timestamp = synced_data.color_data.timestamp;
    cv::Mat color = synced_data.color_data.data;
    Pose p_head2base = synced_data.pose_data.data;
    Pose p_eye2base = p_head2base * p_headprime2head_ * p_eye2head_;

    double time_diff = (timestamp - synced_data.pose_data.timestamp) * 1000;
    if (time_diff > 40) {
        std::cerr << "seg: color pose time diff: " << time_diff << " ms" << std::endl;
    }
    std::cout << "seg: p_eye2base: \n"
              << p_eye2base.toCVMat() << std::endl;

    static uint64_t seg_infer_log_count = 0;
    const bool log_seg_infer = (++seg_infer_log_count <= 5);
    if (log_seg_infer) {
        std::cout << "[VisionNode][seg] TRT inference start #" << seg_infer_log_count << std::endl;
        std::cout.flush();
    }
    const auto seg_infer_start = std::chrono::steady_clock::now();
    auto segmentations = segmentor_->Inference(color);
    if (log_seg_infer) {
        const double seg_ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - seg_infer_start)
                                  .count();
        std::cout << "[VisionNode][seg] TRT inference done #" << seg_infer_log_count
                  << " ms=" << seg_ms << std::endl;
        std::cout.flush();
    }
    std::vector<FieldLineSegment> field_line_segs;
    for (auto &seg : segmentations) {
        // TODO: fit circle line
        if (seg.class_id == 0) continue;
        auto line_segs = FitFieldLineSegments(p_eye2base, intr_, seg.contour, line_segment_area_threshold_);
        for (auto line_seg : line_segs) {
            float inlier_precentage = static_cast<float>(line_seg.inlier_count) / line_seg.contour_2d_points.size();
            if (inlier_precentage < 0.25) {
                continue;
            }
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[0].y);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].x);
            field_line_segs_msg.coordinates.push_back(line_seg.end_points_3d[1].y);

            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[0].x);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[0].y);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[1].x);
            field_line_segs_msg.coordinates_uv.push_back(line_seg.end_points_2d[1].y);

            field_line_segs.push_back(line_seg);
        }
    }
    std::cout << segmentations.size() << " objects segmented." << std::endl;

    field_line_pub_->publish(field_line_segs_msg);
    if (show_seg_) {
        cv::Mat img_out = YoloV8Segmentor::DrawSegmentation(color, segmentations);
        img_out = DrawFieldLineSegments(img_out, field_line_segs);
        cv::imshow("Segmentation", img_out);
        cv::waitKey(1);
    }
}

void VisionNode::SegmentationCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    static uint64_t seg_cb_count = 0;
    seg_cb_count++;
    if (seg_cb_count <= 20 || seg_cb_count % 50 == 0) {
        std::cout << "[VisionNode][seg cb] count=" << seg_cb_count << std::endl;
    }
    if (!segmentor_) {
        std::cerr << "no segmentor loaded." << std::endl;
        return;
    }
    if (!msg) {
        std::cerr << "empty image message." << std::endl;
        return;
    }

    // cv_bridge::CvImagePtr cv_ptr; // 使用cv_bridge将ROS图像消息转换为OpenCV cv::Mat格式
    cv::Mat img;
    try {
        // cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
        img = toCVMat(*msg).clone();
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception: " << e.what() << std::endl;
        return;
    }

    vision_interface::msg::LineSegments field_line_segs_msg;
    field_line_segs_msg.header = msg->header;
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;

    // get synced data
    SyncedDataBlock synced_data = seg_data_syncer_->getSyncedDataBlock(ColorDataBlock(img, timestamp));
    if (seg_cb_count <= 20 || seg_cb_count % 50 == 0) {
        double pose_dt_ms = (timestamp - synced_data.pose_data.timestamp) * 1000;
        std::cout << "[VisionNode][seg sync] ts=" << timestamp
                  << " pose_dt_ms=" << pose_dt_ms
                  << " pose_ts=" << synced_data.pose_data.timestamp << std::endl;
    }
    ProcessSegmentationData(synced_data, field_line_segs_msg);
}

void VisionNode::DepthCallback(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    static uint64_t depth_cb_count = 0;
    depth_cb_count++;
    if (depth_cb_count <= 20 || depth_cb_count % 50 == 0) {
        std::cout << "[VisionNode][depth cb] count=" << depth_cb_count << std::endl;
    }
    // cv_bridge::CvImagePtr cv_ptr;
    cv::Mat img;
    try {
        // TODO(SS): check if the image is 16-bit for zed camera
        // cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1);
        img = toCVMat(*msg).clone();
    } catch (std::exception &e) {
        std::cerr << "cv_bridge exception " << e.what() << std::endl;
        return;
    }

    if (img.empty()) {
        std::cerr << "empty image recevied." << std::endl;
        return;
    }

    // Check if the image is indeed 16-bit
    if (img.depth() != CV_16U && img.depth() != CV_32F) {
        std::cerr << "image is either 16u or 32f." << std::endl;
        return;
    }

    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
    // seg_data_syncer_->AddDepth(DepthDataBlock(img, timestamp));
}

void VisionNode::PoseTFCallBack(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
    double timestamp = msg->header.stamp.sec + static_cast<double>(msg->header.stamp.nanosec) * 1e-9;
    data_syncer_->AddPose(PoseDataBlock(Pose(*msg), timestamp));
    seg_data_syncer_->AddPose(PoseDataBlock(Pose(*msg), timestamp));
}

void VisionNode::PoseCallBack(const geometry_msgs::msg::Pose::SharedPtr msg) {
    static bool first_pose = true;
    if (first_pose) {
        first_pose = false;
        std::cout << "[VisionNode] first /head_pose received" << std::endl;
    }
    auto current_time = this->get_clock()->now();
    double timestamp = static_cast<double>(current_time.nanoseconds()) * 1e-9;

    float x = msg->position.x;
    float y = msg->position.y;
    float z = msg->position.z;
    float qx = msg->orientation.x;
    float qy = msg->orientation.y;
    float qz = msg->orientation.z;
    float qw = msg->orientation.w;
    auto pose = Pose(x, y, z, qx, qy, qz, qw);
    data_syncer_->AddPose(PoseDataBlock(pose, timestamp));
    seg_data_syncer_->AddPose(PoseDataBlock(pose, timestamp));

    if (!offline_mode_) {
        auto tf_msg = pose.toRosTFMsg();
        tf_msg.header.stamp = builtin_interfaces::msg::Time(current_time);

        pose_tf_pub_->publish(tf_msg);
    }
}

void VisionNode::CalParamCallback(const vision_interface::msg::CalParam::SharedPtr msg) {
    float pitch_comp = msg->pitch_compensation;
    float yaw_comp = msg->yaw_compensation;
    float z_comp = msg->z_compensation;
    std::cout << "calParams: " << pitch_comp << " " << yaw_comp << " " << z_comp << std::endl;
    p_headprime2head_ = Pose(0, 0, z_comp, 0, pitch_comp * M_PI / 180, yaw_comp * M_PI / 180);
}

} // namespace booster_vision
