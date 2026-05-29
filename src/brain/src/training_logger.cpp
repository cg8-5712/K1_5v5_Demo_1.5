#include "training_logger.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "brain.h"
#include "brain_data.h"
#include "types.h"
#include "utils/misc.h"
#include "utils/print.h"

namespace {

constexpr uint16_t kSchemaId = 1;
constexpr uint16_t kChannelId = 1;
constexpr char kTopic[] = "/k1/training_frame";

const char *kSchemaJson = R"({"name":"k1.TrainingFrame","version":1,"size":153,"encoding":"little_endian_packed"})";

uint8_t lookupCode(const std::unordered_map<std::string, uint8_t> &table, const std::string &key)
{
    const auto it = table.find(key);
    return it == table.end() ? 0 : it->second;
}

} // namespace

TrainingLogger::TrainingLogger(Brain *brain) : brain_(brain) {}

void TrainingLogger::init()
{
    initialized_ = false;
    if (!brain_->config->trainingLoggerEnabled) {
        return;
    }

    std::string dir = brain_->config->trainingLoggerLogDir;
    if (dir.empty()) {
        dir = ".";
    }
    dir = gen_timestamped_filename(dir,
        format("_T%d_P%d", brain_->config->teamId, brain_->config->playerId));
    if (!mkdir_if_not_exist(dir)) {
        prtErr("TrainingLogger: failed to create log dir: " + dir);
        return;
    }

    const std::string path =
        gen_timestamped_filename(dir, format("_robot%d.mcap", brain_->config->playerId));
    if (!writer_.open(path)) {
        prtErr("TrainingLogger: failed to open " + path);
        return;
    }

    const std::vector<uint8_t> schema_data(kSchemaJson,
        kSchemaJson + std::strlen(kSchemaJson));
    if (!writer_.writeSchema(kSchemaId, "k1.TrainingFrame", "jsonschema", schema_data)
        || !writer_.writeChannel(kChannelId, kSchemaId, kTopic, "k1_training_frame")) {
        prtErr("TrainingLogger: failed to write MCAP schema/channel");
        writer_.close();
        return;
    }

    last_log_time_ = brain_->get_clock()->now();
    initialized_ = true;
    prtDebug("TrainingLogger: writing to " + path);
}

void TrainingLogger::tick()
{
    if (!initialized_ || !writer_.isOpen()) {
        return;
    }
    if (!shouldLogFrame()) {
        return;
    }

    TrainingFrame frame{};
    fillFrame(frame);
    const uint64_t log_time_ns = static_cast<uint64_t>(frame.timestamp_us) * 1000ULL;
    writer_.writeMessage(kChannelId, sequence_++, log_time_ns, &frame, sizeof(frame));
}

bool TrainingLogger::shouldLogFrame()
{
    const double hz = std::max(1.0, brain_->config->trainingLoggerHz);
    const double min_interval_sec = 1.0 / hz;
    const auto now = brain_->get_clock()->now();
    const double elapsed = (now - last_log_time_).seconds();
    if (elapsed < min_interval_sec) {
        return false;
    }
    last_log_time_ = now;
    return true;
}

void TrainingLogger::fillFrame(TrainingFrame &f) const
{
    const auto &data = brain_->data;

    f.timestamp_us = static_cast<uint64_t>(brain_->get_clock()->now().nanoseconds() / 1000ULL);

    if (data->ballDetected) {
        f.raw_ball_robot[0] = static_cast<float>(data->ball.posToRobot.x);
        f.raw_ball_robot[1] = static_cast<float>(data->ball.posToRobot.y);
        f.raw_ball_conf = static_cast<float>(data->ball.confidence / 100.0);
        const auto &bb = data->ball.boundingBox;
        f.bbox_xywh[0] = static_cast<float>(bb.xmin);
        f.bbox_xywh[1] = static_cast<float>(bb.ymin);
        f.bbox_xywh[2] = static_cast<float>(bb.xmax - bb.xmin);
        f.bbox_xywh[3] = static_cast<float>(bb.ymax - bb.ymin);
    }

    if (data->filteredBallField.valid) {
        f.filtered_ball_field[0] = static_cast<float>(data->filteredBallField.x);
        f.filtered_ball_field[1] = static_cast<float>(data->filteredBallField.y);
    }
    if (data->pred100Field.valid) {
        f.pred100_field[0] = static_cast<float>(data->pred100Field.x);
        f.pred100_field[1] = static_cast<float>(data->pred100Field.y);
    }
    if (data->pred300Field.valid) {
        f.pred300_field[0] = static_cast<float>(data->pred300Field.x);
        f.pred300_field[1] = static_cast<float>(data->pred300Field.y);
    }
    f.pred300_valid = data->pred300Valid ? 1 : 0;
    f.using_field_frame = data->usingFieldFrame ? 1 : 0;
    if (data->pred100Robot.valid) {
        f.pred100_robot[0] = static_cast<float>(data->pred100Robot.x);
        f.pred100_robot[1] = static_cast<float>(data->pred100Robot.y);
    }
    f.mode_prob[0] = data->ballModeProb[0];
    f.mode_prob[1] = data->ballModeProb[1];
    f.ball_confidence = data->ballConfidence;

    f.robot_pose[0] = static_cast<float>(data->robotPoseToField.x);
    f.robot_pose[1] = static_cast<float>(data->robotPoseToField.y);
    f.robot_pose[2] = static_cast<float>(data->robotPoseToField.theta);

    double vx = 0.0, vy = 0.0, vtheta = 0.0;
    brain_->client->getLastCommandedVelocity(vx, vy, vtheta);
    f.robot_vel[0] = static_cast<float>(vx);
    f.robot_vel[1] = static_cast<float>(vy);
    f.robot_vel[2] = static_cast<float>(vtheta);

    f.head_pose[0] = static_cast<float>(data->headYaw);
    f.head_pose[1] = static_cast<float>(data->headPitch);
    f.imu_acc[0] = data->imuAcc[0];
    f.imu_acc[1] = data->imuAcc[1];
    f.imu_acc[2] = data->imuAcc[2];

    const std::string tacticalRole = brain_->tree->getEntry<std::string>("tactical_role");
    f.tactical_role = encodeTacticalRole(tacticalRole);
    f.decision = encodeDecision(brain_->tree->getEntry<std::string>("decision"));
    f.is_lead = data->tmImLead ? 1 : 0;
    f.cost = static_cast<float>(data->tmMyCost);

    f.kick_result = data->trainingKickResult;
    f.abort_reason = data->trainingKickAbortReason;

    const int tmSlots = std::min(4, HL_MAX_NUM_PLAYERS);
    for (int i = 0; i < tmSlots; ++i) {
        f.tm_age_ms[i] = static_cast<float>(brain_->msecsSince(data->tmStatus[i].timeLastCom));
    }

    f.fall_state = encodeFallState(static_cast<int>(data->recoveryState));
    f.game_state = encodeGameState(brain_->tree->getEntry<std::string>("gc_game_state"));
}

uint8_t TrainingLogger::encodeTacticalRole(const std::string &role)
{
    static const std::unordered_map<std::string, uint8_t> kTable = {
        {"lead_striker", 1}, {"assist_striker", 2}, {"goal_keeper", 3}};
    return lookupCode(kTable, role);
}

uint8_t TrainingLogger::encodeDecision(const std::string &decision)
{
    static const std::unordered_map<std::string, uint8_t> kTable = {
        {"find", 1},           {"chase", 2},           {"adjust", 3},
        {"kick", 4},           {"cross", 5},           {"shoot", 6},
        {"assist", 7},         {"auto_visual_kick", 8}, {"power_shoot", 9},
        {"kick_select", 10}};
    return lookupCode(kTable, decision);
}

uint8_t TrainingLogger::encodeGameState(const std::string &state)
{
    static const std::unordered_map<std::string, uint8_t> kTable = {
        {"INITIAL", 1}, {"READY", 2}, {"SET", 3}, {"PLAY", 4}, {"END", 5}};
    return lookupCode(kTable, state);
}

uint8_t TrainingLogger::encodeFallState(int recovery_state)
{
    switch (static_cast<RobotRecoveryState>(recovery_state)) {
    case RobotRecoveryState::IS_READY:
        return 0;
    case RobotRecoveryState::IS_FALLING:
        return 1;
    case RobotRecoveryState::HAS_FALLEN:
        return 2;
    case RobotRecoveryState::IS_GETTING_UP:
        return 3;
    default:
        return 255;
    }
}
