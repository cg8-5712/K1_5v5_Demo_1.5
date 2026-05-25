#pragma once

#include <memory>
#include <string>

#include "mcap_writer.h"
#include "training_frame.h"

class Brain;

class TrainingLogger {
public:
    explicit TrainingLogger(Brain *brain);

    void init();
    void tick();

private:
    bool shouldLogFrame();
    void fillFrame(TrainingFrame &frame) const;
    static uint8_t encodeTacticalRole(const std::string &role);
    static uint8_t encodeDecision(const std::string &decision);
    static uint8_t encodeGameState(const std::string &state);
    static uint8_t encodeFallState(int recovery_state);

    Brain *brain_;
    McapWriter writer_;
    uint32_t sequence_ = 0;
    rclcpp::Time last_log_time_;
    bool initialized_ = false;
};
