#pragma once

#include <cstdint>

#pragma pack(push, 1)
struct TrainingFrame {
    uint64_t timestamp_us;
    float raw_ball_robot[2];
    float raw_ball_conf;
    float bbox_xywh[4];

    float filtered_ball_field[2];
    float pred100_field[2];
    float pred300_field[2];
    uint8_t pred300_valid;
    uint8_t using_field_frame;
    float pred100_robot[2];
    float mode_prob[2];
    float ball_confidence;

    float robot_pose[3];
    float robot_vel[3];
    float head_pose[2];
    float imu_acc[3];

    uint8_t tactical_role;
    uint8_t decision;
    uint8_t is_lead;
    float cost;

    int8_t kick_result;
    uint8_t abort_reason;

    float tm_age_ms[4];

    uint8_t fall_state;
    uint8_t game_state;
};
#pragma pack(pop)

static_assert(sizeof(TrainingFrame) == 153, "TrainingFrame packed size");
