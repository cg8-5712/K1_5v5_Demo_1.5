#pragma once

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

class McapWriter {
public:
    McapWriter() = default;
    ~McapWriter();

    bool open(const std::string &path);
    bool isOpen() const { return file_.is_open(); }

    bool writeSchema(uint16_t id, const std::string &name, const std::string &encoding,
                     const std::vector<uint8_t> &data);
    bool writeChannel(uint16_t id, uint16_t schema_id, const std::string &topic,
                      const std::string &message_encoding);
    bool writeMessage(uint16_t channel_id, uint32_t sequence, uint64_t log_time_ns,
                      const void *payload, size_t payload_size);

    void close();

private:
    bool writeRecord(uint8_t opcode, const std::vector<uint8_t> &content);
    static void appendString(std::vector<uint8_t> &buf, const std::string &s);
    static void appendBytes(std::vector<uint8_t> &buf, const void *data, size_t len);
    static void appendU16(std::vector<uint8_t> &buf, uint16_t v);
    static void appendU32(std::vector<uint8_t> &buf, uint32_t v);
    static void appendU64(std::vector<uint8_t> &buf, uint64_t v);

    std::ofstream file_;
    uint32_t message_count_ = 0;
    uint64_t first_log_time_ = 0;
    uint64_t last_log_time_ = 0;
};
