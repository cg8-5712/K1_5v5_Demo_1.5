#include "mcap_writer.h"

#include <cstring>

namespace {

constexpr uint8_t kMagic[8] = {0x89, 'M', 'C', 'A', 'P', '0', '\r', '\n'};
constexpr uint8_t kOpcodeHeader = 0x01;
constexpr uint8_t kOpcodeFooter = 0x02;
constexpr uint8_t kOpcodeSchema = 0x03;
constexpr uint8_t kOpcodeChannel = 0x04;
constexpr uint8_t kOpcodeMessage = 0x05;

} // namespace

McapWriter::~McapWriter()
{
    close();
}

bool McapWriter::open(const std::string &path)
{
    close();
    file_.open(path, std::ios::binary | std::ios::trunc);
    if (!file_) {
        return false;
    }
    file_.write(reinterpret_cast<const char *>(kMagic), sizeof(kMagic));

    std::vector<uint8_t> header;
    appendString(header, "");
    appendString(header, "k1_brain TrainingLogger mcap_writer/1");
    return writeRecord(kOpcodeHeader, header);
}

void McapWriter::close()
{
    if (!file_.is_open()) {
        return;
    }

    std::vector<uint8_t> footer;
    appendU64(footer, 0);
    appendU64(footer, 0);
    appendU32(footer, 0);
    writeRecord(kOpcodeFooter, footer);

    file_.write(reinterpret_cast<const char *>(kMagic), sizeof(kMagic));
    file_.close();
    message_count_ = 0;
    first_log_time_ = 0;
    last_log_time_ = 0;
}

bool McapWriter::writeSchema(uint16_t id, const std::string &name, const std::string &encoding,
                             const std::vector<uint8_t> &data)
{
    std::vector<uint8_t> body;
    appendU16(body, id);
    appendString(body, name);
    appendString(body, encoding);
    appendU32(body, static_cast<uint32_t>(data.size()));
    appendBytes(body, data.data(), data.size());
    return writeRecord(kOpcodeSchema, body);
}

bool McapWriter::writeChannel(uint16_t id, uint16_t schema_id, const std::string &topic,
                              const std::string &message_encoding)
{
    std::vector<uint8_t> body;
    appendU16(body, id);
    appendU16(body, schema_id);
    appendString(body, topic);
    appendString(body, message_encoding);
    appendU32(body, 0);
    return writeRecord(kOpcodeChannel, body);
}

bool McapWriter::writeMessage(uint16_t channel_id, uint32_t sequence, uint64_t log_time_ns,
                              const void *payload, size_t payload_size)
{
    std::vector<uint8_t> body;
    appendU16(body, channel_id);
    appendU32(body, sequence);
    appendU64(body, log_time_ns);
    appendU64(body, log_time_ns);
    appendBytes(body, payload, payload_size);

    if (!writeRecord(kOpcodeMessage, body)) {
        return false;
    }
    ++message_count_;
    if (first_log_time_ == 0 || log_time_ns < first_log_time_) {
        first_log_time_ = log_time_ns;
    }
    if (log_time_ns > last_log_time_) {
        last_log_time_ = log_time_ns;
    }
    return true;
}

bool McapWriter::writeRecord(uint8_t opcode, const std::vector<uint8_t> &content)
{
    if (!file_) {
        return false;
    }
    const uint64_t len = content.size();
    file_.put(static_cast<char>(opcode));
    file_.write(reinterpret_cast<const char *>(&len), sizeof(len));
    if (!content.empty()) {
        file_.write(reinterpret_cast<const char *>(content.data()),
                    static_cast<std::streamsize>(content.size()));
    }
    return static_cast<bool>(file_);
}

void McapWriter::appendString(std::vector<uint8_t> &buf, const std::string &s)
{
    const uint32_t n = static_cast<uint32_t>(s.size());
    appendU32(buf, n);
    if (n > 0) {
        appendBytes(buf, s.data(), s.size());
    }
}

void McapWriter::appendBytes(std::vector<uint8_t> &buf, const void *data, size_t len)
{
    if (len == 0) {
        return;
    }
    const auto *p = static_cast<const uint8_t *>(data);
    buf.insert(buf.end(), p, p + len);
}

void McapWriter::appendU16(std::vector<uint8_t> &buf, uint16_t v)
{
    buf.push_back(static_cast<uint8_t>(v & 0xff));
    buf.push_back(static_cast<uint8_t>((v >> 8) & 0xff));
}

void McapWriter::appendU32(std::vector<uint8_t> &buf, uint32_t v)
{
    for (int i = 0; i < 4; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
}

void McapWriter::appendU64(std::vector<uint8_t> &buf, uint64_t v)
{
    for (int i = 0; i < 8; ++i) {
        buf.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xff));
    }
}
