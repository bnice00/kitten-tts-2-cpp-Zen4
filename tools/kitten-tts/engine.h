#pragma once
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>
namespace kitten {
using args_map = std::map<std::string, std::string>;
using chunk_callback = std::function<bool(const std::vector<float> &)>;
// Persistent model: loads once, then runs many generate calls. Options use the CLI flag names.
class engine {
    struct impl;
    std::unique_ptr<impl> p;
public:
    explicit engine(const args_map & args);
    ~engine();
    std::string metadata() const;
    std::string normalize(const std::string & text) const;
    // on_chunk gets the raw decoder audio of each chunk; return false to stop after that chunk
    std::vector<float> generate(const args_map & args, std::string & report, const chunk_callback & on_chunk = {});
};
}
