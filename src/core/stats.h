/**
*   Timer utility
    @date 2023.12.27
    @author: Qianyue He

    Adapted from my repo: https://github.com/Enigmatisms/scds-pb.git
 */

#pragma once

#include <chrono>
#include <string>

class TicToc {
private:
    std::string name;
    int num;
    std::chrono::system_clock::time_point tp;
public:
    TicToc(std::string name, int n = 1) : name(name), num(n), tp(std::chrono::system_clock::now()) {}
    void tic() {
        tp = std::chrono::system_clock::now();
    }
    ~TicToc() {
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        auto elapsed = static_cast<double>(count) / (1e3 * num);
        printf("`%s` takes time: %.3lf ms per iteration (%d its)\n", name.c_str(), elapsed, num);
    }
};