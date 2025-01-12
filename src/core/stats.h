/**
*   Timer utility
    @date 2023.12.27
    @author: Qianyue He

    Adapted from my repo: https://github.com/Enigmatisms/scds-pb.git
 */

#pragma once

#include <chrono>
#include <string>

class TicTocLocal {
private:
    std::chrono::system_clock::time_point tp;
public:
    TicTocLocal(): tp(std::chrono::system_clock::now()) {}
    void tic() {
        tp = std::chrono::system_clock::now();
    }
    float toc() const noexcept {
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        return static_cast<float>(count) / (1e3f);
    }
};

class TicToc {
private:
    std::string name;
    int num;
    TicTocLocal tictoc;
public:
    TicToc(std::string name, int n = 1) : name(name), num(n) {}
    void tic() {
        tictoc.tic();
    }
    ~TicToc() {
        float elapsed = tictoc.toc();
        printf("`%s` takes time: %.3lf ms per iteration (%d its)\n", name.c_str(), elapsed / float(num), num);
    }
};