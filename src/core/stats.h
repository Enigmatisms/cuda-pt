
/*
    I found this code pretty useful when working on PBRT-v3
    So I adopted it here and made some slight modification
    @date 2023.12.27
    @author: Matt Pharr / Qianyue He

    Adapted from my repo: https://github.com/Enigmatisms/scds-pb.git

    original license strings:

    pbrt source code is Copyright(c) 1998-2016
                        Matt Pharr, Greg Humphreys, and Wenzel Jakob.

    This file is part of pbrt.

    Redistribution and use in source and binary forms, with or without
    modification, are permitted provided that the following conditions are
    met:

    - Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    - Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
    IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
    PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
    HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
    SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
    LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
    DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
    THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
    (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
    OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#pragma once

#include <map>
#include <cfloat>
#include <chrono>
#include <string>
#include <functional>
#include <mutex>

class TicToc {
private:
    std::string name;
    int num;
    std::chrono::system_clock::time_point tp;
public:
    TicToc(std::string name, int n = 1): name(name), num(n), tp(std::chrono::system_clock::now()) {}

    ~TicToc() {
        auto dur = std::chrono::system_clock::now() - tp;
        auto count = std::chrono::duration_cast<std::chrono::microseconds>(dur).count();
        auto elapsed = static_cast<double>(count) / (1e3 * num);
        printf("`%s` takes time: %lf.3 ms\n", name.c_str(), elapsed);
    }
};

// Statistics Declarations
class StatsAccumulator;
class StatRegisterer {
  public:
    // StatRegisterer Public Methods
    StatRegisterer(std::function<void(StatsAccumulator &)> func) {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!funcs)
            funcs = new std::vector<std::function<void(StatsAccumulator &)>>;
        funcs->push_back(func);
    }
    static void CallCallbacks(StatsAccumulator &accum);

  private:
    // StatRegisterer Private Data
    static std::vector<std::function<void(StatsAccumulator &)>> *funcs;
};

void PrintStats(FILE *dest);
void ClearStats();
void ReportThreadStats();

class StatsAccumulator {
  public:
    // StatsAccumulator Public Methods
    void ReportCounter(const std::string &name, int64_t val) {
        counters[name] += val;
    }
    void ReportMemoryCounter(const std::string &name, int64_t val) {
        memoryCounters[name] += val;
    }
    void ReportIntDistribution(const std::string &name, int64_t sum,
                               int64_t count, int64_t min, int64_t max) {
        intDistributionSums[name] += sum;
        intDistributionCounts[name] += count;
        if (intDistributionMins.find(name) == intDistributionMins.end())
            intDistributionMins[name] = min;
        else
            intDistributionMins[name] =
                std::min(intDistributionMins[name], min);
        if (intDistributionMaxs.find(name) == intDistributionMaxs.end())
            intDistributionMaxs[name] = max;
        else
            intDistributionMaxs[name] =
                std::max(intDistributionMaxs[name], max);
    }
    void ReportFloatDistribution(const std::string &name, double sum,
                                 int64_t count, double min, double max) {
        floatDistributionSums[name] += sum;
        floatDistributionCounts[name] += count;
        if (floatDistributionMins.find(name) == floatDistributionMins.end())
            floatDistributionMins[name] = min;
        else
            floatDistributionMins[name] =
                std::min(floatDistributionMins[name], min);
        if (floatDistributionMaxs.find(name) == floatDistributionMaxs.end())
            floatDistributionMaxs[name] = max;
        else
            floatDistributionMaxs[name] =
                std::max(floatDistributionMaxs[name], max);
    }
    void ReportPercentage(const std::string &name, int64_t num, int64_t denom) {
        percentages[name].first += num;
        percentages[name].second += denom;
    }
    void ReportRatio(const std::string &name, int64_t num, int64_t denom) {
        ratios[name].first += num;
        ratios[name].second += denom;
    }

    void Print(FILE *file);
    void Clear();

  private:
    // StatsAccumulator Private Data
    std::map<std::string, int64_t> counters;
    std::map<std::string, int64_t> memoryCounters;
    std::map<std::string, int64_t> intDistributionSums;
    std::map<std::string, int64_t> intDistributionCounts;
    std::map<std::string, int64_t> intDistributionMins;
    std::map<std::string, int64_t> intDistributionMaxs;
    std::map<std::string, double> floatDistributionSums;
    std::map<std::string, int64_t> floatDistributionCounts;
    std::map<std::string, double> floatDistributionMins;
    std::map<std::string, double> floatDistributionMaxs;
    std::map<std::string, std::pair<int64_t, int64_t>> percentages;
    std::map<std::string, std::pair<int64_t, int64_t>> ratios;
};

enum class Prof {
    StartUp,
    DepthRenderingHost,
    DepthRenderingDevice,
    NumProfCategories
};

static_assert((int)Prof::NumProfCategories <= 64,
              "No more than 64 profiling categories may be defined.");

inline uint64_t ProfToBits(Prof p) { return 1ull << (int)p; }

static const char *ProfNames[] = {
    "Program starting up",
    "render_depth()",
    "render_depth_kernel()"
};

static_assert((int)Prof::NumProfCategories ==
                  sizeof(ProfNames) / sizeof(ProfNames[0]),
              "ProfNames[] array and Prof enumerant have different "
              "numbers of entries!");

extern thread_local uint64_t ProfilerState;
inline uint64_t CurrentProfilerState() { return ProfilerState; }

class ProfilePhase {
  public:
    // ProfilePhase Public Methods
    ProfilePhase(Prof p) {
        categoryBit = ProfToBits(p);
        reset = (ProfilerState & categoryBit) == 0;
        ProfilerState |= categoryBit;
    }
    ~ProfilePhase() {
        if (reset) ProfilerState &= ~categoryBit;
    }
    ProfilePhase(const ProfilePhase &) = delete;
    ProfilePhase &operator=(const ProfilePhase &) = delete;

  private:
    // ProfilePhase Private Data
    bool reset;
    uint64_t categoryBit;
};

void InitProfiler();
void SuspendProfiler();
void ResumeProfiler();
void ProfilerWorkerThreadInit();
void ReportProfilerResults(FILE *dest);
void ClearProfiler();
void CleanupProfiler();

// Statistics Macros
#define STAT_COUNTER(title, var)                           \
    static thread_local int64_t var;                  \
    static void STATS_FUNC##var(StatsAccumulator &accum) { \
        accum.ReportCounter(title, var);                   \
        var = 0;                                           \
    }                                                      \
    static StatRegisterer STATS_REG##var(STATS_FUNC##var)
#define STAT_MEMORY_COUNTER(title, var)                    \
    static thread_local int64_t var;                  \
    static void STATS_FUNC##var(StatsAccumulator &accum) { \
        accum.ReportMemoryCounter(title, var);             \
        var = 0;                                           \
    }                                                      \
    static StatRegisterer STATS_REG##var(STATS_FUNC##var)

#define STATS_INT64_T_MIN std::numeric_limits<int64_t>::max()
#define STATS_INT64_T_MAX std::numeric_limits<int64_t>::lowest()
#define STATS_DBL_T_MIN std::numeric_limits<double>::max()
#define STATS_DBL_T_MAX std::numeric_limits<double>::lowest()

#define STAT_INT_DISTRIBUTION(title, var)                                  \
    static thread_local int64_t var##sum;                             \
    static thread_local int64_t var##count;                           \
    static thread_local int64_t var##min = (STATS_INT64_T_MIN);       \
    static thread_local int64_t var##max = (STATS_INT64_T_MAX);       \
    static void STATS_FUNC##var(StatsAccumulator &accum) {                 \
        accum.ReportIntDistribution(title, var##sum, var##count, var##min, \
                                    var##max);                             \
        var##sum = 0;                                                      \
        var##count = 0;                                                    \
        var##min = std::numeric_limits<int64_t>::max();                    \
        var##max = std::numeric_limits<int64_t>::lowest();                 \
    }                                                                      \
    static StatRegisterer STATS_REG##var(STATS_FUNC##var)

#define STAT_FLOAT_DISTRIBUTION(title, var)                                  \
    static thread_local double var##sum;                                \
    static thread_local int64_t var##count;                             \
    static thread_local double var##min = (STATS_DBL_T_MIN);            \
    static thread_local double var##max = (STATS_DBL_T_MAX);            \
    static void STATS_FUNC##var(StatsAccumulator &accum) {                   \
        accum.ReportFloatDistribution(title, var##sum, var##count, var##min, \
                                      var##max);                             \
        var##sum = 0;                                                        \
        var##count = 0;                                                      \
        var##min = std::numeric_limits<double>::max();                       \
        var##max = std::numeric_limits<double>::lowest();                    \
    }                                                                        \
    static StatRegisterer STATS_REG##var(STATS_FUNC##var)

#define ReportValue(var, value)                                   \
    do {                                                          \
        var##sum += value;                                        \
        var##count += 1;                                          \
        var##min = std::min(var##min, decltype(var##min)(value)); \
        var##max = std::max(var##max, decltype(var##min)(value)); \
    } while (0)

#define STAT_PERCENT(title, numVar, denomVar)                 \
    static thread_local int64_t numVar, denomVar;        \
    static void STATS_FUNC##numVar(StatsAccumulator &accum) { \
        accum.ReportPercentage(title, numVar, denomVar);      \
        numVar = denomVar = 0;                                \
    }                                                         \
    static StatRegisterer STATS_REG##numVar(STATS_FUNC##numVar)

#define STAT_RATIO(title, numVar, denomVar)                   \
    static thread_local int64_t numVar, denomVar;        \
    static void STATS_FUNC##numVar(StatsAccumulator &accum) { \
        accum.ReportRatio(title, numVar, denomVar);           \
        numVar = denomVar = 0;                                \
    }                                                         \
    static StatRegisterer STATS_REG##numVar(STATS_FUNC##numVar)
    