/**
 * @file phase_registry.cuh
 * @author Qianyue He
 * @brief Includes all the derived class of BSDF
 * @date 2025-02-09
 * @copyright Copyright (c) 2025
 */

#pragma once

#include "volume/henyey_greenstein.cuh"
#include "volume/rayleigh.cuh"
#include "volume/sggx.cuh"

class PhaseFunctionWrapper {
private:
    std::string type;
    std::unique_ptr<PhaseFunction> ptr;
public:
    PhaseFunctionWrapper(
        std::string _t, 
        std::unique_ptr<PhaseFunction> _ptr
    ): type(_t), ptr(std::move(_ptr)) {}

    template <typename PhaseType>
    PhaseType deref() const {
        static_assert(std::is_base_of_v<PhaseFunction, PhaseType>,
                      "PhaseType must be derived from PhaseFunction");
        auto output = dynamic_cast<PhaseType*>(ptr.get());
        if (output) {
            return *output;
        } else {
            std::cerr << "Dynamic cast failed for phase type '" << id << "'\n";
            throw std::runtime_error("Dynamic cast failed.");
        }
    }
};
