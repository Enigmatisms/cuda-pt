#include <iostream>
#include <vector>

class CompactNode {
private:
    static constexpr uint32_t LOW_5_MASK = 0x1F;           // 000...00011111
    static constexpr uint32_t HIGH_27_MASK = 0xFFFFFFE0;  // 111...11100000
    static constexpr uint32_t LOW_27_MASK = 0x07FFFFFF;  // 00000111...1111
    static constexpr int HIGH_SHIFT = 5;
    uint32_t data;
public:

    CompactNode(int high, uint32_t low) {
        set_high_27bits(high);
        set_low_5bits(low);
    }
    CompactNode(): data(0) {}

    // set high 27 bits (signed)
    void set_high_27bits(int val) {
        // clear the high 27 bits
        data &= LOW_5_MASK;

        // store as uint32
        uint32_t unsigned_val = static_cast<uint32_t>(val) & LOW_27_MASK; // 27 bits
        data |= (unsigned_val << HIGH_SHIFT);
    }

    void set_low_5bits(int val) {
        // clear low 5 bits
        data &= HIGH_27_MASK;
        data |= (val & LOW_5_MASK);
    }
    // signed 27 bits (upper bound: ~68M)
    int get_gmem_index() const noexcept {
        uint32_t high = (data >> HIGH_SHIFT) & 0x07FFFFFF;
        // place the sign bit at bit 31, then perform an arithmetic right shift for sign-extend
        return (static_cast<int32_t>(high << 5)) >> 5;
    }

    // signed 27 bits (upper bound: ~68M)
    int get_beg_idx() const noexcept {
         uint32_t high = (data >> HIGH_SHIFT) & 0x07FFFFFF;

        // place the sign bit at bit 31, then perform an arithmetic right shift for sign-extend
        return (static_cast<int32_t>(high << 5)) >> 5;
    }

    // unsigned 5 bits (upper bound: 31)
    uint32_t get_cached_offset() const noexcept {
        return data & LOW_5_MASK;
    }

    // unsigned 5 bits (upper bound: 31)
    uint32_t set_prim_cnt() const noexcept {
        return data & LOW_5_MASK;
    }
};

int main() {
    std::vector<int> ints = {1, 0, -100, 65536, -70000, 1234567, -1000000, -1};
    std::vector<uint32_t> uints = {0, 10, 31, 20, 16, 4, 1, 2};

    for (size_t i = 0; i < ints.size(); i++) {
        int v_int = ints[i];
        uint32_t v_uint = uints[i];
        CompactNode node(v_int, v_uint);
        int get_int = node.get_beg_idx();
        uint32_t get_uint = node.set_prim_cnt();
        if (get_int == v_int && get_uint == v_uint) {
            printf("Example %llu checks out.\n", i + 1);
        } else {
            printf("Example %llu mismatches: (%d, %d), (%u, %u)\n", i + 1, get_int, v_int, get_uint, v_uint);
        }
    }
    return 0;
}