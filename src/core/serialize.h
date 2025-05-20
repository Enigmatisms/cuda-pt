// Copyright (C) 2025 Qianyue He
//
// This program is free software: you can redistribute it and/or
// modify it under the terms of the GNU Affero General Public License
// as published by the Free Software Foundation, either
// version 3 of the License, or (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See
// the GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General
// Public License along with this program. If not, see
//
//             <https://www.gnu.org/licenses/>.

/**
 * @file serialize.h
 * @author Qianyue He
 * @brief Simple serialization & de-serialization utility functions
 * for a std::vector<char> byte buffer
 * @date 2025-01-07
 * @copyright Copyright (c) 2025
 */

#include <stdexcept>
#include <vector>

class Serializer {
    using ByteBuffer = std::vector<char>;

  public:
    template <typename T> static T get(const ByteBuffer &bbf, size_t index) {
        size_t byte_offset = index * sizeof(T);
        if (byte_offset + sizeof(T) > bbf.size()) {
            throw std::runtime_error("Index into the byte buffer out-of-range");
        }
        return *reinterpret_cast<const T *>(bbf.data() + byte_offset);
    }

    template <typename T>
    static void set(ByteBuffer &bbf, size_t index, T val) {
        size_t byte_offset = index * sizeof(T);
        if (byte_offset + sizeof(T) > bbf.size()) {
            throw std::runtime_error("Index into the byte buffer out-of-range");
        }
        *reinterpret_cast<T *>(bbf.data() + byte_offset) = val;
    }

    template <typename T> static void push(ByteBuffer &bbf, T val) {
        size_t old_size = bbf.size();
        for (size_t i = 0; i < sizeof(T); i++) {
            bbf.push_back(0x00);
        }
        *reinterpret_cast<T *>(bbf.data() + old_size) = val;
    }

    template <typename T> static T pop(ByteBuffer &bbf) {
        if (bbf.size() < sizeof(T)) {
            throw std::runtime_error(
                "Invalid byte buffer popping: insufficient size.");
        }
        size_t new_size = bbf.size() - sizeof(T);
        T val = *reinterpret_cast<const T *>(bbf.data() + new_size);
        bbf.resize(new_size);
        return val;
    }
};
