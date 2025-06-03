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
 * @brief Process Geometries Impl
 * @author Qianyue He
 * @date 2025.6.3
 */
#include "core/proc_geometry.cuh"

CPT_CPU inline bool is_inside(const Vec3 &point, int axis, float boundary_val,
                              bool is_min_boundary) {
    // for min face, keep the greater side, otherwise keep the less side
    return is_min_boundary ? point[axis] >= boundary_val
                           : point[axis] <= boundary_val;
}

// calculate segment-face intersection point
CPT_CPU static Vec3 line_face_intersect(const Vec3 &p1, const Vec3 &p2,
                                        int axis, float boundary_val) {
    // this function is called only if two input points are on the different
    // side of the clipping plane, therefore d will never be 0.
    float d = p2[axis] - p1[axis];
    if (d == 0)
        return p1;

    float t = (boundary_val - p1[axis]) / d;
    Vec3 intersection = p1.advance(p2 - p1, t);
    intersection[axis] = boundary_val; // exact value
    return intersection;
}

// general polygon clipping with only one clipping plane
CPT_CPU std::vector<Vec3> clip_polygon(std::vector<Vec3> &&polygon, int axis,
                                       float boundary_val,
                                       bool is_min_boundary) {
    if (polygon.empty())
        return {};

    std::vector<Vec3> result;
    Vec3 start = polygon.back(); // start with the last vertex

    for (const Vec3 &end : polygon) {
        bool start_inside =
                 is_inside(start, axis, boundary_val, is_min_boundary),
             end_inside = is_inside(end, axis, boundary_val, is_min_boundary);

        if (end_inside) {
            if (!start_inside) {
                // outside to inside: add one intersection vertex
                result.push_back(
                    line_face_intersect(start, end, axis, boundary_val));
            }
            // inside to inside: add vertex directly
            result.push_back(end);
        } else if (start_inside) {
            // inside to outside: add one intersection vertex
            result.push_back(
                line_face_intersect(start, end, axis, boundary_val));
        }
        // outside to outside: skip
        start = end;
    }

    return result;
}

CPT_CPU std::vector<Vec3> aabb_triangle_clipping(const AABB &aabb,
                                                 std::vector<Vec3> &&polygon) {
    bool skip_clip = true;
    for (const Vec3 &vertex : polygon) {
        if (!aabb.covers(vertex, 1e-6f)) {
            skip_clip = false;
            break;
        }
    }
    if (skip_clip)
        return polygon;
    // clipping sequence
    constexpr int clip_seqs[6][2] = {
        {0, 0}, // x-min (axis=0, isMin=true)
        {0, 1}, // x-max (axis=0, isMin=false)
        {1, 0}, // y-min
        {1, 1}, // y-max
        {2, 0}, // z-min
        {2, 1}  // z-max
    };

    for (const auto &clip : clip_seqs) {
        int axis = clip[0];
        bool is_min_boundary = (clip[1] == 0);
        float boundary_val =
            is_min_boundary ? aabb.mini[axis] : aabb.maxi[axis];

        polygon = clip_polygon(std::move(polygon), axis, boundary_val,
                               is_min_boundary);
        if (polygon.size() < 3) {
            return {};
        }
    }

    return polygon;
}

CPT_CPU std::vector<Vec3> aabb_triangle_clipping(const AABB &aabb,
                                                 const Vec3 &p1, const Vec3 &p2,
                                                 const Vec3 &p3) {
    return aabb_triangle_clipping(aabb, {p1, p2, p3});
}
