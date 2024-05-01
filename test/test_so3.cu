#include "core/vec3.cuh"
#include "core/so3.cuh"

int main() {
    Vec3 v1(0.6160316 , 0.71857749, 0.32272506), v2(0.50171796, 0.86103955, 0.08300593);
    auto res = rotation_between(v1, v2);

    print_vec3(res[0]);
    print_vec3(res[1]);
    print_vec3(res[2]);

    return 0;
}