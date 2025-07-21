#include "FrustumCulling.h"
#include <glm/gtc/matrix_access.hpp>

void ExtractFrustumPlanes(const glm::mat4& m, std::array<FrustumPlane, 6>& planes) {
    planes[0].eq = glm::vec4(
        m[0][3] + m[0][0],
        m[1][3] + m[1][0],
        m[2][3] + m[2][0],
        m[3][3] + m[3][0]
    );
    planes[1].eq = glm::vec4(
        m[0][3] - m[0][0],
        m[1][3] - m[1][0],
        m[2][3] - m[2][0],
        m[3][3] - m[3][0]
    );
    planes[2].eq = glm::vec4(
        m[0][3] + m[0][1],
        m[1][3] + m[1][1],
        m[2][3] + m[2][1],
        m[3][3] + m[3][1]
    );
    planes[3].eq = glm::vec4(
        m[0][3] - m[0][1],
        m[1][3] - m[1][1],
        m[2][3] - m[2][1],
        m[3][3] - m[3][1]
    );
    planes[4].eq = glm::vec4(
        m[0][3] + m[0][2],
        m[1][3] + m[1][2],
        m[2][3] + m[2][2],
        m[3][3] + m[3][2]
    );
    planes[5].eq = glm::vec4(
        m[0][3] - m[0][2],
        m[1][3] - m[1][2],
        m[2][3] - m[2][2],
        m[3][3] - m[3][2]
    );
    for (auto& p : planes) {
        float len = glm::length(glm::vec3(p.eq));
        if (len > 0.0f) p.eq /= len;
    }
}

bool SphereInFrustum(const std::array<FrustumPlane, 6>& planes, const glm::vec3& pos, float radius) {
    for (const auto& p : planes) {
        float d = glm::dot(glm::vec3(p.eq), pos) + p.eq.w;
        if (d < -radius) return false;
    }
    return true;
}
