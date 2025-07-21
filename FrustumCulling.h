#pragma once
#include <glm/glm.hpp>
#include <array>

struct FrustumPlane {
    glm::vec4 eq;
};

extern "C" {
void ExtractFrustumPlanes(const glm::mat4& m, std::array<FrustumPlane, 6>& planes);
}

bool SphereInFrustum(const std::array<FrustumPlane, 6>& planes, const glm::vec3& pos, float radius);
