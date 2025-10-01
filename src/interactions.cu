#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    #ifdef COS_HEMISPHERE
    
    float t = u01(rng);
    float u = u01(rng) * TWO_PI;
    float v = sqrt(1.f - t);

    float px = v * cos(u);
    float py = sqrt(t);
    float pz = v * sin(u);

    #else

    float t = u01(rng);
    float u = u01(rng) * TWO_PI;
    float v = sqrt(1.f - t * t);

    float px = v * cos(u);
    float py = t;
    float pz = v * sin(u);

    #endif

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return py * normal
        + px * perpendicularDirection1
        + pz * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    thrust::default_random_engine &rng)
{
    pathSegment.ray.origin = intersect;
    glm::vec3 direct;
    normal = glm::normalize(normal);
    if(m.hasReflective > 0.0) {
        pathSegment.color *= m.color;
        direct = glm::reflect(pathSegment.ray.direction, normal);
    }
    else if(m.hasRefractive == 1.0) {
        pathSegment.ray.origin += 0.0002f * glm::normalize(pathSegment.ray.direction);
        float n1, n2;

        float cos_theta = glm::dot(glm::normalize(pathSegment.ray.direction), normal);
        pathSegment.color *= m.color;

        if(cos_theta < 0.0){
            n1 = 1.0;
            n2 = m.indexOfRefraction;
            cos_theta = -cos_theta;
        }
        else {
            n1 = m.indexOfRefraction;
            n2 = 1.0;
            normal = -normal;
        }

        float F = (n1 - n2) / (n1 + n2);
        F = F * F;
        F = F + (1.0 - F) * powf(1.0 - cos_theta, 5.0);

        thrust::uniform_real_distribution<float> u01(0, 1);
        float judge = u01(rng);
        if(judge > F) { // refraction
            direct = glm::refract(glm::normalize(pathSegment.ray.direction), normal, n1 / n2);
        }
        if(judge <= F || glm::length(direct) < 1e-8f) { // reflect
            direct = glm::reflect(pathSegment.ray.direction, normal);
        }
        // pathSegment.color *= glm::vec3(1.0, 1.0, 1.0);
        // direct = pathSegment.ray.direction;
    }
    else{
        #ifdef COS_HEMISPHERE
        pathSegment.color *= m.color;
        #else
        pathSegment.color *= m.color * glm::dot(glm::normalize(-pathSegment.ray.direction), normal);
        #endif

        direct = calculateRandomDirectionInHemisphere(normal, rng);
    }
    pathSegment.ray.direction = glm::normalize(direct);
}
