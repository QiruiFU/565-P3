#include "intersections.h"
#include <glm/gtx/intersect.hpp>

__host__ __device__ bool BVHIntersectionTest(
    BVHNode bvh,
    Ray r
)
{ 
    float tmin = -1e38f;
    float tmax = 1e38f;

    for (int i = 0; i < 3; ++i)
    {
        float invD = 1.0f / r.direction[i];
        float t0 = (bvh.minCorner[i] - r.origin[i]) * invD;
        float t1 = (bvh.maxCorner[i] - r.origin[i]) * invD;

        if (invD < 0.0f){
            float temp = t0;
            t0 = t1;
            t1 = temp;
        }

        tmin = max(tmin, t0);
        tmax = min(tmax, t1);

        if (tmin > tmax)
            return false;
    }

    return true;
}

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(q.origin + q.direction * tmin, 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        if (tmin <= 0){
            normal = -normal;
        }
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = rt.origin + rt.direction * t;

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float triIntersectionTest(
    Geom triangle,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    glm::vec3 ro = multiplyMV(triangle.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(triangle.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    glm::vec3 bary;
    bool inter_valid = glm::intersectRayTriangle(
        ro, rd,
        triangle.vertices[0],
        triangle.vertices[1],
        triangle.vertices[2],
        bary);

    if (!inter_valid)
    {
        return -1.0f;
    }

    float t = bary.z;
    glm::vec3 objIntersection = rt.origin + rt.direction * t;

    glm::vec3 e1 = triangle.vertices[1] - triangle.vertices[0];
    glm::vec3 e2 = triangle.vertices[2] - triangle.vertices[0];
    glm::vec3 objNormal = glm::normalize(glm::cross(e1, e2));

    intersectionPoint = multiplyMV(triangle.transform, glm::vec4(objIntersection, 1.0f));
    normal = glm::normalize(multiplyMV(triangle.invTranspose, glm::vec4(objNormal, 0.0f)));

    outside = glm::dot(r.direction, normal) < 0.0f;
    // if (!outside)
    // {
    //     normal = -normal;
    // }

    return glm::length(r.origin - intersectionPoint);
}
