#include "scene.h"

#include "utilities.h"

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

using namespace std;
using json = nlohmann::json;

glm::vec3 compute_center(const Geom &geom) {
    if (geom.type == CUBE || geom.type == SPHERE) {
        return glm::vec3(geom.transform * glm::vec4(0.f, 0.f, 0.f, 1.0f));
    }
    else if(geom.type == TRIANGLE) {
        return (geom.vertices[0] + geom.vertices[1] + geom.vertices[2]) / 3.0f;
    }

    return glm::vec3(0.f);
}

void compute_box(const Geom &geom, glm::vec3 &minAABB, glm::vec3 &maxAABB) {
    minAABB = glm::vec3( std::numeric_limits<float>::max() );
    maxAABB = glm::vec3(-std::numeric_limits<float>::max());

    if (geom.type == CUBE || geom.type == SPHERE) {
        glm::vec3 localMin, localMax;
        localMin = glm::vec3(-0.5f);
        localMax = glm::vec3(+0.5f);

        glm::vec3 corners[8] = {
            {localMin.x, localMin.y, localMin.z},
            {localMin.x, localMin.y, localMax.z},
            {localMin.x, localMax.y, localMin.z},
            {localMin.x, localMax.y, localMax.z},
            {localMax.x, localMin.y, localMin.z},
            {localMax.x, localMin.y, localMax.z},
            {localMax.x, localMax.y, localMin.z},
            {localMax.x, localMax.y, localMax.z}
        };


        for (int i = 0; i < 8; i++) {
            glm::vec3 worldPt = glm::vec3(geom.transform * glm::vec4(corners[i], 1.0f));
            minAABB = glm::min(minAABB, worldPt);
            maxAABB = glm::max(maxAABB, worldPt);
        }
    }
    else if(geom.type == TRIANGLE) {
        for(int i = 0; i < 3; i++){
            glm::vec3 worldPt = glm::vec3(geom.transform * glm::vec4(geom.vertices[i], 1.0f));
            minAABB = glm::min(minAABB, worldPt);
            maxAABB = glm::max(maxAABB, worldPt);
        }
    }
}

#ifdef BVH_NUM

int Scene::buildBVHNode(int st, int ed) {
    BVHNode node;
    int node_idx = bvh.size();
    bvh.push_back(node);

    // bounding box
    node.minCorner = glm::vec3( std::numeric_limits<float>::max() );
    node.maxCorner = glm::vec3(-std::numeric_limits<float>::max());

    for(int i = st; i < ed; i++) {
        glm::vec3 minAABB, maxAABB;
        compute_box(geoms[i], minAABB, maxAABB);
        node.minCorner = glm::min(node.minCorner, minAABB);
        node.maxCorner = glm::max(node.maxCorner, maxAABB);
    }

    node.geo_st = st;
    node.geo_ed = ed;

    if(ed - st > BVH_NUM) {
        node.isLeaf = false;

        glm::vec3 diagonal = node.maxCorner - node.minCorner;
        int axis;
        if(diagonal[2] > diagonal[1] && diagonal[2] > diagonal[0]){
            axis = 2;
        }
        else if(diagonal[1] > diagonal[0]){
            axis = 1;
        }
        else {
            axis = 0;
        }

        auto cmpFunc = [axis](const Geom& a, const Geom& b) {
            glm::vec3 center_a = compute_center(a);
            glm::vec3 center_b = compute_center(b);
            return center_a[axis] < center_b[axis];
        };

        std::sort(geoms.begin() + st, geoms.begin() + ed, cmpFunc);

        int mid = (st + ed) >> 1;

        node.leftNode = buildBVHNode(st, mid);
        node.rightNode = buildBVHNode(mid, ed);
    }
    else{
        node.isLeaf = true;
        node.leftNode = node.rightNode = -1;
    }

    bvh[node_idx] = node;
    return node_idx;
}

void Scene::buildBVH() {
    bvh.clear();
    buildBVHNode(0, geoms.size());
}

#endif


Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            float roughness = p["ROUGHNESS"];
            newMaterial.hasReflective = 1.0 - p["ROUGHNESS"];
        }
        else if (p["TYPE"] == "Refraction")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasRefractive = 1.0;
            // newMaterial.hasReflective = 1.0;
            newMaterial.indexOfRefraction = p["IOR"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        // transform 
        Geom newGeom;
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        // geometry
        const auto& type = p["TYPE"];
        if (type == "cube")
        {
            newGeom.type = CUBE;
            geoms.push_back(newGeom);
        }
        else if (type == "sphere")
        {
            newGeom.type = SPHERE;
            geoms.push_back(newGeom);
        }
        else if (type == "obj_file")
        {
            newGeom.type = TRIANGLE;

            // load obj
            std::string filename = p["FILE"];
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> materials;

            std::string warn;
            std::string err;

            bool ret = tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, filename.c_str());

            if (!warn.empty()) {
                std::cout << warn << std::endl;
            }

            if (!err.empty()) {
                std::cerr << err << std::endl;
            }

            if (!ret) {
                std::cout << "faile to load obj" << std::endl;
                exit(1);
            }

            for (size_t s = 0; s < shapes.size(); s++) {
                size_t index_offset = 0;
                for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
                    size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

                    if(fv != 3){
                        printf("not triangle\n");
                    }
                    // Loop over vertices in the face.
                    for (size_t v = 0; v < fv; v++) {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                        tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                        tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                        tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                        newGeom.vertices[v] = glm::vec3(vx, vy, vz);
                    }
                    geoms.push_back(newGeom);
                    index_offset += fv;
                }
            }

        }
    }

    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.aperture = cameraData["APERTURE"];
    camera.focus_dis = cameraData["FOCUS_DIS"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    #ifdef BVH_NUM

    buildBVH();

    #endif
}
