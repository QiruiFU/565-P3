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
        // TODO: handle materials loading differently
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

                    assert(fv == 3);
                    // Loop over vertices in the face.
                    for (size_t v = 0; v < fv; v++) {
                        // access to vertex
                        tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];

                        tinyobj::real_t vx = attrib.vertices[3*size_t(idx.vertex_index)+0];
                        tinyobj::real_t vy = attrib.vertices[3*size_t(idx.vertex_index)+1];
                        tinyobj::real_t vz = attrib.vertices[3*size_t(idx.vertex_index)+2];

                        // glm::vec4 vertex_pos(vx, vy, vz, 1.0f);
                        // vertex_pos = newGeom.transform * vertex_pos;

                        // new_tri.vertices[v] = glm::vec3(vertex_pos[0], vertex_pos[1], vertex_pos[2]);
                        newGeom.vertices[v] = glm::vec3(vx, vy, vz);

                        // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                        // if (idx.texcoord_index >= 0) {
                        //     tinyobj::real_t tx = attrib.texcoords[2*size_t(idx.texcoord_index)+0];
                        //     tinyobj::real_t ty = attrib.texcoords[2*size_t(idx.texcoord_index)+1];
                        // }
                        geoms.push_back(newGeom);
                    }
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
}
