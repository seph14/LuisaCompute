//
// Repro: TLAS instances bound to the wrong BLAS on DX12.
//
// Mirrors the engine test scene (NewTypeEngine src/tests/HitBug.cpp): two
// 2-triangle quads with IDENTICAL buffer sizes emplaced into one Accel:
//   entry 0: 25x25 plane (4 verts, 2 triangles), origin, identity
//   entry 1: 36x18 plane (4 verts, 2 triangles), translate (0, 25, -16)
// In the engine, rays that hit the second (textured) quad return triangle
// geometry from the first quad's BLAS: the backdrop renders with two giant
// smeared triangles instead of its checker texture.
//
// With both meshes at 2 triangles a raw prim range cannot distinguish the
// BLASes, so the probe kernel also records the hit barycentric coordinate
// and the CPU verification reconstructs the hit point from the REPORTED
// instance's own vertex buffer. The reconstruction must land on the probe
// ray; if the traversal intersected the other mesh's geometry, it lands
// several units away. Probes inside entry 1's footprint but outside entry
// 0's translated extent must hit at all.
//
// Exit code 0 = pass, 1 = bug reproduced / verification failed.
//
// Run: example_tlas_instance_binding dx
//

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <algorithm>
#include <cmath>
#include <fstream>

using namespace luisa;
using namespace luisa::compute;

namespace {

// Generates a grid plane in the XY plane at z = 0, wound clockwise when
// viewed from +z (DX12's default front-face convention, rays come from +z).
void make_plane(float cx, float cy, float half_x, float half_y,
                uint nx, uint ny,
                luisa::vector<float3> &vertices,
                luisa::vector<Triangle> &triangles) {
    vertices.reserve((nx + 1u) * (ny + 1u));
    for (uint y = 0u; y <= ny; ++y) {
        for (uint x = 0u; x <= nx; ++x) {
            vertices.emplace_back(
                cx - half_x + 2.f * half_x * static_cast<float>(x) / static_cast<float>(nx),
                cy - half_y + 2.f * half_y * static_cast<float>(y) / static_cast<float>(ny),
                0.f);
        }
    }
    triangles.reserve(nx * ny * 2u);
    for (uint y = 0u; y < ny; ++y) {
        for (uint x = 0u; x < nx; ++x) {
            auto i = y * (nx + 1u) + x;
            triangles.emplace_back(Triangle{i, i + nx + 1u, i + 1u});
            triangles.emplace_back(Triangle{i + 1u, i + nx + 1u, i + nx + 2u});
        }
    }
}

constexpr float3 probe_direction() noexcept { return make_float3(0.f, 0.f, -1.f); }

// Engine HitBug scene: ground 25x25 at origin, backdrop 36x18 pushed off
// to y = 25 / z = -16 (a pure translation, like the engine's (0, 7, -16))
// so the orthographic probe grid can attribute hits unambiguously.
constexpr float kGroundHalf = 12.5f;                 // engine: Plane 25x25
constexpr float kBackdropHalfX = 18.f;               // engine: Plane 36x18
constexpr float kBackdropHalfY = 9.f;
constexpr float3 kBackdropOffset{0.f, 25.f, -16.f};

}// namespace

int main(int argc, char *argv[]) {
    if (argc <= 1) {
        LUISA_INFO("Usage: {} <backend>", argv[0]);
        return 0;
    }

    Context context{argv[0]};
    auto device = context.create_device(argv[1]);
    auto stream = device.create_stream(StreamTag::GRAPHICS);

    //------------------------------------------------------------------
    // Mesh A (entry 0): 25x25 plane, 4 verts / 2 triangles (engine ground)
    // Mesh B (entry 1): 36x18 plane, 4 verts / 2 triangles (engine backdrop)
    // Both meshes have identical vertex/triangle buffer sizes, like the
    // engine scene where the corruption shows up.
    //------------------------------------------------------------------
    luisa::vector<float3> verts_a;
    luisa::vector<Triangle> tris_a;
    make_plane(0.f, 0.f, kGroundHalf, kGroundHalf, 1u, 1u, verts_a, tris_a);

    luisa::vector<float3> verts_b;
    luisa::vector<Triangle> tris_b;
    // Geometry at the origin; the world offset comes from the instance
    // transform below (matches the engine: mesh local, moved by TLAS entry).
    make_plane(0.f, 0.f, kBackdropHalfX, kBackdropHalfY, 1u, 1u, verts_b, tris_b);

    LUISA_INFO("mesh A: {} verts / {} triangles (entry 0)",
               verts_a.size(), tris_a.size());
    LUISA_INFO("mesh B: {} verts / {} triangles (entry 1)",
               verts_b.size(), tris_b.size());

    auto vert_buffer_a = device.create_buffer<float3>(verts_a.size());
    auto tri_buffer_a = device.create_buffer<Triangle>(tris_a.size());
    auto vert_buffer_b = device.create_buffer<float3>(verts_b.size());
    auto tri_buffer_b = device.create_buffer<Triangle>(tris_b.size());
    stream << vert_buffer_a.copy_from(luisa::span{verts_a})
           << tri_buffer_a.copy_from(luisa::span{tris_a})
           << vert_buffer_b.copy_from(luisa::span{verts_b})
           << tri_buffer_b.copy_from(luisa::span{tris_b});

    auto mesh_a = device.create_mesh(vert_buffer_a, tri_buffer_a);
    auto mesh_b = device.create_mesh(vert_buffer_b, tri_buffer_b);
    stream << mesh_a.build() << mesh_b.build();

    Accel accel = device.create_accel({});
    accel.emplace_back(mesh_a, make_float4x4(1.f));
    accel.emplace_back(mesh_b, luisa::translation(kBackdropOffset));
    stream << accel.build() << synchronize();

    //------------------------------------------------------------------
    // Probe kernel: orthographic ray, write (inst, prim, miss, t) + bary
    //------------------------------------------------------------------
    Kernel1D probe_kernel = [&](AccelVar accel,
                                BufferVar<float2> probe_origins,
                                BufferVar<float4> probe_results,
                                BufferVar<float2> probe_bary) noexcept {
        auto i = dispatch_id().x;
        auto xy = probe_origins->read(i);
        Var<Ray> ray = make_ray(make_float3(xy.x, xy.y, 5.f), probe_direction());
        auto hit = accel.intersect(ray, {});
        probe_results->write(i, make_float4(
            cast<float>(hit.inst), cast<float>(hit.prim),
            ite(hit->miss(), 1.f, 0.f), hit.committed_ray_t));
        probe_bary->write(i, hit.bary);
    };

    //------------------------------------------------------------------
    // Render kernel: orthographic debug view, procedural checker tinted
    // per instance (cell = 2 world units, like the engine backdrop).
    //------------------------------------------------------------------
    Kernel2D render_kernel = [&](AccelVar accel, ImageFloat output) noexcept {
        auto coord = dispatch_id().xy();
        auto size = dispatch_size().xy();
        auto uv = (make_float2(coord) + 0.5f) / make_float2(size);
        auto world = make_float2(-20.f, -14.f) + uv * make_float2(40.f, 50.f);
        Var<Ray> ray = make_ray(make_float3(world, 5.f), probe_direction());
        auto hit = accel.intersect(ray, {});
        Float3 color = def(make_float3(0.f));
        $if(!hit->miss()) {
            auto hit_pos = ray->origin() + ray->direction() * hit.committed_ray_t;
            auto cell = floor(hit_pos.x / 2.f) + floor(hit_pos.y / 2.f);
            auto checker = ite(fmod(cell, 2.f) == 0.f, 0.8f, 0.2f);
            auto tint = ite(hit.inst == 0u,
                            make_float3(0.4f, 1.f, 0.4f),   // entry 0: green
                            make_float3(1.f, 0.4f, 0.4f));  // entry 1: red
            color = make_float3(checker) * tint * 2.f;
        };
        output.write(coord, make_float4(color, 1.f));
    };

    //------------------------------------------------------------------
    // Probe grid. Entry 1's footprint is x in [-18, 18], y in [16, 34];
    // entry 0's geometry translated by the entry-1 transform would span
    // x in [-12.5, 12.5], y in [12.5, 37.5]. The |x| = 16.5 probes lie
    // inside entry 1 but outside that translated ground extent: they must
    // hit. The inner 5x5 grid must hit entry 1 with barycentrics that
    // reconstruct onto the ray from entry 1's own 2 triangles. The 3x3
    // grid checks entry 0 the same way.
    //------------------------------------------------------------------
    struct Probe {
        float2 origin;
        uint expected_inst;
        uint max_prim;
    };
    luisa::vector<Probe> probes;
    for (auto iy = 0u; iy < 5u; ++iy) {
        for (auto ix = 0u; ix < 5u; ++ix) {
            probes.emplace_back(Probe{
                float2{-8.f + 4.f * ix, 18.f + 3.5f * iy},
                1u, 1u});
        }
    }
    for (auto corner : {float2{-16.5f, 17.5f}, float2{16.5f, 17.5f},
                        float2{-16.5f, 32.5f}, float2{16.5f, 32.5f}}) {
        probes.emplace_back(Probe{corner, 1u, 1u});
    }
    for (auto iy = 0u; iy < 3u; ++iy) {
        for (auto ix = 0u; ix < 3u; ++ix) {
            probes.emplace_back(Probe{
                float2{-8.f + 8.f * ix, -8.f + 8.f * iy},
                0u, 1u});
        }
    }

    auto probe_count = static_cast<size_t>(probes.size());
    Buffer<float2> probe_origins = device.create_buffer<float2>(probe_count);
    Buffer<float4> probe_results = device.create_buffer<float4>(probe_count);
    Buffer<float2> probe_bary = device.create_buffer<float2>(probe_count);
    luisa::vector<float2> probe_origins_cpu(probe_count);
    for (auto i = 0u; i < probe_count; ++i) {
        probe_origins_cpu[i] = make_float2(probes[i].origin.x, probes[i].origin.y);
    }
    stream << probe_origins.copy_from(luisa::span{probe_origins_cpu});

    auto probe_shader = device.compile(probe_kernel);
    auto render_shader = device.compile(render_kernel);

    luisa::vector<float4> results_cpu(probe_count);
    luisa::vector<float2> bary_cpu(probe_count);
    stream << probe_shader(accel, probe_origins, probe_results, probe_bary)
                  .dispatch(probe_count)
           << probe_results.copy_to(luisa::span{results_cpu})
           << probe_bary.copy_to(luisa::span{bary_cpu})
           << synchronize();

    //------------------------------------------------------------------
    // Verify: every probe must report the instance it targets, a
    // primitive index within that instance's own triangle range, and a
    // barycentric reconstruction that lands on the probe ray using the
    // reported instance's own vertices.
    //------------------------------------------------------------------
    auto failures = 0u;
    auto reproduced = 0u;
    for (auto i = 0u; i < probe_count; ++i) {
        auto &probe = probes[i];
        auto inst = static_cast<uint>(results_cpu[i].x + 0.5f);
        auto prim = static_cast<uint>(results_cpu[i].y + 0.5f);
        auto missed = results_cpu[i].z != 0.f;
        auto t = results_cpu[i].w;
        auto origin = probe.origin;
        if (missed) {
            ++failures;
            ++reproduced;
            LUISA_INFO("probe {:2d} at ({:6.2f}, {:6.2f}): MISS (ray escaped; "
                       "expected instance {}) -- entry {} not bound to its own "
                       "geometry at all",
                       i, origin.x, origin.y, probe.expected_inst,
                       probe.expected_inst);
            continue;
        }
        if (inst != probe.expected_inst) {
            ++failures;
            ++reproduced;
            LUISA_INFO("probe {:2d} at ({:6.2f}, {:6.2f}): hit inst={} "
                       "(expected {}) t={:5.2f} -- WRONG INSTANCE",
                       i, origin.x, origin.y, inst, probe.expected_inst, t);
            continue;
        }
        if (prim > probe.max_prim) {
            ++failures;
            ++reproduced;
            LUISA_INFO("probe {:2d} at ({:6.2f}, {:6.2f}): inst={} prim={} -- "
                       "BUG REPRODUCED: prim out of range (this mesh has {} "
                       "triangles); the hit came from the other mesh's BLAS",
                       i, origin.x, origin.y, inst, prim,
                       probe.max_prim + 1u);
            continue;
        }
        // Reconstruct the hit point from the reported instance's own
        // mesh data (local space) and map it through the instance
        // transform; it must land on the probe ray (the planes are all
        // z = 0 and the ray is a straight -z line through the origin).
        const auto &verts = inst == 0u ? verts_a : verts_b;
        const auto &tris = inst == 0u ? tris_a : tris_b;
        const auto &offset = inst == 0u ? make_float3(0.f) : kBackdropOffset;
        auto &tri = tris[prim];
        auto b1 = bary_cpu[i].x;
        auto b2 = bary_cpu[i].y;
        auto b0 = 1.f - b1 - b2;
        auto p = b0 * verts[tri.i0] + b1 * verts[tri.i1] + b2 * verts[tri.i2] + offset;
        auto err_x = std::abs(p.x - origin.x);
        auto err_y = std::abs(p.y - origin.y);
        auto err_z = std::abs(p.z - (5.f - t));
        if (err_x > 1e-3f || err_y > 1e-3f || err_z > 1e-3f) {
            ++failures;
            ++reproduced;
            LUISA_INFO("probe {:2d} at ({:6.2f}, {:6.2f}): inst={} prim={} "
                       "t={:5.2f} bary=({:4.2f},{:4.2f}) reconstructs to "
                       "({:6.2f},{:6.2f},{:6.2f}) -- BUG REPRODUCED: hit "
                       "geometry does not belong to the reported instance "
                       "(off by {:5.2f} units)",
                       i, origin.x, origin.y, inst, prim, t, b1, b2,
                       p.x, p.y, p.z, std::max(err_x, err_y));
        } else {
            LUISA_INFO("probe {:2d} at ({:6.2f}, {:6.2f}): inst={} prim={} "
                       "t={:5.2f}  ok",
                       i, origin.x, origin.y, inst, prim, t);
        }
    }

    //------------------------------------------------------------------
    // Debug image: procedural checker tinted green (entry 0) / red
    // (entry 1). Misbound entries show the wrong tint / a warped grid.
    //------------------------------------------------------------------
    constexpr auto render_w = 512u;
    constexpr auto render_h = 640u;
    Image<float> render_target = device.create_image<float>(
        PixelStorage::FLOAT4, make_uint2(render_w, render_h));
    luisa::vector<float4> render_cpu(render_w * render_h);
    stream << render_shader(accel, render_target)
                  .dispatch(make_uint2(render_w, render_h))
           << render_target.copy_to(luisa::span{render_cpu})
           << synchronize();

    std::ofstream ppm{"tlas_instance_binding.ppm", std::ios::binary};
    ppm << "P6\n" << render_w << " " << render_h << "\n255\n";
    for (auto y = render_h; y > 0u; --y) {// flip: image row 0 = world +y
        for (auto x = 0u; x < render_w; ++x) {
            auto &v = render_cpu[(y - 1u) * render_w + x];
            auto tonemap = [](float c) noexcept {
                return static_cast<char>(
                    std::min(std::max(c / (1.f + c), 0.f), 1.f) * 255.f);
            };
            char rgb[3] = {tonemap(v.x), tonemap(v.y), tonemap(v.z)};
            ppm.write(rgb, 3);
        }
    }
    LUISA_INFO("wrote tlas_instance_binding.ppm");

    if (failures == 0u) {
        LUISA_INFO("PASS: all {} probes consistent with the emplaced meshes",
                   probe_count);
        return 0;
    }
    LUISA_ERROR("FAIL: {} of {} probes inconsistent", failures, probe_count);
    if (reproduced != 0u) {
        LUISA_INFO("{} probe(s) reported hits inconsistent with the target "
                   "mesh's geometry.",
                   reproduced);
    }
    return 1;
}
