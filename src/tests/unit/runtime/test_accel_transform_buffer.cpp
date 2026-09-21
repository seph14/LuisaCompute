// Test for device-side TLAS instance-transform updates via a transform buffer.
// This test covers:
// - Accel::set_transform_buffer_on_update consumed by a refit (PREFER_UPDATE)
//   and a forced rebuild (FORCE_BUILD)
// - exact matrices: translation (incl. z and hit-distance checks), rotation,
//   and scaling, which distinguishes column-major from row-major encodings
// - sub-ranges of instances and sub-views (byte offsets) of the source buffer
// - GPU-produced transforms: a kernel writes the matrices in the same stream
//   submission as the build that consumes them
// - coexistence with host-side modifications in the same build and
//   preservation of visibility masks and user ids
// - the update_instance_buffer() path carrying the transform source

#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/luisa-compute.h>
#include <luisa/dsl/sugar.h>

#include <array>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

namespace {

struct Probe {
    float3 origin;
    uint mask;
};

struct ProbeResult {
    uint instance;
    bool hit;
    float distance;
    uint user_id;
};

// orthographic -z probes against the accel; one shared trace kernel
class TraceHelper {

private:
    Device &_device;
    Stream &_stream;
    Accel &_accel;
    Buffer<float3> _origins;
    Buffer<uint> _masks;
    Buffer<float4> _results;
    Shader1D<Buffer<float3>, Buffer<uint>, Buffer<float4>, Accel> _shader;

public:
    TraceHelper(Device &device, Stream &stream, Accel &accel, size_t capacity)
        : _device{device}, _stream{stream}, _accel{accel},
          _origins{device.create_buffer<float3>(capacity)},
          _masks{device.create_buffer<uint>(capacity)},
          _results{device.create_buffer<float4>(capacity)},
          _shader{[&device] {
              Kernel1D trace = [](BufferFloat3 origins, BufferUInt masks,
                                  BufferFloat4 results, AccelVar accel) noexcept {
                  auto index = dispatch_id().x;
                  auto ray = make_ray(origins.read(index), make_float3(0.0f, 0.0f, -1.0f));
                  AccelTraceOptions options{.visibility_mask = masks.read(index)};
                  auto hit = accel.intersect(ray, options);
                  auto hit_instance = ite(hit->miss(), 0u, hit.inst);
                  auto user_id = ite(hit->miss(), 0u, accel.instance_user_id(hit.inst));
                  results.write(index, make_float4(
                                           cast<float>(hit_instance),
                                           cast<float>(!hit->miss()),
                                           hit.committed_ray_t,
                                           cast<float>(user_id)));
              };
              return device.compile(trace);
          }()} {}

    // launches a probe batch and returns normalized results
    luisa::vector<ProbeResult> run(std::initializer_list<Probe> probes) {
        luisa::vector<float3> origins(probes.size());
        luisa::vector<uint> masks(probes.size());
        auto p = probes.begin();
        for (auto i = 0u; i < probes.size(); i++) {
            origins[i] = p[i].origin;
            masks[i] = p[i].mask;
        }
        luisa::vector<float4> host_results(probes.size());
        _stream << _origins.view(0u, probes.size()).copy_from(luisa::span{origins})
                << _masks.view(0u, probes.size()).copy_from(luisa::span{masks})
                << _shader(_origins, _masks, _results, _accel).dispatch(probes.size())
                << _results.view(0u, probes.size()).copy_to(luisa::span{host_results})
                << synchronize();
        luisa::vector<ProbeResult> results(probes.size());
        for (auto i = 0u; i < probes.size(); i++) {
            results[i] = ProbeResult{
                .instance = static_cast<uint>(host_results[i].x + 0.5f),
                .hit = host_results[i].y != 0.0f,
                .distance = host_results[i].z,
                .user_id = static_cast<uint>(host_results[i].w + 0.5f)};
        }
        return results;
    }
};

void check_probe(const ProbeResult &result, bool should_hit, uint instance,
                 float distance, uint user_id, luisa::string_view phase) {
    expect(result.hit == should_hit)
        << luisa::format("{} hit mismatch: got {}, expected {}",
                         phase, result.hit, should_hit);
    if (!should_hit) { return; }
    expect(result.instance == instance)
        << luisa::format("{} instance mismatch: got {}, expected {}",
                         phase, result.instance, instance);
    expect(std::abs(result.distance - distance) < 1.0e-3f)
        << luisa::format("{} hit distance mismatch: got {}, expected {}",
                         phase, result.distance, distance);
    expect(result.user_id == user_id)
        << luisa::format("{} user id mismatch: got {}, expected {}",
                         phase, result.user_id, user_id);
}

[[nodiscard]] auto make_unit_triangle_mesh(Device &device, Stream &stream,
                                           Buffer<float3> &vertex_buffer,
                                           Buffer<Triangle> &triangle_buffer) {
    std::array vertices{
        make_float3(-1.0f, -1.0f, 0.0f),
        make_float3(1.0f, -1.0f, 0.0f),
        make_float3(0.0f, 1.0f, 0.0f)};
    std::array triangles{Triangle{0u, 1u, 2u}};
    vertex_buffer = device.create_buffer<float3>(vertices.size());
    triangle_buffer = device.create_buffer<Triangle>(triangles.size());
    auto mesh = device.create_mesh(vertex_buffer, triangle_buffer);
    stream << vertex_buffer.copy_from(luisa::span{vertices})
           << triangle_buffer.copy_from(luisa::span{triangles})
           << mesh.build();
    return mesh;
}

}// namespace

void test_accel_transform_buffer_updates(Device &device) {
    auto stream = device.create_stream();
    Buffer<float3> vertex_buffer;
    Buffer<Triangle> triangle_buffer;
    auto mesh = make_unit_triangle_mesh(device, stream, vertex_buffer, triangle_buffer);

    auto accel = device.create_accel({.allow_update = true});
    for (auto i = 0u; i < 3u; i++) {
        accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, true, 10u + i);
    }
    stream << accel.build() << synchronize();

    TraceHelper tracer{device, stream, accel, 16u};
    auto transforms = device.create_buffer<float4x4>(3u);
    auto upload = [&](std::array<float4x4, 3u> matrices) {
        stream << transforms.copy_from(luisa::span{matrices});
    };
    auto update = [&](Accel::BuildRequest request) {
        accel.set_transform_buffer_on_update(0u, transforms.view());
        stream << accel.build(request) << synchronize();
    };

    // initial state: all three instances overlap the origin triangle
    check_probe(tracer.run({Probe{make_float3(0.0f, 0.0f, 5.0f), 0xffu}})[0],
                true, 0u, 5.0f, 10u, "initial build");

    // refit with pure translations (z = -2 makes hit distance 7)
    upload({translation(4.0f, 0.0f, -2.0f),
            translation(8.0f, 0.0f, -2.0f),
            translation(12.0f, 0.0f, -2.0f)});
    update(Accel::BuildRequest::PREFER_UPDATE);
    auto results = tracer.run({Probe{make_float3(0.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(4.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(8.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(12.0f, 0.5f, 5.0f), 0xffu}});
    check_probe(results[0], false, 0u, 0.0f, 0u, "refit origin vacated");
    check_probe(results[1], true, 0u, 7.0f, 10u, "refit translation 0");
    check_probe(results[2], true, 1u, 7.0f, 11u, "refit translation 1");
    check_probe(results[3], true, 2u, 7.0f, 12u, "refit translation 2");

    // forced rebuild with a 180-degree rotation and a 2x scale; a transposed
    // (row-major) copy would swap these footprints and fail
    upload({rotation(make_float3(0.0f, 0.0f, 1.0f), pi),
            translation(100.0f, 0.0f, 0.0f) * scaling(2.0f),
            translation(200.0f, 0.0f, 0.0f)});
    update(Accel::BuildRequest::FORCE_BUILD);
    results = tracer.run({Probe{make_float3(0.5f, 0.5f, 5.0f), 0xffu},
                          Probe{make_float3(101.5f, -1.5f, 5.0f), 0xffu},
                          Probe{make_float3(0.5f, -0.5f, 5.0f), 0xffu}});
    // rotated triangle: (0.5, 0.5) is only inside after the 180-degree turn
    check_probe(results[0], true, 0u, 5.0f, 10u, "rebuild rotation");
    // 2x-scaled triangle at x = 100: (101.5, -1.5) is only inside after scaling
    check_probe(results[1], true, 1u, 5.0f, 11u, "rebuild scaling");
    // (0.5, -0.5) was inside the original triangle but not the rotated one
    check_probe(results[2], false, 0u, 0.0f, 0u, "rebuild rotation vacancy");

    // second refit on top of the rebuilt state
    upload({translation(-4.0f, 0.0f, 0.0f),
            translation(-8.0f, 0.0f, 0.0f),
            translation(-12.0f, 0.0f, 0.0f)});
    update(Accel::BuildRequest::PREFER_UPDATE);
    results = tracer.run({Probe{make_float3(-4.0f, 0.0f, 5.0f), 0xffu},
                          Probe{make_float3(-8.0f, 0.0f, 5.0f), 0xffu},
                          Probe{make_float3(-12.0f, 0.0f, 5.0f), 0xffu},
                          Probe{make_float3(0.5f, 0.5f, 5.0f), 0xffu}});
    check_probe(results[0], true, 0u, 5.0f, 10u, "second refit 0");
    check_probe(results[1], true, 1u, 5.0f, 11u, "second refit 1");
    check_probe(results[2], true, 2u, 5.0f, 12u, "second refit 2");
    check_probe(results[3], false, 0u, 0.0f, 0u, "second refit vacated");
}

void test_accel_transform_buffer_subrange(Device &device) {
    auto stream = device.create_stream();
    Buffer<float3> vertex_buffer;
    Buffer<Triangle> triangle_buffer;
    auto mesh = make_unit_triangle_mesh(device, stream, vertex_buffer, triangle_buffer);

    // start all instances far to the left at distinct spots
    auto accel = device.create_accel({.allow_update = true});
    for (auto i = 0u; i < 4u; i++) {
        accel.emplace_back(mesh, translation(-20.0f - 4.0f * static_cast<float>(i), 0.0f, 0.0f),
                           0xffu, true, 20u + i);
    }
    stream << accel.build() << synchronize();

    TraceHelper tracer{device, stream, accel, 16u};

    // five matrices; the update maps buffer elements [2, 4) onto instances
    // [1, 3) -- instance 0 and 3 keep their initial placements
    std::array<float4x4, 5u> matrices{
        make_float4x4(1.0f), make_float4x4(1.0f),
        translation(6.0f, 0.0f, 0.0f), translation(10.0f, 0.0f, 0.0f),
        make_float4x4(1.0f)};
    auto transforms = device.create_buffer<float4x4>(matrices.size());
    stream << transforms.copy_from(luisa::span{matrices});
    accel.set_transform_buffer_on_update(1u, transforms.view(2u, 2u));
    stream << accel.build() << synchronize();

    auto results = tracer.run({Probe{make_float3(6.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(10.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(-20.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(-32.0f, 0.0f, 5.0f), 0xffu},
                               Probe{make_float3(0.0f, 0.0f, 5.0f), 0xffu}});
    check_probe(results[0], true, 1u, 5.0f, 21u, "subrange updated 1");
    check_probe(results[1], true, 2u, 5.0f, 22u, "subrange updated 2");
    check_probe(results[2], true, 0u, 5.0f, 20u, "subrange untouched 0");
    check_probe(results[3], true, 3u, 5.0f, 23u, "subrange untouched 3");
    check_probe(results[4], false, 0u, 0.0f, 0u, "subrange origin empty");
}

void test_accel_transform_buffer_producer(Device &device) {
    auto stream = device.create_stream();
    Buffer<float3> vertex_buffer;
    Buffer<Triangle> triangle_buffer;
    auto mesh = make_unit_triangle_mesh(device, stream, vertex_buffer, triangle_buffer);

    auto accel = device.create_accel({.allow_update = true});
    for (auto i = 0u; i < 2u; i++) {
        accel.emplace_back(mesh, make_float4x4(1.0f), 0xffu, true, 30u + i);
    }
    stream << accel.build() << synchronize();

    TraceHelper tracer{device, stream, accel, 16u};
    auto transforms = device.create_buffer<float4x4>(2u);
    auto offsets = device.create_buffer<float3>(2u);

    // the matrices are produced by a kernel in the same stream submission as
    // the build that consumes them: no host round-trip, no synchronize between
    Kernel1D write_transforms = [](BufferFloat4x4 matrices,
                                   BufferFloat3 translations) noexcept {
        auto i = dispatch_id().x;
        auto offset = translations.read(i);
        auto matrix = make_float4x4(
            make_float4(1.0f, 0.0f, 0.0f, 0.0f),
            make_float4(0.0f, 1.0f, 0.0f, 0.0f),
            make_float4(0.0f, 0.0f, 1.0f, 0.0f),
            make_float4(offset.x, offset.y, offset.z, 1.0f));
        matrices.write(i, matrix);
    };
    auto writer = device.compile(write_transforms);

    for (auto frame = 0u; frame < 2u; frame++) {
        auto x0 = 6.0f + 10.0f * static_cast<float>(frame);
        std::array frame_offsets{
            make_float3(x0, 0.0f, -1.0f),
            make_float3(x0 + 4.0f, 0.0f, -1.0f)};
        stream << offsets.copy_from(luisa::span{frame_offsets})
               << writer(transforms, offsets).dispatch(2u);
        accel.set_transform_buffer_on_update(0u, transforms.view());
        stream << accel.build(Accel::BuildRequest::PREFER_UPDATE) << synchronize();

        auto phase = luisa::format("producer frame {}", frame);
        auto results = tracer.run({Probe{make_float3(x0, 0.0f, 5.0f), 0xffu},
                                   Probe{make_float3(x0 + 4.0f, 0.0f, 5.0f), 0xffu}});
        check_probe(results[0], true, 0u, 6.0f, 30u, phase);
        check_probe(results[1], true, 1u, 6.0f, 31u, phase);
    }
}

void test_accel_transform_buffer_fields(Device &device) {
    auto stream = device.create_stream();
    Buffer<float3> vertex_buffer;
    Buffer<Triangle> triangle_buffer;
    auto mesh = make_unit_triangle_mesh(device, stream, vertex_buffer, triangle_buffer);

    auto accel = device.create_accel({.allow_update = true});
    accel.emplace_back(mesh, make_float4x4(1.0f), 0x1u, true, 40u);
    accel.emplace_back(mesh, make_float4x4(1.0f), 0x2u, true, 41u);
    accel.emplace_back(mesh, make_float4x4(1.0f), 0x4u, true, 42u);
    stream << accel.build() << synchronize();

    TraceHelper tracer{device, stream, accel, 16u};
    auto transforms = device.create_buffer<float4x4>(3u);

    // host-side visibility change rides in the same build as the transform
    // buffer; the transform copy must not clobber masks or user ids
    std::array matrices{
        translation(4.0f, 0.0f, 0.0f),
        translation(8.0f, 0.0f, 0.0f),
        translation(12.0f, 0.0f, 0.0f)};
    stream << transforms.copy_from(luisa::span{matrices});
    accel.set_visibility_on_update(0u, 0x8u);
    accel.set_transform_buffer_on_update(0u, transforms.view());
    stream << accel.build(Accel::BuildRequest::PREFER_UPDATE) << synchronize();

    auto results = tracer.run({Probe{make_float3(4.0f, 0.0f, 5.0f), 0x8u},
                               Probe{make_float3(4.0f, 0.0f, 5.0f), 0x1u},
                               Probe{make_float3(8.0f, 0.0f, 5.0f), 0x2u},
                               Probe{make_float3(12.0f, 0.0f, 5.0f), 0x4u},
                               Probe{make_float3(8.0f, 0.0f, 5.0f), 0x1u}});
    check_probe(results[0], true, 0u, 5.0f, 40u, "fields new mask visible");
    check_probe(results[1], false, 0u, 0.0f, 0u, "fields old mask hidden");
    check_probe(results[2], true, 1u, 5.0f, 41u, "fields mask 1");
    check_probe(results[3], true, 2u, 5.0f, 42u, "fields mask 2");
    check_probe(results[4], false, 0u, 0.0f, 0u, "fields wrong mask hidden");

    // update_instance_buffer() carries the transform source too; a following
    // refit build picks up the already-copied rows
    std::array moved{
        translation(-4.0f, 0.0f, 0.0f),
        translation(-8.0f, 0.0f, 0.0f),
        translation(-12.0f, 0.0f, 0.0f)};
    stream << transforms.copy_from(luisa::span{moved});
    accel.set_transform_buffer_on_update(0u, transforms.view());
    stream << accel.update_instance_buffer()
           << accel.build(Accel::BuildRequest::PREFER_UPDATE)
           << synchronize();
    results = tracer.run({Probe{make_float3(-4.0f, 0.0f, 5.0f), 0x8u},
                          Probe{make_float3(-8.0f, 0.0f, 5.0f), 0x2u},
                          Probe{make_float3(-12.0f, 0.0f, 5.0f), 0x4u}});
    check_probe(results[0], true, 0u, 5.0f, 40u, "instance-buffer-only 0");
    check_probe(results[1], true, 1u, 5.0f, 41u, "instance-buffer-only 1");
    check_probe(results[2], true, 2u, 5.0f, 42u, "instance-buffer-only 2");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(
        argc, const_cast<const char **>(argv));
    // device-side transform-buffer updates are currently implemented by the
    // DirectX backend only; the Vulkan backend rejects the command and the
    // CPU fallback warns and ignores it
    if (dc->device.backend_name() != "dx") {
        LUISA_INFO("Skipping accel transform-buffer test: backend '{}' does "
                   "not support device-side instance updates.",
                   dc->device.backend_name());
        return 0;
    }
    test_accel_transform_buffer_updates(dc->device);
    test_accel_transform_buffer_subrange(dc->device);
    test_accel_transform_buffer_producer(dc->device);
    test_accel_transform_buffer_fields(dc->device);
}
