#pragma once

#include <luisa/core/dll_export.h>
#include <luisa/core/spin_mutex.h>
#include <luisa/core/basic_types.h>
#include <luisa/core/stl/unordered_map.h>
#include <luisa/runtime/rhi/resource.h>
#include <luisa/runtime/rhi/command.h>

namespace luisa::compute {

class Mesh;
class Curve;
class ProceduralPrimitive;
class MotionInstance;
template<typename T>
class BufferView;

namespace detail {
// for DSL
class AccelExprProxy;
}// namespace detail

// Accel is top-level acceleration structure(TLAS) for ray-tracing
class LUISA_RUNTIME_API Accel final : public Resource {
    friend class ManagedAccel;

public:
    using BuildRequest = AccelBuildRequest;
    using Modification = AccelBuildCommand::Modification;

private:
    struct TransformSource {
        uint64_t buffer{};
        uint64_t offset{};
        uint first_instance{};
        uint count{};
        [[nodiscard]] explicit operator bool() const noexcept { return buffer != 0u; }
    };
    luisa::unordered_map<size_t, Modification> _modifications;
    mutable luisa::spin_mutex _mtx;
    size_t _instance_count{};
    TransformSource _transform_source;

private:
    friend class Device;
    friend class Mesh;
    explicit Accel(DeviceInterface *device, const AccelOption &option) noexcept;
    luisa::unique_ptr<Command> _build(Accel::BuildRequest request,
                                      bool update_instance_buffer_only) noexcept;

public:
    Accel() noexcept = default;
    ~Accel() noexcept override;
    Accel(Accel &&) noexcept;
    Accel(Accel const &) noexcept = delete;
    Accel &operator=(Accel &&rhs) noexcept;
    Accel &operator=(Accel const &) noexcept = delete;
    using Resource::operator bool;
    using Resource::release;

    // number of instances
    [[nodiscard]] size_t size() const noexcept;

    // whether there are any stashed updates
    [[nodiscard]] bool dirty() const noexcept;

    // low-level interfaces
    void emplace_back_handle(uint64_t mesh_handle,
                             float4x4 const &transform,
                             uint8_t visibility_mask,
                             bool opaque,
                             uint user_id) noexcept;

    void set_handle(size_t index, uint64_t mesh_handle,
                    float4x4 const &transform,
                    uint8_t visibility_mask, bool opaque, uint user_id) noexcept;

    void set_prim_handle(size_t index, uint64_t prim_handle) noexcept;

    // host interfaces
    // operations is committed by update_instance() or build()
    void emplace_back(const Mesh &mesh,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true,
                      uint user_id = 0) noexcept;

    void emplace_back(const Curve &curve,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true,
                      uint user_id = 0) noexcept;

    void emplace_back(const ProceduralPrimitive &prim,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      uint user_id = 0) noexcept;

    void emplace_back(const MotionInstance &instance,
                      float4x4 transform = make_float4x4(1.f),
                      uint8_t visibility_mask = 0xffu,
                      bool opaque = true,
                      uint user_id = 0) noexcept;

    void set(size_t index, const Mesh &mesh,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true,
             uint user_id = 0) noexcept;

    void set(size_t index, const Curve &curve,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true,
             uint user_id = 0) noexcept;

    void set(size_t index, const ProceduralPrimitive &prim,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             uint user_id = 0) noexcept;

    void set(size_t index, const MotionInstance &instance,
             float4x4 transform = make_float4x4(1.f),
             uint8_t visibility_mask = 0xffu,
             bool opaque = true,
             uint user_id = 0) noexcept;

    void set_mesh(size_t index, const Mesh &mesh) noexcept;
    void set_curve(size_t index, const Curve &curve) noexcept;
    void set_procedural_primitive(size_t index, const ProceduralPrimitive &prim) noexcept;
    void set_motion_instance(size_t index, const MotionInstance &instance) noexcept;

    void pop_back() noexcept;
    void set_transform_on_update(size_t index, float4x4 transform) noexcept;
    void set_visibility_on_update(size_t index, uint8_t visibility_mask) noexcept;
    void set_opaque_on_update(size_t index, bool opaque) noexcept;
    void set_instance_user_id_on_update(size_t index, uint user_id) noexcept;

    // set a device buffer of column-major float4x4 matrices to be copied into
    // the instance-transform rows [first_instance, first_instance + count)
    // by the backend during the next build() / update_instance_buffer(); the
    // matrices never round-trip through the host. Only the transform rows are
    // touched (visibility, opacity, user id, and the bound primitive are
    // preserved). The stash is consumed (cleared) by the next build; if
    // host-side modifications also target transform rows inside the range in
    // the same build, the buffer contents win. Backends without GPU-side
    // instance updates reject the build; a zero view clears a pending stash.
    void set_transform_buffer_on_update(size_t first_instance,
                                        BufferView<float4x4> transforms) noexcept;

    // update top-level acceleration structure's instance data without build
    [[nodiscard]] luisa::unique_ptr<Command> update_instance_buffer() noexcept;
    // update top-level acceleration structure's instance data and build
    [[nodiscard]] luisa::unique_ptr<Command> build(BuildRequest request = BuildRequest::PREFER_UPDATE) noexcept;

    // DSL interface
    [[nodiscard]] const detail::AccelExprProxy *operator->() const noexcept;
};

}// namespace luisa::compute
