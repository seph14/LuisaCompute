#pragma once
#include <DXRuntime/Device.h>
#include <luisa/runtime/rhi/command.h>
#include <luisa/runtime/device.h>
#include <Resource/Resource.h>
#include <DXRuntime/EnhancedBarrierTracker.h>
using namespace luisa::compute;
namespace lc::dx {
class DefaultBuffer;
class Buffer;
class BottomAccel;
class BboxAccel;
class CommandBufferBuilder;
class EnhancedBarrierTracker;
class Mesh;
class BottomAccel;
class MeshHandle;
struct alignas(16) PackedModifier {
    float affine[12];
    uint64_t primitive;
    uint index : 24;
    uint vis_mask : 8;
    uint user_id : 24;
    uint flags : 8;
};
class TopAccel : public Resource {

    friend class BottomAccel;
    friend class BboxAccel;

public:
    // device-side transform source: column-major float4x4 matrices copied
    // into the instance rows [first, first + count) before the build
    struct TransformSource {
        Buffer const *buffer = nullptr;
        uint64_t offset = 0;
        uint first = 0;
        uint count = 0;
        [[nodiscard]] explicit operator bool() const noexcept { return buffer != nullptr; }
    };

private:
    TransformSource transformSource;
    vstd::unique_ptr<DefaultBuffer> instBuffer;
    vstd::unique_ptr<DefaultBuffer> accelBuffer;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO topLevelPrebuildInfo;
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC topLevelBuildDesc;
    struct Instance {
        MeshHandle *handle = nullptr;
    };
    vstd::vector<Instance> allInstance;
    void ResizeAllInstance(size_t size);
    vstd::unordered_map<uint64, MeshHandle *> setMap;
    vstd::vector<PackedModifier> setDesc;
    void SetMesh(BottomAccel *mesh, uint64 index);
    uint compactSize = 0;
    bool requireBuild = false;
    bool update = false;
    void UpdateMesh(
        MeshHandle *handle);
    bool GenerateNewBuffer(
        char const *name,
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        vstd::unique_ptr<DefaultBuffer> &oldBuffer, size_t newSize, bool needCopy, D3D12_RESOURCE_STATES state);
    void InitSetDesc(vstd::span<AccelBuildCommand::Modification const> const &modifications);
    void ProcessSetDesc(EnhancedBarrierTracker &tracker);
    void ProcessSetMap();

public:
    bool RequireCompact() const;
    TopAccel(Device *device, luisa::compute::AccelOption const &option);
    uint Length() const { return topLevelBuildDesc.Inputs.NumDescs; }
    Tag get_tag() const override { return Tag::Accel; }

    DefaultBuffer const *GetAccelBuffer() const {
        return accelBuffer.get();
    }
    DefaultBuffer const *GetInstBuffer() const {
        return instBuffer.get();
    }
    size_t PreProcess(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        uint64 size,
        vstd::span<AccelBuildCommand::Modification const> const &modifications,
        TransformSource const &transformSrc,
        bool update);
    void PreProcessInst(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        uint64 size,
        vstd::span<AccelBuildCommand::Modification const> const &modifications,
        TransformSource const &transformSrc);
    void Build(
        EnhancedBarrierTracker &tracker,
        CommandBufferBuilder &builder,
        BufferView const *scratchBuffer);
    void FinalCopy(
        CommandBufferBuilder &builder,
        BufferView const &scratchBuffer);
    bool CheckAccel(
        CommandBufferBuilder &builder);
    ~TopAccel();
};
}// namespace lc::dx
