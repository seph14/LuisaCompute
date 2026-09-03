// Repro: texture uploaded+synced on a temporary GRAPHICS stream that is then
// destroyed, with compression + bindless update happening later on a second
// long-lived stream — mirrors NewTypeEngine's TextureConverter nullptr-stream
// path followed by MaterialPool::createMaterial (DX backend).
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/backends/ext/tex_compress_ext.h>
#include <luisa/core/logging.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

static constexpr uint kSize = 512u;

static luisa::vector<half4> make_pixels() {
    luisa::vector<half4> pixels(kSize * kSize);
    for (uint y = 0; y < kSize; y++) {
        for (uint x = 0; x < kSize; x++) {
            auto i = y * kSize + x;
            pixels[i] = half4{half{0.5f}, half{0.25f}, half{0.75f}, half{1.f}};
        }
    }
    return pixels;
}

// Variant A — engine nullptr path: upload+sync on temp stream, destroy it,
// then compress + bindless on the pool stream.
void test_cross_stream(Device &device, Stream &pool_stream) {
    auto tex = device.create_image<float>(PixelStorage::HALF4, make_uint2(kSize));

    {
        auto temp = device.create_stream(StreamTag::GRAPHICS);
        auto pixels = make_pixels();
        temp << tex.copy_from(luisa::span{pixels}) << synchronize();
        LUISA_INFO("uploaded on temp stream");
    } // temp stream destroyed here (already synchronized)

    // compressTexture step 1: download level 0 on the pool stream
    luisa::vector<half4> back(kSize * kSize);
    pool_stream << tex.view(0).copy_to(luisa::span{back}) << synchronize();
    LUISA_INFO("downloaded on pool stream");

    // compressTexture step 2: BC7 compress via TexCompressExt
    if (auto ext = device.extension<TexCompressExt>()) {
        auto mipped = device.create_image<float>(PixelStorage::BYTE4, make_uint2(kSize), 1u);
        luisa::vector<uint8_t> quantized(kSize * kSize * 4u);
        pool_stream << mipped.view(0).copy_from(quantized.data());
        Buffer<uint> staging = device.create_buffer<uint>(kSize * kSize / 4u);
        auto result = ext->compress_bc7(pool_stream, mipped.view(0), staging.view(0, kSize * kSize / 4u), 0.0f);
        pool_stream << synchronize();
        LUISA_INFO("BC7 compress result: {}", static_cast<int>(result));
    } else {
        LUISA_INFO("TexCompressExt not available, skipping");
    }

    // uploadMaterialTextures: bindless update on pool stream
    auto heap = device.create_bindless_array(4u);
    heap.emplace_on_update(0u, tex, Sampler::linear_linear_mirror());
    pool_stream << heap.update() << synchronize();
    LUISA_INFO("bindless update on pool stream succeeded");
}

// Variant B — engine pool-stream path (known-good control): upload, compress
// and bindless all on the same long-lived stream.
void test_same_stream(Device &device, Stream &pool_stream) {
    auto tex = device.create_image<float>(PixelStorage::HALF4, make_uint2(kSize));
    auto pixels = make_pixels();
    pool_stream << tex.copy_from(luisa::span{pixels});

    if (auto ext = device.extension<TexCompressExt>()) {
        auto mipped = device.create_image<float>(PixelStorage::BYTE4, make_uint2(kSize), 1u);
        luisa::vector<uint8_t> quantized(kSize * kSize * 4u);
        pool_stream << mipped.view(0).copy_from(quantized.data());
        Buffer<uint> staging = device.create_buffer<uint>(kSize * kSize / 4u);
        auto result = ext->compress_bc7(pool_stream, mipped.view(0), staging.view(0, kSize * kSize / 4u), 0.0f);
        pool_stream << synchronize();
        LUISA_INFO("BC7 compress result: {}", static_cast<int>(result));
    }

    auto heap = device.create_bindless_array(4u);
    heap.emplace_on_update(0u, tex, Sampler::linear_linear_mirror());
    pool_stream << heap.update() << synchronize();
    LUISA_INFO("bindless update on pool stream succeeded");
}

static void run_tests(Device &device) {
    auto pool_stream = device.create_stream(StreamTag::GRAPHICS);
    test_same_stream(device, pool_stream);
    test_cross_stream(device, pool_stream);
    pool_stream << synchronize();
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    // --validation: recreate the device through the validation layer, as
    // NewTypeEngine does when NT_PROFILING is on.
    bool validation = argc > 2 && std::string_view{argv[2]} == "--validation";
    if (validation) {
        auto dev = dc->context.create_device(argv[1], nullptr, true);
        LUISA_INFO("running with validation layer");
        run_tests(dev);
    } else {
        run_tests(dc->device);
    }
    LUISA_INFO("all tests done");
    return 0;
}
