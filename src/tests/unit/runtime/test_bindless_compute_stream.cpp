// Repro: bindless array update on a COMPUTE-tagged stream (DX backend).
#include "ut/ut.hpp"
#include "test_device.h"

#include <luisa/runtime/context.h>
#include <luisa/runtime/device.h>
#include <luisa/runtime/stream.h>
#include <luisa/runtime/image.h>
#include <luisa/runtime/buffer.h>
#include <luisa/runtime/bindless_array.h>
#include <luisa/dsl/syntax.h>
#include <luisa/core/logging.h>

using namespace luisa;
using namespace luisa::compute;
using namespace boost::ut;
using namespace boost::ut::literals;

void test_bindless_compute_stream(Device &device) {
    auto tex = device.create_image<float>(PixelStorage::FLOAT4, make_uint2(16u));
    auto buf = device.create_buffer<float4>(16u);
    auto heap = device.create_bindless_array(2u);
    // compute-tagged stream, as in the crash report
    auto stream = device.create_stream(StreamTag::COMPUTE);
    auto pixels = luisa::vector<float>{};
    pixels.resize(16u * 16u * 4u);
    stream << tex.copy_from(luisa::span{pixels});
    stream << synchronize();
    LUISA_INFO("updating bindless array");
    stream << heap.emplace_on_update(0u, tex, Sampler::linear_linear_repeat())
                  .emplace_on_update(1u, buf)
                  .update();
    LUISA_INFO("synchronizing");
    stream << synchronize();
    LUISA_INFO("bindless update succeeded");
}

int main(int argc, char *argv[]) {
    auto dc = luisa::test::create_device_from_ut(argc, argv);
    if (!dc) { return 0; }
    boost::ut::detail::cfg::parse_arg_with_fallback(argc, const_cast<const char **>(argv));
    test_bindless_compute_stream(dc->device);
}
