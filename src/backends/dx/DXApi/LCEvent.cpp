#include <DXApi/LCEvent.h>
#include <DXRuntime/CommandQueue.h>
#include <DXRuntime/DStorageCommandQueue.h>
// WaitOnAddress/WakeByAddressAll are exported by the OneCore umbrella
// import library only (documented library for synchapi.h).
#pragma comment(lib, "OneCore.lib")
namespace lc::dx {
LCEvent::LCEvent(Device *device, bool shared)
    : Resource(device) {
    ThrowIfFailed(device->device->CreateFence(
        0,
        shared ? D3D12_FENCE_FLAG_SHARED : D3D12_FENCE_FLAG_NONE,
        IID_PPV_ARGS(&_fence)));
}
LCEvent::~LCEvent() {
    if (!_fence) return;
    HANDLE eventHandle = CreateEventExW(nullptr, nullptr, 0, EVENT_ALL_ACCESS);
    auto d = vstd::scope_exit([&] {
        CloseHandle(eventHandle);
    });
    if (_fence->GetCompletedValue() < last_fence) {
        ThrowIfFailed(_fence->SetEventOnCompletion(last_fence, eventHandle));
        WaitForSingleObject(eventHandle, INFINITE);
    }
}

void LCEvent::sync(uint64_t fenceIdx) const {
    // WaitOnAddress re-checks the value before blocking, so a wake between
    // the load and the wait cannot be missed. Wakers: the executor threads
    // after storing finished_event (under event_mtx).
    auto cur = finished_event.load(std::memory_order_acquire);
    while (cur < fenceIdx) {
        WaitOnAddress(&finished_event, &cur, sizeof(cur), INFINITE);
        cur = finished_event.load(std::memory_order_acquire);
    }
}
void LCEvent::signal(CommandQueue *queue, uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    if (!device->device_settings || !device->device_settings->SignalFence(queue->queue(), _fence.Get(), fenceIdx))
        ThrowIfFailed(queue->queue()->Signal(_fence.Get(), fenceIdx));
    last_fence = std::max(last_fence, fenceIdx);
    queue->add_event(this, fenceIdx);
}
void LCEvent::signal(DStorageCommandQueue *queue, uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    queue->Signal(_fence.Get(), fenceIdx);
    last_fence = std::max(last_fence, fenceIdx);
    queue->AddEvent(this, fenceIdx);
}
void LCEvent::wait(CommandQueue *queue, uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    if (!device->device_settings || !device->device_settings->WaitFence(queue->queue(), _fence.Get(), fenceIdx))
        ThrowIfFailed(queue->queue()->Wait(_fence.Get(), fenceIdx));
}
bool LCEvent::is_complete(uint64_t fenceIdx) const {
    std::lock_guard<luisa::spin_mutex> lck(event_mtx);
    return finished_event >= fenceIdx;
}
}// namespace lc::dx
