#pragma once
#include <dstorage/dstorage.h>
#include <DXRuntime/Device.h>
#include <luisa/vstl/lockfree_array_queue.h>
#include <DXRuntime/DxPtr.h>
#include <DXApi/CmdQueueBase.h>
#include <luisa/runtime/command_list.h>
#include <luisa/backends/ext/dstorage_ext_interface.h>

namespace lc::dx {

class LCEvent;
class DStorageFileImpl : public vstd::IOperatorNewBase {
public:
    ComPtr<IDStorageFile> file;
    size_t size_bytes;
    DStorageFileImpl(ComPtr<IDStorageFile> &&file, size_t size_bytes) : file{std::move(file)}, size_bytes{size_bytes} {}
};
class DStorageCommandQueue : public CmdQueueBase {
    struct WaitQueueHandle {
        HANDLE handle;
    };
    struct CallbackEvent {
        using Variant = vstd::variant<
            WaitQueueHandle,
            vstd::vector<vstd::function<void()>>,
            LCEvent const *>;
        Variant evt;
        uint64_t fence;
        bool wakeupThread;
        template<typename Arg>
            requires(luisa::is_constructible_v<Variant, Arg &&>)
        CallbackEvent(Arg &&arg,
                      uint64_t fence,
                      bool wakeupThread)
            : evt{std::forward<Arg>(arg)}, fence{fence}, wakeupThread{wakeupThread} {}
    };
    std::atomic_bool enabled = true;
    std::atomic_uint64_t executedFrame = 0;
    std::atomic_uint64_t lastFrame = 0;
    luisa::spin_mutex mtx;
    luisa::spin_mutex exec_mtx;
    DSTORAGE_REQUEST_SOURCE_TYPE sourceType;
    ComPtr<IDStorageQueue2> queue;
    vstd::SingleThreadArrayQueue<CallbackEvent> executedAllocators;
    void ExecuteThread();

public:
    size_t staging_buffer_size = DSTORAGE_STAGING_BUFFER_SIZE_32MB;
    void Signal(ID3D12Fence *fence, UINT64 value);
    uint64_t LastFrame() const { return lastFrame; }
    DStorageCommandQueue(IDStorageFactory *factory, Device *device, luisa::compute::DStorageStreamSource source);
    void AddEvent(LCEvent const *evt, uint64_t fenceIdx);
    uint64_t Execute(
        vstd::span<const luisa::unique_ptr<luisa::compute::Command>> commands,
        luisa::vector<luisa::move_only_function<void()>> &&funcs);
    void Complete(uint64_t fence);
    void Complete();
    KILL_MOVE_CONSTRUCT(DStorageCommandQueue)
    KILL_COPY_CONSTRUCT(DStorageCommandQueue)
    ~DStorageCommandQueue();
private:
    // One token released (under mtx) per enqueue and one by the destructor
    // before join(); lets the executor block on an empty queue. Must be
    // constructed before thd starts.
    HANDLE wakeSem = CreateSemaphoreW(nullptr, 0, 0x7fffffff, nullptr);
    // make sure thread always construct after all members
    std::thread thd;
};
}// namespace lc::dx
