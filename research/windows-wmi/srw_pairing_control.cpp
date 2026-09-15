// Original isolated negative control. Operates only on this process's local lock.
// It never opens WMI, games, services, or any other process. No blocking re-entry.
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cstdint>

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const bool wrong = std::strcmp(argv[1], "mismatch") == 0;
    if (!wrong && std::strcmp(argv[1], "matched") != 0) return 2;
    SRWLOCK lock = SRWLOCK_INIT;
    AcquireSRWLockExclusive(&lock);
    if (wrong) ReleaseSRWLockShared(&lock);
    else ReleaseSRWLockExclusive(&lock);
    uintptr_t raw = 0;
    static_assert(sizeof(raw) == sizeof(lock));
    std::memcpy(&raw, &lock, sizeof(raw));
    const BOOL acquired = TryAcquireSRWLockExclusive(&lock);
    if (acquired) ReleaseSRWLockExclusive(&lock);
    std::printf("{\"release\":\"%s\",\"raw_after_release\":\"0x%016llx\",\"try_exclusive\":%s}\n",
                wrong ? "mismatched_shared" : "matched_exclusive",
                static_cast<unsigned long long>(raw), acquired ? "true" : "false");
    return 0;
}
