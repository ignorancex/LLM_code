#include <alpaqa/util/dl-flags.hpp>

#if !_WIN32
#include <dlfcn.h>
#endif

namespace alpaqa {

DynamicLoadFlags::operator int() const {
#if _WIN32
    return 0;
#else
    return (global ? RTLD_GLOBAL : RTLD_LOCAL) | //
           (lazy ? RTLD_LAZY : RTLD_NOW) |       //
           (nodelete ? RTLD_NODELETE : 0) |
#ifdef RTLD_DEEPBIND
           (deepbind ? RTLD_DEEPBIND : 0) |
#endif
           0;
#endif
}

} // namespace alpaqa
