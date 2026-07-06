#pragma once

/*
 * Temporary workaround for CUDA 12.9 + GCC13 + SCALE on Hudson.
 *
 * CUDA's host_defines.h defines __noinline__ as a macro before libstdc++
 * is parsed. GCC13's libstdc++ uses __attribute__((__noinline__)),
 * which then expands into invalid syntax.
 *
 * Undefining the macro restores the compiler builtin attribute lookup.
 *
 * Remove once the SCALE/CUDA toolchain no longer requires this workaround.
 */

#ifdef __noinline__
#undef __noinline__
#endif
