#ifndef DILITHIUM3_AVX2_CDECL_H
#define DILITHIUM3_AVX2_CDECL_H

#define QINV 58728449 // q^(-1) mod 2^32
#define MONT (-4186625) // 2^32 mod q
#define DIV 41978 // mont^2/256
#define DIV_QINV (-8395782)

#define _8XQ          0
#define _8XQINV       8
#define _8XDIV_QINV  16
#define _8XDIV       24
#define _ZETAS_QINV  32
#define _ZETAS      328

/* The C ABI on MacOS exports all symbols with a leading
 * underscore. This means that any symbols we refer to from
 * C files (functions) can't be found, and all symbols we
 * refer to from ASM also can't be found (nttconsts.c).
 *
 * This define helps us get around this
 */

#define _cdecl(s) _##s
#define cdecl(s) s

#endif
