# This Makefile can be used with GNU Make or BSD Make

LIB=libdilithium3_avx2.a
HEADERS=align.h api.h cdecl.h consts.h fips202x4.h ntt.h packing.h params.h poly.h polyvec.h rejsample.h rounding.h sign.h symmetric.h  fips202.h
OBJECTS=consts.o fips202x4.o packing.o poly.o polyvec.o rejsample.o rounding.o sign.o symmetric-shake.o  f1600x4.o invntt.o ntt.o pointwise.o shuffle.o
KECCAK4XDIR=keccak4x
KECCAK4XOBJ=KeccakP-1600-times4-SIMD256.o
KECCAK4X=$(KECCAK4XDIR)/$(KECCAK4XOBJ)
SOURCES = sign.c packing.c polyvec.c poly.c ntt.S invntt.S pointwise.S consts.c rejsample.c rounding.c\
fips202x4.c  symmetric-shake.c  f1600x4.S  shuffle.S fips202.c randombytes.h fips202x8.c fips202x8.h ntt512.c inv_ntt512.c pointwise512.c\
common/keccak/keccak8x.c common/keccak/keccak8x.h common/keccak/KeccakP-1600-AVX512.c common/keccak/KeccakP-1600-SnP.h common/keccak/KeccakP-1600-times8-SIMD512.c common/keccak/KeccakP-1600-times8-SnP.h
KECCAK_SOURCES = $(SOURCES)  keccak4x/KeccakP-1600-times4-SIMD256.o
KECCAK_ALL=keccak4x/KeccakP-1600-times4-SIMD256.c \
           keccak4x/KeccakP-1600-times4-SnP.h \
           keccak4x/SIMD256-config.h \
           keccak4x/align.h \
           keccak4x/brg_endian.h

CFLAGS=-Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
         -Wshadow -Wpointer-arith -mavx2 -mpopcnt -maes -mbmi2 -march=native -mtune=native -O3



all: test_speed

test_speed: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c
	$(CC) $(CFLAGS) \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	    $(SOURCES) $(KECCAK_ALL) $(HEADERS)


keccak4x/KeccakP-1600-times4-SIMD256.o: \
  keccak4x/KeccakP-1600-times4-SIMD256.c \
  keccak4x/KeccakP-1600-times4-SnP.h \
  keccak4x/KeccakP-1600-unrolling.macros \
  keccak4x/SIMD256-config.h \
  keccak4x/align.h \
  keccak4x/brg_endian.h
	$(CC) $(CFLAGS) -c $< -o $@



%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(OBJECTS) $(KECCAK4X)
	$(AR) -r $@  $(KECCAK4X)

$(KECCAK4X):
	$(MAKE) -C $(KECCAK4XDIR) $(KECCAK4XOBJ)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(LIB)
	$(RM) test_speed
	$(RM) keccak4x/KeccakP-1600-times4-SIMD256.o
