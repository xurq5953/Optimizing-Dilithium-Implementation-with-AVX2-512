# This Makefile can be used with GNU Make or BSD Make

LIB=libdilithium5_avx2.a
HEADERS=align.h api.h  consts.h ntt.h packing.h params.h poly.h polyvec.h rejsample.h rounding.h sign.h symmetric.h common/keccak/brg_endian.h
OBJECTS=consts.o  packing.o poly.o polyvec.o rejsample.o rounding.o sign.o symmetric-shake.o   invntt.o ntt.o pointwise.o
KECCAK4XDIR=common/keccak
KECCAK4XOBJ=KeccakP-1600-times4-SIMD256.o
KECCAK4X=$(KECCAK4XDIR)/$(KECCAK4XOBJ)
SOURCES=sign.c packing.c polyvec.c poly.c  consts.c rejsample.c rounding.c \
  common/fips202.c \
common/keccak/keccak8x.c common/keccak/keccak8x.h common/keccak/KeccakP-1600-AVX512.c common/keccak/KeccakP-1600-SnP.h \
fips202x8.c fips202x8.h ntt512.c inv_ntt512.c pointwise512.c uniform_eta_pack.c  common/keccak/KeccakP-1600-times8-SIMD512.c common/keccak/KeccakP-1600-times8-SnP.h \
 PSPMTEE.c PSPMTEE.h

CFLAGS=-mavx2 -mpopcnt -O3 -Wall -Wextra -Wpedantic\
          -Wmissing-prototypes -Wredundant-decls \
          -Wpointer-arith -Wshadow \
          -mavx512f -mavx512vl -mavx512cd -mavx512bw -mavx512dq  -mavx512vbmi2 -mavx512vbmi\
          -std=c99

all: test_speed

test_speed: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c
	$(CC) $(CFLAGS) \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	   $(KECCAK4X) $(HEADERS) $(SOURCES)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

%.o: %.S $(HEADERS)
	$(CC) $(CFLAGS) -c -o $@ $<

$(LIB): $(OBJECTS) $(KECCAK4X)
	$(AR) -r $@ $(OBJECTS) $(KECCAK4X)

$(KECCAK4X):
	$(MAKE) -C $(KECCAK4XDIR) $(KECCAK4XOBJ)

clean:
	$(RM) $(OBJECTS)
	$(RM) $(LIB)
	$(RM) test_speed