# This Makefile can be used with GNU Make or BSD Make

LIB=libdilithium2_avx2.a
HEADERS=align.h api.h cdecl.h consts.h fips202x4.h ntt.h packing.h params.h poly.h polyvec.h rejsample.h rounding.h sign.h symmetric.h uniform_eta_pack.h shuffle.inc keccak/KeccakP-1600-times8-SnP.h
OBJECTS=consts.o fips202x4.o packing.o poly.o polyvec.o rejsample.o rounding.o sign.o symmetric-shake.o  f1600x4.o invntt.o ntt.o pointwise.o shuffle.o 
KECCAK4XDIR=keccak/
KECCAK4XOBJ=KeccakP-1600-times4-SIMD256.o
KECCAK4X=$(KECCAK4XDIR)/$(KECCAK4XOBJ)
SOURCES = sign.c packing.c polyvec.c poly.c ntt.S invntt.S pointwise.S consts.c rejsample.c rounding.c \
fips202x4.c  symmetric-shake.c  f1600x4.S  shuffle.S ntt512.c inv_ntt512.c pointwise512.c uniform_eta_pack.c PSPMTEE.c PSPMTEE.h
KECCAK = fips202.c fips202.h keccak/keccak8x.c keccak/keccak8x.h fips202x8.c fips202x8.h keccak/KeccakP-1600-AVX512.c keccak/KeccakP-1600-SnP.h keccak/KeccakP-1600-times8-SIMD512.c

CFLAGS=-mavx2 -mpopcnt -O3 -march=native -mtune=native -Wall -Wextra -Wpedantic \
          -Wmissing-prototypes -Wredundant-decls \
          -Wpointer-arith -Wshadow \
          -mavx512f -mavx512vl -mavx512cd -mavx512bw -mavx512dq -mavx512vbmi2 -mavx512vbmi\


all: test_speed

test_speed: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c
	$(CC) $(CFLAGS) \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	   $(SOURCES) $(KECCAK)


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
