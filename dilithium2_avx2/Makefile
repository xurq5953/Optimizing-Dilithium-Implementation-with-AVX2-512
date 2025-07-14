# This Makefile can be used with GNU Make or BSD Make

LIB=libdilithium2_avx2.a
HEADERS=align.h cdecl.h consts.h keccak/fips202x4.h ntt/ntt.h packing.h params.h poly.h polyvec.h rejsample.h rounding.h sign.h  keccak/fips202.h
OBJECTS=consts.o keccak/fips202x4.o packing.o poly.o polyvec.o rejsample.o rounding.o sign.o  keccak/f1600x4.o ntt/invntt.o ntt/ntt.o pointwise.o
KECCAK4XDIR=keccak
KECCAK4XOBJ=KeccakP-1600-times4-SIMD256.o
KECCAK4X=$(KECCAK4XDIR)/$(KECCAK4XOBJ)
SOURCES = sign.c packing.c polyvec.c poly.c pointwise.S consts.c rejsample.c rounding.c\
keccak/fips202x4.c  keccak/f1600x4.S keccak/fips202.c ntt/ntt_avx2.c ntt/intt_avx2.c randombytes.h
KECCAK_SOURCES = $(SOURCES)  keccak/KeccakP-1600-times4-SIMD256.o
KECCAK_ALL=keccak/KeccakP-1600-times4-SIMD256.c \
           keccak/KeccakP-1600-times4-SnP.h \
           keccak/SIMD256-config.h \
           keccak/align.h \
           keccak/brg_endian.h

CFLAGS=-Wall -Wextra -Wpedantic -Wmissing-prototypes -Wredundant-decls \
         -Wshadow -Wpointer-arith -mavx2 -mpopcnt -maes -mbmi2 -march=native -mtune=native -O3



all: test_speed

test_speed: test/test_speed.c test/speed_print.c test/speed_print.h \
  test/cpucycles.c test/cpucycles.h randombytes.c
	$(CC) $(CFLAGS) \
	  -o $@ $< test/speed_print.c test/cpucycles.c randombytes.c \
	    $(SOURCES) $(KECCAK_ALL) $(HEADERS)


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
	$(RM) keccak/KeccakP-1600-times4-SIMD256.o
