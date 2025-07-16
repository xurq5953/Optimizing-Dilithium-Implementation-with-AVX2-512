# Optimizing-Dilithium-Implementation-with-AVX2-512
The source code of the paper Optimizing Dilithium Implementation with AVX2/-512. 


Our implementation is based on the reference Dilithium implementation (https://csrc.nist.gov/Projects/post-quantum-cryptography/selected-algorithms-2022) and the PQClean project (https://github.com/PQClean/PQClean).

The performance of the 4-way Keccak implementations under AVX2 adopted in the reference Dilithium implementation (KeccakP1600times4_PermuteAll_24rounds) and the PQClean project (PQCLEAN_DILITHIUM2_AVX2_f1600x4) seems to depend on platforms and compilers. So we keep the code for both implementations in our project, which can be chosen in functions as needed.
