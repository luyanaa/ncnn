#ifndef NCNN_PS2_VU1_GEMM_H
#define NCNN_PS2_VU1_GEMM_H

/*
 * VU1 microprogram GEMM engine for the Sony PlayStation 2 (EE side driver).
 *
 * Computes the blocked product   C(4x4) = A(4xK) * B(Kx4)   in fp32 on VU1,
 * blocking the EE until the result is ready.
 *
 * VU1 data memory layout, in quadwords (16 bytes), used by the microprogram
 * src/layer/mips/ps2_vu1_gemm.vsm:
 *
 *   qword 0    .. K-1      : A stored k-major ("column packed"):
 *                            qword k = {A[0][k], A[1][k], A[2][k], A[3][k]}
 *                            so one lq loads the k-th column of A as 4 lanes.
 *   qword 256  .. 256+K-1  : B row-major; qword k = {B[k][0..3]}
 *   qword 512  .. 515      : C output; qword 512+i = C[i][0..3]
 *   qword 1023             : header; word 3 (the "w" slot) = K as an integer
 *
 * K is zero-padded by the driver to a multiple of 4 (needed so that the
 * per-k column qword always contains 4 valid A values). K must be in
 * 1..256. Results are read back from the IO-mapped VU1 data memory
 * (0x1100C000) after the microprogram's E-bit, so no reverse DMA is used.
 */

#ifdef __cplusplus
extern "C" {
#endif

/* Uploads the microprogram to VU1 and initializes DMA channel VIF1.
 * Call once before ps2_vu1_gemm_4xKx4(). Returns 0 on success. */
int ps2_vu1_gemm_init(void);

/* C(4x4) = A(4xK) * B(Kx4), row-major fp32, blocking.
 * A: 4*K floats (zero-padded internally to a multiple of 4 in K).
 * B: K*4 floats.  C: 16 floats.  K in 1..256.
 * Returns 0 on success, negative on error. */
int ps2_vu1_gemm_4xKx4(const float* A, const float* B, float* C, int K);

/* Releases the DMA packet buffer (microcode stays in VU1). */
void ps2_vu1_gemm_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* NCNN_PS2_VU1_GEMM_H */
