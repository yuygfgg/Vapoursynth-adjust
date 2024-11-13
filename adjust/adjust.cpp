#include <VapourSynth4.h>
#include <VSHelper4.h>
#include <algorithm>
#include <cmath>
#include <cstring>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#define CLAMP(value, min, max) ((value) < (min) ? (min) : ((value) > (max) ? (max) : (value)))

struct TweakData {
    VSNode* node;
    float hue;
    float sat;
    float bright;
    float cont;
    bool coring;
};

// 手动复制一个平面的数据
static void copyPlaneData_uint8(const uint8_t* src, uint8_t* dst, int width, int height,
                          ptrdiff_t src_stride, ptrdiff_t dst_stride) {
    for (int y = 0; y < height; y++) {
        memcpy(dst, src, width * sizeof(uint8_t));
        src += src_stride;
        dst += dst_stride;
    }
}

static void copyPlaneData_uint16(const uint16_t* src, uint16_t* dst, int width, int height,
                          ptrdiff_t src_stride, ptrdiff_t dst_stride) {
    for (int y = 0; y < height; y++) {
        memcpy(dst, src, width * sizeof(uint16_t));
        src += src_stride;
        dst += dst_stride;
    }
}

static void copyPlaneData_float(const float* src, float* dst, int width, int height,
                          ptrdiff_t src_stride, ptrdiff_t dst_stride) {
    for (int y = 0; y < height; y++) {
        memcpy(dst, src, width * sizeof(float));
        src += src_stride;
        dst += dst_stride;
    }
}

// 16-bit luma processing with NEON
void process_luma_uint16(const uint16_t* srcp, uint16_t* dstp, const int width, const int height,
                             const ptrdiff_t src_stride, const ptrdiff_t dst_stride,
                             const float cont, const float bright,
                             const uint16_t luma_min, const uint16_t luma_max) {
    if (width <= 0 || height <= 0)
        return;

    if (std::abs(bright) < 1e-6f && std::abs(cont - 1.0f) < 1e-6f) {
        copyPlaneData_uint16(srcp, dstp, width, height, src_stride, dst_stride);
        return;
    }
#ifdef __ARM_NEON__
    float32x4_t v_cont = vdupq_n_f32(cont);
    float32x4_t v_bright = vdupq_n_f32(bright);
    float32x4_t v_min = vdupq_n_f32(luma_min);
    float32x4_t v_max = vdupq_n_f32(luma_max);
#endif
    for (int y = 0; y < height; y++) {
        int x = 0;

        // Process 8 pixels at a time
#ifdef __ARM_NEON__
        for (; x <= width - 8; x += 8) {

            // Load 8 16-bit pixels
            uint16x8_t v_src = vld1q_u16(srcp + x);
            
            // Split into two 32-bit vectors and convert to float
            uint32x4_t v_src_low = vmovl_u16(vget_low_u16(v_src));
            uint32x4_t v_src_high = vmovl_u16(vget_high_u16(v_src));
            float32x4_t v_float_low = vcvtq_f32_u32(v_src_low);
            float32x4_t v_float_high = vcvtq_f32_u32(v_src_high);

            // Apply contrast and brightness
            v_float_low = vmlaq_f32(v_bright, vsubq_f32(v_float_low, v_min), v_cont);
            v_float_high = vmlaq_f32(v_bright, vsubq_f32(v_float_high, v_min), v_cont);

            // Add back minimum
            v_float_low = vaddq_f32(v_float_low, v_min);
            v_float_high = vaddq_f32(v_float_high, v_min);

            // Clamp values
            v_float_low = vminq_f32(vmaxq_f32(v_float_low, v_min), v_max);
            v_float_high = vminq_f32(vmaxq_f32(v_float_high, v_min), v_max);

            // Convert back to uint16
            uint16x8_t v_result = vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(v_float_low)),
                vqmovn_u32(vcvtq_u32_f32(v_float_high))
            );

            // Store result
            vst1q_u16(dstp + x, v_result);
        }
#endif
        // Process remaining pixels
        
        for (; x < width; x++) {
            float val = static_cast<float>((srcp[x] - luma_min)) * cont + bright + luma_min;
            dstp[x] = static_cast<uint16_t>(CLAMP(val, static_cast<float>(luma_min), static_cast<float>(luma_max)));
        }

        srcp += src_stride;
        dstp += dst_stride;
    }
}

// 16-bit chroma processing with NEON
void process_chroma_uint16(const uint16_t* srcp_u, const uint16_t* srcp_v, uint16_t* dstp_u, uint16_t* dstp_v,
                               const int width, const int height,
                               const ptrdiff_t src_stride_u, const ptrdiff_t src_stride_v,
                               const ptrdiff_t dst_stride_u, const ptrdiff_t dst_stride_v,
                               const float hue_sin, const float hue_cos, const float sat,
                               const uint16_t chroma_min, const uint16_t chroma_max, const uint16_t gray) {
    if (width <= 0 || height <= 0)
        return;

    if (std::abs(sat - 1.0f) < 1e-6f && std::abs(hue_sin) < 1e-6f) {
        copyPlaneData_uint16(srcp_u, dstp_u, width, height, src_stride_u, dst_stride_u);
        copyPlaneData_uint16(srcp_v, dstp_v, width, height, src_stride_v, dst_stride_v);
        return;
    }
#ifdef __ARM_NEON__
    float32x4_t v_hue_sin = vdupq_n_f32(hue_sin);
    float32x4_t v_hue_cos = vdupq_n_f32(hue_cos);
    float32x4_t v_sat = vdupq_n_f32(sat);
    float32x4_t v_gray = vdupq_n_f32(gray);
    float32x4_t v_min = vdupq_n_f32(chroma_min);
    float32x4_t v_max = vdupq_n_f32(chroma_max);
#endif
    for (int y = 0; y < height; y++) {
        int x = 0;
        // Process 4 pixels at a time
        #ifdef __ARM_NEON__
        for (; x <= width - 4; x += 4) {
            // Load 4 pixels from each plane
            uint16x4_t v_src_u = vld1_u16(srcp_u + x);
            uint16x4_t v_src_v = vld1_u16(srcp_v + x);

            // Convert to float32
            float32x4_t v_u = vcvtq_f32_u32(vmovl_u16(v_src_u));
            float32x4_t v_v = vcvtq_f32_u32(vmovl_u16(v_src_v));

            // Subtract gray
            v_u = vsubq_f32(v_u, v_gray);
            v_v = vsubq_f32(v_v, v_gray);

            // Calculate new U and V values
            float32x4_t v_new_u = vmlaq_f32(
                vmulq_f32(v_u, vmulq_f32(v_hue_cos, v_sat)),
                v_v, vmulq_f32(v_hue_sin, v_sat)
            );
            float32x4_t v_new_v = vmlsq_f32(
                vmulq_f32(v_v, vmulq_f32(v_hue_cos, v_sat)),
                v_u, vmulq_f32(v_hue_sin, v_sat)
            );

            // Add back gray
            v_new_u = vaddq_f32(v_new_u, v_gray);
            v_new_v = vaddq_f32(v_new_v, v_gray);

            // Clamp values
            v_new_u = vminq_f32(vmaxq_f32(v_new_u, v_min), v_max);
            v_new_v = vminq_f32(vmaxq_f32(v_new_v, v_min), v_max);

            // Convert back to uint16
            uint16x4_t v_result_u = vqmovn_u32(vcvtq_u32_f32(v_new_u));
            uint16x4_t v_result_v = vqmovn_u32(vcvtq_u32_f32(v_new_v));

            // Store results
            vst1_u16(dstp_u + x, v_result_u);
            vst1_u16(dstp_v + x, v_result_v);
        }
#endif
        // Process remaining pixels
        
        for (; x < width; x++) {
            const float u = static_cast<float>(srcp_u[x]) - gray;
            const float v = static_cast<float>(srcp_v[x]) - gray;
            
            float new_u = u * hue_cos * sat + v * hue_sin * sat + gray;
            float new_v = v * hue_cos * sat - u * hue_sin * sat + gray;
            
            dstp_u[x] = static_cast<uint16_t>(CLAMP(new_u, static_cast<float>(chroma_min), static_cast<float>(chroma_max)));
            dstp_v[x] = static_cast<uint16_t>(CLAMP(new_v, static_cast<float>(chroma_min), static_cast<float>(chroma_max)));
        }

        srcp_u += src_stride_u;
        srcp_v += src_stride_v;
        dstp_u += dst_stride_u;
        dstp_v += dst_stride_v;
    }
}

// Float processing with NEON
void process_luma_float(const float* srcp, float* dstp, const int width, const int height,
                            const ptrdiff_t src_stride, const ptrdiff_t dst_stride,
                            const float cont, const float bright,
                            const float luma_min, const float luma_max) {
    if (width <= 0 || height <= 0)
        return;

    if (std::abs(bright) < 1e-6f && std::abs(cont - 1.0f) < 1e-6f) {
        copyPlaneData_float(srcp, dstp, width, height, src_stride, dst_stride);
        return;
    }
#ifdef __ARM_NEON__
    float32x4_t v_cont = vdupq_n_f32(cont);
    float32x4_t v_bright = vdupq_n_f32(bright);
    float32x4_t v_min = vdupq_n_f32(luma_min);
    float32x4_t v_max = vdupq_n_f32(luma_max);
#endif
    for (int y = 0; y < height; y++) {
        int x = 0;
        // Process 4 pixels at a time
        #ifdef __ARM_NEON__
        for (; x <= width - 4; x += 4) {
            float32x4_t v_src = vld1q_f32(srcp + x);

            // Apply contrast and brightness
            float32x4_t v_result = vmlaq_f32(v_bright, vsubq_f32(v_src, v_min), v_cont);
            v_result = vaddq_f32(v_result, v_min);

            // Clamp values
            v_result = vminq_f32(vmaxq_f32(v_result, v_min), v_max);

            // Store result
            vst1q_f32(dstp + x, v_result);
        }
        #endif
        // Process remaining pixels
        
        for (; x < width; x++) {
            float val = (srcp[x] - luma_min) * cont + bright + luma_min;
            dstp[x] = CLAMP(val, luma_min, luma_max);
        }

        srcp += src_stride;
        dstp += dst_stride;
    }
}

void process_chroma_float(const float* srcp_u, const float* srcp_v, float* dstp_u, float* dstp_v,
                              const int width, const int height,
                              const ptrdiff_t src_stride_u, const ptrdiff_t src_stride_v,
                              const ptrdiff_t dst_stride_u, const ptrdiff_t dst_stride_v,
                              const float hue_sin, const float hue_cos, const float sat,
                              const float chroma_min, const float chroma_max, const float gray) {
    if (width <= 0 || height <= 0)
        return;

    if (std::abs(sat - 1.0f) < 1e-6f && std::abs(hue_sin) < 1e-6f) {
        copyPlaneData_float(srcp_u, dstp_u, width, height, src_stride_u, dst_stride_u);
        copyPlaneData_float(srcp_v, dstp_v, width, height, src_stride_v, dst_stride_v);
        return;
    }
#ifdef __ARM_NEON__
    float32x4_t v_hue_sin = vdupq_n_f32(hue_sin);
    float32x4_t v_hue_cos = vdupq_n_f32(hue_cos);
    float32x4_t v_sat = vdupq_n_f32(sat);
    float32x4_t v_gray = vdupq_n_f32(gray);
    float32x4_t v_min = vdupq_n_f32(chroma_min);
    float32x4_t v_max = vdupq_n_f32(chroma_max);
#endif
    for (int y = 0; y < height; y++) {
        int x = 0;
        // Process 4 pixels at a time
#ifdef __ARM_NEON__
        for (; x <= width - 4; x += 4) {
            float32x4_t v_u = vld1q_f32(srcp_u + x);
            float32x4_t v_v = vld1q_f32(srcp_v + x);

            // Subtract gray
            v_u = vsubq_f32(v_u, v_gray);
            v_v = vsubq_f32(v_v, v_gray);

            // Calculate new U and V values
            float32x4_t v_new_u = vmlaq_f32(
                vmulq_f32(v_u, vmulq_f32(v_hue_cos, v_sat)),
                v_v, vmulq_f32(v_hue_sin, v_sat)
            );
            float32x4_t v_new_v = vmlsq_f32(
                vmulq_f32(v_v, vmulq_f32(v_hue_cos, v_sat)),
                v_u, vmulq_f32(v_hue_sin, v_sat)
            );

            // Add back gray
            v_new_u = vaddq_f32(v_new_u, v_gray);
            v_new_v = vaddq_f32(v_new_v, v_gray);

            // Clamp values
            v_new_u = vminq_f32(vmaxq_f32(v_new_u, v_min), v_max);
            v_new_v = vminq_f32(vmaxq_f32(v_new_v, v_min), v_max);

            // Store results
            vst1q_f32(dstp_u + x, v_new_u);
            vst1q_f32(dstp_v + x, v_new_v);
        }
#endif
        // Process remaining pixels
        
        for (; x < width; x++) {
            const float u = srcp_u[x] - gray;
            const float v = srcp_v[x] - gray;
            
            float new_u = u * hue_cos * sat + v * hue_sin * sat + gray;
            float new_v = v * hue_cos * sat - u * hue_sin * sat + gray;
            
            dstp_u[x] = CLAMP(new_u, chroma_min, chroma_max);
            dstp_v[x] = CLAMP(new_v, chroma_min, chroma_max);
        }

        srcp_u += src_stride_u;
        srcp_v += src_stride_v;
        dstp_u += dst_stride_u;
        dstp_v += dst_stride_v;
    }
}

// 8-bit luma processing with NEON
void process_luma_uint8(const uint8_t* srcp, uint8_t* dstp, const int width, const int height,
                            const ptrdiff_t src_stride, const ptrdiff_t dst_stride,
                            const float cont, const float bright,
                            const uint8_t luma_min, const uint8_t luma_max) {
    if (width <= 0 || height <= 0)
        return;

    if (std::abs(bright) < 1e-6f && std::abs(cont - 1.0f) < 1e-6f) {
        copyPlaneData_uint8(srcp, dstp, width, height, src_stride, dst_stride);
        return;
    }
#ifdef __ARM_NEON__
    // Convert parameters to vectors
    float32x4_t v_cont = vdupq_n_f32(cont);
    float32x4_t v_bright = vdupq_n_f32(bright);
    float32x4_t v_min = vdupq_n_f32(luma_min);
    float32x4_t v_max = vdupq_n_f32(luma_max);
#endif
    for (int y = 0; y < height; y++) {
        int x = 0;
        // Process 16 pixels at a time
#ifdef __ARM_NEON__
        for (; x <= width - 16; x += 16) {
            // Load 16 pixels
            uint8x16_t v_src = vld1q_u8(srcp + x);
            
            // Convert first 8 pixels to float
            uint8x8_t v_src_low = vget_low_u8(v_src);
            float32x4_t v_float_0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v_src_low))));
            float32x4_t v_float_1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(v_src_low))));
            
            // Convert second 8 pixels to float
            uint8x8_t v_src_high = vget_high_u8(v_src);
            float32x4_t v_float_2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v_src_high))));
            float32x4_t v_float_3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(v_src_high))));

            // Apply contrast and brightness
            v_float_0 = vmlaq_f32(v_bright, vsubq_f32(v_float_0, v_min), v_cont);
            v_float_1 = vmlaq_f32(v_bright, vsubq_f32(v_float_1, v_min), v_cont);
            v_float_2 = vmlaq_f32(v_bright, vsubq_f32(v_float_2, v_min), v_cont);
            v_float_3 = vmlaq_f32(v_bright, vsubq_f32(v_float_3, v_min), v_cont);

            // Add back minimum
            v_float_0 = vaddq_f32(v_float_0, v_min);
            v_float_1 = vaddq_f32(v_float_1, v_min);
            v_float_2 = vaddq_f32(v_float_2, v_min);
            v_float_3 = vaddq_f32(v_float_3, v_min);

            // Clamp values
            v_float_0 = vminq_f32(vmaxq_f32(v_float_0, v_min), v_max);
            v_float_1 = vminq_f32(vmaxq_f32(v_float_1, v_min), v_max);
            v_float_2 = vminq_f32(vmaxq_f32(v_float_2, v_min), v_max);
            v_float_3 = vminq_f32(vmaxq_f32(v_float_3, v_min), v_max);

            // Convert back to uint8
            uint16x8_t v_u16_low = vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(v_float_0)),
                vqmovn_u32(vcvtq_u32_f32(v_float_1))
            );
            uint16x8_t v_u16_high = vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(v_float_2)),
                vqmovn_u32(vcvtq_u32_f32(v_float_3))
            );
            
            uint8x16_t v_result = vcombine_u8(
                vqmovn_u16(v_u16_low),
                vqmovn_u16(v_u16_high)
            );

            // Store result
            vst1q_u8(dstp + x, v_result);
        }
#endif
        // Process remaining pixels
        
        for (; x < width; x++) {
            float val = static_cast<float>((srcp[x] - luma_min)) * cont + bright + luma_min;
            dstp[x] = static_cast<uint8_t>(CLAMP(val, static_cast<float>(luma_min), static_cast<float>(luma_max)));
        }

        srcp += src_stride;
        dstp += dst_stride;
    }
}

// 8-bit chroma processing with NEON
void process_chroma_uint8(const uint8_t* srcp_u, const uint8_t* srcp_v, uint8_t* dstp_u, uint8_t* dstp_v,
                              const int width, const int height,
                              const ptrdiff_t src_stride_u, const ptrdiff_t src_stride_v,
                              const ptrdiff_t dst_stride_u, const ptrdiff_t dst_stride_v,
                              const float hue_sin, const float hue_cos, const float sat,
                              const uint8_t chroma_min, const uint8_t chroma_max, const uint8_t gray) {
    if (width <= 0 || height <= 0)
        return;

    if (std::abs(sat - 1.0f) < 1e-6f && std::abs(hue_sin) < 1e-6f) {
        copyPlaneData_uint8(srcp_u, dstp_u, width, height, src_stride_u, dst_stride_u);
        copyPlaneData_uint8(srcp_v, dstp_v, width, height, src_stride_v, dst_stride_v);
        return;
    }
#ifdef __ARM_NEON__
    // Convert parameters to vectors
    float32x4_t v_hue_sin = vdupq_n_f32(hue_sin);
    float32x4_t v_hue_cos = vdupq_n_f32(hue_cos);
    float32x4_t v_sat = vdupq_n_f32(sat);
    float32x4_t v_gray = vdupq_n_f32(gray);
    float32x4_t v_min = vdupq_n_f32(chroma_min);
    float32x4_t v_max = vdupq_n_f32(chroma_max);
#endif
    for (int y = 0; y < height; y++) {
        int x = 0;
#ifdef __ARM_NEON__
        // Process 8 pixels at a time
        for (; x <= width - 8; x += 8) {
            // Load 8 pixels from each plane
            uint8x8_t v_src_u = vld1_u8(srcp_u + x);
            uint8x8_t v_src_v = vld1_u8(srcp_v + x);

            // Convert to float
            float32x4_t v_u_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v_src_u))));
            float32x4_t v_u_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(v_src_u))));
            float32x4_t v_v_low = vcvtq_f32_u32(vmovl_u16(vget_low_u16(vmovl_u8(v_src_v))));
            float32x4_t v_v_high = vcvtq_f32_u32(vmovl_u16(vget_high_u16(vmovl_u8(v_src_v))));

            // Subtract gray
            v_u_low = vsubq_f32(v_u_low, v_gray);
            v_u_high = vsubq_f32(v_u_high, v_gray);
            v_v_low = vsubq_f32(v_v_low, v_gray);
            v_v_high = vsubq_f32(v_v_high, v_gray);

            // Calculate new U and V values
            float32x4_t v_new_u_low = vmlaq_f32(
                vmulq_f32(v_u_low, vmulq_f32(v_hue_cos, v_sat)),
                v_v_low, vmulq_f32(v_hue_sin, v_sat)
            );
            float32x4_t v_new_u_high = vmlaq_f32(
                vmulq_f32(v_u_high, vmulq_f32(v_hue_cos, v_sat)),
                v_v_high, vmulq_f32(v_hue_sin, v_sat)
            );

            float32x4_t v_new_v_low = vmlsq_f32(
                vmulq_f32(v_v_low, vmulq_f32(v_hue_cos, v_sat)),
                v_u_low, vmulq_f32(v_hue_sin, v_sat)
            );
            float32x4_t v_new_v_high = vmlsq_f32(
                vmulq_f32(v_v_high, vmulq_f32(v_hue_cos, v_sat)),
                v_u_high, vmulq_f32(v_hue_sin, v_sat)
            );

            // Add back gray
            v_new_u_low = vaddq_f32(v_new_u_low, v_gray);
            v_new_u_high = vaddq_f32(v_new_u_high, v_gray);
            v_new_v_low = vaddq_f32(v_new_v_low, v_gray);
            v_new_v_high = vaddq_f32(v_new_v_high, v_gray);

            // Clamp values
            v_new_u_low = vminq_f32(vmaxq_f32(v_new_u_low, v_min), v_max);
            v_new_u_high = vminq_f32(vmaxq_f32(v_new_u_high, v_min), v_max);
            v_new_v_low = vminq_f32(vmaxq_f32(v_new_v_low, v_min), v_max);
            v_new_v_high = vminq_f32(vmaxq_f32(v_new_v_high, v_min), v_max);

            // Convert back to uint8
            uint8x8_t v_result_u = vqmovn_u16(vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(v_new_u_low)),
                vqmovn_u32(vcvtq_u32_f32(v_new_u_high))
            ));
            uint8x8_t v_result_v = vqmovn_u16(vcombine_u16(
                vqmovn_u32(vcvtq_u32_f32(v_new_v_low)),
                vqmovn_u32(vcvtq_u32_f32(v_new_v_high))
            ));

            // Store results
            vst1_u8(dstp_u + x, v_result_u);
            vst1_u8(dstp_v + x, v_result_v);
        }
#endif
        // Process remaining pixels
        
        for (; x < width; x++) {
            const float u = static_cast<float>(srcp_u[x]) - gray;
            const float v = static_cast<float>(srcp_v[x]) - gray;
            
            float new_u = u * hue_cos * sat + v * hue_sin * sat + gray;
            float new_v = v * hue_cos * sat - u * hue_sin * sat + gray;
            
            dstp_u[x] = static_cast<uint8_t>(CLAMP(new_u, static_cast<float>(chroma_min), static_cast<float>(chroma_max)));
            dstp_v[x] = static_cast<uint8_t>(CLAMP(new_v, static_cast<float>(chroma_min), static_cast<float>(chroma_max)));
        }

        srcp_u += src_stride_u;
        srcp_v += src_stride_v;
        dstp_u += dst_stride_u;
        dstp_v += dst_stride_v;
    }
}

static const VSFrame* VS_CC tweakGetFrame(int n, int activationReason, void* instanceData, void** frameData,
                                        VSFrameContext* frameCtx, VSCore* core, const VSAPI* vsapi) {
    TweakData* d = static_cast<TweakData*>(instanceData);
    
    if (activationReason == arInitial) {
        vsapi->requestFrameFilter(n, d->node, frameCtx);
        return nullptr;
    }
    
    if (activationReason != arAllFramesReady)
        return nullptr;
    
    const VSFrame* src = vsapi->getFrameFilter(n, d->node, frameCtx);
    const VSVideoFormat* fi = vsapi->getVideoFrameFormat(src);
    
    // 如果参数都是默认值，直接返回源帧
    if (fi->colorFamily == cfGray) {
        if (d->bright == 0.0f && d->cont == 1.0f) {
            return src;
        }
    } else if (d->bright == 0.0f && d->cont == 1.0f && d->hue == 0.0f && d->sat == 1.0f) {
        return src;
    }
    
    VSFrame* dst = vsapi->newVideoFrame(fi, vsapi->getFrameWidth(src, 0), 
                                        vsapi->getFrameHeight(src, 0), src, core);
    
    const float hue_rad = d->hue * M_PI / 180.0f;
    const float hue_sin = std::sin(hue_rad);
    const float hue_cos = std::cos(hue_rad);
    
    if (fi->sampleType == stInteger) {
        const int bits = fi->bitsPerSample;
        const int gray = 128 << (bits - 8);
        const int chroma_min = d->coring ? (16 << (bits - 8)) : 0;
        const int chroma_max = d->coring ? (240 << (bits - 8)) : (1 << bits) - 1;
        const int luma_min = d->coring ? (16 << (bits - 8)) : 0;
        const int luma_max = d->coring ? (235 << (bits - 8)) : (1 << bits) - 1;
        
        if (bits == 8) {
            // 处理色度平面
            if (fi->colorFamily != cfGray && (d->hue != 0.0f || d->sat != 1.0f)) {
                const int chroma_width = vsapi->getFrameWidth(src, 1);
                const int chroma_height = vsapi->getFrameHeight(src, 1);
                
                process_chroma_uint8(reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(src, 1)),
                            reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(src, 2)),
                            reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, 1)),
                            reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, 2)),
                            chroma_width, chroma_height,
                            vsapi->getStride(src, 1), vsapi->getStride(src, 2),
                            vsapi->getStride(dst, 1), vsapi->getStride(dst, 2),
                            hue_sin, hue_cos, d->sat,
                            static_cast<uint8_t>(chroma_min),
                            static_cast<uint8_t>(chroma_max),
                            static_cast<uint8_t>(gray));
            } else if (fi->colorFamily != cfGray) {
                // 直接复制色度平面
                const int chroma_width = vsapi->getFrameWidth(src, 1);
                const int chroma_height = vsapi->getFrameHeight(src, 1);
                
                copyPlaneData_uint8(reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(src, 1)),
                            reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, 1)),
                            chroma_width, chroma_height,
                            vsapi->getStride(src, 1),
                            vsapi->getStride(dst, 1));
                            
                copyPlaneData_uint8(reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(src, 2)),
                            reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, 2)),
                            chroma_width, chroma_height,
                            vsapi->getStride(src, 2),
                            vsapi->getStride(dst, 2));
            }
            
            // 处理亮度平面
            if (d->bright != 0.0f || d->cont != 1.0f) {
                process_luma_uint8(reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(src, 0)),
                            reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, 0)),
                            vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0),
                            vsapi->getStride(src, 0), vsapi->getStride(dst, 0),
                            d->cont, d->bright,
                            static_cast<uint8_t>(luma_min),
                            static_cast<uint8_t>(luma_max));
            } else {
                // 直接复制亮度平面
                copyPlaneData_uint8(reinterpret_cast<const uint8_t*>(vsapi->getReadPtr(src, 0)),
                            reinterpret_cast<uint8_t*>(vsapi->getWritePtr(dst, 0)),
                            vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0),
                            vsapi->getStride(src, 0),
                            vsapi->getStride(dst, 0));
            }
        } else {
            // 处理色度平面
            if (fi->colorFamily != cfGray && (d->hue != 0.0f || d->sat != 1.0f)) {
                const int chroma_width = vsapi->getFrameWidth(src, 1);
                const int chroma_height = vsapi->getFrameHeight(src, 1);
                
                process_chroma_uint16(reinterpret_cast<const uint16_t*>(vsapi->getReadPtr(src, 1)),
                            reinterpret_cast<const uint16_t*>(vsapi->getReadPtr(src, 2)),
                            reinterpret_cast<uint16_t*>(vsapi->getWritePtr(dst, 1)),
                            reinterpret_cast<uint16_t*>(vsapi->getWritePtr(dst, 2)),
                            chroma_width, chroma_height,
                            vsapi->getStride(src, 1) / 2, vsapi->getStride(src, 2) / 2,
                            vsapi->getStride(dst, 1) / 2, vsapi->getStride(dst, 2) / 2,
                            hue_sin, hue_cos, d->sat,
                            static_cast<uint16_t>(chroma_min),
                            static_cast<uint16_t>(chroma_max),
                            static_cast<uint16_t>(gray));
            } else if (fi->colorFamily != cfGray) {
                // 直接复制色度平面
                const int chroma_width = vsapi->getFrameWidth(src, 1);
                const int chroma_height = vsapi->getFrameHeight(src, 1);
                
                copyPlaneData_uint16(reinterpret_cast<const uint16_t*>(vsapi->getReadPtr(src, 1)),
                            reinterpret_cast<uint16_t*>(vsapi->getWritePtr(dst, 1)),
                            chroma_width, chroma_height,
                            vsapi->getStride(src, 1) / 2,
                            vsapi->getStride(dst, 1) / 2);
                            
                copyPlaneData_uint16(reinterpret_cast<const uint16_t*>(vsapi->getReadPtr(src, 2)),
                            reinterpret_cast<uint16_t*>(vsapi->getWritePtr(dst, 2)),
                            chroma_width, chroma_height,
                            vsapi->getStride(src, 2) / 2,
                            vsapi->getStride(dst, 2) / 2);
            }
            
            // 处理亮度平面
            if (d->bright != 0.0f || d->cont != 1.0f) {
                process_luma_uint16(reinterpret_cast<const uint16_t*>(vsapi->getReadPtr(src, 0)),
                            reinterpret_cast<uint16_t*>(vsapi->getWritePtr(dst, 0)),
                            vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0),
                            vsapi->getStride(src, 0) / 2, vsapi->getStride(dst, 0) / 2,
                            d->cont, d->bright,
                            static_cast<uint16_t>(luma_min),
                            static_cast<uint16_t>(luma_max));
            } else {
                // 直接复制亮度平面
                copyPlaneData_uint16(reinterpret_cast<const uint16_t*>(vsapi->getReadPtr(src, 0)),
                            reinterpret_cast<uint16_t*>(vsapi->getWritePtr(dst, 0)),
                            vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0),
                            vsapi->getStride(src, 0) / 2,
                            vsapi->getStride(dst, 0) / 2);
            }
        }
    } else {
        // 处理色度平面
        if (fi->colorFamily != cfGray && (d->hue != 0.0f || d->sat != 1.0f)) {
            const int chroma_width = vsapi->getFrameWidth(src, 1);
            const int chroma_height = vsapi->getFrameHeight(src, 1);
            
            process_chroma_float(reinterpret_cast<const float*>(vsapi->getReadPtr(src, 1)),
                        reinterpret_cast<const float*>(vsapi->getReadPtr(src, 2)),
                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1)),
                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2)),
                        chroma_width, chroma_height,
                        vsapi->getStride(src, 1) / sizeof(float),
                        vsapi->getStride(src, 2) / sizeof(float),
                        vsapi->getStride(dst, 1) / sizeof(float),
                        vsapi->getStride(dst, 2) / sizeof(float),
                        hue_sin, hue_cos, d->sat,
                        -0.5f, 0.5f, 0.0f);
        } else if (fi->colorFamily != cfGray) {
            // 直接复制色度平面
            const int chroma_width = vsapi->getFrameWidth(src, 1);
            const int chroma_height = vsapi->getFrameHeight(src, 1);
            
            copyPlaneData_float(reinterpret_cast<const float*>(vsapi->getReadPtr(src, 1)),
                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 1)),
                        chroma_width, chroma_height,
                        vsapi->getStride(src, 1) / sizeof(float),
                        vsapi->getStride(dst, 1) / sizeof(float));
                        
            copyPlaneData_float(reinterpret_cast<const float*>(vsapi->getReadPtr(src, 2)),
                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 2)),
                        chroma_width, chroma_height,
                        vsapi->getStride(src, 2) / sizeof(float),
                        vsapi->getStride(dst, 2) / sizeof(float));
        }
        
        // 处理亮度平面
        if (d->bright != 0.0f || d->cont != 1.0f) {
            process_luma_float(reinterpret_cast<const float*>(vsapi->getReadPtr(src, 0)),
                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0)),
                        vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0),
                        vsapi->getStride(src, 0) / sizeof(float),
                        vsapi->getStride(dst, 0) / sizeof(float),
                        d->cont, d->bright, 0.0f, 1.0f);
        } else {
            // 直接复制亮度平面
            copyPlaneData_float(reinterpret_cast<const float*>(vsapi->getReadPtr(src, 0)),
                        reinterpret_cast<float*>(vsapi->getWritePtr(dst, 0)),
                        vsapi->getFrameWidth(src, 0), vsapi->getFrameHeight(src, 0),
                        vsapi->getStride(src, 0) / sizeof(float),
                        vsapi->getStride(dst, 0) / sizeof(float));
        }
    }
    
    vsapi->freeFrame(src);
    return dst;
}

static void VS_CC tweakFree(void* instanceData, VSCore* core, const VSAPI* vsapi) {
    TweakData* d = static_cast<TweakData*>(instanceData);
    vsapi->freeNode(d->node);
    free(d);
}

static void VS_CC tweakCreate(const VSMap* in, VSMap* out, void* userData, VSCore* core, const VSAPI* vsapi) {
    TweakData d;
    TweakData* data;
    int err;

    d.node = vsapi->mapGetNode(in, "clip", 0, &err);
    const VSVideoInfo* vi = vsapi->getVideoInfo(d.node);

    if (!vsh::isConstantVideoFormat(vi)) {
        vsapi->mapSetError(out, "Tweak: only clips with constant format are accepted");
        vsapi->freeNode(d.node);
        return;
    }

    if (vi->format.colorFamily == cfRGB) {
        vsapi->mapSetError(out, "Tweak: RGB clips are not accepted");
        vsapi->freeNode(d.node);
        return;
    }

    err = 0;
    d.hue = static_cast<float>(vsapi->mapGetFloat(in, "hue", 0, &err));
    if (err)
        d.hue = 0.0f;

    err = 0;
    d.sat = static_cast<float>(vsapi->mapGetFloat(in, "sat", 0, &err));
    if (err)
        d.sat = 1.0f;

    err = 0;
    d.bright = static_cast<float>(vsapi->mapGetFloat(in, "bright", 0, &err));
    if (err)
        d.bright = 0.0f;

    err = 0;
    d.cont = static_cast<float>(vsapi->mapGetFloat(in, "cont", 0, &err));
    if (err)
        d.cont = 1.0f;

    err = 0;
    d.coring = !!vsapi->mapGetInt(in, "coring", 0, &err);
    if (err)
        d.coring = true;

    data = static_cast<TweakData*>(malloc(sizeof(TweakData)));
    if (!data) {
        vsapi->mapSetError(out, "Tweak: malloc failed");
        vsapi->freeNode(d.node);
        return;
    }
    *data = d;

    VSFilterDependency deps[] = {{d.node, rpGeneral}};
    vsapi->createVideoFilter(out, "Tweak", vi, tweakGetFrame, tweakFree, fmParallel, deps, 1, data, core);
}

VS_EXTERNAL_API(void) VapourSynthPluginInit2(VSPlugin* plugin, const VSPLUGINAPI* vspapi) {
    vspapi->configPlugin("com.yuygfgg.adjust", "adjust", "VapourSynth Tweak Filter", VS_MAKE_VERSION(1, 0), VAPOURSYNTH_API_VERSION, 0, plugin);
    vspapi->registerFunction("Tweak",
                            "clip:vnode;"
                            "hue:float:opt;"
                            "sat:float:opt;"
                            "bright:float:opt;"
                            "cont:float:opt;"
                            "coring:int:opt;",
                            "clip:vnode;",
                            tweakCreate, nullptr, plugin);
}