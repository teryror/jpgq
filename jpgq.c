/*******************************************************************************
Author: Tristan Dannenberg
Notice: No warranty is offered or implied; use this code at your own risk.
********************************************************************************
TODOs:
- Color science experiments
  - Try different color distance metric altogether
- Improve the way the final pixel values are determined
  - Down-scale the image after quantization
  - Add dithering to the output image
  - Try using edge detection if dithering is too hard to control  
- Improve the way the color palette is calculated
  - Mess with the color_counts so that small regions with distinct colors are
    not hurt so much (start with sqrt on all values?)
  - Try different perceived-image-difference metrics, e.g. SSIM, __or__
    just use the color_dist function we already have lying around???
- Speed this thing up; I want to do (at least) Full HD input images and a
  256-color palette, running as many iterations as necessary to early-exit,
  and I _never_ want to wait more than a couple seconds
  - possibly multi-threading, though the need for synchronization may kill us
  - (optionally) shrink the search space (e.g. 15-bit RGB color palette)
    => This actually takes _longer_ to converge. We may still want to do it for
       stylistic purposes, though.
- Improve the CLI so that no recompilation is necessary for changing an
  effect parameter, turning features on/off, etc.
*******************************************************************************/

#include "stdint.h"
#include "float.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PALETTE_SIZE  32
#define BEST_PAL_OF    8
#define ITERATIONS   128
#define CONVERGENCE_THRESHOLD 50.0f
#define YCBCR
#define SIMD_OPT

#undef assert
#define assert(E) do {         \
    if (!(E)) *((int *)0) = 0; \
} while (0)

#define clamp(min, v, max) do { \
    if (v < (min)) {            \
        v = (min);              \
    } else if (v > (max)) {     \
        v = (max);              \
    }                           \
} while (0)

#define mfree(P) do { \
    assert(P);        \
    free(P); P = 0;   \
} while (0)

#include "windows.h"

typedef struct {
    LARGE_INTEGER perf_counter;
    int64_t       cycle_count;
} TimeStamp;

typedef struct {
    double  milliseconds;
    int64_t cycles;
} ElapsedTime;

// NOTE: When we parallelize, we need one of these per thread to get correct timing values
static double perf_counter_freq = 0;

static inline void init_time() {
    LARGE_INTEGER counter_frequency;
    QueryPerformanceFrequency(&counter_frequency);
    
    perf_counter_freq = (double)(counter_frequency.QuadPart) / 1000.0;
}

static inline TimeStamp start_timer() {
    TimeStamp result;
    
    QueryPerformanceCounter(&result.perf_counter);
    result.cycle_count = __rdtsc();
    
    return result;
}

static inline ElapsedTime end_timer(TimeStamp start) {
    int64_t end_cycles = __rdtsc();
    
    LARGE_INTEGER end_counter;
    QueryPerformanceCounter(&end_counter);
    
    ElapsedTime result;
    result.cycles = end_cycles - start.cycle_count;
    result.milliseconds = (double)(end_counter.QuadPart - start.perf_counter.QuadPart) / perf_counter_freq;
    
    return result;
}

typedef uint64_t Random;
static inline uint32_t next_rand(Random * rng) {
    // This code is stolen from http://www.pcg-random.org/download.html
    
    // *Really* minimal PCG32 code / (c) 2014 M.E. O'Neill / pcg-random.org
    // Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)
    
    uint64_t oldstate = *rng;
    *rng = oldstate * 0x5851F42D4C957F2DULL + (0xFF34C1F839BE198FULL);
    
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rotation   = oldstate >> 59u;
    return  (xorshifted >> rotation) | (xorshifted << ((-rotation) & 31));
}

static inline float color_dist(uint32_t a, uint32_t b) {
    float a_y = (float)(a >> 20);
    float b_y = (float)(b >> 20);
    float delta_y = a_y - b_y; delta_y *= delta_y;
    
    float a_Cb = (float)((a >> 10) & 0x3FF);
    float b_Cb = (float)((b >> 10) & 0x3FF);
    float delta_Cb = a_Cb - b_Cb; delta_Cb *= delta_Cb;
    
    float a_Cr = (float)(a & 0x3FF);
    float b_Cr = (float)(b & 0x3FF);
    float delta_Cr = a_Cr - b_Cr; delta_Cr *= delta_Cr;
    
    float result = delta_y + delta_Cb + delta_Cr;
    if (a == b) assert(result == 0);
    
    return result;
}

static int color_cmp(const void * a, const void * b) {
    // NOTE: We don't care about the exact ordering semantics, we just want all
    // identical values to be adjacent in the array, so any consistent comparison
    // function will do.
    
    return ( *(int32_t*)a - *(int32_t*)b );
}

static inline uint32_t img_read(unsigned char * img_data, uint32_t pixel_idx) {
    img_data += pixel_idx * 3;
    
    // Linearize input (TODO: Make gamma configurable)
    float R = powf((float)img_data[0] / 255.0f, 0.45f) * 1023.0f;
    float G = powf((float)img_data[1] / 255.0f, 0.45f) * 1023.0f;
    float B = powf((float)img_data[2] / 255.0f, 0.45f) * 1023.0f;
    
    assert(R >= 0 && R <= 1023.0f);
    assert(G >= 0 && G <= 1023.0f);
    assert(B >= 0 && B <= 1023.0f);
    
#ifndef YCBCR
    uint32_t r = (uint32_t) round(R); assert(r < 0x400);
    uint32_t g = (uint32_t) round(G); assert(g < 0x400);
    uint32_t b = (uint32_t) round(B); assert(b < 0x400);
    
    uint32_t result = 0;
    result |= r << 20;
    result |= g << 10;
    result |= b;
#else
    // Convert to 10-bit-per-channel YCbCr
    float Y = 0.299f*R + 0.587f*G + 0.114f*B;
    clamp(0, Y, 1023.0f);
    
    float Cb = 512 - 0.168735892f*R - 0.331264108f*G + 0.5f*B;
    clamp(0, Cb, 1023.0f);
    
    float Cr = 512 + 0.5f*R - 0.418687589f*G - 0.081312411f*B;
    clamp(0, Cr, 1023.0f);
    
    // Pack into 30 bits
    uint32_t y  = (uint32_t) round(Y);  assert(y  < 0x400);
    uint32_t cb = (uint32_t) round(Cb); assert(cb < 0x400);
    uint32_t cr = (uint32_t) round(Cr); assert(cr < 0x400);
    
    uint32_t result = 0;
    result |= y << 20;
    result |= cb << 10;
    result |= cr;
#endif
    
    return result;
}

#ifdef SIMD_OPT
#include <emmintrin.h>

static inline void
simd_find_closest_color(uint32_t * colors,
                        uint32_t * palette,
                        size_t palette_size,
                        unsigned char * pal_idxs)
{
    // Load 4 colors and unpack their components into separate vectors
    // (so we have (Y0 Y1 Y2 Y3), (Cb0 Cb1 Cb2 Cb3), (Cr0 Cr1 Cr2 Cr3))
    __m128i ucolors = _mm_load_si128((__m128i const *)colors);
    __m128i ten_bit_mask = _mm_set1_epi32(0x3FF);
    
    __m128 uclrs_y = _mm_cvtepi32_ps(_mm_srli_epi32(ucolors, 20));
    __m128 uclrs_cb = _mm_cvtepi32_ps(
        _mm_and_si128(_mm_srli_epi32(ucolors, 10), ten_bit_mask));
    __m128 uclrs_cr = _mm_cvtepi32_ps(
        _mm_and_si128(ucolors, ten_bit_mask));
    
    // The actual search for palette colors with minimal distances:
    __m128i min_idxs = _mm_set1_epi32(0);
    __m128 min_dsts = _mm_set_ps1(FLT_MAX);
    for (int j = 0; j < palette_size; ++j) {
        uint32_t pal_color = palette[j];
        
        __m128 p_y = _mm_set_ps1((float)(pal_color >> 20));
        __m128 p_cb = _mm_set_ps1((float)((pal_color >> 10) & 0x3FF));
        __m128 p_cr = _mm_set_ps1((float)(pal_color & 0x3FF));
        
        __m128 delta_y = _mm_sub_ps(uclrs_y, p_y);
        delta_y = _mm_mul_ps(delta_y, delta_y);
        
        __m128 delta_cb = _mm_sub_ps(uclrs_cb, p_cb);
        delta_cb = _mm_mul_ps(delta_cb, delta_cb);
        
        __m128 delta_cr = _mm_sub_ps(uclrs_cr, p_cr);
        delta_cr = _mm_mul_ps(delta_cr, delta_cr);
        
        // NOTE: Since we're only comparing distances (not doing any
        // arithmetic), and sqrt is monotonically increasing,
        // skipping it should not change results.
        
        __m128 dsts = _mm_add_ps(delta_y, _mm_add_ps(delta_cb, delta_cr));
        __m128 mask = _mm_cmplt_ps(dsts, min_dsts);
        
        dsts = _mm_and_ps(mask, dsts);
        min_dsts = _mm_andnot_ps(mask, min_dsts);
        min_dsts = _mm_or_ps(min_dsts, dsts);
        
        min_idxs = _mm_andnot_si128(_mm_castps_si128(mask), min_idxs);
        min_idxs = _mm_or_si128(min_idxs, _mm_and_si128(
            _mm_castps_si128(mask), _mm_set1_epi32(j)));
    }
    
    pal_idxs[0] = (unsigned char)_mm_extract_epi16(min_idxs, 0);
    pal_idxs[1] = (unsigned char)_mm_extract_epi16(min_idxs, 2);
    pal_idxs[2] = (unsigned char)_mm_extract_epi16(min_idxs, 4);
    pal_idxs[3] = (unsigned char)_mm_extract_epi16(min_idxs, 6);
}
#endif

int main(int argc, char ** argv) {
    init_time();
    
    if (argc < 2) {
        printf("jpgq: You need to provide an image file as an argument.");
        return -1;
    }
    
    char *filename = argv[1];
    
    int width, height, channel_count;
    unsigned char *img_data = stbi_load(filename, &width, &height, &channel_count, 0);
    
    int img_size = width * height;
    
    if (img_data == 0) {
        printf("jpgq: An error occured while reading the file '%s'!\n", filename);
        return -2;
    } else if (channel_count != 3) {
        printf("jpgq: Only images with exactly 3 color channels are accepted!\n");
        return -3;
    }
    
    // ------------------------------------------------------------------------
    printf(">>> Converting Between Color Spaces ...\n");
    TimeStamp conv_start_time = start_timer();
    
    uint32_t * px_values = (uint32_t *) malloc(sizeof(uint32_t) * img_size);
    uint32_t * img_padded = (uint32_t *) malloc(sizeof(uint32_t) * img_size);
    for (int i = 0; i < img_size; ++i) {
        uint32_t px = img_read(img_data, i);
        px_values[i] = px;
        img_padded[i] = px;
    }
    
    ElapsedTime conv_time_elapsed = end_timer(conv_start_time);
    printf("Completed in %f ms (%lld cycles)\n",
           conv_time_elapsed.milliseconds,
           conv_time_elapsed.cycles);
    // ------------------------------------------------------------------------
    printf(">>> Histogramming Pixel Values ...\n");
    TimeStamp hist_start_time = start_timer();
    
    size_t arr_size = sizeof(uint32_t) << 24;
    uint32_t * unique_colors = (uint32_t *) malloc(arr_size);
    uint32_t * color_counts  = (uint32_t *) malloc(arr_size);
    size_t unique_colors_count = 0;
    
    qsort(px_values, img_size, sizeof(px_values[0]), color_cmp);
    
    uint32_t current_val = px_values[0];
    uint32_t current_count = 1;
    
    for (int i = 1; i < img_size; ++i) {
        if (px_values[i] == current_val) {
            current_count += 1;
        } else {
            unique_colors[unique_colors_count] = current_val;
            color_counts[unique_colors_count] = current_count;
            unique_colors_count += 1;
            
            current_val = px_values[i];
            current_count = 1;
        }
    }
    
    // The loop above does not count the final run, so we do that here
    unique_colors[unique_colors_count] = current_val;
    color_counts[unique_colors_count] = current_count;
    unique_colors_count += 1;
    
    // We have the histogram now, so we don't need the sorted copy anymore
    mfree(px_values);
    
    uint64_t test_sum = 0;
    for (int i = 0; i < unique_colors_count; ++i) {
        test_sum += color_counts[i];
    }
    assert(test_sum == img_size);
    
    unsigned char *pal_idxs = (char *) malloc(unique_colors_count);
    
    ElapsedTime hist_time_elapsed = end_timer(hist_start_time);
    printf("Completed in %f ms (%lld cycles). Counted %zd unique colors in %d pixels.\n",
           hist_time_elapsed.milliseconds, hist_time_elapsed.cycles,
           unique_colors_count, img_size);
    
    // 
    // Begin Clustering
    // 
    
    Random rng = 0x9709FDD653807BFEULL;
    uint32_t best_palette[PALETTE_SIZE];
    double best_psnr = 0;
    
    double total_step_time = 0;
    double total_assign_time = 0;
    double total_accumulate_time = 0;
    double total_divide_time = 0;
    
    for (int l = 0; l < BEST_PAL_OF; ++l) {
        printf(">>> Generating Palette %d ...\t", l);
        uint32_t palette[PALETTE_SIZE];
        
        // ------------------------------------------------------------------------
        TimeStamp init_start_time = start_timer();
        
        // Uniformly draw PALETTE_SIZE samples from unique_colors without
        // replacement using Knuth's algorithm (TAOCP Vol.2, I think?)
        // 
        // This is necessary because, with entirely random starting values,
        // you often get empty clusters, which the algorithm as implemented
        // below can't deal with; the palette size effectively gets smaller,
        // and to minimize errors (in YCbCr), converges on mostly gray-scale.
        int palette_colors_drawn = 0;
        
        for (int i = 0; i < unique_colors_count &&
             palette_colors_drawn < PALETTE_SIZE; ++i)
        {
            // NOTE: This is not _truly_ uniform, but probably close enough
            uint32_t nr = next_rand(&rng) % (unique_colors_count - i);
            
            if (nr < (PALETTE_SIZE - palette_colors_drawn)) {
                palette[palette_colors_drawn] = unique_colors[i];
                palette_colors_drawn += 1;
            }
            palette[palette_colors_drawn] = nr & 0x3FFFFFFF;
        }
        
        assert(palette_colors_drawn == PALETTE_SIZE);
        
        ElapsedTime init_time_elapsed = end_timer(init_start_time);
        // ------------------------------------------------------------------------
        int steps = 0;
        double best_step_time = DBL_MAX;
        
        for (; steps < ITERATIONS; ++steps) {
            TimeStamp iter_start_time = start_timer();
            
            // Find closest palette color for each unique color:
            TimeStamp pal_assign_start_time = start_timer();
            int i = 0;
#ifdef SIMD_OPT
            assert(sizeof(uint64_t) == sizeof(uint32_t *));
            assert((uint64_t)unique_colors % 16 == 0);
            
            for (; i < (unique_colors_count & ~3llu); i += 4) {
                simd_find_closest_color(&unique_colors[i],
                                        palette, PALETTE_SIZE,
                                        &pal_idxs[i]);
            }
#endif
            for (; i < unique_colors_count; ++i) {
                unsigned char min_idx = 0;
                float min_dst = color_dist(unique_colors[i], palette[0]);
                
                for (int j = 1; j < PALETTE_SIZE; ++j) {
                    float dst = color_dist(unique_colors[i], palette[j]);
                    if (dst < min_dst) {
                        min_idx = (unsigned char)j;
                        min_dst = dst;
                    }
                }
                
                pal_idxs[i] = min_idx;
            }
            ElapsedTime pal_assign_time_elapsed = end_timer(
                pal_assign_start_time);
            
            // Adjust color palette:
            uint64_t counts[PALETTE_SIZE];
            uint64_t sums_r[PALETTE_SIZE];
            uint64_t sums_g[PALETTE_SIZE];
            uint64_t sums_b[PALETTE_SIZE];
            
            for (int j = 0; j < PALETTE_SIZE; ++j) {
                sums_r[j] = 0;
                sums_g[j] = 0;
                sums_b[j] = 0;
                counts[j] = 0;
            }
            
            TimeStamp pal_accumulate_start_time = start_timer();
            for (int i = 0; i < unique_colors_count; ++i) {
                unsigned char pal_idx = pal_idxs[i];
                
                uint32_t col_r = unique_colors[i] >> 20;
                sums_r[pal_idx] += col_r * color_counts[i];
                
                uint32_t col_g = (unique_colors[i] >> 10) & 0x3FF;
                sums_g[pal_idx] += col_g * color_counts[i];
                
                uint32_t col_b = unique_colors[i] & 0x3FF;
                sums_b[pal_idx] += col_b * color_counts[i];
                
                counts[pal_idx] += color_counts[i];
            }
            ElapsedTime pal_accumulate_time_elapsed = end_timer(
                pal_accumulate_start_time);
            
            TimeStamp pal_divide_start_time = start_timer();
            int updated = 0;
            for (int j = 0; j < PALETTE_SIZE; ++j) {
                // TODO: This could produce some palette colors that are never used,
                // which is wasteful. Maybe try selecting two other colors randomly
                // and use their mid-point or something?
                if (counts[j] == 0) continue;
                
                int32_t avg_r = (int32_t) round((double)sums_r[j] / (double)counts[j]);
                int32_t avg_g = (int32_t) round((double)sums_g[j] / (double)counts[j]);
                int32_t avg_b = (int32_t) round((double)sums_b[j] / (double)counts[j]);
                
                assert(avg_r >= 0 && avg_r < 1024);
                assert(avg_g >= 0 && avg_g < 1024);
                assert(avg_b >= 0 && avg_b < 1024);
                
                uint32_t palette_old = palette[j];
                
                palette[j] = 0;
                palette[j] |= (uint32_t)avg_r << 20;
                palette[j] |= (uint32_t)avg_g << 10;
                palette[j] |= (uint32_t)avg_b;
                
                if (color_dist(palette[j], palette_old) > CONVERGENCE_THRESHOLD) {
                    updated = 1;
                }
            }
            ElapsedTime pal_divide_time_elapsed = end_timer(
                pal_divide_start_time);
            
            ElapsedTime iter_time_elapsed = end_timer(iter_start_time);
            if (iter_time_elapsed.milliseconds < best_step_time) {
                best_step_time = iter_time_elapsed.milliseconds;
            }
            
            total_step_time += iter_time_elapsed.milliseconds;
            total_assign_time += pal_assign_time_elapsed.milliseconds;
            total_accumulate_time += pal_accumulate_time_elapsed.milliseconds;
            total_divide_time += pal_divide_time_elapsed.milliseconds;
            
            if (!updated) break;
        }
        
        printf("Converged after %d steps.\t", steps);
        // ------------------------------------------------------------------------
        TimeStamp psnr_start_time = start_timer();
        
        uint64_t square_error_sum = 0;
        for (int i = 0; i < unique_colors_count; ++i) {
            unsigned char pal_idx = pal_idxs[i];
            uint32_t pal_color = palette[pal_idx];
            uint32_t tru_color = unique_colors[i];
            
            int64_t pal_0 = pal_color >> 20;
            int64_t tru_0 = tru_color >> 20;
            int64_t d0 = (pal_0 - tru_0);
            
            int64_t pal_1 = (pal_color >> 10) & 0x3FF;
            int64_t tru_1 = (tru_color >> 10) & 0x3FF;
            int64_t d1 = (pal_1 - tru_1);
            
            int64_t pal_2 = pal_color & 0x3FF;
            int64_t tru_2 = tru_color & 0x3FF;
            int64_t d2 = (pal_2 - tru_2);
            
            uint64_t square_error = (uint64_t)(d0 * d0 + d1 * d1 + d2 * d2);
            square_error_sum += square_error * color_counts[i];
        }
        
        double mean_square_error = (double)square_error_sum / (double)(img_size * 3);
        double lg10_1023_x_20 = 20*log10(1023);
        double peak_signal_noise_ratio = lg10_1023_x_20 - 10*log10(mean_square_error);
        
        // TEMPORARY HACK: the check for 0 lets us copy palettes without having
        // to debug the PSNR=NaN issue, but that needs to be fixed, too!
        if (l == 0 || peak_signal_noise_ratio > best_psnr) {
            best_psnr = peak_signal_noise_ratio;
            for (int j = 0; j < PALETTE_SIZE; ++j) {
                best_palette[j] = palette[j];
            }
        }
        
        ElapsedTime psnr_time_elapsed = end_timer(psnr_start_time);
        
        printf("PSNR = %f db.\n", peak_signal_noise_ratio);
        printf("    Init Time: %f ms\tBest Step Time: %f ms\tPSNR Time: %f ms\n",
               init_time_elapsed.milliseconds,
               best_step_time,
               psnr_time_elapsed.milliseconds);
    }
    // ------------------------------------------------------------------------
    printf(">>> Deciding Final Pixel Values ...\n");
    TimeStamp write_start_time = start_timer();
    
    // Convert color palette back to 8-bit RGB for fast writing
    uint32_t rgb_palette[PALETTE_SIZE];
    for (int j = 0; j < PALETTE_SIZE; ++j) {
#ifndef YCBCR
        uint32_t RGB = best_palette[j];
        float R = (float)(RGB >> 20);
        float G = (float)((RGB >> 10) & 0x3FF);
        float B = (float)(RGB & 0x3FF);
#else
        uint32_t YCbCr = best_palette[j];
        float Y  = (float)(YCbCr >> 20);
        float Cb = (float)((YCbCr >> 10) & 0x3FF);
        float Cr = (float)(YCbCr & 0x3FF);
        
        float R = Y                          + 1.402000f * (Cr - 512);
        float G = Y - 0.344136f * (Cb - 512) - 0.714136f * (Cr - 512);
        float B = Y + 1.772000f * (Cb - 512);
        
        clamp(0, R, 1023.0f);
        clamp(0, G, 1023.0f);
        clamp(0, B, 1023.0f);
#endif
        uint32_t r = (uint32_t) round(powf(R / 1023.0f, 2.2f) * 255.0f);
        uint32_t g = (uint32_t) round(powf(G / 1023.0f, 2.2f) * 255.0f);
        uint32_t b = (uint32_t) round(powf(B / 1023.0f, 2.2f) * 255.0f);
        assert(r < 0x100); assert(g < 0x100); assert(b < 0x100);
        
        uint32_t rgb = 0;
        rgb |= r << 16;
        rgb |= g << 8;
        rgb |= b;
        
        rgb_palette[j] = rgb;
    }
    
    // TEMPORARY: Overwrite img_data with paletized image data
    int i = 0;
#ifdef SIMD_OPT
    for (; i < (img_size & ~3llu); i += 4) {
        unsigned char pal_idxs[4];
        simd_find_closest_color(
            img_padded + i, best_palette, PALETTE_SIZE, pal_idxs);
        
        for (int j = 0; j < 4; ++j) {
            unsigned char * offset = img_data + ((i + j) * 3);
            offset[0] = (unsigned char)(rgb_palette[pal_idxs[j]] >> 16);
            offset[1] = (unsigned char)((rgb_palette[pal_idxs[j]] >> 8) & 0xFF);
            offset[2] = (unsigned char)(rgb_palette[pal_idxs[j]] & 0xFF);
        }
    }
#endif
    for (; i < img_size; ++i) {
        // Find closest palette color for current pixel:
        unsigned char pal_idx = 0;
        float min_dst = color_dist(img_padded[i], best_palette[0]);
        
        for (int j = 1; j < PALETTE_SIZE; ++j) {
            float dst = color_dist(img_padded[i], best_palette[j]);
            if (dst < min_dst) {
                pal_idx = (unsigned char)j;
                min_dst = dst;
            }
        }
        
        unsigned char * offset = img_data + (i * 3);
        offset[0] = (unsigned char)(rgb_palette[pal_idx] >> 16);
        offset[1] = (unsigned char)((rgb_palette[pal_idx] >> 8) & 0xFF);
        offset[2] = (unsigned char)(rgb_palette[pal_idx] & 0xFF);
    }
    
    stbi_write_bmp("out3.bmp", width, height, channel_count, img_data);
    
    ElapsedTime write_time_elapsed = end_timer(write_start_time);
    printf("Completed in %f ms (%lld cycles)\n",
           write_time_elapsed.milliseconds,
           write_time_elapsed.cycles);
    
    // NOTE: Accumulating timings in doubles is super-bad practice, but since I
    // suspected total_assign_time to be the biggest by far, this was accurate
    // enough to test that hypothesis.
    
    printf("\nTotal Step Time: %f ms\n", total_step_time);
    printf(">   Assigning Palettes:\t%f ms (%f %%)\n",
           total_assign_time, total_assign_time / total_step_time * 100.0);
    printf(">   Accumulating:\t%f ms (%f %%)\n",
           total_accumulate_time, total_accumulate_time / total_step_time * 100.0);
    printf(">   Dividing:\t\t%f ms (%f %%)\n",
           total_divide_time, total_divide_time / total_step_time * 100.0);
    
    printf("\nDONE.\n");
    return 0;
}