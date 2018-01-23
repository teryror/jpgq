/*******************************************************************************
Author: Tristan Dannenberg
Notice: No warranty is offered or implied; use this code at your own risk.
********************************************************************************
TODOs:
- Color science experiments
  - Linearize input image colors
  - Try converting to different color space
  - Try different color distance metric altogether
- Improve the way the final pixel values are determined
  - Down-scale the image after quantization
  - Add dithering to the output image
  - Try using edge detection if dithering is too hard to control  
- Improve the way the color palette is calculated
  - Try different initialization schemes to replace the random starting values,
    e.g. a lattice structure.
  - Try different perceived-image-difference metrics, e.g. SSIM
- Speed this thing up; I want to do (at least) Full HD input images and a
  256-color palette, running as many iterations as necessary to early-exit,
  and I _never_ want to wait more than a couple seconds
  - write SIMD variations of the most expensive functions (presumably starting
    with the loop finding the closest palette color, followed by the averaging
  - possibly multi-threading, though the need for synchronization may kill us
  - (optionally) shrink the search space (e.g. 15-bit RGB color palette)
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

#undef assert
#define assert(E) do {     \
    if (!(E)) *((int *)0); \
} while (0);

#define mfree(P) do { \
    assert(P);        \
    free(P); P = 0;   \
} while (0);

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
    int32_t ra = (int32_t)(a >> 16);
    int32_t rb = (int32_t)(b >> 16);
    int32_t delta_r = ra - rb;
    
    int32_t ga = (int32_t)((a >> 8) & 0xFF);
    int32_t gb = (int32_t)((b >> 8) & 0xFF);
    int32_t delta_g = ra - rb;
    
    int32_t ba = (int32_t)(a & 0xFF);
    int32_t bb = (int32_t)(b & 0xFF);
    int32_t delta_b = ba - bb;
    
    float result = sqrt(delta_r * delta_r +
                        delta_g * delta_g +
                        delta_b * delta_b);
    
    return result;
}

static int color_cmp(const void * a, const void * b) {
    // NOTE: We don't care about the exact ordering semantics, we just want all
    // identical values to be adjacent in the array, so any consistent comparison
    // function will do.
    
    return ( *(int32_t*)a - *(int32_t*)b );
}

static inline uint32_t img_read(unsigned char * img_data, uint32_t pixel_idx) {
    uint32_t result = 0;
    img_data += pixel_idx * 3;
    result |= img_data[0] << 16;
    result |= img_data[1] << 8;
    result |= img_data[2];
    return result;
}


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
    
    // NOTE: We may want a second copy of the converted values that remains
    // in-order, for deciding the final pixel values more quickly
    
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
    
    for (int l = 0; l < BEST_PAL_OF; ++l) {
        printf(">>> Generating Palette %d ...\t", l);
        uint32_t palette[PALETTE_SIZE];
        
        // ------------------------------------------------------------------------
        TimeStamp init_start_time = start_timer();
        
#if 1
        // Choose fully random starting values
        for (int j = 0; j < PALETTE_SIZE; ++j) {
            uint32_t nr = next_rand(&rng);
            palette[j] = nr & 0x00FFFFFF;
        }
#else
        // Uniformly draw PALETTE_SIZE samples from unique_colors without
        // replacement using Knuth's algorithm (TAOCP Vol.2, I think?)
        // Note that we're not using the histogram to adjust the weighting.
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
        }
        
        assert(palette_colors_drawn == PALETTE_SIZE);
#endif
        
        ElapsedTime init_time_elapsed = end_timer(init_start_time);
        // ------------------------------------------------------------------------
        int steps = 0;
        double best_step_time = DBL_MAX;
        
        for (; steps < ITERATIONS; ++steps) {
            TimeStamp iter_start_time = start_timer();
            
            // Find closest palette color for each unique color:
            for (int i = 0; i < unique_colors_count; ++i) {
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
            
            for (int i = 0; i < unique_colors_count; ++i) {
                unsigned char pal_idx = pal_idxs[i];
                
                uint32_t col_r = unique_colors[i] >> 16;
                sums_r[pal_idx] += col_r * color_counts[i];
                
                uint32_t col_g = (unique_colors[i] >> 8) & 0xFF;
                sums_g[pal_idx] += col_g * color_counts[i];
                
                uint32_t col_b = unique_colors[i] & 0xFF;
                sums_b[pal_idx] += col_b * color_counts[i];
                
                counts[pal_idx] += color_counts[i];
            }
            
            int updated = 0;
            for (int j = 0; j < PALETTE_SIZE; ++j) {
                assert(counts[j]);
                
                int32_t avg_r = (int32_t) round((float)sums_r[j] / (float)counts[j]);
                int32_t avg_g = (int32_t) round((float)sums_g[j] / (float)counts[j]);
                int32_t avg_b = (int32_t) round((float)sums_b[j] / (float)counts[j]);
                
                assert(avg_r > 0 && avg_r < 256);
                assert(avg_g > 0 && avg_g < 256);
                assert(avg_b > 0 && avg_b < 256);
                
                uint32_t palette_old = palette[j];
                
                palette[j] = 0;
                palette[j] |= (uint32_t)avg_r << 16;
                palette[j] |= (uint32_t)avg_g << 8;
                palette[j] |= (uint32_t)avg_b;
                
                if (palette[j] != palette_old) {
                    updated = 1;
                }
            }
            
            ElapsedTime iter_time_elapsed = end_timer(iter_start_time);
            if (iter_time_elapsed.milliseconds < best_step_time) {
                best_step_time = iter_time_elapsed.milliseconds;
            }
            
            if (!updated) break;
        }
        
        printf("Converged after %d steps.\t", steps);
        // ------------------------------------------------------------------------
        TimeStamp psnr_start_time = start_timer();
        
        int64_t square_error_sum = 0;
        for (int i = 0; i < unique_colors_count; ++i) {
            unsigned char pal_idx = pal_idxs[i];
            uint32_t pal_color = palette[pal_idx];
            uint32_t tru_color = unique_colors[i];
            
            int32_t pal_r = pal_color >> 16;
            int32_t tru_r = tru_color >> 16;
            int32_t dr = pal_r - tru_r;
            
            int32_t pal_g = (pal_color >> 8) & 0xFF;
            int32_t tru_g = (tru_color >> 8) & 0xFF;
            int32_t dg = pal_g - tru_g;
            
            int32_t pal_b = pal_color & 0xFF;
            int32_t tru_b = tru_color & 0xFF;
            int32_t db = pal_b - tru_b;
            
            int64_t square_error = dr * dr + dg * dg + db * db;
            square_error_sum += square_error * color_counts[i];
        }
        
        double mean_square_error = (double)square_error_sum / (double)(img_size * 3);
        double lg10_255_x_20 = 48.130803608679103412429178057179;
        double peak_signal_noise_ratio = lg10_255_x_20 - 10*log10(mean_square_error);
        
        if (peak_signal_noise_ratio > best_psnr) {
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
    
    // TEMPORARY: Overwrite img_data with paletized image data
    for (int i = 0; i < img_size; ++i) {
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
        offset[0] = (unsigned char)(best_palette[pal_idx] >> 16);
        offset[1] = (unsigned char)((best_palette[pal_idx] >> 8) & 0xFF);
        offset[2] = (unsigned char)(best_palette[pal_idx] & 0xFF);
    }
    
    stbi_write_bmp("out3.bmp", width, height, channel_count, img_data);
    
    ElapsedTime write_time_elapsed = end_timer(write_start_time);
    printf("Completed in %f ms (%lld cycles)\n",
           write_time_elapsed.milliseconds,
           write_time_elapsed.cycles);
    
    printf("\nDONE.\n");
    return 0;
}