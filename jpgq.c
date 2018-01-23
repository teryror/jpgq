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
  - Down-scale the image down after quantization
  - Add dithering to the output image
  - Try using edge detection if dithering is too hard to control
- Speed this thing up; I want to do (at least) Full HD input images and a
  256-color palette, running as many iterations as necessary to early-exit,
  and I _never_ want to wait more than a couple seconds
  - measure time per iteration, as well as perf of individual steps 
  - do the equivalent of a sort + RLE on the input image to speed up the
    palette adjustment step
  - write SIMD variations of the most expensive functions (presumably starting
    with the loop finding the closest palette color, followed by the averaging
  - possibly multi-threading, though the need for synchronization may kill us
  - (optionally) shrink the search space (e.g. 15-bit RGB color palette)
- Improve the CLI so that no recompilation is necessary for changing an
  effect parameter, turning features on/off, etc.
- Try different initialization schemes to replace the random starting values,
  e.g. a lattice structure, or randomly selected, unique samples from the image.
*******************************************************************************/

#include "stdint.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_LINEAR
#define STBI_NO_HDR
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define PALETTE_SIZE  32
#define ITERATIONS   128

#undef assert
#define assert(E) do {     \
    if (!(E)) *((int *)0); \
} while (0);

typedef uint64_t Random;
static inline uint32_t next_rand(Random * rng) {
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

static inline uint32_t img_read(unsigned char * img_data, uint32_t pixel_idx) {
    uint32_t result = 0;
    img_data += pixel_idx * 3;
    result |= img_data[0] << 16;
    result |= img_data[1] << 8;
    result |= img_data[2];
    return result;
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        printf("jpgq: You need to provide an image file as an argument.");
        return -1;
    }
    
    char *filename = argv[1];
    
    int width, height, channel_count;
    unsigned char *img_data = stbi_load(filename, &width, &height, &channel_count, 0);
    
    int img_size = width * height;
    unsigned char *pal_idxs = (char *) malloc(img_size);
    
    if (img_data == 0) {
        printf("jpgq: An error occured while reading the file '%s'!\n", filename);
        return -2;
    } else if (channel_count != 3) {
        printf("jpgq: Only images with exactly 3 color channels are accepted!\n");
        return -3;
    }
    
    // Initialize color palette to random colors:
    Random rng = 0x9709FDD653807BFEULL;
    
    uint32_t palette[PALETTE_SIZE];
    for (int j = 0; j < PALETTE_SIZE; ++j) {
        uint32_t nr = next_rand(&rng);
        palette[j] = nr & 0x00FFFFFF;
    }
    
    for (int k = 0; k < ITERATIONS; ++k) {
        printf(">>>\tIteration %d...\n", k + 1);
        
        // Find closest palette color for each pixel:
        for (int i = 0; i < img_size; ++i) {
            unsigned char min_idx = 0;
            float min_dst = color_dist(img_read(img_data, i), palette[0]);
            
            for (int j = 1; j < PALETTE_SIZE; ++j) {
                float dst = color_dist(img_read(img_data, i), palette[j]);
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
        
        for (int i = 0; i < img_size; ++i) {
            unsigned char pal_idx = pal_idxs[i];
            sums_r[pal_idx] += img_data[3*i + 0];
            sums_g[pal_idx] += img_data[3*i + 1];
            sums_b[pal_idx] += img_data[3*i + 2];
            counts[pal_idx] += 1;
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
        
        if (!updated) break;
    }
    
    // TEMPORARY: Overwrite img_data with paletized image data
    // TODO: We really want to scale down the image at some point in the process...
    for (int i = 0; i < img_size; ++i) {
        unsigned char pal_idx = pal_idxs[i];
        unsigned char * offset = img_data + (i * 3);
        offset[0] = (unsigned char)(palette[pal_idx] >> 16);
        offset[1] = (unsigned char)((palette[pal_idx] >> 8) & 0xFF);
        offset[2] = (unsigned char)(palette[pal_idx] & 0xFF);
    }
    
    stbi_write_bmp("out2.bmp", width, height, channel_count, img_data);
    
    printf("\nDONE.\n");
    return 0;
}