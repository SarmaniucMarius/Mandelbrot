#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include <SDL2/SDL.h>
#include <SDL_TTF.h>

#define ARRAY_COUNT(a) (sizeof(a) / sizeof(a[0]))

#define SCREEN_WIDTH  1024 // 640
#define SCREEN_HEIGHT 728  // 480

#define MANDELBROT_SIMD_FLOAT  1
#define MANDELBROT_SIMD_DOUBLE 0
#define MANDELBROT_PRECISION 256

struct Job_Data
{
	SDL_Surface *surface;
	int tile_start_x, tile_start_y; 
	int tile_end_x, tile_end_y;
	float x_min, x_max, y_min, y_max;
};

SDL_atomic_t current_job = {};
Job_Data jobs[32] = {};

SDL_sem* renderer_sem;

float map(float x, float in_min, float in_max, float out_min, float out_max)
{
	return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
}

// 8x single precision (float) mapping
__m256 map(__m256 x, __m256 in_min, __m256 in_max, __m256 out_min, __m256 out_max)
{
	__m256 ymm0 = _mm256_sub_ps(x, in_min);			// x - in_min
	__m256 ymm1 = _mm256_sub_ps(out_max, out_min);  // out_max - out_min
	__m256 ymm2 = _mm256_mul_ps(ymm0, ymm1);		// (x - in_min) * (out_max - out_min)
	__m256 ymm3 = _mm256_sub_ps(in_max, in_min);	// in_max - in_min
	__m256 ymm4 = _mm256_div_ps(ymm2, ymm3);		// (x - in_min) * (out_max - out_min) / (in_max - in_min)
	// return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	return _mm256_add_ps(ymm4, out_min);
}

// 4x double precision mapping
__m256d map(__m256d x, __m256d in_min, __m256d in_max, __m256d out_min, __m256d out_max)
{
	__m256d ymm0 = _mm256_sub_pd(x, in_min);		 // x - in_min
	__m256d ymm1 = _mm256_sub_pd(out_max, out_min);  // out_max - out_min
	__m256d ymm2 = _mm256_mul_pd(ymm0, ymm1);		 // (x - in_min) * (out_max - out_min)
	__m256d ymm3 = _mm256_sub_pd(in_max, in_min);	 // in_max - in_min
	__m256d ymm4 = _mm256_div_pd(ymm2, ymm3);		 // (x - in_min) * (out_max - out_min) / (in_max - in_min)
	// return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
	return _mm256_add_pd(ymm4, out_min);
}

void draw_pixel(SDL_Surface& surface, int x, int y, SDL_Color color)
{
	if (x < 0) return;
	else if (x >= surface.w) return;
	if (y < 0) return;
	else if (y >= surface.h) return;

	uint32_t* pixel = (uint32_t*)surface.pixels + x + y * surface.w;
	// NOTE: I think this is windows specific (This should be checked)
	*pixel = ((color.a << 24) | 
			  (color.r << 16) |
			  (color.g <<  8) |
			  (color.b <<  0));
}

void slow_mandelbrot(SDL_Surface& surface, float x_min, float x_max, float y_min, float y_max)
{
	for (int y = 0; y < SCREEN_HEIGHT; y++)
	{
		for (int x = 0; x < SCREEN_WIDTH; x++)
		{
			float x0 = map((float)x, .0f, float(SCREEN_WIDTH), x_min, x_max);
			float y0 = map((float)y, float(SCREEN_HEIGHT), .0f, y_min, y_max);

			// the real coeficients of a complex number (a + bi)
			float a = .0f;
			float b = .0f;

			int current_iteration = 0;
			int max_iterations = MANDELBROT_PRECISION;
			for(; current_iteration < max_iterations; current_iteration++)
			{
				float a_temp = a * a - b * b + x0;
				b = 2 * a*b + y0;
				a = a_temp;

				if (a*a + b * b > 2.0f*2.0f)
					break;
			}

			float color = float(current_iteration) / float(max_iterations);
			SDL_Color c = SDL_Color{ uint8_t(255.0f*color), uint8_t(255.0f*color), uint8_t(255.0f*color), 255 };
			draw_pixel(surface, x, y, c);
		}
	}
}

void simd_float_mandelbrot(SDL_Surface& surface, 
						   int tile_start_x, int tile_start_y, int tile_end_x, int tile_end_y,
						   float x_min, float x_max, float y_min, float y_max)
{
#define SIMD_LINE_WIDTH 8

	assert(SCREEN_WIDTH % SIMD_LINE_WIDTH == 0);
	const float sw = float(SCREEN_WIDTH);
	const float sh = float(SCREEN_HEIGHT);
	for (int y = tile_start_y; y < tile_end_y; y++)
	{
		for (int x = tile_start_x; x < tile_end_x; x += SIMD_LINE_WIDTH)
		{
			// float x0 = map((float)x, .0f, float(SCREEN_WIDTH), x_min, x_max);
			__m256 x0 = map(_mm256_set_ps(float(x), float(x + 1), float(x + 2), float(x + 3), float(x + 4), float(x + 5), float(x + 6), float(x + 7)),
							_mm256_set1_ps(.0f), _mm256_set1_ps(sw),
							_mm256_set1_ps(x_min), _mm256_set1_ps(x_max));
			// float y0 = map((float)y, float(SCREEN_HEIGHT), .0f, y_min, y_max);
			__m256 y0 = map(_mm256_set1_ps(float(y)),
							_mm256_set1_ps(sh), _mm256_set1_ps(.0f),
							_mm256_set1_ps(y_min), _mm256_set1_ps(y_max));

			// the real coeficients of a complex number (a + bi)
			// float a = .0f;
			// float b = .0f;
			__m256 a = _mm256_set1_ps(.0f);
			__m256 b = _mm256_set1_ps(.0f);

			// helper variables
			__m256i one = _mm256_set1_epi32(1);
			__m256 two = _mm256_set1_ps(2.0f);
			__m256 four = _mm256_set1_ps(4.0f);

			// int current_iteration = 0;
			// int max_iterations = 100;
			__m256i current_iterations = _mm256_set1_epi32(0);
			__m256i max_iterations = _mm256_set1_epi32(MANDELBROT_PRECISION);
			// for (; current_iteration < max_iterations; current_iteration++)
			{
				repeat:
				// float a_temp = a * a - b * b + x0;
				__m256 a_squared = _mm256_mul_ps(a, a);
				__m256 b_squared = _mm256_mul_ps(b, b);
				__m256 a_temp = _mm256_add_ps(_mm256_sub_ps(a_squared, b_squared), x0);

				// b = 2 * a*b + y0;
				__m256 ab = _mm256_mul_ps(a, b);
				b = _mm256_add_ps(_mm256_mul_ps(ab, two), y0);

				// a = a_temp;
				a = a_temp;

				// if(a*a + b*b < 2.0f*2.0f)
				a_squared = _mm256_mul_ps(a, a);
				b_squared = _mm256_mul_ps(b, b);
				__m256 length_sq = _mm256_add_ps(a_squared, b_squared);
				__m256 mask1 = _mm256_cmp_ps(length_sq, four, _CMP_LT_OQ);
				__m256i incrementer = _mm256_and_si256(_mm256_castps_si256(mask1), one);
				current_iterations = _mm256_add_epi32(current_iterations, incrementer);

				__m256i mask2 = _mm256_cmpgt_epi32(max_iterations, current_iterations);
				mask2 = _mm256_and_si256(mask2, _mm256_castps_si256(mask1));
				if (_mm256_movemask_epi8(mask2) != 0)
					goto repeat;
			}

			int j = 0;
			for (int i = SIMD_LINE_WIDTH - 1; i >= 0; i--)
			{
				float color = float(current_iterations.m256i_i32[i]) / float(max_iterations.m256i_i32[i]);
				SDL_Color c = SDL_Color{ uint8_t(255.0f*color), uint8_t(255.0f*color), uint8_t(255.0f*color), 255 };
				draw_pixel(surface, x + j, y, c);
				j++;
			}
		}
	}
#undef SIMD_LINE_WIDTH
}

void simd_double_mandelbrot(SDL_Surface& surface, double x_min, double x_max, double y_min, double y_max)
{
#define SIMD_LINE_WIDTH 4

	assert(SCREEN_WIDTH % SIMD_LINE_WIDTH == 0);
	const double sw = double(SCREEN_WIDTH);
	const double sh = double(SCREEN_HEIGHT);
	for (int y = 0; y < SCREEN_HEIGHT; y++)
	{
		for (int x = 0; x < SCREEN_WIDTH; x += SIMD_LINE_WIDTH)
		{
			// float x0 = map((float)x, .0f, float(SCREEN_WIDTH), x_min, x_max);
			__m256d x0 = map(_mm256_set_pd(double(x), double(x + 1), double(x + 2), double(x + 3)),
							 _mm256_set1_pd(.0), _mm256_set1_pd(sw), 
							 _mm256_set1_pd(x_min), _mm256_set1_pd(x_max));
			// float y0 = map((float)y, float(SCREEN_HEIGHT), .0f, y_min, y_max);
			__m256d y0 = map(_mm256_set1_pd(double(y)), 
							 _mm256_set1_pd(sh), _mm256_set1_pd(.0f), 
							 _mm256_set1_pd(y_min), _mm256_set1_pd(y_max));

			// the real coeficients of a complex number (a + bi)
			// float a = .0f;
			// float b = .0f;
			__m256d a = _mm256_set1_pd(.0);
			__m256d b = _mm256_set1_pd(.0);

			// helper variables
			__m256i one = _mm256_set1_epi64x(1);
			__m256d two = _mm256_set1_pd(2.0);
			__m256d four = _mm256_set1_pd(4.0);

			// int current_iteration = 0;
			// int max_iterations = 100;
			__m256i current_iterations = _mm256_set1_epi64x(0);
			__m256i max_iterations = _mm256_set1_epi64x(MANDELBROT_PRECISION);
			// for (; current_iteration < max_iterations; current_iteration++)
			{
			repeat:
				// float a_temp = a * a - b * b + x0;
				__m256d a_squared = _mm256_mul_pd(a, a);
				__m256d b_squared = _mm256_mul_pd(b, b);
				__m256d a_temp = _mm256_add_pd(_mm256_sub_pd(a_squared, b_squared), x0);

				// b = 2 * a*b + y0;
				__m256d ab = _mm256_mul_pd(a, b);
				b = _mm256_add_pd(_mm256_mul_pd(ab, two), y0);

				// a = a_temp;
				a = a_temp;

				// if(a*a + b*b < 2.0f*2.0f)
				a_squared = _mm256_mul_pd(a, a);
				b_squared = _mm256_mul_pd(b, b);
				__m256d length_sq = _mm256_add_pd(a_squared, b_squared);
				__m256d mask1 = _mm256_cmp_pd(length_sq, four, _CMP_LT_OQ);
				__m256i incrementer = _mm256_and_si256(_mm256_castpd_si256(mask1), one);
				current_iterations = _mm256_add_epi64(current_iterations, incrementer);

				__m256i mask2 = _mm256_cmpgt_epi32(max_iterations, current_iterations);
				mask2 = _mm256_and_si256(mask2, _mm256_castpd_si256(mask1));
				if (_mm256_movemask_epi8(mask2) != 0)
					goto repeat;
			}

			int j = 0;
			for (int i = SIMD_LINE_WIDTH - 1; i >= 0; i--)
			{
				double color = double(current_iterations.m256i_i64[i]) / double(max_iterations.m256i_i64[i]);
				SDL_Color c = SDL_Color{ uint8_t(255.0f*color), uint8_t(255.0f*color), uint8_t(255.0f*color), 255 };
				draw_pixel(surface, x + j, y, c);
				j++;
			}
		}
	}
#undef SIMD_LINE_WIDTH
}

bool render_tile()
{
	bool did_work = false;
	if(current_job.value > 0)
	{
		Job_Data *job = jobs + current_job.value - 1;
		if (SDL_AtomicCAS(&current_job, current_job.value, current_job.value - 1))
		{
			simd_float_mandelbrot(*job->surface,
								  job->tile_start_x, job->tile_start_y,
								  job->tile_end_x, job->tile_end_y,
								  job->x_min, job->x_max, job->y_min, job->y_max);

			did_work = true;
		}
	}
	return did_work;
}

int worker_render_tile(void* ptr)
{
	for (;;)
	{
		if (!render_tile())
		{
			SDL_SemWait(renderer_sem);
		}
		SDL_SemWait(renderer_sem);
	}
}

void add_job(SDL_Surface* surface,
             int tile_start_x, int tile_start_y,
             int tile_end_x, int tile_end_y,
             float x_min, float x_max,
             float y_min, float y_max)
{	
	// simd_float_mandelbrot(*surface, 0, 0, tile_width, tile_height, (float)x_min, (float)x_max, (float)y_min, (float)y_max);
	assert(current_job.value < ARRAY_COUNT(jobs));
	Job_Data* job = jobs + current_job.value;
	
	job->surface = surface;
	job->tile_start_x = tile_start_x;
	job->tile_start_y = tile_start_y;
	job->tile_end_x = tile_end_x;
	job->tile_end_y = tile_end_y;
	job->x_min = x_min;
	job->x_max = x_max;
	job->y_min = y_min;
	job->y_max = y_max;

	SDL_AtomicIncRef(&current_job);
	SDL_SemPost(renderer_sem);
}

int main(int argc, char* args[]) 
{
	if (SDL_Init(SDL_INIT_VIDEO) < 0) 
	{
		fprintf(stderr, "Could not initialize sdl2: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);
	}
	if (TTF_Init() == -1) 
	{
		printf("Coult not initialize TTF: %s\n", TTF_GetError());
		exit(EXIT_FAILURE);
	}

	SDL_Window* window = SDL_CreateWindow("Mandelbrot", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
	if (window == NULL)
	{
		fprintf(stderr, "Could not create window: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);
	}
	SDL_Surface* surface = SDL_GetWindowSurface(window);
	if (surface == NULL)
	{
		fprintf(stderr, "Coult not return a surface: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);
	}

	TTF_Font* font = TTF_OpenFont("C:/Windows/Fonts/arial.ttf", 25);
	if (font == NULL)
	{
		fprintf(stderr, "TTF_OpenFont error: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);
	}

	int core_count = SDL_GetCPUCount();
	renderer_sem = SDL_CreateSemaphore(0);
	if(renderer_sem == NULL)
	{
		fprintf(stderr, "Could not create a semaphore: %s\n", SDL_GetError());
		exit(EXIT_FAILURE);	
	}

	for(int thread_index = 1; thread_index < core_count; thread_index++)
	{
		if(SDL_CreateThread(worker_render_tile, NULL, NULL) == NULL)
		{
			fprintf(stderr, "Could not create a thread: %s\n", SDL_GetError());
			exit(EXIT_FAILURE);	
		}
	}
	
	bool is_running = true;
	float y_min = -1.0f;
	float y_max = 1.0f;
	float x_min = -2.0f;
	float x_max = 1.0f;
	float scale = 1.0f;
	bool mouse_pressed = false;
	int prev_mouse_x = 0;
	int prev_mouse_y = 0; 
	while (is_running)
	{
		uint32_t current_timer = SDL_GetTicks();

		SDL_Event event;
		while (SDL_PollEvent(&event))
		{
			switch (event.type)
			{
				// TODO: Zooming must be dependent on scale factor!
				case SDL_MOUSEWHEEL:
				{
					if (event.wheel.y > 0) // scroll up
					{
						float x_dif_1 = x_max - x_min;
						float x_dif_2 = x_min - x_max;
						x_min += x_dif_1 * .1f;
						x_max += x_dif_2 * .1f;
						float y_dif_1 = y_max - y_min;
						float y_dif_2 = y_min - y_max;
						y_min += y_dif_1 * .1f;
						y_max += y_dif_2 * .1f;
						
						scale *= 1.25f;
					}
					else if (event.wheel.y < 0) // scroll down
					{
						float x_dif_1 = x_max - x_min;
						float x_dif_2 = x_min - x_max;
						x_min -= x_dif_1 * .1f;
						x_max -= x_dif_2 * .1f;
						float y_dif_1 = y_max - y_min;
						float y_dif_2 = y_min - y_max;
						y_min -= y_dif_1 * .1f;
						y_max -= y_dif_2 * .1f;
						
						scale *= 0.80f;
					}
				} break;

				// TODO: Panning must be dependent on scale factor!
				case SDL_MOUSEBUTTONDOWN:
				{
					int cur_mouse_x = 0;
					int cur_mouse_y = 0;
					SDL_GetMouseState(&cur_mouse_x, &cur_mouse_y);
					prev_mouse_x = cur_mouse_x;
					prev_mouse_y = cur_mouse_y;
					mouse_pressed = true;
				} break;

				case SDL_MOUSEBUTTONUP:
				{
					mouse_pressed = false;
				} break;

				case SDL_QUIT:
					is_running = false;
					break;
			}
		}

		if (mouse_pressed)
		{
			int cur_mouse_x = 0;
			int cur_mouse_y = 0;
			SDL_GetMouseState(&cur_mouse_x, &cur_mouse_y);

			x_min -= (cur_mouse_x - prev_mouse_x) / (500.0f * scale);
			x_max -= (cur_mouse_x - prev_mouse_x) / (500.0f * scale);
			y_min += (cur_mouse_y - prev_mouse_y) / (500.0f * scale);
			y_max += (cur_mouse_y - prev_mouse_y) / (500.0f * scale);

			prev_mouse_x = cur_mouse_x;
			prev_mouse_y = cur_mouse_y;
		}

		// TODO: Better tile division
		int tile_width = SCREEN_WIDTH / core_count;
		int tile_height = SCREEN_HEIGHT;

		for(int i = 0; i < core_count; i++)
		{
			add_job(surface, tile_width*i, 0, tile_width*(i + 1), tile_height, x_min, x_max, y_min, y_max);
		}

		while(current_job.value > 0)
		{
			render_tile();
			SDL_SemTryWait(renderer_sem);
		}
#if 0
		printf("Job done: %d | %d\n", current_job.value, SDL_SemValue(renderer_sem));
#endif

		uint32_t last_timer = SDL_GetTicks();
		uint32_t elapsed_time = last_timer - current_timer; // miliseconds passed

		SDL_Color color = { 255, 0, 0, 255 };
		char elapsed_time_string[128] = {};
		sprintf_s(elapsed_time_string, "Simulation time: %dms", elapsed_time);
		SDL_Surface* fps_text_surface = TTF_RenderText_Solid(font, elapsed_time_string, color);
		// SDL_BlitSurface(fps_text_surface, NULL, surface, NULL);

		SDL_UpdateWindowSurface(window);
	}

	return 0;
}