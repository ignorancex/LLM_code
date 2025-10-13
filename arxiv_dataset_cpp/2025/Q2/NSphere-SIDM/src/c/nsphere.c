/*
 * Copyright 2025 Kris Sigurdson
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <stdint.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_spline.h>
#include <gsl/gsl_interp.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <time.h>
#include <sys/time.h>  /* For gettimeofday */
#include <unistd.h>    /* For getpid */
#ifdef _OPENMP
#include <omp.h>
#else
// OpenMP function stubs when compiled without OpenMP
// Use __attribute__((unused)) to prevent unused function warnings
static int __attribute__((unused)) omp_get_max_threads(void) { return 1; }
static int __attribute__((unused)) omp_get_num_procs(void) { return 1; }
static int __attribute__((unused)) omp_get_thread_num(void) { return 0; }
static void __attribute__((unused)) omp_set_max_active_levels(int x) { (void)x; }
static void __attribute__((unused)) omp_set_num_threads(int x) { (void)x; }
static double __attribute__((unused)) omp_get_wtime(void) {
    // Use higher precision time function for non-OpenMP builds
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + (double)tv.tv_usec / 1000000.0;
}
#endif
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h> // For usleep function
#include "nsphere_sort.h" // Clean custom wrapper around quadsort
#include <ctype.h>
#include <fftw3.h>
#include <float.h> // For DBL_MAX
#ifdef _WIN32
#include <windows.h>
#else
#include <sys/statvfs.h>
#endif

/**
 * @brief Three-dimensional vector structure for particle physics calculations.
 * @details Represents velocity and position vectors in 3D space for self-interacting
 *          dark matter simulations. Components are stored in Cartesian coordinates.
 */
typedef struct {
  double x; ///< x-component of the vector.
  double y; ///< y-component of the vector.
  double z; ///< z-component of the vector.
} threevector;

/**
 * @brief Container for self-interacting dark matter scattering event data.
 * @details Stores particle indices and final velocity vectors for a single scattering
 *          interaction. Used to buffer scattering results before applying velocity updates,
 *          particularly useful for parallel implementations where race conditions must be avoided.
 */
typedef struct {
    int i;              ///< Index of the first particle in the scattering pair.
    int m_offset;       ///< Offset of the scattering partner relative to particle i (e.g., partner is i + m_offset).
    threevector Vifinal;///< Final 3D velocity vector of particle i.
    threevector Vmfinal;///< Final 3D velocity vector of the partner particle.
} ScatterEvent;

/**
 * @brief Parameters for the NFW distribution function integrand fEintegrand_nfw.
 * @details This structure passes all necessary data, including pre-computed splines
 *          for r(Psi) and M(r), physical constants, and profile-specific parameters,
 *          to the GSL integration routine for calculating I(E).
 */
typedef struct {
    double E_current_shell;         ///< Energy E of the current shell for which I(E) is being computed.
    gsl_spline *spline_r_of_Psi;    ///< Spline for r(-Psi_true), i.e., radius as a function of negated true potential.
    gsl_interp_accel *accel_r_of_Psi; ///< Accelerator for the r(-Psi_true) spline.
    gsl_spline *spline_M_of_r;      ///< Spline for M(r), enclosed mass as a function of radius.
    gsl_interp_accel *accel_M_of_r;   ///< Accelerator for the M(r) spline.
    double const_G_universal;       ///< Universal gravitational constant G.
    double profile_rc_const;        ///< Scale radius (rc) of the NFW profile.
    double profile_nt_norm_const;   ///< Density normalization constant (nt_nfw) for the NFW profile.
    double profile_falloff_C_const; ///< Falloff transition factor C for power-law cutoff in NFW profile.
    double Psimin_global;           ///< Minimum potential value for physical range validation.
    double Psimax_global;           ///< Maximum potential value for physical range validation.
} fE_integrand_params_NFW_t;


// =========================================================================
// Windows‑compatibility shims
// =========================================================================
// Provide POSIX‑style helpers for MinGW/Clang:
//   • mkdir(path,mode)   → _mkdir(path)
//   • drand48 / srand48  → wrappers around ANSI rand
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
  #include <direct.h>
  #include <stdlib.h>

  /** Accept 1‑ or 2‑argument forms of mkdir on Windows. */
  #define mkdir(path, ...) _mkdir(path)

  #define NSPHERE_WINDOWS_SHIMS_DONE 1
#endif
/* ========================================================================= */

// SIDM vector mathematics and cross-section function declarations
threevector make_threevector(double x, double y, double z);
double dotproduct(threevector X, threevector Y);
threevector crossproduct(threevector X, threevector Y);
double sigmatotal(double vrel, int npts, double halo_mass_for_calc, double rc_for_calc);

// Serial SIDM scattering integration function declaration
void perform_sidm_scattering_serial(double **particles, int npts, double dt, double current_time, gsl_rng *rng, long long *Nscatter_total_step, double halo_mass_for_sidm, double rc_for_sidm);

// Forward declaration for the parallel SIDM scattering function
void perform_sidm_scattering_parallel(double **particles, int npts, double dt, double current_time, gsl_rng **rng_per_thread_list, int num_threads_for_rng, long long *Nscatter_total_step, double halo_mass_for_sidm, double rc_for_sidm);


static char g_file_suffix[256] = ""; ///< Global file suffix string.
static gsl_rng *g_rng = NULL; ///< GSL Random Number Generator state.
static gsl_rng **g_rng_per_thread = NULL; ///< Array of GSL RNG states, one per OpenMP thread.
static int g_max_omp_threads_for_rng = 1; ///< Number of threads for which RNGs are allocated.

// =========================================================================
// PERSISTENT SORT BUFFER CONFIGURATION
// =========================================================================
// 
// The simulation exclusively uses a persistent global buffer (`g_sort_columns_buffer`)
// for particle data transposition during sorting operations. This strategy 
// minimizes memory allocation/deallocation overhead.

/** Global persistent buffer for particle data transposition during sorting. */
static double **g_sort_columns_buffer = NULL; 
/** Number of particles the persistent buffer was allocated for; updated if npts changes. */
static int g_sort_columns_buffer_npts = 0;

/** Global NFW mass spline for force calculations. */
static gsl_spline *g_nfw_splinemass_for_force = NULL;
/** Global NFW mass spline accelerator for force calculations. */
static gsl_interp_accel *g_nfw_enclosedmass_accel_for_force = NULL;

// =========================================================================
// PARALLEL SORT ALGORITHM CONFIGURATION
// =========================================================================
// Constants controlling the behavior of parallel sorting algorithms.
// These parameters tune the parallel sorting operations used when 
// OpenMP is available, affecting the partitioning of data across threads
// and the overlap required for correct merging of sorted sections.

/** 
 * Default number of sections when OpenMP is unavailable or reports few threads.
 * Provides a baseline level of partitioning even in limited thread environments.
 */
static const int PARALLEL_SORT_DEFAULT_SECTIONS = 8;

/**
 * Number of sort sections per OpenMP thread for workload distribution.
 * Multiplier used to determine total section count from available threads.
 */
static const int PARALLEL_SORT_SECTIONS_PER_THREAD = 2;

/**
 * Divisor for calculating proportional overlap between sort sections.
 * Overlap is calculated as chunk_size / OVERLAP_DIVISOR to scale with data size.
 */
static const int PARALLEL_SORT_OVERLAP_DIVISOR = 8;

/** 
 * Minimum required overlap between adjacent sort sections (in elements).
 * This ensures sufficient overlap for correct merging of sorted sections,
 * even with sparse data distributions and when calculated proportional
 * overlap would be too small.
 */
static const int PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP = 32;


/**
 * Minimum average chunk size threshold for parallel sort operation.
 * If the calculated size of each chunk (total elements / number of sections)
 * falls below this threshold, the algorithm reverts to serial sorting to
 * avoid overhead from managing very small parallel tasks.
 */
static const int PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD = 128;

/**
 * @brief Applies the global suffix to a filename.
 * @details For .dat files, inserts suffix before the extension.
 *          For other files, appends suffix to the end of filename.
 *
 * Parameters
 * ----------
 * base_filename : const char*
 *     Original filename.
 * with_suffix : int
 *     Flag indicating whether to apply the suffix (1=yes, 0=no).
 * buffer : char*
 *     Output buffer for the resulting filename.
 * bufsize : size_t
 *     Size of the output buffer.
 *
 * Returns
 * -------
 * None
 */
void get_suffixed_filename(const char *base_filename, int with_suffix, char *buffer, size_t bufsize)
{
    if (!with_suffix || g_file_suffix[0] == '\0')
    {
        // No suffix to apply
        strncpy(buffer, base_filename, bufsize - 1);
        buffer[bufsize - 1] = '\0';
        return;
    }

    const char *ext = strrchr(base_filename, '.');
    if (ext && strcmp(ext, ".dat") == 0)
    {
        // For .dat files: insert suffix before extension
        size_t basename_len = ext - base_filename;
        if (basename_len + strlen(g_file_suffix) + strlen(ext) + 1 > bufsize)
        {
            // Buffer too small
            strncpy(buffer, base_filename, bufsize - 1);
            buffer[bufsize - 1] = '\0';
            return;
        }

        strncpy(buffer, base_filename, basename_len);
        buffer[basename_len] = '\0';
        strcat(buffer, g_file_suffix);
        strcat(buffer, ext);
    }
    else
    {
        // For non-dat files: append suffix to filename
        if (strlen(base_filename) + strlen(g_file_suffix) + 1 > bufsize)
        {
            // Buffer too small
            strncpy(buffer, base_filename, bufsize - 1);
            buffer[bufsize - 1] = '\0';
            return;
        }

        strcpy(buffer, base_filename);
        strcat(buffer, g_file_suffix);
    }
}

#ifndef MYBINIO_H
#define MYBINIO_H
#endif

/**
 * @brief Writes a formatted message to the log file with timestamp and severity level.
 *
 * Parameters
 * ----------
 * level : const char*
 *     Severity level (e.g., "INFO", "WARNING", "ERROR").
 * format : const char*
 *     Printf-style format string.
 * ... :
 *     Variable arguments for the format string.
 *
 * Returns
 * -------
 * None
 *
 * @note Creates the "log" directory if it doesn't exist.
 * @warning Prints an error to stderr if the log file cannot be opened.
 *          Logging only occurs if the global `g_enable_logging` flag is set.
 * @see g_enable_logging
 */
// Forward declaration of global logging flag
extern int g_enable_logging;

void log_message(const char *level, const char *format, ...)
{
    // Only write to log file if logging is enabled
    if (g_enable_logging)
    {
        // Create log directory if it doesn't exist
        struct stat st = {0};
        if (stat("log", &st) == -1)
        {
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
            mkdir("log"); // Windows
#else
            mkdir("log", 0755); // Unix-like systems
#endif
        }

        // Always use a single log file regardless of file suffix
        const char *log_filename = "log/nsphere.log";

        FILE *logfile = fopen(log_filename, "a");
        if (logfile)
        {
            time_t now;
            time(&now);
            char timestamp[64];
            strftime(timestamp, sizeof(timestamp), "%Y-%m-%d %H:%M:%S", localtime(&now));

            // Include file suffix in log entries if available
            if (g_file_suffix[0] != '\0')
            {
                fprintf(logfile, "[%s] [%s] [%s] ", timestamp, level, g_file_suffix);
            }
            else
            {
                fprintf(logfile, "[%s] [%s] ", timestamp, level);
            }

            va_list args;
            va_start(args, format);
            vfprintf(logfile, format, args);
            va_end(args);

            fprintf(logfile, "\n");
            fclose(logfile);
        }
        else
        {
            // Print an error message to stderr if the log file cannot be opened
            fprintf(stderr, "Warning: Failed to open log file '%s'\n", log_filename);
        }
    }
}

/**
 * @brief Formats a byte count into a human-readable string with appropriate units.
 *
 * Parameters
 * ----------
 * size_in_bytes : long
 *     The size in bytes to format.
 * buffer : char*
 *     Output buffer for the formatted string.
 * buffer_size : size_t
 *     Size of the output buffer.
 *
 * Returns
 * -------
 * None
 */
void format_file_size(long size_in_bytes, char *buffer, size_t buffer_size)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit_index = 0;
    double size = (double)size_in_bytes;

    // Find appropriate unit
    while (size >= 1024.0 && unit_index < 4)
    {
        size /= 1024.0;
        unit_index++;
    }

    // Format with appropriate precision based on size
    if (unit_index == 0)
    {
        // Bytes: no decimal places needed
        snprintf(buffer, buffer_size, "%ld %s", (long)size, units[unit_index]);
    }
    else if (size >= 10)
    {
        // Larger sizes: one decimal place
        snprintf(buffer, buffer_size, "%.1f %s", size, units[unit_index]);
    }
    else
    {
        // Small sizes: two decimal places
        snprintf(buffer, buffer_size, "%.2f %s", size, units[unit_index]);
    }
}

/**
 * @def CLEAN_EXIT(code)
 * @brief Thread-safe exit macro that properly cleans up allocated resources.
 *
 * @details This macro ensures proper resource cleanup before program termination:
 *          - Uses OpenMP critical section to ensure only one thread performs cleanup.
 *          - Calls cleanup_all_particle_data() and free_local_snap_arrays()
 *            to free dynamically allocated memory.
 *          - Exits with the specified error code.
 *
 * @param code The exit code for the program.
 *
 * @par Example
 * @code
 * if (error_condition) {
 *     CLEAN_EXIT(1);
 * }
 * @endcode
 */
#define CLEAN_EXIT(code)                 \
    do                                   \
    {                                    \
        _Pragma("omp critical")          \
        {                                \
            cleanup_all_particle_data(); \
            free_local_snap_arrays();    \
            exit(code);                  \
        }                                \
    } while (0)

// --- Global Configuration and Macros ---

/** @def imin(a, b) Minimum of two integer values. */
#define imin(a, b) ((a) < (b) ? (a) : (b))

/**
 * @brief Binary file I/O function declarations.
 * @details These functions provide platform-independent binary file I/O operations
 *          similar to fprintf and fscanf, but ensure consistent binary format
 *          regardless of host architecture.
 */
int fprintf_bin(FILE *fp, const char *format, ...);
int fscanf_bin(FILE *fp, const char *format, ...);

/**
 * @brief Global simulation feature flags.
 * @details These control various optional behaviors and optimizations
 *          in the simulation. Each flag can be set via command line arguments.
 */
int g_doDebug = 1;           ///< Enable detailed debug output (default: on).
int g_doDynPsi = 1;          ///< Enable dynamic potential recalculation (default: on).
int g_doDynRank = 1;         ///< Enable dynamic rank calculation per step (default: on).
int g_doAllParticleData = 1; ///< Save complete particle evolution history (default: on).
int g_doRestart = 0;         ///< Enable simulation restart from checkpoint.
int skip_file_writes = 0;    ///< Skip file writes during simulation restart.
int g_enable_logging = 0;    ///< Enable logging to file (controlled by `--log` flag).
int g_enable_sidm_scattering = 0;    ///< Enable SIDM scattering physics (0=no, 1=yes). Default is OFF.
int g_sidm_execution_mode = 1;       ///< SIDM execution mode: 0 for serial, 1 for parallel (default).
long long g_total_sidm_scatters = 0; ///< Global counter for total SIDM scatters.
static double g_sidm_kappa = 50.0;           ///< SIDM opacity kappa (cm^2/g), default 50.0.
static int    g_sidm_kappa_provided = 0;     ///< Flag: 1 if `--sidm-kappa` was given by the user.
static int *g_particle_scatter_state = NULL; ///< Tracks recent scatter history for Adams-Bashforth integrator state reset. Indexed by original particle ID. 0=normal AB, 1=just scattered (use AB1-like step), 2=one step after scatter (use AB2-like step).

// Seed Management Globals
static unsigned long int g_master_seed = 0;         ///< Master seed for the simulation, if provided.
static unsigned long int g_initial_cond_seed = 0;   ///< Seed used for generating initial conditions.
static unsigned long int g_sidm_seed = 0;           ///< Seed used for SIDM calculations.
static int g_master_seed_provided = 0;              ///< Flag: 1 if `--master-seed` was given by the user.
static int g_initial_cond_seed_provided = 0;        ///< Flag: 1 if `--init-cond-seed` was given by the user.
static int g_sidm_seed_provided = 0;                ///< Flag: 1 if `--sidm-seed` was given by the user.
static int g_attempt_load_seeds = 0;                ///< Flag: 1 if we should try to load seeds from files if not provided.

static const char* g_initial_cond_seed_filename_base = "data/last_initial_seed"; ///< Base name for IC seed file.
static const char* g_sidm_seed_filename_base = "data/last_sidm_seed";             ///< Base name for SIDM seed file.

// Profile parameter macros used by profile variables
#define RC 100.0                  ///< Core radius in kpc.
#define RC_NFW_DEFAULT 1.18       ///< Default NFW profile scale radius (kpc).
#define HALO_MASS_NFW 1.15e9      ///< Default NFW profile halo mass in solar masses (Msun).
#define HALO_MASS 1.0e12          ///< Default general halo mass (used for Cored profile by default) in Msun.
#define CUTOFF_FACTOR_NFW_DEFAULT 85.0  ///< Default rmax factor for NFW profile (rmax = factor * rc).
#define CUTOFF_FACTOR_CORED_DEFAULT 85.0 ///< Default rmax factor for Cored profile (rmax = factor * rc).
#define FALLOFF_FACTOR_NFW_DEFAULT 19.0 ///< Default falloff factor for NFW profile (C_cutoff_factor).
#define NUM_MINI_SUBSTEPS_BOOTSTRAP 20 ///< Number of mini Euler steps per bootstrap full step.

// Profile Selection and NFW-Specific Parameters
static int g_use_nfw_profile = 0; ///< Flag to use NFW-like profile for ICs: 0 = Cored (default), 1 = NFW.

static double g_nfw_profile_rc = RC_NFW_DEFAULT;    ///< NFW-specific scale radius (kpc); set by `--scale-radius` if NFW active, else defaults to RC_NFW_DEFAULT.
static double g_nfw_profile_halo_mass = HALO_MASS_NFW; ///< NFW-specific halo mass (Msun); set by `--halo-mass` if NFW active, else defaults to HALO_MASS_NFW.
static double g_nfw_profile_rmax_norm_factor = CUTOFF_FACTOR_NFW_DEFAULT; ///< NFW-specific r_max factor for IC norm/grid; set by `--cutoff-factor` if NFW active, else defaults to CUTOFF_FACTOR_NFW_DEFAULT.
static double g_nfw_profile_falloff_factor = FALLOFF_FACTOR_NFW_DEFAULT; ///< NFW-specific falloff transition C factor; set by `--falloff-factor` if NFW active, else defaults to FALLOFF_FACTOR_NFW_DEFAULT.


// Generalized Profile Parameters (set by new command line flags)
static double g_scale_radius_param = RC;        ///< Generalized scale radius (kpc), defaults to RC macro.
static double g_halo_mass_param = HALO_MASS;    ///< Generalized halo mass (Msun), defaults to HALO_MASS macro.
static double g_cutoff_factor_param = CUTOFF_FACTOR_CORED_DEFAULT; ///< Generalized rmax factor, defaults to Cored's default.
static char   g_profile_type_str[16] = "nfw";   ///< Profile type string ("nfw" or "cored"), default "nfw".

static int g_scale_radius_param_provided = 0;   ///< Flag: 1 if `--scale-radius` was given by the user.
static int g_halo_mass_param_provided = 0;      ///< Flag: 1 if `--halo-mass` was given by the user.
static int g_cutoff_factor_param_provided = 0;  ///< Flag: 1 if `--cutoff-factor` was given by the user.
static double g_falloff_factor_param = FALLOFF_FACTOR_NFW_DEFAULT; ///< Generalized falloff factor, defaults to NFW's default.
static int    g_falloff_factor_param_provided = 0;                  ///< Flag: 1 if `--falloff-factor` was given by the user.
static int g_profile_type_str_provided = 0;     ///< Flag: 1 if `--profile` was given by the user.

// Cored Plummer-like Profile Specific Parameters (populated from generalized flags)
static double g_cored_profile_rc = RC;              ///< Cored-profile-specific scale radius (kpc); set by `--scale-radius` if Cored active, else defaults to RC macro.
static double g_cored_profile_halo_mass = HALO_MASS; ///< Cored-profile-specific halo mass (Msun); set by `--halo-mass` if Cored active, else defaults to HALO_MASS macro.
static double g_cored_profile_rmax_factor = CUTOFF_FACTOR_CORED_DEFAULT;   ///< Cored-profile-specific r_max factor for IC norm/grid; set by `--cutoff-factor` if Cored active, else defaults to CUTOFF_FACTOR_CORED_DEFAULT.

/**
 * @brief Conditional compilation macros for feature flags.
 * @details These macros provide a cleaner syntax for conditional code blocks
 *          that depend on the global feature flags.
 *
 * @def IF_DEBUG
 * @brief Macro for code blocks executed only if `g_doDebug` is true.
 * @def IF_DYNPSI
 * @brief Macro for code blocks executed only if `g_doDynPsi` is true.
 * @def IF_DYNRANK
 * @brief Macro for code blocks executed only if `g_doDynRank` is true.
 * @def IF_ALL_PART
 * @brief Macro for code blocks executed only if `g_doAllParticleData` is true.
 */
#define IF_DEBUG if (g_doDebug)
#define IF_DYNPSI if (g_doDynPsi)
#define IF_DYNRANK if (g_doDynRank)
#define IF_ALL_PART if (g_doAllParticleData)

/**
 * @brief Convolution method selection for density smoothing.
 * @details 0 = FFT-based convolution (default, faster for large datasets).
 *          1 = Direct spatial convolution (more accurate but slower).
 */
int debug_direct_convolution = 0;

// =========================================================================
// PHYSICAL CONSTANTS AND ASTROPHYSICAL PARAMETERS
// =========================================================================
//
// Core constants and unit conversion factors for astrophysical calculations
#define PI 3.14159265358979323846 ///< Mathematical constant Pi.
#define G_CONST 4.3e-6           ///< Newton's gravitational constant in kpc (km/sec)^2/Msun.
/** @def sqr(x) Calculates the square of a value. */
#define sqr(x) ((x) * (x))
/** @def cube(x) Calculates the cube of a value. */
#define cube(x) ((x) * (x) * (x))
#define kmsec_to_kpcmyr 1.02271e-3 ///< Conversion factor: km/s to kpc/Myr.
#define VEL_CONV_SQ (kmsec_to_kpcmyr * kmsec_to_kpcmyr) ///< Velocity conversion squared (kpc/Myr)^2 per (km/s)^2.

static double g_active_halo_mass = HALO_MASS; ///< Active halo mass for N-body force calculations.

/**
 * @brief Angular momentum selection configuration for particle filtering.
 * @details Mode 0: Select particles with the 5 lowest L values.
 *          Mode 1: Select particles with L values closest to Lcompare.
 */
static int use_closest_to_Lcompare = 1; ///< Mode selector (0 or 1).
static double Lcompare = 0.05;          ///< Reference L value for closest-match mode (Mode 1).

// =========================================================================
// GLOBAL PARTICLE DATA ARRAYS
// =========================================================================

/** @brief Global particle data arrays for snapshot processing. */
static int *Rank_partdata_snap = NULL;   ///< Particle rank (sorted position) data for a snapshot.
static float *R_partdata_snap = NULL;    ///< Radial position data for a snapshot.
static float *Vrad_partdata_snap = NULL; ///< Radial velocity data for a snapshot.
static float *L_partdata_snap = NULL;    ///< Angular momentum data for a snapshot.

/** @brief Global arrays for particle data processing (block storage). */
static float *L_block = NULL;    ///< Angular momentum block.
static int *Rank_block = NULL;   ///< Particle rank (sorted position) block.
static float *R_block = NULL;    ///< Radial position block.
static float *Vrad_block = NULL; ///< Radial velocity block.

/** @brief Variables for tracking low angular momentum particles. */
static int nlowest = 5;           ///< Number of lowest angular momentum particles to track.
static int *chosen = NULL;        ///< Array of indices (original IDs) for selected low-L particles.
static double **lowestL_r = NULL; ///< Radial positions of tracked low-L particles over time [particle][time_step].
static double **lowestL_E = NULL; ///< Energy values of tracked low-L particles over time [particle][time_step].
static double **lowestL_L = NULL; ///< Angular momenta of tracked low-L particles over time [particle][time_step].

/**
 * @brief Frees all global arrays used for particle data processing.
 * @details Frees L_block, Rank_block, R_block, Vrad_block, and chosen.
 *          Does NOT free lowestL_* arrays as they are handled elsewhere.
 */
void cleanup_all_particle_data(void)
{
    free(L_block);
    free(Rank_block);
    free(R_block);
    free(Vrad_block);

    // Free low angular momentum tracking arrays
    free(chosen);

    // lowestL arrays are freed in the main function after they're used
}

/**
 * @brief Frees local arrays used for snapshot processing.
 * @details Frees Rank_partdata_snap, R_partdata_snap, Vrad_partdata_snap, L_partdata_snap.
 */
static void free_local_snap_arrays(void)
{
    free(Rank_partdata_snap);
    free(R_partdata_snap);
    free(Vrad_partdata_snap);
    free(L_partdata_snap);
}

/**
 * @def CLEAN_LOCAL_EXIT(code)
 * @brief Non-thread-safe exit macro for cleanup and termination.
 *
 * @details Performs cleanup and exits in single-threaded contexts:
 *          - Frees local snapshot arrays via free_local_snap_arrays().
 *          - Cleans up all global particle data arrays via cleanup_all_particle_data().
 *          - Exits with the specified error code.
 *
 * @warning Not thread-safe. Use only in single-threaded contexts
 *          or where thread safety is guaranteed by other means.
 *
 * @param code The exit code for the program.
 *
 * @par Example
 * @code
 * if (error_condition) {
 *     CLEAN_LOCAL_EXIT(1);
 * }
 * @endcode
 */
#define CLEAN_LOCAL_EXIT(code)       \
    do                               \
    {                                \
        free_local_snap_arrays();    \
        cleanup_all_particle_data(); \
        exit(code);                  \
    } while (0)

/**
 * @brief Gravitational force calculation control flag.
 * @details Used for testing and debugging orbital dynamics:
 *          - 0 = Normal gravitational force calculation (default).
 *          - 1 = Zero gravity (particles move in straight lines).
 *
 * @note Setting this to 1 is useful for validating the integration scheme
 *       independent of gravitational physics.
 */
static int use_identity_gravity = 0;

/**
 * @brief Calculates gravitational acceleration at a given radius.
 *
 * Parameters
 * ----------
 * r : double
 *     Radius in kpc.
 * current_rank : int
 *     Particle rank (0 to npts-1), used for M(r) approximation.
 * npts : int
 *     Total number of particles.
 * G_value : double
 *     Gravitational constant value (e.g., G_CONST).
 * halo_mass_value : double
 *     Total halo mass (e.g., HALO_MASS).
 *
 * Returns
 * -------
 * double
 *     Gravitational acceleration (force per unit mass) in simulation units (kpc/Myr^2).
 *
 * @note Returns 0.0 if `use_identity_gravity` is set to 1.
 * @note M(r) is approximated as `(current_rank / npts) * halo_mass_value`.
 */
static inline double gravitational_force(double r, int current_rank, int npts, double G_value, double halo_mass_value)
{
    if (use_identity_gravity)
    {
        // Testing mode: no gravitational force
        return 0.0;
    }
    else
    {
        // Calculate gravitational force: F = -G * M(r) / r²
        // where M(r) is proportional to particle rank
        return -(VEL_CONV_SQ * G_value) * ((double)current_rank / (double)npts) * halo_mass_value / (r * r);
    }
}

/**
 * @brief Computes the effective centrifugal acceleration due to angular momentum.
 *
 * Parameters
 * ----------
 * r : double
 *     Radius (kpc).
 * ell : double
 *     Angular momentum per unit mass (kpc^2/Myr).
 *
 * Returns
 * -------
 * double
 *     Centrifugal acceleration: L²/r³ (kpc/Myr^2).
 */
static inline double effective_angular_force(double r, double ell)
{
    return (ell * ell) / (r * r * r);
}

/**
 * @brief Alternative gravitational force calculation using transformed coordinates.
 * @details Used in the Levi-Civita regularization scheme. Calculates F/m in rho coordinates.
 *
 * Parameters
 * ----------
 * rho : double
 *     Transformed radial coordinate (sqrt(r)). Units: sqrt(kpc).
 * current_rank : int
 *     Particle rank (0 to npts-1).
 * npts : int
 *     Total number of particles.
 * G_value : double
 *     Gravitational constant value (e.g., G_CONST).
 * halo_mass_value : double
 *     Total halo mass (e.g., HALO_MASS).
 *
 * Returns
 * -------
 * double
 *     Gravitational acceleration in transformed coordinates (units related to kpc^(3/2)/Myr^2).
 *
 * @note Returns 0.0 if `use_identity_gravity` is set to 1.
 * @see gravitational_force
 * @see doLeviCivitaLeapfrog
 */
static inline double gravitational_force_rho_v(double rho, int current_rank, int npts, double G_value, double halo_mass_value)
{
    if (use_identity_gravity)
    {
        // Testing mode: no gravitational force
        return 0.0;
    }
    else
    {
        // Gravitational force in transformed coordinates
        return -(VEL_CONV_SQ * G_value) * ((double)current_rank / (double)npts) * halo_mass_value / (rho * rho);
    }
}

/**
 * @brief Computes effective centrifugal acceleration in transformed coordinates.
 * @details Used in the Levi-Civita regularization scheme. Calculates L^2/r^3 in rho coordinates.
 *
 * Parameters
 * ----------
 * rho : double
 *     Transformed radial coordinate (sqrt(r)). Units: sqrt(kpc).
 * ell : double
 *     Angular momentum per unit mass (kpc^2/Myr).
 *
 * Returns
 * -------
 * double
 *     Centrifugal acceleration in transformed coordinates (units related to kpc^(3/2)/Myr^2).
 *
 * @see effective_angular_force
 * @see doLeviCivitaLeapfrog
 */
static inline double effective_angular_force_rho_v(double rho, double ell)
{
    return (ell * ell) / (rho * rho * rho * rho);
}

// =========================================================================
// Energy Debugging and Validation Subsystem
// =========================================================================

#define DEBUG_PARTICLE_ID 4    // Particle ID to track for debugging
#define DEBUG_MAX_STEPS 100000 // Maximum number of debug energy snapshots

/** @brief Arrays for tracking energy components through simulation for debugging. */
static double dbg_approxE[DEBUG_MAX_STEPS]; ///< Theoretical model energy (per unit mass).
static double dbg_dynE[DEBUG_MAX_STEPS];    ///< Actual dynamical energy (per unit mass).
static double dbg_kinE[DEBUG_MAX_STEPS];    ///< Kinetic energy component (per unit mass).
static double dbg_potE[DEBUG_MAX_STEPS];    ///< Potential energy component (per unit mass).
static double dbg_time[DEBUG_MAX_STEPS];    ///< Simulation time at each snapshot (Myr).
static double dbg_radius[DEBUG_MAX_STEPS];  ///< Particle radius at each snapshot (kpc).
static int dbg_count = 0;                   ///< Number of debug snapshots recorded.

/**
 * @brief Records the theoretical model energy for a debug snapshot.
 *
 * Parameters
 * ----------
 * snapIndex : int
 *     Index of the snapshot (0 to DEBUG_MAX_STEPS - 1).
 * E_value : double
 *     Theoretical energy value (per unit mass).
 * time_val : double
 *     Simulation time (Myr).
 *
 * Returns
 * -------
 * None
 *
 * @note Updates `dbg_count` if `snapIndex` extends the recorded range.
 * @see store_debug_dynE_components
 * @see finalize_debug_energy_output
 */
static void store_debug_approxE(int snapIndex, double E_value, double time_val)
{
    if (snapIndex < 0 || snapIndex >= DEBUG_MAX_STEPS)
        return;

    dbg_approxE[snapIndex] = E_value;
    dbg_time[snapIndex] = time_val;

    // Update the total count if needed
    if (snapIndex >= dbg_count)
    {
        dbg_count = snapIndex + 1;
    }
}

/**
 * @brief Records the actual dynamical energy and its components for a debug snapshot.
 *
 * Parameters
 * ----------
 * snapIndex : int
 *     Index of the snapshot (0 to DEBUG_MAX_STEPS - 1).
 * totalE : double
 *     Total energy (KE + PE) per unit mass.
 * kinE : double
 *     Kinetic energy component (per unit mass).
 * potE : double
 *     Potential energy component (per unit mass).
 * time_val : double
 *     Simulation time (Myr).
 * radius_val : double
 *     Particle radius (kpc).
 *
 * Returns
 * -------
 * None
 *
 * @note Updates `dbg_count` if `snapIndex` extends the recorded range.
 * @see store_debug_approxE
 * @see finalize_debug_energy_output
 */
static void store_debug_dynE_components(int snapIndex,
                                        double totalE,
                                        double kinE,
                                        double potE,
                                        double time_val,
                                        double radius_val)
{
    if (snapIndex < 0 || snapIndex >= DEBUG_MAX_STEPS)
        return;

    dbg_dynE[snapIndex] = totalE;
    dbg_kinE[snapIndex] = kinE;
    dbg_potE[snapIndex] = potE;
    dbg_time[snapIndex] = time_val;
    dbg_radius[snapIndex] = radius_val;

    // Update the total count if needed
    if (snapIndex >= dbg_count)
    {
        dbg_count = snapIndex + 1;
    }
}

/**
 * @brief Writes the collected debug energy comparison data to a file.
 * @details This function is called at the end of the simulation if debug mode (`g_doDebug`)
 *          is enabled. It iterates through the stored debug snapshots collected during
 *          the simulation and post-processing, writing the following data for the
 *          tracked particle (DEBUG_PARTICLE_ID) to `data/debug_energy_compare.dat`
 *          (with the appropriate file suffix applied):
 *          - Snapshot index
 *          - Simulation time (Myr)
 *          - Particle radius (kpc)
 *          - Approximate (theoretical) energy (per unit mass)
 *          - Dynamic (calculated) energy (per unit mass)
 *          - Difference between dynamic and approximate energy
 *          - Kinetic energy component (per unit mass)
 *          - Potential energy component (per unit mass)
 *
 * @note The file writing operation is skipped if the simulation is in restart mode
 *       and file writes are disabled (`skip_file_writes` is true).
 *
 * @see store_debug_approxE
 * @see store_debug_dynE_components
 */
static void finalize_debug_energy_output(void)
{
    // Skip file writes if in restart mode and file writes are skipped
    if (!skip_file_writes)
    {
        // Create filename with suffix
        char filename[256];
        get_suffixed_filename("data/debug_energy_compare.dat", 1, filename, sizeof(filename));

        FILE *fp = fopen(filename, "wb"); // Binary mode for fprintf_bin
        if (!fp)
        {
            // Use log_message for consistency if logging is enabled
            log_message("ERROR", "Cannot open %s for writing debug energy data.", filename);
            // Fallback to printf if logging might be off or failed
            printf("Error: cannot open %s\n", filename);
            return;
        }

        /* Output 8 columns, including KE & PE. */
        fprintf_bin(fp,
                    "# snapIdx  time(Myr)  radius(kpc)   approxE   dynE   (dyn-approx)    KE       PE\n");

        for (int i = 0; i < dbg_count; i++)
        {
            double eA = dbg_approxE[i];
            double eD = dbg_dynE[i];
            double ke = dbg_kinE[i];
            double pe = dbg_potE[i];
            double tVal = dbg_time[i];
            double rVal = dbg_radius[i];

            double dff = eD - eA;
            fprintf_bin(fp, "%4d  %.6f  %.6f  %.8g  %.8g  %.8g  %.8g  %.8g\n",
                        i, tVal, rVal, eA, eD, dff, ke, pe);
        }
        fclose(fp);

        // Only log the debug message, never print to console
        log_message("DEBUG", "Wrote %s with %d lines.", filename, dbg_count);
    }
    else
    {
        log_message("INFO", "Skipped writing debug energy file due to restart/skip_file_writes flag.");
    }
}

static double normalization; ///< Mass normalization factor for energy calculations. Calculated based on the integral of the density profile.

// =========================================================================
// ENERGY CALCULATION AND INTEGRATION STRUCTURES
// =========================================================================

/**
 * @brief Parameters for energy integration calculations.
 * @details Used as `void* params` argument in GSL integration routines,
 *          specifically for distribution function calculations (`fEintegrand`).
 */
typedef struct
{
    double E;                      ///< Energy value (relative energy).
    gsl_spline *splinePsi;         ///< Interpolation spline for potential Psi(r).
    gsl_spline *splinemass;        ///< Interpolation spline for enclosed mass M(r).
    gsl_interp_accel *rofPsiarray; ///< Accelerator for radius lookup from potential r(Psi).
    gsl_interp_accel *massarray;   ///< Accelerator for mass lookups M(r).
} fEintegrand_params;

/**
 * @brief Parameters for Psiintegrand to support profile-specific mass integrands.
 * @details Allows Psiintegrand to call the appropriate mass integrand function
 *          based on the selected density profile.
 */
typedef struct {
    double (*massintegrand_func)(double, void *); ///< Function pointer to profile-specific mass integrand
    void *params_for_massintegrand;               ///< Parameters for the mass integrand function
} Psiintegrand_params;

/**
 * @brief Structure to track angular momentum with particle index and direction.
 * @details Used in sorting and selection of particles by angular momentum,
 *          especially when finding particles closest to a reference L.
 */
typedef struct
{
    double L; ///< Angular momentum value (or squared difference from Lcompare).
    int idx;  ///< Original particle index (before sorting by L).
    int sign; ///< Direction indicator (+1 or -1) or sign of (L - Lcompare).
} LAndIndex;

/**
 * @brief Comparison function for sorting double values in ascending order.
 * @details This function is designed to be used with `qsort` or other
 *          standard library sorting functions that require a comparator.
 *          It takes two void pointers, casts them to `const double*`,
 *          dereferences them, and compares their values.
 *
 * @param a [in] Pointer to the first double value.
 * @param b [in] Pointer to the second double value.
 * @return int -1 if the first double is less than the second,
 *              1 if the first double is greater than the second,
 *              0 if they are equal.
 */
int double_cmp(const void *a, const void *b)
{
    double da = *(const double *)a;
    double db = *(const double *)b;

    if (da < db)
        return -1;
    if (da > db)
        return 1;
    return 0;
}

// =========================================================================
// PARTICLE SORTING AND ORDERING OPERATIONS
// =========================================================================

/**
 * @brief Comparison function for qsort to order particles based on radius.
 *
 * Parameters
 * ----------
 * a : const void*
 *     Pointer to the first particle data array (double**).
 * b : const void*
 *     Pointer to the second particle data array (double**).
 *
 * Returns
 * -------
 * int
 *     Integer less than, equal to, or greater than zero if the radius
 *     (first element) of particle `a` is found, respectively, to be
 *     less than, to match, or be greater than the radius of particle `b`.
 *
 * @note Compares based on the first element (index 0) of the column data,
 *       which represents the radius.
 */
int compare_particles(const void *a, const void *b);

/**
 * @brief Sorts particle data using the default sorting algorithm (g_defaultSortAlg).
 *
 * Parameters
 * ----------
 * particles : double**
 *     Particle data array to sort (array of pointers to component arrays).
 *     Assumed structure: particles[component][particle_index].
 * npts : int
 *     Number of particles.
 *
 * Returns
 * -------
 * None (sorts the array in-place).
 *
 * @see sort_particles_with_alg
 * @see g_defaultSortAlg
 */
void sort_particles(double **particles, int npts);

/**
 * @brief Sorts particle data using the specified sorting algorithm.
 *
 * Parameters
 * ----------
 * particles : double**
 *     Particle data array to sort (array of pointers to component arrays).
 *     Assumed structure: particles[component][particle_index].
 * npts : int
 *     Number of particles.
 * sortAlg : const char*
 *     Algorithm identifier string: "quadsort", "quadsort_parallel",
 *     "insertion", "insertion_parallel".
 *
 * Returns
 * -------
 * None (sorts the array in-place).
 *
 * @see sort_particles
 * @see compare_particles
 */
void sort_particles_with_alg(double **particles, int npts, const char *sortAlg);

/**
 * @brief Sorts radius and psi arrays in tandem based on radius values.
 * @details Ensures that the correspondence between radius and potential values
 *          is maintained after sorting, which is necessary for spline creation.
 *
 * Parameters
 * ----------
 * rrA_spline : double*
 *     Array of radial coordinates to be sorted.
 * psiAarr_spline : double*
 *     Array of corresponding Psi values.
 * npts : int
 *     Number of points (arrays are typically of size npts or npts+1).
 *
 * Returns
 * -------
 * None (sorts the arrays in-place).
 */
void sort_rr_psi_arrays(double *rrA_spline, double *psiAarr_spline, int npts);

// =========================================================================
// PHYSICS CALCULATION FUNCTIONS
// =========================================================================

/**
 * @brief Integrand for calculating enclosed mass: `r^2 * rho(r)`.
 * @details Used in GSL integration routines to compute M(r).
 *
 * Parameters
 * ----------
 * r : double
 *     Radial coordinate (kpc).
 * params : void*
 *     Integration parameters (unused here, but required by GSL).
 *
 * Returns
 * -------
 * double
 *     Value of the mass integrand `r^2 * rho(r)` at radius `r`.
 *
 * @note Assumes a density profile `rho(r)` proportional to `1 / (1 + (r/RC)^2)^3`.
 */
double massintegrand(double r, void *params);
double massintegrand_profile_nfwcutoff(double r, void *params);
double drhodr_profile_nfwcutoff(double r, double rc, double nt_nfw, double falloff_C_param);
double fEintegrand_nfw(double t_integration_var, void *params);

/**
 * @brief Calculates the derivative of the density profile with respect to radius.
 *
 * Parameters
 * ----------
 * r : double
 *     Radial coordinate (kpc).
 *
 * Returns
 * -------
 * double
 *     Density gradient d(rho)/dr at radius `r`.
 *
 * @note Assumes a density profile `rho(r)` proportional to `1 / (1 + (r/RC)^2)^3`.
 */
double drhodr(double r);

/**
 * @brief Integrand for calculating gravitational potential: `massintegrand(r) / r`.
 * @details Used in GSL integration routines to compute Psi(r) via integral from r to infinity.
 *
 * Parameters
 * ----------
 * rp : double
 *     Radial coordinate (kpc).
 * params : void*
 *     Integration parameters (contains alpha, currently unused but required by GSL).
 *
 * Returns
 * -------
 * double
 *     Value of the potential integrand `M(r) / r` (related to `r * rho(r)`).
 *
 * @see massintegrand
 */
double Psiintegrand(double rp, void *params);

/**
 * @brief Evaluates a GSL spline at a given value with bounds checking.
 * @details Safely evaluates the spline, clamping the input value to the spline's
 *          defined range if necessary to prevent GSL errors.
 *
 * Parameters
 * ----------
 * spline : gsl_spline*
 *     Pointer to the initialized GSL spline object.
 * acc : gsl_interp_accel*
 *     Pointer to the GSL interpolation accelerator.
 * value : double
 *     The value at which to evaluate the spline.
 *
 * Returns
 * -------
 * double
 *     The interpolated value from the spline. Returns the boundary value if
 *     `value` is outside the spline's range. Returns 0.0 if spline or acc is NULL.
 */
double evaluatespline(gsl_spline *spline, gsl_interp_accel *acc, double value);

/**
 * @brief Integrand for energy distribution function `f(E)` calculation using Eddington's formula.
 * @details Computes `2 * drho/dpsi` as a function of `t = sqrt(E - Psi_min)`.
 *          Used in GSL integration to find the inner integral in Eddington's formula.
 *
 * Parameters
 * ----------
 * t : double
 *     Integration variable related to energy `E` and potential `Psi`.
 * params : void*
 *     Pointer to a `fEintegrand_params` struct containing splines and energy `E`.
 *
 * Returns
 * -------
 * double
 *     Value of the integrand `2 * drho/dpsi` at the corresponding `Psi`.
 *
 * @see fEintegrand_params
 */
double fEintegrand(double t, void *params);

/**
 * @brief Comparison function for sorting LAndIndex structures by the `L` member (ascending).
 *
 * Parameters
 * ----------
 * a : const void*
 *     Pointer to the first LAndIndex structure.
 * b : const void*
 *     Pointer to the second LAndIndex structure.
 *
 * Returns
 * -------
 * int
 *     -1 if a->L < b->L, 1 if a->L > b->L, 0 if a->L == b->L.
 *
 * @see LAndIndex
 */
int cmp_LAI(const void *a, const void *b);

/**
 * @brief Checks if a given string represents a valid integer.
 * @details Allows an optional leading '+' or '-' sign. Validates that all
 *          subsequent characters in the string are digits. Returns 0 (false)
 *          for empty strings, strings containing only a sign, or strings with
 *          non-digit characters after the optional sign.
 *
 * @param str [in] The null-terminated string to check.
 * @return int 1 if the string is a valid integer, 0 otherwise.
 */
static int isInteger(const char *str)
{
    if (*str == '-' || *str == '+')
    {
        str++;
    }
    if (!*str)
    {
        return 0;
    }
    while (*str)
    {
        if (!isdigit((unsigned char)*str))
        {
            return 0;
        }
        str++;
    }
    return 1;
}

/**
 * @brief Checks if a given string represents a valid floating-point number.
 * @details Validates if the input string conforms to common floating-point number
 *          formats, including an optional leading sign ('+' or '-'), digits,
 *          at most one decimal point (if not in exponent part), and an optional
 *          exponent part (e.g., "e+10", "E-5").
 *          The function requires at least one digit to be present for a number to be
 *          considered valid (e.g., "." or "+." are not valid floats).
 *
 * @param str [in] The null-terminated string to check.
 * @return int 1 if the string is a valid float, 0 otherwise.
 */
static int isFloat(const char *str)
{
    if (*str == '-' || *str == '+')
    {
        str++;
    }
    if (!*str)
    {
        return 0;
    }

    int has_digit = 0;
    int has_decimal = 0;
    int has_exponent = 0;

    while (*str)
    {
        if (isdigit((unsigned char)*str))
        {
            has_digit = 1;
        }
        else if (*str == '.' && !has_decimal && !has_exponent)
        {
            has_decimal = 1;
        }
        else if ((*str == 'e' || *str == 'E') && !has_exponent && has_digit)
        {
            has_exponent = 1;
            str++;
            // Check for optional sign after exponent
            if (*str == '-' || *str == '+')
            {
                str++;
            }
            if (!*str || !isdigit((unsigned char)*str))
            {
                return 0; // Exponent must have at least one digit
            }
            // Don't reset has_digit - we already have valid digits before exponent
        }
        else
        {
            return 0;
        }
        str++;
    }

    return has_digit;
}

// =========================================================================
// COMMAND LINE ARGUMENT PROCESSING
// =========================================================================

/**
 * @brief Displays detailed usage information for command-line arguments.
 * @details Prints a comprehensive help message to stderr showing all available
 *          command-line options, their default values, and brief descriptions.
 *          Includes information about integration methods, sorting algorithms,
 *          data saving modes, and basic usage examples.
 *
 * Parameters
 * ----------
 * prog : const char*
 *     The program name to display in the usage message (typically argv[0]).
 *
 * Returns
 * -------
 * None (prints to stderr).
 *
 * @note Called when the user specifies `--help` or when errors occur during
 *       argument parsing.
 */
static void printUsage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --help                        Show this usage message.\n"
            "  --log                         [Default Off] Enable writing detailed logs to log/nsphere.log\n"
            "  --tag <string>                [Default Off] Add a custom tag to output filenames (e.g. \"run1\")\n"
            "  --save <subargs>              [Default all] Enable various data-saving modes (may combine any):\n"
            "                                          all           => output everything\n"
            "                                          raw-data      => only raw particle data\n"
            "                                          psi-snaps     => plus Psi snapshots\n"
            "                                          full-snaps    => plus full data snapshots\n"
            "                                          debug-energy  => plus energy diagnostics\n"
            "                                          If multiple subargs are given, the highest priority\n"
            "                                          one overrides the lower: raw-data < psi-snaps <\n"
            "                                          full-snaps < all/debug-energy.\n"
            "  --restart                     [Default Off] Restart processing from the last written snapshot\n"
            "\n"
            "  --nparticles <int>            [Default 100000] Number of particles\n"
            "  --ntimesteps <int>            [Default 10000] Requested total timesteps\n"
            "                                     Note: Ntimes will be adjusted to the minimum value that\n"
            "                                     satisfies the constraint (Ntimes - 1) = k*(dtwrite)*(nout)\n"
            "  --dtwrite <int>               [Default 100] Low level diskwrite interval in timesteps\n"
            "  --nout <int>                  [Default 100] Number of post-processing output data snapshot times\n"
            "  --tfinal <int>                [Default 5] Final simulation time in units of the dynamical time\n"
            "\n"
            "  --readinit <file>             [Default Off] Read initial conditions from <file> (binary)\n"
            "  --writeinit <file>            [Default Off] Write initial conditions to <file> (binary)\n"
            "  --master-seed <int>           [Default Random] Set a master seed to derive other seeds.\n"
            "  --load-seeds                  [Default Off] Load seeds from previous run's output files\n"
            "  --init-cond-seed <int>        [Default Random/Master] Set seed for IC generation.\n"
            "                                     Overrides derivation from master-seed.\n"
            "\n"
            "  --method <int>                [Default 1] Integration method (1..9):\n"
            "                                          1   Adaptive Leapfrog with Adaptive Levi-Civita\n"
            "                                          2   Full-step adaptive Leapfrog + Levi-Civita\n"
            "                                          3   Full-step adaptive Leapfrog\n"
            "                                          4   Yoshida 4th-order\n"
            "                                          5   Adams-Bashforth 3\n"
            "                                          6   Leapfrog (vel half-step)\n"
            "                                          7   Leapfrog (pos half-step)\n"
            "                                          8   Classic RK4\n"
            "                                          9   Euler\n"
            "  --methodtag                   [Default Off] Include method string in output filenames\n"
            "  --sort <int>                  [Default 1] Sorting algorithm (1..4):\n"
            "                                          1   Parallel Quadsort\n"
            "                                          2   Sequential Quadsort\n"
            "                                          3   Parallel Insertion Sort\n"
            "                                          4   Sequential Insertion Sort\n"
            "\n"
            "  --halo-mass <float>           [Default 1.15e9] Total halo mass in M☉ for the selected profile.\n"
            "  --profile <type>              [Default nfw] Profile type for ICs: 'nfw' or 'cored'.\n"
            "  --scale-radius <float>        [Default 1.18] Scale radius in kpc for the selected profile.\n"
            "  --cutoff-factor <float>       [Default 85.0] Absolute r_max in units of scale radius.\n"
            "  --falloff-factor <float>      [Default 19.0] NFW concentration parameter transition factor.\n"
            "  --ftidal <float>              [Default 0.0] Set tidal fraction outer stripping value (0.0 to 1.0)\n"
            "\n"
            "  --sidm                        [Default Off] Enable self-interacting scattering physics\n"
            "  --sidm-seed <int>             [Default Random/Master] Set seed for SIDM calculations.\n"
            "                                     Overrides derivation from master-seed.\n"
            "  --sidm-mode <serial|parallel> [Default parallel] Select SIDM execution mode.\n"
            "                                     Parallel mode requires OpenMP.\n"
            "  --sidm-kappa <float>          [Default 50.0] SIDM opacity kappa in cm^2/g.\n"
            "\n"
            "Example:\n"
            "  %s --nparticles 50000 --ntimesteps 20000 --tfinal 5 \\\n"
            "     --nout 100 --dtwrite 10 --method 6 --sort 2 --readinit initial_conditions.bin\n",
            prog, prog);
}

/**
 * @brief Displays an error message, suggests `--help`, and terminates the program.
 * @details Formats and prints an error message to stderr, includes a suggestion
 *          to use the `--help` flag for usage information, performs necessary
 *          cleanup of allocated resources, and then exits with a non-zero status.
 *
 * Parameters
 * ----------
 * msg : const char*
 *     The error message to display.
 * arg : const char*
 *     Optional argument value that caused the error (shown in quotes),
 *     or NULL to omit this part of the message.
 * prog : const char*
 *     The program name to display in the `--help` usage suggestion.
 *
 * Returns
 * -------
 * None (calls exit(1) and never returns).
 *
 * @note Calls cleanup_all_particle_data() before exiting to free allocated memory.
 * @warning This function does not return; execution is terminated.
 */
static void errorAndExit(const char *msg, const char *arg, const char *prog)
{
    if (arg != NULL)
    {
        fprintf(stderr, "Error: %s '%s'\n", msg, arg);
    }
    else
    {
        fprintf(stderr, "Error: %s\n", msg);
    }
    fprintf(stderr, "Use '%s --help' for usage information.\n", prog ? prog : "./nsphere");
    cleanup_all_particle_data();
    exit(1);
}

// =========================================================================
// PARTICLE DATA STRUCTURES AND OPERATIONS
// =========================================================================

/**
 * @brief Compact data structure for particle properties used in sorting and analysis.
 * @details Contains essential physical properties (radial position, velocity, 
 *          angular momentum) and tracking metadata (rank, original index) for each
 *          particle in the simulation. Used extensively for sorting, file I/O, and
 *          data analysis operations.
 * 
 * @note Uses compact `float` types for physical quantities to reduce memory usage
 *       when processing large particle counts.
 * @note The `rank` field is assigned during radial sorting, while `original_index`
 *       preserves the initial array position for tracking particles across snapshots.
 */
struct PartData
{
    int rank;           // Particle rank (sorted position)
    float rad;          // Radial position
    float vrad;         // Radial velocity
    float angmom;       // Angular momentum
    int original_index; // Original position in array before sorting
};

/**
 * Sorts particle data in ascending order by radial position
 *
 * Performs a quick sort of the particle data structure array,
 * using the radial position as the primary sorting key.
 * Preserves the original indices to allow tracking particles across time steps.
 *
 * @param array  Array of particle data structures to be sorted
 * @param npts   Number of particles in the array
 */
void sort_by_rad(struct PartData *array, int npts);

/**
 * @brief Appends a block of full particle data to the specified output file.
 * @details This function writes a chunk of particle data, corresponding to `block_size`
 *          timesteps, to the given binary file. The data for each particle (rank, radius,
 *          radial velocity, angular momentum) is written sequentially for each timestep
 *          within the block (step-major order). This means all particle data for step `s`
 *          is contiguous, followed by all data for step `s+1`, etc., within the block.
 *          The file is opened in append binary mode ("ab").
 *          This is used for creating the `all_particle_data.dat` file which stores
 *          the complete evolution history when `g_doAllParticleData` is enabled.
 *
 * @param filename   [in] Path to the output binary file (e.g., "data/all_particle_data<suffix>.dat").
 * @param npts       [in] Number of particles.
 * @param block_size [in] Number of timesteps of data contained in the provided `_block` arrays.
 * @param L_block    [in] Pointer to the block of angular momentum data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @param Rank_block [in] Pointer to the block of particle rank data (int array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @param R_block    [in] Pointer to the block of radial position data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @param Vrad_block [in] Pointer to the block of radial velocity data (float array).
 *                        Assumed to be `[step_in_block * npts + particle_orig_id]`.
 * @note Exits via `CLEAN_EXIT(1)` if the file cannot be opened for appending.
 */
static void append_all_particle_data_chunk_to_file(const char *filename, int npts, int block_size,
                                                   float *L_block, int *Rank_block, float *R_block, float *Vrad_block)
{
    FILE *f = fopen(filename, "ab");
    if (!f)
    {
        printf("Error: cannot open %s for appending all_particle_data\n", filename);
        CLEAN_EXIT(1);
    }

    // Write particle data in step-major order
    for (int step = 0; step < block_size; step++)
    {
        for (int i = 0; i < npts; i++)
        {
            int rankval = Rank_block[step * npts + i];
            float rval = R_block[step * npts + i];
            float vval = Vrad_block[step * npts + i];
            float lval = L_block[step * npts + i];

            // Write particle data in fixed order
            fwrite(&rankval, sizeof(int), 1, f);
            fwrite(&rval, sizeof(float), 1, f);
            fwrite(&vval, sizeof(float), 1, f);
            fwrite(&lval, sizeof(float), 1, f);
        }
    }

    fclose(f);
}

/**
 * @brief Retrieves particle data for a specific snapshot from the `all_particle_data.dat` binary file.
 * @details This function reads the data (rank, radius, radial velocity, angular momentum)
 *          for all `npts` particles corresponding to a single snapshot number (`snap`)
 *          from the specified binary file. The file is expected to be in step-major order,
 *          where each record per particle consists of an int (rank) and three floats (R, Vrad, L).
 *          It calculates the correct file offset to seek to the desired snapshot.
 *          To ensure thread safety when called in parallel (e.g., during post-processing
 *          of snapshots), file I/O (seeking and reading) is performed within an
 *          OpenMP critical section named `file_access`. Temporary local buffers are used
 *          for reading, and data is then copied to the caller-provided output arrays.
 *
 * @param filename    [in] Path to the binary data file (e.g., "data/all_particle_data<suffix>.dat").
 * @param snap        [in] The snapshot number (0-indexed, corresponding to write events) to retrieve.
 * @param npts        [in] Number of particles per snapshot.
 * @param block_size  [in] The number of snapshots that were written per block to the file by
 *                       `append_all_particle_data_chunk_to_file`. Used for calculating seek offset.
 * @param L_out       [out] Pointer to an array (size `npts`) to store the retrieved angular momentum values.
 * @param Rank_out    [out] Pointer to an array (size `npts`) to store the retrieved particle ranks.
 * @param R_out       [out] Pointer to an array (size `npts`) to store the retrieved radial positions.
 * @param Vrad_out    [out] Pointer to an array (size `npts`) to store the retrieved radial velocities.
 * @note Exits via `CLEAN_EXIT(1)` on memory allocation failure, file open failure, fseek failure, or unexpected EOF.
 */
static void retrieve_all_particle_snapshot(
    const char *filename,
    int snap,
    int npts,
    int block_size,
    float *L_out,
    int *Rank_out,
    float *R_out,
    float *Vrad_out)
{
    // Allocate local (thread-private) arrays
    float *tmpL = (float *)malloc(npts * sizeof(float));
    int *tmpRank = (int *)malloc(npts * sizeof(int));
    float *tmpR = (float *)malloc(npts * sizeof(float));
    float *tmpV = (float *)malloc(npts * sizeof(float));

    if (!tmpL || !tmpRank || !tmpR || !tmpV)
    {
        fprintf(stderr, "Error: out of memory in retrieve_all_particle_snapshot!\n");
        CLEAN_EXIT(1);
    }

    // Status messages are handled in the ordered section of the parallel loop.

// Read from file in a critical section
#pragma omp critical(file_access)
    {
        FILE *f = fopen(filename, "rb");
        if (!f)
        {
            fprintf(stderr, "Error: cannot open %s for reading\n", filename);
            CLEAN_EXIT(1);
        }

        // Compute offset in file.
        int block_number = snap / block_size;
        int index_in_block = snap % block_size;

        long long step_data_size = (long long)npts * 16; // 16 bytes per record.
        long long block_data_size = (long long)block_size * step_data_size;
        long long offset = block_data_size * block_number + step_data_size * index_in_block;

        if (fseek(f, offset, SEEK_SET) != 0)
        {
            fprintf(stderr, "Error: fseek failed for snap=%d\n", snap);
            fclose(f);
            CLEAN_EXIT(1);
        }

        // Read npts records into local buffers.
        for (int i = 0; i < npts; i++)
        {
            int rankval;
            float rval, vval, lval;

            if (fread(&rankval, sizeof(int), 1, f) != 1 ||
                fread(&rval, sizeof(float), 1, f) != 1 ||
                fread(&vval, sizeof(float), 1, f) != 1 ||
                fread(&lval, sizeof(float), 1, f) != 1)
            {
                fprintf(stderr, "Error: unexpected EOF while reading snap=%d (particle %d)\n", snap, i);
                fclose(f);
                CLEAN_EXIT(1);
            }
            tmpRank[i] = rankval;
            tmpR[i] = rval;
            tmpV[i] = vval;
            tmpL[i] = lval;
        }
        fclose(f);
    } // End critical(file_access).

    // Copy from local buffers to output arrays
    memcpy(Rank_out, tmpRank, npts * sizeof(int));
    memcpy(R_out, tmpR, npts * sizeof(float));
    memcpy(Vrad_out, tmpV, npts * sizeof(float));
    memcpy(L_out, tmpL, npts * sizeof(float));

    // Free the temporary local arrays
    free(tmpL);
    free(tmpRank);
    free(tmpR);
    free(tmpV);
}

/**
 * @brief Adjusts the total number of timesteps to align with desired output snapshot intervals.
 * @details This function calculates an adjusted number of total simulation timesteps, \f$N'_{times}\f$,
 *          such that it is greater than or equal to the initially requested `Ntimes_initial` (\f$N\f$)
 *          and satisfies the constraint: \f$(N'_{times} - 1)\f$ must be an integer multiple of
 *          \f$(M - 1) \times p\f$. Here, \f$M\f$ is `nout` (number of desired output snapshot points,
 *          which means \f$M-1\f$ intervals) and \f$p\f$ is `dtwrite` (the low-level write interval
 *          in terms of simulation timesteps).
 *          This alignment ensures that exactly `nout` snapshots can be produced at intervals
 *          that are multiples of `dtwrite` and that also evenly span the total adjusted simulation duration.
 *
 * @param Ntimes_initial [in] Initially requested total number of simulation timesteps (\f$N\f$).
 * @param nout           [in] Number of desired output snapshot points (\f$M\f$). Must be >= 2 for adjustment to apply.
 * @param dtwrite        [in] The interval (in timesteps) at which low-level data is potentially written (\f$p\f$). Must be >= 1.
 * @return int The adjusted total number of timesteps (\f$N'_{times}\f$). Returns `Ntimes_initial`
 *             if `nout < 2` or `dtwrite < 1` or other edge cases where the constraint cannot be met.
 */
static int adjust_ntimesteps(int Ntimes_initial, int nout, int dtwrite)
{
    // Notation: M = nout, p = dtwrite, N = Ntimes_initial.
    // Find the smallest N' >= N such that (N' - 1) is a multiple of (M - 1) * p.

    int M = nout;
    int p = dtwrite;
    int N = Ntimes_initial;

    // Edge cases:
    if (M < 2)
    {
        // If only 0 or 1 snapshot requested, no interval constraint applies.
        return N;
    }
    if (p < 1)
    {
        // Invalid write interval.
        return N;
    }

    // The total number of intervals between M snapshots is (M - 1).
    // The total number of steps spanning these intervals must be a multiple of p.
    // Therefore, the total number of steps (N' - 1) must be a multiple of (M - 1) * p.
    // Find the smallest integer k >= 1 such that (M - 1) * k * p >= (N - 1).
    double required_steps = (double)(N - 1);
    double steps_per_output_cycle = (M - 1) * (double)p;

    // Handle case where denominator is zero (e.g., M=1 or p=0, caught above but added safety)
    if (steps_per_output_cycle <= 0) {
        return N; // Cannot satisfy constraint
    }

    double ratio = required_steps / steps_per_output_cycle;
    int k = (int)ceil(ratio);
    if (k < 1)
    {
        k = 1; // Ensure at least one full output cycle.
    }

    int Nprime_minus_1 = (M - 1) * k * p;
    int Nprime = Nprime_minus_1 + 1;

    return Nprime;
}

/**
 * @brief Comparison function for sorting LAndIndex structures by the L member.
 * @details Sorts LAndIndex structures in ascending order based on their 'L'
 *          (angular momentum or squared difference from a reference L) value.
 *          Used with qsort for ordering particles by their L values, typically
 *          for selecting particles with lowest L or L closest to a target.
 *
 * @param a [in] Pointer to the first LAndIndex structure.
 * @param b [in] Pointer to the second LAndIndex structure.
 * @return int -1 if a->L < b->L, 1 if a->L > b->L, 0 if equal.
 */
int cmp_LAI(const void *a, const void *b)
{
    double La = ((const LAndIndex *)a)->L;
    double Lb = ((const LAndIndex *)b)->L;

    if (La < Lb)
        return -1;
    if (La > Lb)
        return 1;
    return 0;
}

// =========================================================================
// ADAPTIVE FULL LEAPFROG STEP: r(n), v(n) --> r(n+1), v(n+1)
// =========================================================================
//
// Physics-based time integration method with adaptive step refinement:
// - Subdivide the time step h = ΔT in powers-of-2 "micro-steps"
// - Compare a (2N+1)-step "coarse" vs. a (4N+1)-step "fine" integration
// - If within tolerance, return one of {coarse, fine, rich (Richardson extrapolation)}
// - Otherwise, double N and repeat until convergence or max subdivision reached
/**
 * @brief Performs a leapfrog integration step using a fixed number of micro-steps.
 * @details This helper function implements the leapfrog (Kick-Drift-Kick) integration
 *          over a total time interval `h` by dividing it into a sequence of micro-steps.
 *          The number of micro-steps is `subSteps` (e.g., \f$2N+1\f$ for a "coarse" pass or
 *          \f$4N+1\f$ for a "fine" pass in an adaptive scheme, where \f$N\f$ is `N_subdivision_factor`).
 *          The micro-timestep sizes for kicks and drifts are adjusted based on `N_subdivision_factor`
 *          and whether it's a coarse or fine integration sequence.
 *          Sequence: Initial half-kick, \f$((\text{subSteps}-1)/2 - 1)\f$ full Drift-Kick pairs,
 *          a final full Drift, and a final half-Kick.
 *
 * @param i                   [in] Particle index (0 to npts-1), for rank in gravitational force.
 * @param npts                [in] Total number of particles.
 * @param r_in                [in] Input radial position (kpc) at the start of the total interval `h`.
 * @param v_in                [in] Input radial velocity (kpc/Myr) at the start of `h`.
 * @param ell                 [in] Angular momentum per unit mass (kpc^2/Myr).
 * @param h                   [in] Total physical time interval for this leapfrog sequence (Myr).
 * @param N_subdivision_factor [in] Base subdivision factor \f$N\f$ used to determine micro-timestep sizes.
 * @param subSteps            [in] Total number of Kicks/Drifts (e.g., \f$2N+1\f$ or \f$4N+1\f$).
 * @param grav                [in] Gravitational constant G (simulation units).
 * @param r_out               [out] Pointer to store the output radial position (kpc) after time `h`.
 * @param v_out               [out] Pointer to store the output radial velocity (kpc/Myr) after time `h`.
 */
static void doMicroLeapfrog(
    int i, int npts,
    double r_in, double v_in, double ell,
    double h,     // Total "big step".
    int N,        // Micro-subdivision.
    int subSteps, // 2N+1 or 4N+1.
    double grav,
    double *r_out, double *v_out)
{
    // Initialize current state with input values
    double r_curr = r_in;
    double v_curr = v_in;

    // Set up timestep sizes based on subdivision level
    double halfKick, midStep;
    if (subSteps == (2 * N + 1))
    {
        // Coarse integration (2N+1 substeps)
        halfKick = h / (2.0 * N);
        midStep = h / (1.0 * N);
    }
    else
    {
        // Fine integration (4N+1 substeps)
        halfKick = h / (4.0 * N);
        midStep = h / (2.0 * N);
    }

    // Initial half-kick (velocity update)
    double force = gravitational_force(r_curr, i, npts, grav, g_active_halo_mass);
    double dvdt = force + effective_angular_force(r_curr, ell);
    v_curr += halfKick * dvdt;

    // Middle pattern of drift-kick pairs
    int pairs = (subSteps - 1) / 2; // Total number of drift-kick pairs
    // Execute all but the last pair (last kick handled separately)
    for (int pp = 1; pp <= (pairs - 1); pp++)
    {
        // Drift: update position using current velocity
        r_curr += midStep * v_curr;

        // Kick: update velocity using forces at new position
        force = gravitational_force(r_curr, i, npts, grav, g_active_halo_mass);
        dvdt = force + effective_angular_force(r_curr, ell);
        v_curr += midStep * dvdt;
    }

    // Final full drift
    r_curr += midStep * v_curr;

    // Final half-kick
    force = gravitational_force(r_curr, i, npts, grav, g_active_halo_mass);
    dvdt = force + effective_angular_force(r_curr, ell);
    v_curr += halfKick * dvdt;

    *r_out = r_curr;
    *v_out = v_curr;
}

/**
 * @brief Performs an adaptive full leapfrog step with error control over a physical timestep `h`.
 * @details This function implements an adaptive leapfrog algorithm to advance a particle's
 *          state \f$(r, v_{rad})\f$ over a full physical timestep `h` (\f$\Delta T_{phys}\f$). It iteratively
 *          refines the integration by comparing a "coarse" integration (using \f$2N+1\f$
 *          micro-steps via `doMicroLeapfrog`) with a "fine" integration (using \f$4N+1\f$
 *          micro-steps). The subdivision factor \f$N\f$ starts at 1 and is doubled if the
 *          relative differences in final radius and velocity between coarse and fine passes
 *          exceed `radius_tol` and `velocity_tol`, respectively. This process repeats up
 *          to a maximum subdivision factor `max_subdiv`.
 *          The final state \f$(r_{out}, v_{out})\f$ for the step `h` is chosen based on `out_type`
 *          (coarse, fine, or Richardson extrapolation) once convergence is met or
 *          `max_subdiv` is reached.
 *
 * @param i             [in] Particle index (0 to npts-1), for rank in gravitational force calculation.
 * @param npts          [in] Total number of particles in the simulation.
 * @param r_in          [in] Initial radial position (kpc) at the start of the step `h`.
 * @param v_in          [in] Initial radial velocity (kpc/Myr) at the start of `h`.
 * @param ell           [in] Angular momentum per unit mass (kpc^2/Myr), conserved.
 * @param h             [in] Full physical timestep size \f$\Delta T_{phys}\f$ (Myr).
 * @param radius_tol    [in] Relative convergence tolerance for radius comparison.
 * @param velocity_tol  [in] Relative convergence tolerance for velocity comparison.
 * @param max_subdiv    [in] Maximum allowed subdivision factor \f$N\f$ for micro-steps.
 * @param grav          [in] Gravitational constant G (simulation units, e.g., G_CONST).
 * @param out_type      [in] Result selection mode for converged integration:
 *                         0 for coarse result, 1 for fine result, 2 for Richardson extrapolation.
 * @param r_out         [out] Pointer to store the final radial position (kpc) after time `h`.
 * @param v_out         [out] Pointer to store the final radial velocity (kpc/Myr) after time `h`.
 */
static void doAdaptiveFullLeap(
    int i,               // Particle index for force computation
    int npts,            // Total number of particles in simulation
    double r_in,         // Initial radius at start of step
    double v_in,         // Initial velocity at start of step
    double ell,          // Angular momentum (conserved during integration)
    double h,            // Full physical timestep size ΔT
    double radius_tol,   // Convergence tolerance for radius
    double velocity_tol, // Convergence tolerance for velocity
    int max_subdiv,      // Maximum subdivision factor allowed
    double grav,         // Gravitational constant (renamed from G to avoid macro collision)
    int out_type,        // Result selection: 0=coarse, 1=fine, 2=Richardson extrapolation
    double *r_out,       // Output parameter for final radius
    double *v_out        // Output parameter for final velocity
)
{
    int N = 1; // Start with N=1 micro-steps.

    // Initialize coarse/fine results (first coarse pass).
    double r_coarse = r_in, v_coarse = v_in;
    double r_fine = r_in, v_fine = v_in;

    while (N <= max_subdiv)
    {
        // Coarse pass (2N+1 steps)
        doMicroLeapfrog(
            i, npts, r_in, v_in, ell,
            h, N, (2 * N + 1),
            grav,
            &r_coarse, &v_coarse);

        // Fine pass (4N+1 steps)
        doMicroLeapfrog(
            i, npts, r_in, v_in, ell,
            h, N, (4 * N + 1),
            grav,
            &r_fine, &v_fine);

        // Compare radius, velocity.
        double radius_diff = fabs(r_fine - r_coarse) / (fabs(r_fine) + 1.0e-30);
        double velocity_diff = fabs(v_fine - v_coarse) / (fabs(v_fine) + 1.0e-30);

        if ((radius_diff < radius_tol) && (velocity_diff < velocity_tol))
        {
            // Tolerances met => output according to out_type.
            if (out_type == 0)
            {
                *r_out = r_coarse;
                *v_out = v_coarse;
            }
            else if (out_type == 1)
            {
                *r_out = r_fine;
                *v_out = v_fine;
            }
            else
            {
                // Richardson extrapolation for higher-order result
                *r_out = 4.0 * r_fine - 3.0 * r_coarse;
                *v_out = 4.0 * v_fine - 3.0 * v_coarse;
            }
            return; // Done.
        }
        else
        {
            // Not converged, increase refinement
            N *= 2;
        }
    }

    // If we exit the while(N <= max_subdiv) loop, it means we never converged.
    // Return the last available result based on out_type.
    if (out_type == 0)
    {
        *r_out = r_coarse;
        *v_out = v_coarse;
    }
    else if (out_type == 1)
    {
        *r_out = r_fine;
        *v_out = v_fine;
    }
    else
    {
        *r_out = 4.0 * r_fine - 3.0 * r_coarse;
        *v_out = 4.0 * v_fine - 3.0 * v_coarse;
    }
}

// =========================================================================
// LEVI-CIVITA REGULARIZATION
// =========================================================================
//
// Physics-based regularized time integration method:
// - Transforms coordinates (r -> ρ = √r) to handle close encounters
// - Uses fictitious time τ to integrate equations of motion
// - Maps back to physical coordinates and time after integration
// - Provides enhanced stability for high-eccentricity orbits

/**
 * @brief Calculates \f$d\rho/d\tau\f$, the derivative of the regularized coordinate \f$\rho\f$ with respect to fictitious time \f$\tau\f$.
 * @details In Levi-Civita regularization, \f$d\rho/d\tau = \frac{1}{2} \rho v_{rad}\f$, where \f$\rho = \sqrt{r}\f$
 *          and \f$v_{rad}\f$ is the radial velocity in physical units (though often represented as \f$v\f$ or \f$v_{\rho}\f$
 *          in transformed equations of motion depending on the specific formulation).
 *          This function implements this relationship.
 *
 * @param rhoVal [in] The current value of the regularized radial coordinate \f$\rho = \sqrt{r}\f$.
 * @param vVal   [in] The current radial velocity \f$v_{rad}\f$ (kpc/Myr).
 * @return double The value of \f$d\rho/d\tau\f$.
 */
static inline double dRhoDtaufun(double rhoVal, double vVal)
{
    // dρ/dτ = 0.5 * ρ * v
    return 0.5 * rhoVal * vVal;
}

/**
 * @brief Calculates the total effective force per unit mass in Levi-Civita transformed coordinates.
 * @details This function computes \f$F_{\rho}/m = (F_{grav,\rho} + F_{centrifugal,\rho})/m\f$,
 *          where \f$F_{grav,\rho}\f$ is the gravitational force and \f$F_{centrifugal,\rho}\f$ is the
 *          effective centrifugal force, both expressed in the regularized radial coordinate \f$\rho = \sqrt{r}\f$.
 *          It calls `gravitational_force_rho_v` and `effective_angular_force_rho_v`.
 *          This combined force is used in the equations of motion for Levi-Civita regularization.
 *
 * @param i          [in] Particle index (0 to npts-1), for rank in gravitational force calculation.
 * @param npts       [in] Total number of particles.
 * @param totalmass  [in] Total halo mass of the system (Msun) used for gravitational force.
 * @param grav       [in] Gravitational constant G (simulation units).
 * @param ell        [in] Angular momentum per unit mass (kpc^2/Myr).
 * @param rhoVal     [in] Current value of the regularized radial coordinate \f$\rho = \sqrt{r}\f$.
 * @return double    The total transformed force per unit mass \f$F_{\rho}/m\f$.
 */
static inline double forceLCfun(
    int i, int npts,
    double totalmass,
    double grav,
    double ell,
    double rhoVal)
{
    double gravPart = gravitational_force_rho_v(rhoVal, i, npts, grav, totalmass);
    double angPart = effective_angular_force_rho_v(rhoVal, ell);
    return gravPart + angPart;
}

/**
 * @brief Performs integration of particle motion over a physical time `dt` using Levi-Civita regularization.
 * @details This function implements a leapfrog-like integration scheme in Levi-Civita
 *          regularized coordinates \f$(ρ, v_{rad})\f$ and fictitious time \f$τ\f$. The physical
 *          coordinates are \f$r = ρ^2\f$, and physical time \f$t_{phys}\f$ is related to \f$τ\f$ by \f$dt_{phys} = ρ^2 dτ\f$.
 *          The integration proceeds by taking variable \f$Δτ\f$ steps (estimated based on `N_taumin`)
 *          until the accumulated physical time `t_phys` reaches or exceeds the target physical
 *          timestep `dt`. If a step overshoots `dt`, linear interpolation is used to obtain
 *          the state precisely at the target physical time.
 *          This method is particularly effective for handling close encounters where \f$r → 0\f$.
 *
 * @param i         [in] Particle index (0 to npts-1), for rank in force calculation.
 * @param npts      [in] Total number of particles.
 * @param r_in      [in] Initial physical radial position (kpc) at the start of the physical step `dt`.
 * @param v_in      [in] Initial physical radial velocity (kpc/Myr) at the start of `dt`.
 * @param ell       [in] Angular momentum per unit mass (kpc^2/Myr) for the force calculation.
 * @param dt        [in] The full physical timestep \f$\Delta T_{phys}\f$ (Myr) to advance the particle.
 * @param N_taumin  [in] Target number of fictitious \f$τ\f$-steps within the `dt` interval; influences
 *                     the initial guess for \f$Δτ\f$.
 * @param grav      [in] Gravitational constant G (simulation units).
 * @param r_out     [out] Pointer to store the final physical radial position (kpc) after time `dt`.
 * @param v_out     [out] Pointer to store the final physical radial velocity (kpc/Myr) after time `dt`.
 */
static void doLeviCivitaLeapfrog(
    int i,
    int npts,
    double r_in,
    double v_in,
    double ell,
    double dt,
    int N_taumin,
    double grav,
    double *r_out,
    double *v_out)
{
    // Transform to Levi-Civita coordinates: rho = sqrt(r)
    double rho = sqrt(r_in);
    double v_rad = v_in;
    double tau = 0.0;
    double t_cur = 0.0;

    // Estimate initial tau step size based on N_taumin
    double deltaTau = 0.0;
    if (r_in > 1.0e-30 && N_taumin > 0)
    {
        deltaTau = dt / (2.0 * r_in * N_taumin);
    }
    else
    {
        deltaTau = dt / 100.0;
    }

    // Initialize integration state variables
    double rho_cur = rho;
    double v_cur = v_rad;
    double tau_cur = tau;
    double t_phys = t_cur;

    int stepCount = 0;
    int stepMax = 200000000; // Maximum step count to prevent infinite loops.

    while (1)
    {
        if (t_phys >= dt)
        {
            // Exit loop when physical time reaches target
            break;
        }

        // Leapfrog step 1: Evaluate force at current position
        double fval = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_cur);

        // Leapfrog step 2: First half-kick for velocity
        double v_half = v_cur + 0.5 * deltaTau * fval;

        // Leapfrog step 3: Full drift for position
        double rho_next = rho_cur + deltaTau * dRhoDtaufun(rho_cur, v_half);

        // Leapfrog step 4: Second half-kick with force at new position
        double fval2 = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_next);
        double v_next = v_half + 0.5 * deltaTau * fval2;

        // Update physical time using midpoint rho value
        double rho_mid = 0.5 * (rho_cur + rho_next);
        double t_next = t_phys + deltaTau * (rho_mid * rho_mid);

        double tau_next = tau_cur + deltaTau;

        // Handle case where step overshoots target time
        if (t_next >= dt)
        {
            double alpha = 0.0;
            if (fabs(t_next - t_phys) > 1.0e-30)
            {
                alpha = (dt - t_phys) / (t_next - t_phys);
            }
            else
            {
                alpha = 1.0;
            }

            // Interpolate state to exact target time
            double rho_f = rho_cur + alpha * (rho_next - rho_cur);
            double v_f = v_cur + alpha * (v_next - v_cur);

            // Transform back to physical coordinates
            double r_f = rho_f * rho_f;
            *r_out = r_f;
            *v_out = v_f;
            return;
        }

        // Update state for next iteration
        rho_cur = rho_next;
        v_cur = v_next;
        tau_cur = tau_next;
        t_phys = t_next;

        stepCount++;
        if (stepCount > stepMax)
        {
            // Safety exit if maximum iteration count exceeded
            *r_out = rho_cur * rho_cur;
            *v_out = v_cur;
            return;
        }
    }

    // Transform final state back to physical coordinates
    *r_out = rho_cur * rho_cur;
    *v_out = v_cur;

    return;
}

// =========================================================================
// ADAPTIVE FULL LEVI-CIVITA REGULARIZATION
// =========================================================================
//
// Enhanced regularization scheme that combines adaptive step sizing with
// Levi-Civita coordinate transformation for optimal performance near
// the coordinate origin.

/**
 * @brief Performs Levi-Civita regularized integration using a fixed number of micro-steps.
 * @details This helper function advances the particle state in regularized coordinates
 *          \f$(ρ, v_{rad}, t_{phys})\f$ over a total fictitious time interval `h_tau` (\f$Δτ_{total}\f$)
 *          by taking a specified number of `subSteps` fixed-size micro-steps (\f$δτ = Δτ_{total} / \text{subSteps}\f$).
 *          Each micro-step uses a leapfrog-like scheme (Kick-Drift-Kick for \f$ρ, v_{rad}\f$)
 *          and updates the accumulated physical time \f$t_{phys}\f$ using \f$dt_{phys} = δτ \cdot ρ_{mid}^2\f$.
 *          This function is called by `doSingleTauStepAdaptiveLeviCivita` for its coarse and fine passes.
 *
 * @param i         [in] Particle index (0 to npts-1), for rank in force calculation.
 * @param npts      [in] Total number of particles.
 * @param rho_in    [in] Initial \f$ρ = \sqrt{r}\f$ at the start of the `h_tau` interval.
 * @param v_in      [in] Initial radial velocity \f$v_{rad}\f$ at the start of `h_tau`.
 * @param t_in      [in] Initial accumulated physical time \f$t_{phys}\f$ at the start of `h_tau`.
 * @param subSteps  [in] Number of fixed micro-steps to perform over `h_tau`.
 * @param h_tau     [in] Total fictitious time interval \f$Δτ_{total}\f$ for this integration sequence.
 * @param grav      [in] Gravitational constant G (simulation units).
 * @param ell       [in] Angular momentum per unit mass (kpc^2/Myr).
 * @param rho_out   [out] Pointer to store the final \f$ρ\f$ after `h_tau`.
 * @param v_out     [out] Pointer to store the final \f$v_{rad}\f$ after `h_tau`.
 * @param t_out     [out] Pointer to store the final accumulated \f$t_{phys}\f$ after `h_tau`.
 */
static void doMicroLeviCivita(
    int i,
    int npts,
    double rho_in,
    double v_in,
    double t_in,
    int subSteps,
    double h_tau,
    double grav,
    double ell,
    double *rho_out,
    double *v_out,
    double *t_out)
{

    double dtau = h_tau / (double)subSteps;
    double rho_curr = rho_in;
    double v_curr = v_in;
    double t_curr = t_in;

    for (int ss = 0; ss < subSteps; ss++)
    {
        // Half-kick.
        double fLC = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_curr);
        double v_half = v_curr + 0.5 * dtau * fLC;

        // Drift for rho.
        double rho_next = rho_curr + dtau * (0.5 * rho_curr * v_half);

        // Second half-kick.
        double fLC2 = forceLCfun(i, npts, g_active_halo_mass, grav, ell, rho_next);
        double v_next = v_half + 0.5 * dtau * fLC2;

        // Accumulate physical time t(τ).
        // Simplest approach: use midpoint for rho =>  ρ_mid^2.
        double rho_mid = 0.5 * (rho_curr + rho_next);
        double dt_phys = dtau * (rho_mid * rho_mid);

        t_curr += dt_phys;
        rho_curr = rho_next;
        v_curr = v_next;
    }

    *rho_out = rho_curr;
    *v_out = v_curr;
    *t_out = t_curr;
}

/**
 * @brief Performs a single adaptive step in Levi-Civita coordinates over a proposed fictitious time `h_guess`.
 * @details This function integrates the equations of motion in regularized \f$(ρ, v_{rad}, t_{phys})\f$
 *          coordinates over a proposed fictitious time interval `h_guess` (\f$Δτ_{guess}\f$).
 *          It employs an adaptive refinement strategy by comparing a "coarse" integration
 *          (using \f$2N+1\f$ micro-steps via `doMicroLeviCivita`) with a "fine" integration
 *          (using \f$4N+1\f$ micro-steps). The subdivision factor \f$N\f$ starts at 1 and is
 *          doubled if the relative differences in \f$ρ\f$ and \f$v_{rad}\f$ between coarse and fine
 *          results exceed `radius_tol` and `velocity_tol`, respectively. This continues
 *          up to `max_subdiv`. The final state for the \f$Δτ_{guess}\f$ step is chosen based on
 *          `out_type` (coarse, fine, or Richardson extrapolation).
 *          The function outputs the final \f$ρ_{out}\f$, \f$v_{out}\f$ (radial velocity), and
 *          accumulated physical time \f$t_{out}\f$ corresponding to this adaptive \f$Δτ\f$ step.
 *
 * @param i             [in] Particle index (0 to npts-1), for rank in force calculation.
 * @param npts          [in] Total number of particles.
 * @param rho_in        [in] Initial regularized radial coordinate \f$ρ = \sqrt{r}\f$ at the start of \f$Δτ_{guess}\f$.
 * @param v_in          [in] Initial radial velocity \f$v_{rad}\f$ at the start of \f$Δτ_{guess}\f$.
 * @param t_in          [in] Initial accumulated physical time \f$t_{phys}\f$ at the start of \f$Δτ_{guess}\f$.
 * @param h_guess       [in] The proposed total fictitious time step \f$Δτ_{guess}\f$ for this adaptive step.
 * @param radius_tol    [in] Relative convergence tolerance for \f$ρ\f$ comparison.
 * @param velocity_tol  [in] Relative convergence tolerance for \f$v_{rad}\f$ comparison.
 * @param max_subdiv    [in] Maximum subdivision factor \f$N\f$ for micro-steps within `doMicroLeviCivita`.
 * @param grav          [in] Gravitational constant G (simulation units).
 * @param ell           [in] Angular momentum per unit mass (kpc^2/Myr).
 * @param out_type      [in] Result selection for converged micro-integration: 0=coarse, 1=fine, 2=Richardson.
 * @param rho_out       [out] Pointer to store the final \f$ρ\f$ after the adaptive \f$Δτ_{guess}\f$ step.
 * @param v_out         [out] Pointer to store the final \f$v_{rad}\f$ after the adaptive \f$Δτ_{guess}\f$ step.
 * @param t_out         [out] Pointer to store the final accumulated physical time \f$t_{phys}\f$ after this step.
 */
static void doSingleTauStepAdaptiveLeviCivita(
    int i, int npts,
    double rho_in,
    double v_in,
    double t_in,
    double h_guess,
    double radius_tol,
    double velocity_tol,
    int max_subdiv,
    double grav,
    double ell,
    int out_type,
    double *rho_out,
    double *v_out,
    double *t_out)
{
    // Initialize with the smallest subdivision factor N=1
    int N = 1;

    while (N <= max_subdiv)
    {
        // COARSE integration: use subSteps = 2N+1
        double rhoC, vC, tC;
        doMicroLeviCivita(
            i, npts,
            rho_in, v_in,
            t_in,
            (2 * N + 1),
            h_guess,
            grav,
            ell,
            &rhoC, &vC, &tC);

        // FINE integration: use subSteps = 4N+1
        double rhoF, vF, tF;
        doMicroLeviCivita(
            i, npts,
            rho_in, v_in,
            t_in,
            (4 * N + 1),
            h_guess,
            grav,
            ell,
            &rhoF, &vF, &tF);

        // Compare final radius and velocity for convergence
        // Calculate relative differences between coarse and fine solutions

        double rF = rhoF * rhoF;

        double rhodif = fabs(rhoF - rhoC) / (fabs(rF) + 1.0e-30); // Relative radial difference
        double vdif = fabs(vF - vC) / (fabs(vF) + 1.0e-30);       // Relative velocity difference

        if ((rhodif < radius_tol) && (vdif < velocity_tol))
        {
            if (out_type == 0)
            {
                *rho_out = rhoC;
                *v_out = vC;
                *t_out = tC;
            }
            else if (out_type == 1)
            {
                *rho_out = rhoF;
                *v_out = vF;
                *t_out = tF;
            }
            else
            {
                // Apply Richardson extrapolation formula: result = 4*fine - 3*coarse
                // This provides a higher-order approximation by eliminating leading error terms
                double rho_rich = 4.0 * rhoF - 3.0 * rhoC; // Extrapolated ρ value
                double v_rich = 4.0 * vF - 3.0 * vC;       // Extrapolated velocity
                double t_rich = 4.0 * tF - 3.0 * tC;       // Extrapolated time value

                // Ensure non-negative radius
                if (rho_rich < 0.0)
                    rho_rich = 0.0;
                *rho_out = rho_rich;
                *v_out = v_rich;
                *t_out = t_rich;
            }
            return;
        }
        else
        {
            // Not converged, double subdivision factor and try again
            N *= 2;
        }
    }

    // Convergence not achieved within max_subdiv iterations
    // Use highest-resolution fine integration as fallback result
    {
        double rhoF, vF, tF;
        doMicroLeviCivita(
            i, npts,
            rho_in, v_in,
            t_in,
            (4 * N + 1),
            h_guess,
            grav,
            ell,
            &rhoF, &vF, &tF);

        *rho_out = rhoF;
        *v_out = vF;
        *t_out = tF;
    }
}

/**
 * @brief Performs a full adaptive leapfrog step using Levi-Civita regularization over a physical time interval `dt`.
 * @details This function integrates a particle's motion over a physical timestep `dt` (\f$\Delta T_{phys}\f$)
 *          by taking multiple adaptive steps in fictitious Levi-Civita time \f$τ\f$.
 *          It repeatedly calls `doSingleTauStepAdaptiveLeviCivita` to advance the state in
 *          \f$(ρ, v_{rad}, t_{phys})\f$ coordinates, where \f$ρ = \sqrt{r}\f$. Each call to
 *          `doSingleTauStepAdaptiveLeviCivita` takes an adaptive \f$Δτ\f$ step.
 *          The loop continues until the accumulated physical time `t_cur` (from summing \f$Δt_{phys}\f$
 *          corresponding to each \f$Δτ\f$) reaches or exceeds the target `dt`.
 *          If a \f$Δτ\f$ step overshoots `dt`, linear interpolation is used to find the
 *          state precisely at `t_cur = dt`. The final regularized state \f$(ρ_f, v_f)\f$ is then
 *          transformed back to physical coordinates \f$(r_{out}, v_{out})\f$.
 *          An initial guess for \f$Δτ\f$ is made based on `N_taumin` and the initial radius.
 *
 * @param i             [in] Particle index (0 to npts-1), used for rank in gravitational force calculation.
 * @param npts          [in] Total number of particles in the simulation.
 * @param r_in          [in] Initial physical radial position (kpc) at the start of the physical step `dt`.
 * @param v_in          [in] Initial physical radial velocity (kpc/Myr) at the start of `dt`.
 * @param ell           [in] Angular momentum per unit mass (kpc^2/Myr), conserved during integration.
 * @param dt            [in] The full physical timestep \f$\Delta T_{phys}\f$ (Myr) to advance the particle.
 * @param N_taumin      [in] Target number of fictitious \f$τ\f$-steps within `dt`; influences the initial \f$Δτ\f$ guess.
 * @param radius_tol    [in] Relative convergence tolerance for \f$ρ\f$ comparison within each adaptive \f$τ\f$-step.
 * @param velocity_tol  [in] Relative convergence tolerance for velocity comparison within each adaptive \f$τ\f$-step.
 * @param max_subdiv    [in] Maximum allowed subdivision factor N for micro-steps within each adaptive \f$τ\f$-step.
 * @param grav          [in] Gravitational constant G (simulation units, e.g., G_CONST).
 * @param out_type      [in] Result selection mode for micro-steps within `doSingleTauStepAdaptiveLeviCivita`:
 *                         0 for coarse, 1 for fine, 2 for Richardson extrapolation.
 * @param r_out         [out] Pointer to store the final physical radial position (kpc) after time `dt`.
 * @param v_out         [out] Pointer to store the final physical radial velocity (kpc/Myr) after time `dt`.
 */
static void doAdaptiveFullLeviCivita(
    int i,
    int npts,
    double r_in,
    double v_in,
    double ell,
    double dt, // big step in physical time
    int N_taumin,
    double radius_tol,
    double velocity_tol,
    int max_subdiv,
    double grav,
    int out_type, // 0=coarse,1=fine,2=Richardson
    double *r_out,
    double *v_out)
{
    // Handle near-zero radius edge case
    if (r_in < 1.0e-30)
    {
        *r_out = r_in;
        *v_out = v_in;
        return;
    }

    // Transform from physical to regularized coordinates
    double rho_current = sqrt(r_in);
    double v_current = v_in;

    double t_cur = 0.0; // Current physical time.

    // Estimate initial fictitious time step deltaTau
    double deltaTau = 0.0;
    if (r_in > 1.0e-30 && N_taumin > 0)
    {
        deltaTau = dt / (2.0 * r_in * N_taumin);
    }
    else
    {
        deltaTau = dt / 100.0;
    }

    int stepCount = 0;
    int stepMax = 100000000; // Safety limit on iteration count

    while (1)
    {
        if (t_cur >= dt)
            break; // Exit when physical time target is reached

        double rho_next, v_next, t_next;
        doSingleTauStepAdaptiveLeviCivita(
            i, npts,
            rho_current, v_current,
            t_cur,    // physical time in
            deltaTau, // guess for τ
            radius_tol,
            velocity_tol,
            max_subdiv,
            grav,
            ell,
            out_type,
            &rho_next, &v_next, &t_next);

        if (t_next > dt)
        {
            // Handle overshoot case with linear interpolation
            double alpha = 0.0;
            if (fabs(t_next - t_cur) > 1.0e-30)
                alpha = (dt - t_cur) / (t_next - t_cur);

            // Interpolate to exact target time dt
            double rho_final = rho_current + alpha * (rho_next - rho_current);
            double v_final = v_current + alpha * (v_next - v_current);

            double r_fin = rho_final * rho_final;
            *r_out = r_fin;
            *v_out = v_final;
            return;
        }
        else
        {
            // Update state variables for next iteration
            rho_current = rho_next;
            v_current = v_next;
            t_cur = t_next;
        }

        // Check for iteration limit
        stepCount++;
        if (stepCount > stepMax)
        {
            // Return best available result if maximum iterations reached
            double r_fin = rho_current * rho_current;
            *r_out = r_fin;
            *v_out = v_current;
            return;
        }
    }

    // Transform final regularized state back to physical coordinates
    double r_fin = rho_current * rho_current;
    *r_out = r_fin;
    *v_out = v_current;
}

// =========================================================================
// FILE I/O AND DATA MANAGEMENT SUBSYSTEM
// =========================================================================
//
// Functions for saving, loading, and managing simulation data including:
// - Initial condition generation and I/O
// - Snapshot file management
// - Binary file format utilities

static int doReadInit = 0;            ///< Flag indicating whether to read initial conditions from file (1=yes, 0=no).
static int doWriteInit = 0;           ///< Flag indicating whether to write initial conditions to file (1=yes, 0=no).
static const char *readInitFilename = NULL; ///< Filename to read initial conditions from (if doReadInit=1).
static const char *writeInitFilename = NULL; ///< Filename to write initial conditions to (if doWriteInit=1).

// =========================================================================
// INITIAL CONDITION FILE I/O FUNCTIONS
// =========================================================================

/**
 * @brief Writes particle initial conditions to a binary file.
 * @details Stores the complete initial particle state (radius, velocity, angular momentum,
 *          original index, orientation parameter mu) and the particle count (`npts`)
 *          to a binary file for later retrieval via `read_initial_conditions`.
 *          Opens the file in write binary mode ("wb"). First writes the integer `npts`,
 *          then writes the 5 double-precision values for each particle sequentially.
 *
 * Parameters
 * ----------
 * particles : double**
 *     2D array containing particle properties [component][particle_index].
 *     Expected components: 0=rad, 1=vel, 2=angmom, 3=orig_idx, 4=mu.
 * npts : int
 *     Number of particles to write.
 * filename : const char*
 *     Path to the output binary file.
 *
 * Returns
 * -------
 * None
 *
 * @note The binary file format is: `npts` (int32), followed by `npts` records,
 *       each consisting of 5 `double` values (radius, velocity, ang. mom., orig. index, mu).
 * @warning Prints an error to stderr if the file cannot be opened.
 *
 * @see read_initial_conditions
 */
static void write_initial_conditions(double **particles, int npts, const char *filename)
{
    FILE *fp = fopen(filename, "wb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open '%s' for writing initial conditions.\n", filename);
        return;
    }

    // Write the number of particles first
    fwrite(&npts, sizeof(int), 1, fp);

    for (int i = 0; i < npts; i++)
    {
        double r_val = particles[0][i];
        double v_val = particles[1][i];
        double ell_val = particles[2][i];
        double idx_val = particles[3][i];
        double mu_val = particles[4][i];

        fwrite(&r_val, sizeof(double), 1, fp);
        fwrite(&v_val, sizeof(double), 1, fp);
        fwrite(&ell_val, sizeof(double), 1, fp);
        fwrite(&idx_val, sizeof(double), 1, fp);
        fwrite(&mu_val, sizeof(double), 1, fp);
    }

    fclose(fp);
    log_message("INFO", "Wrote initial conditions (%d particles) to '%s'", npts, filename);
}

/**
 * @brief Reads particle initial conditions from a binary file.
 * @details Loads the complete particle state (radius, velocity, angular momentum,
 *          original index, orientation parameter mu) from a binary file previously
 *          created by `write_initial_conditions`. Verifies that the number of
 *          particles read from the file matches the expected count `npts`.
 *          Opens the file in read binary mode ("rb").
 *
 * Parameters
 * ----------
 * particles : double**
 *     2D array to store the loaded particle properties [component][particle_index].
 *     Must be pre-allocated with dimensions [5][npts].
 * npts : int
 *     Expected number of particles to read.
 * filename : const char*
 *     Path to the input binary file.
 *
 * Returns
 * -------
 * None (populates the `particles` array).
 *
 * @note See `write_initial_conditions` for file format details.
 * @warning Prints an error to stderr if the file cannot be opened, if the particle
 *          count doesn't match `npts` (returns early), or if a read error occurs.
 *
 * @see write_initial_conditions
 */
static void read_initial_conditions(double **particles, int npts, const char *filename)
{
    FILE *fp = fopen(filename, "rb");
    if (!fp)
    {
        fprintf(stderr, "Error: cannot open '%s' for reading initial conditions.\n", filename);
        return;
    }

    int file_npts;
    if (fread(&file_npts, sizeof(int), 1, fp) != 1)
    {
        fprintf(stderr, "Error: failed to read npts from '%s'.\n", filename);
        fclose(fp);
        return;
    }
    if (file_npts != npts)
    {
        fprintf(stderr, "Warning: file npts=%d doesn't match current npts=%d.\n", file_npts, npts);
        fclose(fp);
        return;
    }

    for (int i = 0; i < npts; i++)
    {
        double r_val, v_val, ell_val, idx_val, mu_val;
        if (fread(&r_val, sizeof(double), 1, fp) != 1 || fread(&v_val, sizeof(double), 1, fp) != 1 || fread(&ell_val, sizeof(double), 1, fp) != 1 || fread(&idx_val, sizeof(double), 1, fp) != 1 || fread(&mu_val, sizeof(double), 1, fp) != 1)
        {
            fprintf(stderr, "Error: partial read at i=%d in '%s'.\n", i, filename);
            fclose(fp);
            return;
        }

        particles[0][i] = r_val;
        particles[1][i] = v_val;
        particles[2][i] = ell_val;
        particles[3][i] = idx_val;
        particles[4][i] = mu_val;
    }

    fclose(fp);
    log_message("INFO", "Read initial conditions (%d particles) from '%s'", npts, filename);
}

// =========================================================================
// SORTING ALGORITHM CONFIGURATION
// =========================================================================
///< Default sorting algorithm identifier string. Set based on command-line options.
static const char *g_defaultSortAlg = "quadsort_parallel";

/**
 * @brief Returns a human-readable description for a sort algorithm identifier string.
 * @details Maps internal sort algorithm identifiers (e.g., "quadsort_parallel")
 *          to user-friendly descriptive names (e.g., "Parallel Quadsort").
 *          This is primarily used for display purposes in command-line output
 *          or logs, providing more context than the internal short string identifiers.
 *
 * @param sort_alg [in] The internal algorithm identifier string.
 * @return const char* A descriptive name for the algorithm. If no match is found,
 *                     the input `sort_alg` string itself is returned as a fallback.
 */
const char *get_sort_description(const char *sort_alg)
{
    if (strcmp(sort_alg, "quadsort_parallel") == 0)
        return "Parallel Quadsort";
    if (strcmp(sort_alg, "quadsort") == 0)
        return "Sequential Quadsort";
    if (strcmp(sort_alg, "insertion_parallel") == 0)
        return "Parallel Insertion Sort";
    if (strcmp(sort_alg, "insertion") == 0)
        return "Sequential Insertion Sort";
    return sort_alg; // Default fallback.
}

// =========================================================================
// BINARY FILE I/O UTILITIES
// =========================================================================

/**
 * @brief Writes binary data to a file using a printf-like format string.
 * @details Parses a format string containing simplified specifiers (`%d`, `%f`, `%g`, `%e`).
 *          Writes corresponding arguments from the variadic list (`...`) as binary data.
 *          Integer types (`%d`) are written as `int`.
 *          Floating-point types (`%f`, `%g`, `%e`) are read as `double` from args but
 *          written as `float` to the file for storage efficiency.
 *
 * Parameters
 * ----------
 * fp : FILE*
 *     File pointer to write to (must be opened in binary mode).
 * format : const char*
 *     Format string with specifiers (`%d`, `%f`, `%g`, `%e`). Other characters are ignored.
 * ... :
 *     Variable arguments matching the format specifiers.
 *
 * Returns
 * -------
 * int
 *     The number of items successfully written according to the format string.
 *
 * @note Skips optional width/precision specifiers in the format string.
 * @see fscanf_bin
 */
int fprintf_bin(FILE *fp, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    int count_items = 0;
    const char *p = format;

    while (*p != '\0')
    {
        if (*p == '%')
        {
            p++; // Move past '%'
            
            // Skip format modifiers until we find a type specifier
            while (*p && !strchr("dfge", *p) && !(*p == 'l'))
            {
                p++;
            }

            if (*p == 'd')
            {
                // Process integer format
                int val = va_arg(args, int);
                fwrite(&val, sizeof(val), 1, fp);
                count_items++;
            }
            else if (*p == 'f' || *p == 'g' || *p == 'e')
            {
                // Process float format (stored as 4-byte float)
                double tmp = va_arg(args, double);
                float val = (float)tmp;
                fwrite(&val, sizeof(float), 1, fp);
                count_items++;
            }
        }
        if (*p)
            p++;
    }

    va_end(args);
    return count_items;
}

/**
 * @brief Reads binary data from a file using a scanf-like format string.
 * @details Parses a format string containing simplified specifiers (`%d`, `%f`, `%g`, `%e`).
 *          Reads corresponding binary data from the file and stores it in the
 *          pointer arguments provided in the variadic list (`...`).
 *          Integer types (`%d`) are read as `int`.
 *          Floating-point types (`%f`, `%g`, `%e`) are read as `float` from the file
 *          but stored into `double*` arguments provided by the caller.
 *
 * Parameters
 * ----------
 * fp : FILE*
 *     File pointer to read from (must be opened in binary mode).
 * format : const char*
 *     Format string with specifiers (`%d`, `%f`, `%g`, `%e`). Other characters are ignored.
 * ... :
 *     Variable pointer arguments matching the format specifiers (e.g., `int*`, `double*`).
 *
 * Returns
 * -------
 * int
 *     The number of items successfully read and assigned according to the format string.
 *     Stops reading on the first failure or EOF.
 *
 * @note Skips optional width/precision specifiers in the format string.
 * @see fprintf_bin
 */
int fscanf_bin(FILE *fp, const char *format, ...)
{
    va_list args;
    va_start(args, format);

    int count_items = 0;
    const char *p = format;

    while (*p != '\0')
    {
        if (*p == '%')
        {
            p++; // Move past '%'
            
            // Skip format modifiers until we reach a type specifier
            while (*p && !strchr("dfge", *p) && !(*p == 'l'))
            {
                p++;
            }

            if (*p == 'd')
            {
                // Process integer format
                int *iptr = va_arg(args, int *);
                size_t nread = fread(iptr, sizeof(int), 1, fp);
                if (nread == 1)
                    count_items++;
                else
                    return count_items;
            }
            else if (*p == 'f' || *p == 'g' || *p == 'e')
            {
                // Read 4-byte float from file
                float fval;
                size_t nread = fread(&fval, sizeof(float), 1, fp);
                if (nread == 1)
                {
                    // Store in caller's double pointer with type conversion
                    double *dptr = va_arg(args, double *);
                    *dptr = (double)fval;
                    count_items++;
                }
                else
                {
                    return count_items; // Stop on read failure
                }
            }
        }
        if (*p)
            p++;
    }

    va_end(args);
    return count_items;
}

/**
 * @brief Parses sub-arguments for the `--save` command-line option.
 * @details This function is called when the `--save` option is encountered during
 *          command-line argument parsing. It reads subsequent arguments (until another
 *          option starting with '-' is found, or arguments end) which specify the
 *          level or type of data to save. It then sets the corresponding global data
 *          output flags (`g_doDebug`, `g_doDynPsi`, `g_doDynRank`, `g_doAllParticleData`)
 *          based on the highest priority valid sub-argument encountered.
 *          Valid sub-arguments and their priority (lowest to highest):
 *          - "raw-data": Enables `g_doAllParticleData`.
 *          - "psi-snaps": Enables `g_doAllParticleData`, `g_doDynPsi`.
 *          - "full-snaps": Enables `g_doAllParticleData`, `g_doDynPsi`, `g_doDynRank`.
 *          - "all" or "debug-energy": Enables all flags (`g_doDebug`, `g_doDynPsi`, `g_doDynRank`, `g_doAllParticleData`).
 *
 * @param argc   [in] The total argument count from `main()`.
 * @param argv   [in] The argument array from `main()`.
 * @param pIndex [in,out] Pointer to the current index in `argv`. On input, it points to the
 *                       `--save` option. On output, it is updated to point to the last
 *                       sub-argument consumed by this function.
 * @note Exits the program with an error message if an unknown sub-argument to `--save` is found.
 */
static void parseSaveArgs(int argc, char *argv[], int *pIndex)
{
    // Use an integer priority to track the highest level of saving requested
    static int savePriority = 0; // 0=none, 1=raw, 2=psi, 3=full, 4=all/debug

    // Start checking arguments after "--save"
    int i = *pIndex + 1;

    // Process arguments until the end or another option (starting with '-') is found
    while (i < argc && argv[i][0] != '-')
    {
        const char *subarg = argv[i];

        if (strcmp(subarg, "all") == 0 || strcmp(subarg, "debug-energy") == 0)
        {
            savePriority = 4; // Highest priority
        }
        else if (strcmp(subarg, "full-snaps") == 0)
        {
            if (savePriority < 3)
                savePriority = 3;
        }
        else if (strcmp(subarg, "psi-snaps") == 0)
        {
            if (savePriority < 2)
                savePriority = 2;
        }
        else if (strcmp(subarg, "raw-data") == 0)
        {
            if (savePriority < 1)
                savePriority = 1;
        }
        else
        {
            fprintf(stderr, "Error: unknown argument to --save '%s'\n", subarg);
            exit(1);
        }

        i++;
    }

    // Set global flags based on the highest priority encountered
    switch (savePriority)
    {
    case 4: // All or debug-energy
        g_doDebug = 1;
        g_doDynPsi = 1;
        g_doDynRank = 1;
        g_doAllParticleData = 1;
        break;
    case 3: // Full-snaps.
        g_doDebug = 0;
        g_doDynPsi = 1;
        g_doDynRank = 1;
        g_doAllParticleData = 1;
        break;
    case 2: // Psi-snaps.
        g_doDebug = 0;
        g_doDynPsi = 1;
        g_doDynRank = 0;
        g_doAllParticleData = 1;
        break;
    case 1: // Raw-data.
        g_doDebug = 0;
        g_doDynPsi = 0;
        g_doDynRank = 0;
        g_doAllParticleData = 1;
        break;
    default: // No saving option specified, all flags remain 0
        break;
    }

    // Update index to point to the last processed argument
    *pIndex = i - 1;
}

/**
 * @brief Remaps original particle IDs to their zero-based rank among a given set of particles.
 * @details This function is typically used after a process like tidal stripping, where a
 *          subset of particles remains. The `orig_ids` array at this point contains the
 *          original, potentially non-contiguous, IDs of these remaining `n` particles.
 *          The function sorts these original IDs and then replaces each ID in the input
 *          array with its new rank (0 to n-1) within this sorted sequence.
 *          This effectively transforms arbitrary original ID values into a compact,
 *          contiguous sequence of rank IDs for the final set of particles.
 *          The input `orig_ids` array (which is `particles[3]` in `main`) is modified in-place.
 *
 * @param orig_ids [in,out] Pointer to an array of particle IDs (stored as doubles).
 *                        These are modified in-place to become rank IDs.
 * @param n        [in] The number of elements in the `orig_ids` array (i.e., `npts` after stripping).
 * @note Uses `qsort` and a temporary array for sorting. Exits via `exit(1)` if memory
 *       allocation for the temporary array fails.
 */
static void reassign_orig_ids_with_rank(double *orig_ids, int n)
{
    if (n <= 0) return; // Handle empty or invalid input

    double *temp = (double *)malloc(n * sizeof(double));
    if (temp == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed in reassign_orig_ids_with_rank\n");
        exit(1);
    }
    
    // Step 1: Copy original IDs to temporary array
    for (int i = 0; i < n; i++)
    {
        temp[i] = orig_ids[i];
    }
    
    // Step 2: Sort temporary array to establish rank order
    qsort(temp, n, sizeof(double), double_cmp);
    
    // Step 3: Find rank of each original ID using binary search
    for (int i = 0; i < n; i++)
    {
        int low = 0, high = n - 1, rank = -1;
        while (low <= high)
        {
            int mid = (low + high) / 2;
            if (temp[mid] == orig_ids[i])
            {
                rank = mid;
                break;
            }
            else if (temp[mid] < orig_ids[i])
            {
                low = mid + 1;
            }
            else
            {
                high = mid - 1;
            }
        }
        if (rank == -1)
        {
            // Use insertion point as fallback if exact match not found
            rank = low;
        }
        
        // Step 4: Replace original ID with its rank
        orig_ids[i] = (double)rank;
    }
    free(temp);
}

// =========================================================================
// SIGNAL PROCESSING AND FILTERING UTILITIES
// =========================================================================
//
// Advanced numerical processing utilities for density field handling including:
// - FFT-based convolution for density smoothing
// - Direct Gaussian convolution for smaller datasets
// - Signal filtering and processing functions
// =========================================================================
// FFT METHODS AND CONVOLUTION IMPLEMENTATIONS
// =========================================================================
/**
 * @brief Applies Gaussian smoothing using FFT-based convolution (thread-safe via critical section).
 * @details Smooths a density field defined on a potentially non-uniform grid
 *          (`log_r_grid`) using FFT convolution with a Gaussian kernel of width
 *          `sigma_log` (defined in log-space). Uses zero-padding to avoid
 *          wrap-around artifacts. This is generally faster than direct convolution
 *          for large `grid_size`. Assumes log_r_grid is uniformly spaced.
 *
 * Parameters
 * ----------
 * density_grid : const double*
 *     Input density grid array (values corresponding to `log_r_grid`).
 * grid_size : int
 *     Number of points in the input grid and density arrays.
 * log_r_grid : const double*
 *     Array of logarithmic radial grid coordinates (log10(r)). Must be uniformly spaced.
 * sigma_log : double
 *     Width (standard deviation) of the Gaussian kernel in log10-space.
 * result : double*
 *     Output array (pre-allocated, size `grid_size`) for the smoothed density field.
 *
 * Returns
 * -------
 * None (populates the `result` array).
 *
 * @note Uses FFTW library for Fast Fourier Transforms (`fftw_malloc`, `fftw_plan_dft_r2c_1d`, etc.).
 *       Requires FFTW to be installed. FFTW operations are protected by `omp critical(fftw)`.
 * @note Resulting smoothed density is clamped to a minimum value of 1e-10.
 * @warning Prints errors to stderr and returns early on memory allocation failures or
 *          FFTW plan creation failures.
 * @see direct_gaussian_convolution
 * @see gaussian_convolution
 */
void fft_gaussian_convolution(
    const double *density_grid,
    int grid_size,
    const double *log_r_grid,
    double sigma_log,
    double *result)
{
#pragma omp critical(fftw)
    {
        // Create zero-padded arrays (step 1)
        int padded_size = 2 * grid_size;

        // Allocate memory for padded input array
        double *padded_input = (double *)fftw_malloc(sizeof(double) * padded_size);
        if (!padded_input)
        {
            fprintf(stderr, "Error: Failed to allocate memory for padded_input\n");
            return;
        }

        double *padded_kernel = (double *)fftw_malloc(sizeof(double) * padded_size);
        if (!padded_kernel)
        {
            fprintf(stderr, "Error: Failed to allocate memory for padded_kernel\n");
            fftw_free(padded_input);
            return;
        }

        double *padded_output = (double *)fftw_malloc(sizeof(double) * padded_size);
        if (!padded_output)
        {
            fprintf(stderr, "Error: Failed to allocate memory for padded_output\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            return;
        }

        // Initialize padded arrays with zeros
        for (int i = 0; i < padded_size; i++)
        {
            padded_input[i] = 0.0;
            padded_kernel[i] = 0.0;
        }

        // Copy input data to first half of padded array
        for (int i = 0; i < grid_size; i++)
        {
            padded_input[i] = density_grid[i];
        }

        // Create Gaussian kernel in spatial domain
        double dlog = log_r_grid[1] - log_r_grid[0]; // Grid spacing in log space.
        double norm = 0.0;

        for (int i = 0; i < grid_size; i++)
        {
            // Distance in log space.
            double x = i * dlog;

            // Gaussian kernel centered at 0.
            double kernel_val = (1.0 / (sigma_log * sqrt(2.0 * M_PI))) *
                                exp(-0.5 * (x / sigma_log) * (x / sigma_log));

            padded_kernel[i] = kernel_val;
            norm += kernel_val;
        }

        // Normalize the kernel for unit sum
        for (int i = 0; i < grid_size; i++)
        {
            padded_kernel[i] /= norm;
        }

        // Prepare for FFT computation (step 2)
        fftw_complex *fft_input = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (padded_size / 2 + 1));
        if (!fft_input)
        {
            fprintf(stderr, "Error: Failed to allocate memory for fft_input\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            return;
        }

        fftw_complex *fft_kernel = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (padded_size / 2 + 1));
        if (!fft_kernel)
        {
            fprintf(stderr, "Error: Failed to allocate memory for fft_kernel\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            return;
        }

        fftw_complex *fft_output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (padded_size / 2 + 1));
        if (!fft_output)
        {
            fprintf(stderr, "Error: Failed to allocate memory for fft_output\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            return;
        }

        fftw_plan plan_forward_input = fftw_plan_dft_r2c_1d(padded_size, padded_input, fft_input, FFTW_ESTIMATE);
        if (!plan_forward_input)
        {
            fprintf(stderr, "Error: Failed to create forward FFTW plan for input\n");
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            fftw_free(fft_output);
            return;
        }

        fftw_plan plan_forward_kernel = fftw_plan_dft_r2c_1d(padded_size, padded_kernel, fft_kernel, FFTW_ESTIMATE);
        if (!plan_forward_kernel)
        {
            fprintf(stderr, "Error: Failed to create forward FFTW plan for kernel\n");
            fftw_destroy_plan(plan_forward_input);
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            fftw_free(fft_output);
            return;
        }

        fftw_plan plan_backward = fftw_plan_dft_c2r_1d(padded_size, fft_output, padded_output, FFTW_ESTIMATE);
        if (!plan_backward)
        {
            fprintf(stderr, "Error: Failed to create backward FFTW plan\n");
            fftw_destroy_plan(plan_forward_input);
            fftw_destroy_plan(plan_forward_kernel);
            fftw_free(padded_input);
            fftw_free(padded_kernel);
            fftw_free(padded_output);
            fftw_free(fft_input);
            fftw_free(fft_kernel);
            fftw_free(fft_output);
            return;
        }

        // Execute forward FFTs.
        fftw_execute(plan_forward_input);
        fftw_execute(plan_forward_kernel);

        // Perform complex multiplication in frequency domain (convolution in spatial domain)
        // For each frequency component, multiply signal and kernel transforms
        for (int i = 0; i < padded_size / 2 + 1; i++)
        {
            double re_in = fft_input[i][0];   // Real part of input transform
            double im_in = fft_input[i][1];   // Imaginary part of input transform
            double re_ker = fft_kernel[i][0]; // Real part of kernel transform
            double im_ker = fft_kernel[i][1]; // Imaginary part of kernel transform

            // Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
            fft_output[i][0] = re_in * re_ker - im_in * im_ker; // Real component
            fft_output[i][1] = re_in * im_ker + im_in * re_ker; // Imaginary component
        }

        // Execute inverse FFT.
        fftw_execute(plan_backward);

        // Apply normalization to compensate for FFTW's unnormalized inverse transform
        // FFTW's implementation requires division by array length to get properly scaled result
        for (int i = 0; i < padded_size; i++)
        {
            padded_output[i] /= padded_size; // Scale by 1/N to get normalized values
        }

        // Extract the valid portion of the convolution result
        // Only the first grid_size elements contain the actual result (rest is padding)
        for (int i = 0; i < grid_size; i++)
        {
            result[i] = padded_output[i];

            // Enforce minimum density threshold to avoid numerical instability
            // Consistent with minimum threshold in direct convolution method
            if (result[i] < 1e-10)
            {
                result[i] = 1e-10;
            }
        }

        // Release FFTW resources
        fftw_destroy_plan(plan_forward_input);
        fftw_destroy_plan(plan_forward_kernel);
        fftw_destroy_plan(plan_backward);

        fftw_free(padded_input);
        fftw_free(padded_kernel);
        fftw_free(padded_output);
        fftw_free(fft_input);
        fftw_free(fft_kernel);
        fftw_free(fft_output);
    } // End of critical section.
}

/**
 * @brief Applies Gaussian smoothing using direct convolution.
 * @details Smooths a density field defined on a potentially non-uniform grid
 *          (`log_r_grid`) using direct spatial convolution with a Gaussian kernel
 *          of width `sigma_log` (defined in log-space).
 *          For each point `i` in the output `result` array, it computes a weighted
 *          sum of the input `density_grid` values:
 *          `result[i] = sum(density_grid[j] * kernel(log_r_grid[i] - log_r_grid[j])) / sum(kernel(...))`
 *          where the `kernel` is a Gaussian function `G(x) = (1/(σ√2π)) * exp(-0.5*(x/σ)²)`,
 *          with `x` being the distance `log_r_grid[i] - log_r_grid[j]` and `σ = sigma_log`.
 *          This method is generally more accurate than FFT-based convolution, especially
 *          for non-uniform grids or near boundaries, but has a higher computational
 *          cost (O(N²)) which makes it slower for large `grid_size`.
 *
 * Parameters
 * ----------
 * density_grid : const double*
 *     Input density grid array (values corresponding to `log_r_grid`).
 * grid_size : int
 *     Number of points in the input grid and density arrays.
 * log_r_grid : const double*
 *     Array of logarithmic radial grid coordinates (log10(r)). Can be non-uniformly spaced.
 * sigma_log : double
 *     Width (standard deviation) of the Gaussian kernel in log10-space.
 * result : double*
 *     Output array (pre-allocated, size `grid_size`) for the smoothed density field.
 *
 * Returns
 * -------
 * None (populates the `result` array).
 *
 * @note Resulting smoothed density is clamped to a minimum value of 1e-10.
 *       This function is inherently thread-safe as it only reads inputs and writes
 *       to distinct elements of the output array without shared intermediate state.
 * @see fft_gaussian_convolution
 * @see gaussian_convolution
 */
void direct_gaussian_convolution(
    const double *density_grid,
    int grid_size,
    const double *log_r_grid,
    double sigma_log,
    double *result)
{
    // Precompute normalization factor for Gaussian kernel if sigma is valid
    double kernel_norm_factor = (sigma_log > 1e-15) ? (1.0 / (sigma_log * sqrt(2.0 * M_PI))) : 1.0;
    double sig_sq_inv = (sigma_log > 1e-15) ? (1.0 / (sigma_log * sigma_log)) : 0.0; // Avoid division by zero

    // Perform direct convolution
    for (int i = 0; i < grid_size; i++)
    {
        double sum = 0.0;
        double norm = 0.0;

        for (int j = 0; j < grid_size; j++)
        {
            double dlog_r = log_r_grid[i] - log_r_grid[j];
            double kernel = (sigma_log > 1e-15) ? 
                (kernel_norm_factor * exp(-0.5 * dlog_r * dlog_r * sig_sq_inv)) : 
                ((i == j) ? 1.0 : 0.0); // Delta function if sigma=0

            sum += density_grid[j] * kernel;
            norm += kernel;
        }

        // Normalize the result, handle potential division by zero if norm is too small
        result[i] = (norm > 1e-15) ? (sum / norm) : density_grid[i]; // Fallback to original value if norm is zero

        // Clamp to minimum density
        if (result[i] < 1e-10)
        {
            result[i] = 1e-10;
        }
    }
}

/**
 * @brief Controls the convolution method selection for density smoothing.
 * @details Global flag that determines whether to use direct convolution (1)
 *          or FFT-based convolution (0). Direct convolution is more accurate
 *          but slower for large grids, while FFT-based convolution is faster
 *          but may introduce artifacts.
 *
 * @note External declaration, defined elsewhere.
 * @see gaussian_convolution
 */
extern int debug_direct_convolution;

/**
 * @brief Performs Gaussian smoothing by selecting the appropriate convolution method.
 * @details Acts as a routing function that delegates density smoothing to either
 *          `direct_gaussian_convolution` or `fft_gaussian_convolution` based on the
 *          value of the global `debug_direct_convolution` flag.
 *          - If `debug_direct_convolution` is non-zero, `direct_gaussian_convolution`
 *            is called (more accurate, O(N²) complexity, suitable for smaller or
 *            non-uniform grids).
 *          - If `debug_direct_convolution` is zero (default), `fft_gaussian_convolution`
 *            is called (faster for large grids, O(N log N) complexity, requires
 *            uniformly spaced logarithmic grid).
 *
 * Parameters
 * ----------
 * density_grid : const double*
 *     Input density grid array (values corresponding to `log_r_grid`).
 * grid_size : int
 *     Number of points in the input grid and density arrays.
 * log_r_grid : const double*
 *     Array of logarithmic radial grid coordinates (log10(r)). Must be uniform if FFT is used.
 * sigma_log : double
 *     Width (standard deviation) of the Gaussian kernel in log10-space.
 * result : double*
 *     Output array (pre-allocated, size `grid_size`) for the smoothed density field.
 *
 * Returns
 * -------
 * None (populates the `result` array).
 *
 * @see direct_gaussian_convolution
 * @see fft_gaussian_convolution
 * @see debug_direct_convolution
 */
void gaussian_convolution(
    const double *density_grid,
    int grid_size,
    const double *log_r_grid,
    double sigma_log,
    double *result)
{
    // Select convolution method based on global configuration flag.
    if (debug_direct_convolution)
    {
        // Use direct spatial-domain convolution.
        direct_gaussian_convolution(density_grid, grid_size, log_r_grid, sigma_log, result);
    }
    else
    {
        // Use FFT-based frequency-domain convolution.
        fft_gaussian_convolution(density_grid, grid_size, log_r_grid, sigma_log, result);
    }
}

// =========================================================================
// RESTART AND RECOVERY MANAGEMENT
// =========================================================================

/**
 * @brief Finds the index of the last successfully processed and written snapshot in restart mode.
 * @details This function is called when `--restart` is active. It checks for the existence
 *          and basic integrity of snapshot data files (e.g., `Rank_Mass_Rad_VRad_unsorted_t%05d.dat`
 *          and `Rank_Mass_Rad_VRad_sorted_t%05d.dat`) corresponding to the timesteps listed
 *          in `snapshot_steps`.
 *          Integrity is checked by comparing file sizes against those of the first snapshot's
 *          files (snapshot_steps[0]), allowing for a small percentage tolerance (typically +/- 5%).
 *          The function aims to determine from which snapshot index `s` (in `snapshot_steps`)
 *          the post-processing (e.g., Rank file generation) should resume.
 *          It assumes `g_file_suffix` is correctly set to identify the relevant run's files.
 *
 * @param snapshot_steps [in] Array of integer timestep numbers for which snapshot data
 *                           was expected to be written (often these are the `dtwrite` intervals).
 * @param noutsnaps      [in] The total number of snapshot indices in the `snapshot_steps` array.
 *
 * @return int The index `s` into `snapshot_steps` corresponding to the *first snapshot that
 *             needs to be processed* (i.e., last valid snapshot index + 1).
 *             - Returns -1 if no valid snapshot files are found (implying processing should start from index 0).
 *             - Returns -2 if all expected snapshot files for all unique snapshot numbers exist and
 *               appear valid (implying no further snapshot processing is needed for this phase).
 *             Prints status messages to stdout and logs warnings/errors.
 */
static int find_last_processed_snapshot(int *snapshot_steps, int noutsnaps)
{
    // Validate input parameters before proceeding
    if (snapshot_steps == NULL || noutsnaps <= 0)
    {
        printf("ERROR: Invalid snapshot_steps or noutsnaps in find_last_processed_snapshot\n");
        return -1; // Start from beginning.
    }

    int last_valid_snap = -1;
    int last_valid_index = -1;
    long reference_unsorted_size = 0;
    long reference_sorted_size = 0;

    printf("Restart mode: Checking for existing data products with suffix '%s'...\n\n", g_file_suffix);

    // Initialize to track if we've checked all snapshots.
    int all_snapshots_checked = 1; // Assume all checked until we find a problem.
    int checked_count = 0;         // Count how many files we've actually checked.
    int unique_snapshots = 0;      // Count how many unique snapshot numbers we have.

    // Calculate how many unique snapshot numbers there are (might be less than noutsnaps).
    int total_writes = 0; // Determine max snapshot number.
    for (int i = 0; i < noutsnaps; i++)
    {
        if (snapshot_steps[i] > total_writes)
        {
            total_writes = snapshot_steps[i];
        }
    }
    total_writes++; // Convert from 0-based index to count.

    {
        int *seen = (int *)calloc(total_writes, sizeof(int)); // Use calloc to initialize to 0.
        if (seen)
        {
            for (int i = 0; i < noutsnaps; i++)
            {
                int snap = snapshot_steps[i];
                if (snap >= 0 && snap < total_writes && !seen[snap])
                {
                    seen[snap] = 1;
                    unique_snapshots++;
                }
            }
            free(seen);
        }
        else
        {
            unique_snapshots = noutsnaps; // Fallback if memory allocation fails.
        }
    }

    int total_expected = unique_snapshots * 2; // Total files expected (unsorted + sorted for each unique snapshot).
    printf("Detected %d unique snapshot numbers out of %d indices. Expecting %d files total.\n\n",
           unique_snapshots, noutsnaps, total_expected);

    // First, see if we even have the reference snapshot file.
    if (noutsnaps > 0)
    {
        int snap = snapshot_steps[0];
        char fname_unsorted[256];
        char fname_sorted[256];

        char base_filename_1[256];
        snprintf(base_filename_1, sizeof(base_filename_1), "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_1, 1, fname_unsorted, sizeof(fname_unsorted));
        char base_filename_2[256];
        snprintf(base_filename_2, sizeof(base_filename_2), "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_2, 1, fname_sorted, sizeof(fname_sorted));

        FILE *fun = fopen(fname_unsorted, "rb");
        FILE *fsort = fopen(fname_sorted, "rb");

        if (fun && fsort)
        {
            // Get file sizes.
            fseek(fun, 0, SEEK_END);
            reference_unsorted_size = ftell(fun);
            fseek(fsort, 0, SEEK_END);
            reference_sorted_size = ftell(fsort);

            // If both files have non-zero size, use as reference.
            if (reference_unsorted_size > 0 && reference_sorted_size > 0)
            {
                printf("Found reference file sizes from snapshot %d (unsorted: %ld bytes, sorted: %ld bytes)\n\n",
                       snap, reference_unsorted_size, reference_sorted_size);
                last_valid_snap = snap;
                last_valid_index = 0;
                checked_count += 2; // Count these two files.
            }

            fclose(fun);
            fclose(fsort);
        }
        else
        {
            // Only close if non-NULL.
            if (fun)
                fclose(fun);
            if (fsort)
                fclose(fsort);

            // If the very first files cannot be opened, processing should not continue.
            printf("Could not find initial snapshot files for index 0. Will start from the beginning.\n");
            return -1;
        }
    }
    else
    {
        // No snapshots to check.
        return -1;
    }

    // If reference sizes cannot be found, proper validation is not possible.
    if (reference_unsorted_size == 0 || reference_sorted_size == 0)
    {
        printf("Could not find valid reference file sizes. Will start from the beginning.\n\n");
        return -1;
    }

    printf("Checking all %d snapshots for completeness...\n\n", noutsnaps);

    // Now check ALL snapshots (except index 0 which we already checked).
    for (int i = 1; i < noutsnaps; i++)
    {
        int snap = snapshot_steps[i];

        // If this snapshot index maps to the same snapshot number as a previous index,.
        // We might be seeing duplicated snapshot numbers in the calculation.
        if (snap == snapshot_steps[0])
        {
            printf("Warning: Duplicate snapshot number %d (index 0 and %d)\n", snap, i);
        }

        char fname_unsorted[256];
        char fname_sorted[256];

        char base_filename_3[256];
        snprintf(base_filename_3, sizeof(base_filename_3), "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_3, 1, fname_unsorted, sizeof(fname_unsorted));

        char base_filename_4[256];
        snprintf(base_filename_4, sizeof(base_filename_4), "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_4, 1, fname_sorted, sizeof(fname_sorted));

        FILE *fun = fopen(fname_unsorted, "rb");
        FILE *fsort = fopen(fname_sorted, "rb");

        if (fun && fsort)
        {
            // Get file sizes.
            fseek(fun, 0, SEEK_END);
            long unsorted_size = ftell(fun);
            fseek(fsort, 0, SEEK_END);
            long sorted_size = ftell(fsort);

            // Compare to reference sizes.
            // Allow for some small variation (±5%).
            double unsorted_ratio = (double)unsorted_size / reference_unsorted_size;
            double sorted_ratio = (double)sorted_size / reference_sorted_size;

            if (unsorted_size > 0 && sorted_size > 0 &&
                unsorted_ratio >= 0.95 && unsorted_ratio <= 1.05 &&
                sorted_ratio >= 0.95 && sorted_ratio <= 1.05)
            {
                last_valid_snap = snap;
                last_valid_index = i;
                log_message("INFO", "Verified valid files for snapshot %d (unsorted: %ld bytes, sorted: %ld bytes)",
                            snap, unsorted_size, sorted_size);
                checked_count += 2; // Count both files as checked.
            }
            else
            {
                log_message("WARNING", "Found invalid files for snapshot %d (unsorted: %ld bytes, sorted: %ld bytes) - expected ~%ld and ~%ld bytes",
                            snap, unsorted_size, sorted_size, reference_unsorted_size, reference_sorted_size);
                fclose(fun);
                fclose(fsort);
                all_snapshots_checked = 0; // Not all snapshots are valid.
                printf("Invalid snapshot found. Will restart from this point.\n\n");
                break; // Stop at first invalid snapshot.
            }

            fclose(fun);
            fclose(fsort);
        }
        else
        {
            // File doesn't exist at all for this snapshot.
            log_message("WARNING", "Missing files for snapshot %d", snap);
            // Only close if non-NULL.
            if (fun)
                fclose(fun);
            if (fsort)
                fclose(fsort);
            all_snapshots_checked = 0; // Not all snapshots are valid.
            printf("Missing snapshot found. Will restart from this point.\n");
            break; // Stop at first missing snapshot.
        }
    }

    printf("End of file check: last_valid_snap=%d, last_valid_index=%d, noutsnaps=%d, all_snapshots_checked=%d\n\n",
           last_valid_snap, last_valid_index, noutsnaps, all_snapshots_checked);
    printf("Files checked: %d out of %d expected (unique snapshots: %d)\n\n",
           checked_count, total_expected, unique_snapshots);

    if (last_valid_snap == -1)
    {
        printf("No valid data products found. Will start from the beginning.\n\n");
        return -1;
    }

    // Check if we've verified more files than expected - this can happen if we have duplicate snapshot numbers.
    if (checked_count > total_expected)
    {
        printf("WARNING: Checked more files (%d) than expected (%d) - likely due to duplicate snapshot numbers.\n",
               checked_count, total_expected);
    }

    // Determine if we need to proceed with Rank file generation.
    if (unique_snapshots == 1 && all_snapshots_checked)
    {
        // Special case: Only one unique snapshot number (usually 0), and it's valid.
        printf("WARNING: Only one unique snapshot number found (%d). There's likely an issue with the calculation.\n",
               snapshot_steps[0]);
        printf("Only 1 Rank file (snapshot %d) exists. Starting from the beginning to create all files.\n",
               snapshot_steps[0]);
        return -1; // Start from beginning.
    }
    // Only say "all files exist" if:
    // 1. We have more than one unique snapshot, and.
    // 2. We've checked all expected files and found them valid.
    else if (unique_snapshots > 1 && checked_count >= total_expected && all_snapshots_checked)
    {
        printf("All %d data product files (for %d unique snapshots) already exist and are valid. Nothing to do.\n\n",
               checked_count, unique_snapshots);
        return -2; // Special code for "all done".
    }
    else
    {
        printf("Will restart processing from snapshot index %d (after snapshot %d)\n\n",
               last_valid_index + 1, last_valid_snap);
        return last_valid_index + 1; // Return the index of the NEXT snapshot to process.
    }
}

/**
 * @brief Main entry point for the n-sphere dark matter simulation program.
 * @details Orchestrates the overall simulation workflow:
 *          1. Parses command-line arguments using a flexible option system
 *          2. Sets up initial conditions (particle distribution, system parameters)
 *          3. Executes the selected integration method for trajectory evolution
 *          4. Outputs data products (particle states, phase diagrams, energy tracking)
 *          5. Handles restart/resume functionality when requested
 *          
 * Parameters
 * ----------
 * argc : int
 *     Standard argument count from the command line.
 * argv : char**
 *     Standard array of argument strings from the command line.
 *
 * Returns
 * -------
 * int
 *     Exit code: 0 for successful execution, non-zero for errors.
 *
 * @note This is a highly parallelized application that takes advantage of OpenMP
 *       when available. Performance scales with the number of available cores.
 * @warning Some simulation configurations can be very memory-intensive. For large
 *          particle counts (>10^6), ensure sufficient RAM is available.
 */

// Forward declarations for structures and functions used in diagnostic loop
struct RrPsiPair
{
    double rr;  ///< Radius value or x-axis value for sorting
    double psi; ///< Corresponding potential value or y-axis value
};

int compare_by_rr(const void *a, const void *b);

/**
 * @brief Check if an array is strictly monotonically increasing.
 * 
 * @param arr Array to check
 * @param n Number of elements
 * @param name Name of the array for debug messages
 * @return 1 if strictly monotonic, 0 otherwise
 */
static int check_strict_monotonicity(const double *arr, int n, const char *name) {
    int i;
    for (i = 1; i < n; i++) {
        if (arr[i] <= arr[i-1]) {
            fprintf(stderr, "MONOTONICITY_CHECK FAILED for '%s': arr[%d]=%.17e <= arr[%d]=%.17e\n", 
                       name, i, arr[i], i-1, arr[i-1]);
            fflush(stderr);
            // Print a few surrounding values for context
            for (int k = (i > 2 ? i - 2 : 0); k < (i + 3 < n ? i + 3 : n); k++) {
                fprintf(stderr, "  Context: %s[%d] = %.17e\n", name, k, arr[k]);
                fflush(stderr);
            }
            return 0;
        }
    }
    return 1;
}

/**
 * @brief Handles the SIDM scattering phase for a single timestep.
 * @details Checks if SIDM is enabled and if not in a bootstrap phase that should skip SIDM. 
 *          If proceeding, it resets particle scatter flags, selects serial or parallel execution 
 *          based on `g_sidm_execution_mode`, calls the appropriate core scattering function 
 *          (`perform_sidm_scattering_serial` or `perform_sidm_scattering_parallel`), 
 *          updates the global total scatter count, and logs debug information if scatters occurred
 *          and debugging is enabled. The core scattering functions are responsible for updating
 *          the `g_particle_scatter_state` flags for particles that underwent scattering.
 * 
 * @param particles         [in,out] The main particle data array: `particles[component][current_sorted_index]`.
 *                              Modified in-place with post-scattering velocities/angular momenta.
 * @param npts              [in] Total number of particles.
 * @param dt                [in] The simulation timestep (Myr).
 * @param current_sim_time  [in] The current simulation time at the beginning of this step (Myr).
 * @param active_profile_rc [in] The scale radius (kpc) of the currently active profile (NFW or Cored),
 *                              passed to `sigmatotal`.
 * @param current_method_display_num [in] The user-facing display number of the current integration method (for logging).
 * @param bootstrap_phase_active [in] Flag (0 or 1) indicating if a bootstrap phase (e.g., for Adams-Bashforth)
 *                               is active. If 1, SIDM scattering is skipped for this step.
 * @note This function modifies the `particles` array in-place.
 * @note It uses global variables: `g_enable_sidm_scattering`, `g_sidm_execution_mode`, 
 *       `g_rng_per_thread`, `g_max_omp_threads_for_rng`, `g_rng`, `g_total_sidm_scatters`,
 *       `g_active_halo_mass`, `g_doDebug`, and `g_particle_scatter_state`.
 */
static void handle_sidm_step(double **particles, int npts, double dt, double current_sim_time, 
                             double active_profile_rc, int current_method_display_num, 
                             int bootstrap_phase_active) 
{
    if (!g_enable_sidm_scattering || bootstrap_phase_active) {
        return; // Skip SIDM if disabled or in a bootstrap phase that should skip SIDM
    }

    long long Nscatters_in_this_step = 0;

    if (g_sidm_execution_mode == 1) { // Parallel
        #ifdef _OPENMP
            if (g_rng_per_thread != NULL && g_max_omp_threads_for_rng > 0) {
                perform_sidm_scattering_parallel(particles, npts, dt, current_sim_time, 
                                               g_rng_per_thread, g_max_omp_threads_for_rng, 
                                               &Nscatters_in_this_step, g_active_halo_mass, active_profile_rc);
            } else {
                log_message("ERROR", "SIDM Parallel mode selected but per-thread RNGs not available. Skipping SIDM for step.");
                Nscatters_in_this_step = 0;
            }
        #else
            // Serial fallback if OpenMP not compiled but parallel mode selected
            log_message("WARNING", "SIDM Parallel mode selected but OpenMP not enabled. Running SIDM serially.");
            gsl_rng *rng_for_serial_fallback = (g_rng_per_thread != NULL && g_rng_per_thread[0] != NULL) ? g_rng_per_thread[0] : g_rng;
            if (rng_for_serial_fallback != NULL) {
                perform_sidm_scattering_serial(particles, npts, dt, current_sim_time, rng_for_serial_fallback,
                                             &Nscatters_in_this_step, g_active_halo_mass, active_profile_rc);
            } else {
                log_message("ERROR", "SIDM Serial fallback: No suitable RNG available. Skipping SIDM for step.");
                Nscatters_in_this_step = 0;
            }
        #endif
    } else { // Serial SIDM execution
        gsl_rng *rng_for_serial = (g_rng_per_thread != NULL && g_rng_per_thread[0] != NULL) ? g_rng_per_thread[0] : g_rng;
        if (rng_for_serial != NULL) {
            perform_sidm_scattering_serial(particles, npts, dt, current_sim_time, rng_for_serial,
                                         &Nscatters_in_this_step, g_active_halo_mass, active_profile_rc);
        } else {
            log_message("ERROR", "SIDM Serial mode: No suitable RNG available. Skipping SIDM for step.");
            Nscatters_in_this_step = 0;
        }
    }
    
    g_total_sidm_scatters += Nscatters_in_this_step;
    
    if (Nscatters_in_this_step > 0 && g_doDebug) {
        log_message("DEBUG", "Method %d Step: %lld SIDM scatters this step, %lld total", 
                    current_method_display_num, Nscatters_in_this_step, g_total_sidm_scatters);
    }
}

/**
 * @brief Gets the available disk space for the filesystem containing the given path.
 * @details This function uses platform-specific APIs to determine the free space
 *          available to the current user on the filesystem where `path` resides.
 *          On Windows, it uses `GetDiskFreeSpaceEx`. On POSIX-compliant systems
 *          (Linux, macOS), it uses `statvfs`.
 *
 * @param path [in] A path to a file or directory on the filesystem to check.
 *                For Windows, this can be a root directory like "C:\\".
 *                For POSIX, any path within the target filesystem, e.g., "data/".
 * @return long long Available disk space in bytes. Returns -1 on error or if the
 *                   functionality is not implemented for the current platform.
 */
long long get_available_disk_space(const char *path) {
    #ifdef _WIN32
        ULARGE_INTEGER freeBytesAvailable;
        if (GetDiskFreeSpaceEx(path, &freeBytesAvailable, NULL, NULL)) {
            return (long long)freeBytesAvailable.QuadPart;
        }
    #else
        struct statvfs stat;
        if (statvfs(path, &stat) == 0) {
            return (long long)stat.f_bavail * (long long)stat.f_frsize;
        }
    #endif
    return -1;
}

/**
 * @brief Prompts the user with a yes/no question and reads their response from stdin.
 * @details Displays the given `prompt` string followed by "[y/N]: ".
 *          Reads a line of input from the user.
 *          - Returns 1 (yes) if the first character of the input is 'y' or 'Y'.
 *          - Returns 0 (no) if the first character is 'n', 'N', or if the input is
 *            an empty line (user just pressed Enter, defaulting to No).
 *          - If any other input is received, the prompt is repeated.
 *          Handles potential EOF or read errors by defaulting to No.
 *
 * @param prompt [in] The question/prompt message to display to the user.
 * @return int 1 if the user confirms (yes), 0 otherwise (no/default).
 */
int prompt_yes_no(const char *prompt) {
    int response;
    
    while (1) {
        printf("%s [y/N]: ", prompt);
        fflush(stdout);
        
        // Read entire line
        char buffer[256];
        if (fgets(buffer, sizeof(buffer), stdin) == NULL) {
            // EOF or error - treat as 'N'
            return 0;
        }
        
        // Check first character
        response = buffer[0];
        
        if (response == 'y' || response == 'Y') {
            return 1;
        } else if (response == 'n' || response == 'N' || response == '\n') {
            // Empty line (just Enter) or explicit 'n'/'N'
            return 0;
        }
        // Any other input repeats the prompt
    }
}

/**
 * @brief Main entry point for the n-sphere dark matter simulation program.
 * @details Orchestrates the overall simulation workflow:
 *          1. Parses command-line arguments.
 *          2. Sets up global parameters and logging.
 *          3. Initializes random number generators.
 *          4. Generates or loads initial conditions (ICs) for either NFW or Cored Plummer-like profiles.
 *             - Includes theoretical calculations for density, mass, potential, and f(E) splines.
 *             - Includes a diagnostic loop to test IC generation with varied numerical parameters.
 *             - Performs particle sampling based on the derived distribution function.
 *          5. Optionally performs tidal stripping and re-assigns particle IDs.
 *          6. Converts particle velocities to physical simulation units.
 *          7. Executes the main N-body timestepping loop using a selected integration method.
 *             - Performs gravitational updates.
 *             - Optionally performs SIDM scattering via `handle_sidm_step`.
 *             - Tracks particle trajectories and energies.
 *             - Periodically writes simulation data and progress.
 *          8. If `g_doAllParticleData` is enabled, processes all particle data to generate
 *             snapshot files for Rank/Mass/Radius/Velocity/Potential/Energy/Density.
 *          9. Writes final summary plots and theoretical profiles.
 *          10. Cleans up allocated resources.
 *          Handles restart/resume functionality by checking for existing data products.
 * 
 * @param argc [in] Standard argument count from the command line.
 * @param argv [in] Standard array of argument strings from the command line.
 * @return int Exit code: 0 for successful execution, non-zero for errors.
 *
 * @note This application supports OpenMP for parallelization in various sections.
 * @warning Large particle counts or long simulations can be memory and CPU intensive.
 *          Disk space requirements for full data output can also be significant.
 */
int main(int argc, char *argv[])
{
// Print the tool header first, regardless of OpenMP status
printf("\n===================================================================================================\n");
printf("NSphere Simulation Tool\n");
printf("===================================================================================================\n");
printf("  \n");

#ifdef _OPENMP
    /** @note OpenMP section: Configures parallel execution environment when compiled with OpenMP. */
    // OpenMP is available - configure parallel execution environment
    int max_threads = omp_get_max_threads();
    int num_processors = omp_get_num_procs();

    // Enable nested parallelism with max_active_levels (replacing deprecated omp_set_nested)
    omp_set_max_active_levels(10); // Allow up to 10 levels of nested parallelism

    // Set OpenMP to use maximum available thread parallelism
    omp_set_num_threads(max_threads);

    printf("OpenMP Status: ENABLED (%d logical processors, using %d threads)\n", 
           num_processors, max_threads);
    printf("\n\n");

    log_message("INFO", "Using maximum thread parallelism: %d threads", max_threads);
#else
    /** @warning OpenMP section: Warns user when compiled without OpenMP support. */
    // OpenMP is not available - warn user about performance implications
    printf("WARNING: OpenMP is NOT ENABLED in this build!\n");
    printf("This will result in significantly reduced performance.\n");
    printf("For better performance, please install OpenMP and recompile with -fopenmp flag.\n\n\n");

    log_message("WARNING", "OpenMP not available - running in single-threaded mode");
    
    // Check if --help flag is used (don't delay in that case)
    int help_requested = 0;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            help_requested = 1;
            break;
        }
    }
    
    // Add a delay to ensure the warning is noticed, but only when not showing help
    if (!help_requested) {
        printf("Continuing in single-threaded mode");
        fflush(stdout);
        for (int i = 0; i < 5; i++) {
            usleep(500000); // 500ms * 5 = 2.5 seconds
            printf(".");
            fflush(stdout);
        }
        printf("\n\n");
    }

    // Define a single thread variable for the code below to use
    int max_threads __attribute__((unused)) = 1;
#endif

#ifdef _OPENMP
    /** @note Initializes FFTW thread support if compiled with OpenMP. */
    // Initialize FFTW thread support (only effective if FFTW was compiled with threading)
    fftw_init_threads();
    fftw_plan_with_nthreads(max_threads);
#endif

    int npts = 100000;
    int Ntimes = 10000;
    int tfinal_factor = 5;
    int nout = 100;
    int dtwrite = 100;
    double tidal_fraction = 0.0;
    int noutsnaps;
    g_total_sidm_scatters = 0; // Initialize global SIDM scatter counter

    int method_select = 1;            // Default: option 1 (Adaptive Leapfrog with Adaptive Levi-Civita)
    int display_sort = 1;             // Default: option 1 (Parallel Quadsort)
    int include_method_in_suffix = 0; // Default: exclude method from filenames.
    char custom_tag[256] = {0};       // Default: no custom tag.

    /** @note Check for the `--help` argument first before parsing other options. */
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--help") == 0)
        {
            printUsage(argv[0]);
            return 0;
        }
    }

    /** @note Handle the case where no command-line arguments are provided. */
    if (argc == 1)
    {
        printf("No command-line arguments given. Using default parameters.\n");
        printf("Run `%s --help` to learn how to adjust parameters.\n\n", argv[0]);
    }

    /** @note Main command-line argument parsing loop (strict parsing). */
    for (int i = 1; i < argc; i++)
    {
        /** @note Forbid using '=' within options; require space separation. */
        // Check for "--option=value" format, which is disallowed.
        if (strncmp(argv[i], "--", 2) == 0 && strstr(argv[i], "=") != NULL)
        {
            errorAndExit("use space, not '=' after option", argv[i], argv[0]);
        }

        if (strcmp(argv[i], "--nparticles") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--nparticles requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --nparticles", argv[i + 1], argv[0]);
            }
            npts = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--ntimesteps") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--ntimesteps requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --ntimesteps", argv[i + 1], argv[0]);
            }
            Ntimes = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tfinal") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--tfinal requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --tfinal", argv[i + 1], argv[0]);
            }
            tfinal_factor = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--nout") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--nout requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --nout", argv[i + 1], argv[0]);
            }
            nout = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--dtwrite") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--dtwrite requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --dtwrite", argv[i + 1], argv[0]);
            }
            dtwrite = atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--tag") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--tag requires a string argument", NULL, argv[0]);
            }

            strncpy(custom_tag, argv[++i], 255);
            custom_tag[255] = '\0'; // Ensure null-termination.
        }
        else if (strcmp(argv[i], "--method") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--method requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --method", argv[i + 1], argv[0]);
            }
            method_select = atoi(argv[++i]);

            if (method_select < 1 || method_select > 9)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "method must be in [1..9]");
                errorAndExit(buf, NULL, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--sort") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--sort requires an integer argument", NULL, argv[0]);
            }
            if (!isInteger(argv[i + 1]))
            {
                errorAndExit("invalid integer for --sort", argv[i + 1], argv[0]);
            }
            int sort_val = atoi(argv[++i]);

            if (sort_val < 1 || sort_val > 4)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "sort must be in [1..4]");
                errorAndExit(buf, NULL, argv[0]);
            }

            display_sort = sort_val;

            switch (sort_val)
            {
            case 1:
                g_defaultSortAlg = "quadsort_parallel"; // Formerly case 1.
                break;
            case 2:
                g_defaultSortAlg = "quadsort"; // Formerly case 0.
                break;
            case 3:
                g_defaultSortAlg = "insertion_parallel"; // Formerly case 2.
                break;
            case 4:
                g_defaultSortAlg = "insertion"; // Formerly case 3.
                break;
            }
        }
        else if (strcmp(argv[i], "--readinit") == 0)
        {
            if (i + 1 >= argc) // Check if filename argument exists
            {
                errorAndExit("--readinit requires a file argument", NULL, argv[0]);
            }

            /** @warning Check for incompatibility with `--restart` and `--writeinit`. */
            if (g_doRestart)
            {
                errorAndExit("--readinit is incompatible with --restart. These options cannot be used together. Use either --restart OR --readinit, not both.", NULL, argv[0]);
            }
            if (doWriteInit)
            {
                errorAndExit("--readinit is incompatible with --writeinit. These options cannot be used together. Use either --readinit OR --writeinit, not both.", NULL, argv[0]);
            }

            const char* user_filename = argv[++i]; // consume next arg
            // Prefix the path with "init/" directory
            static char prefixed_read_path[512]; // Static buffer for the path
            snprintf(prefixed_read_path, sizeof(prefixed_read_path), "init/%s", user_filename);
            readInitFilename = prefixed_read_path; // Assign the prefixed path
            doReadInit = 1;
        }
        else if (strcmp(argv[i], "--writeinit") == 0)
        {
            if (i + 1 >= argc) // Check if filename argument exists
            {
                errorAndExit("--writeinit requires a file argument", NULL, argv[0]);
            }

            /** @warning Check for incompatibility with `--restart` and `--readinit`. */
            if (g_doRestart)
            {
                errorAndExit("--writeinit is incompatible with --restart. These options cannot be used together. Use either --restart OR --writeinit, not both.", NULL, argv[0]);
            }
            if (doReadInit)
            {
                errorAndExit("--writeinit is incompatible with --readinit. These options cannot be used together. Use either --writeinit OR --readinit, not both.", NULL, argv[0]);
            }

            const char* user_filename = argv[++i]; // consume next arg
            // Prefix the path with "init/" directory
            static char prefixed_write_path[512]; // Static buffer for the path
            snprintf(prefixed_write_path, sizeof(prefixed_write_path), "init/%s", user_filename);
            writeInitFilename = prefixed_write_path; // Assign the prefixed path
            doWriteInit = 1;
        }
        else if (strcmp(argv[i], "--restart") == 0)
        {
            /** @warning Check for incompatibility with `--readinit` and `--writeinit`. */
            if (doReadInit)
            {
                errorAndExit("--restart is incompatible with --readinit. These options cannot be used together. Use either --restart OR --readinit, not both.", NULL, argv[0]);
            }
            if (doWriteInit)
            {
                errorAndExit("--restart is incompatible with --writeinit. These options cannot be used together. Use either --restart OR --writeinit, not both.", NULL, argv[0]);
            }

            g_doRestart = 1;
            printf("Restart mode enabled. Will look for existing data products to resume processing.\n\n");
        }
        else if (strcmp(argv[i], "--save") == 0)
        {
            /** @note Delegate parsing of subsequent arguments to parseSaveArgs. */
            parseSaveArgs(argc, argv, &i);
            // The main loop continues from the updated index 'i'.
        }
        else if (strcmp(argv[i], "--ftidal") == 0)
        {
            if (i + 1 >= argc)
            {
                errorAndExit("--ftidal requires a float argument", NULL, argv[0]);
            }
            if (!isFloat(argv[i + 1]))
            {
                errorAndExit("invalid float for --ftidal", argv[i + 1], argv[0]);
            }
            tidal_fraction = atof(argv[++i]);

            /** @warning Check range [0.0, 1.0]. */
            if (tidal_fraction < 0.0 || tidal_fraction > 1.0)
            {
                char buf[256];
                snprintf(buf, sizeof(buf), "tidal_fraction must be in [0.0..1.0]");
                errorAndExit(buf, NULL, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--methodtag") == 0)
        {
            /** @note Flag to include integration method string in output filename suffix. */
            include_method_in_suffix = 1;
        }
        else if (strcmp(argv[i], "--log") == 0)
        {
            /** @note Flag to enable logging to log/nsphere.log. */
            g_enable_logging = 1;
        }
        else if (strcmp(argv[i], "--sidm") == 0)
        {
            /** @note Flag to enable Self-Interacting Dark Matter physics. */
            g_enable_sidm_scattering = 1;
            // This flag does not take a value, so 'i' is not incremented further.
        }
        else if (strcmp(argv[i], "--sidm-mode") == 0)
        {
            if (i + 1 >= argc) {
                errorAndExit("--sidm-mode requires an argument (serial or parallel)", NULL, argv[0]);
            }
            char* mode_arg = argv[++i];
            if (strcmp(mode_arg, "serial") == 0) {
                g_sidm_execution_mode = 0;
            } else if (strcmp(mode_arg, "parallel") == 0) {
                #ifndef _OPENMP
                    printf("Warning: OpenMP is not enabled in this build. SIDM will run serially despite '--sidm-mode parallel'.\n");
                    log_message("WARNING", "OpenMP not enabled, SIDM forced to serial despite --sidm-mode parallel request.");
                    g_sidm_execution_mode = 0; // Force serial if no OpenMP
                #else
                    g_sidm_execution_mode = 1;
                #endif
            } else {
                errorAndExit("Invalid argument for --sidm-mode. Use 'serial' or 'parallel'.", mode_arg, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--sidm-kappa") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--sidm-kappa requires a float argument", argv[i + 1], argv[0]);
            }
            g_sidm_kappa = atof(argv[++i]);
            if (g_sidm_kappa < 0) { // Kappa can be 0 (no interaction) but not negative
                errorAndExit("--sidm-kappa must be non-negative", NULL, argv[0]);
            }
            g_sidm_kappa_provided = 1;
        }
        else if (strcmp(argv[i], "--master-seed") == 0) {
            if (i + 1 >= argc || !isInteger(argv[i + 1])) {
                errorAndExit("--master-seed requires an integer argument", argv[i + 1], argv[0]);
            }
            g_master_seed = strtoul(argv[++i], NULL, 10);
            g_master_seed_provided = 1;
        } else if (strcmp(argv[i], "--init-cond-seed") == 0) {
            if (i + 1 >= argc || !isInteger(argv[i + 1])) {
                errorAndExit("--init-cond-seed requires an integer argument", argv[i + 1], argv[0]);
            }
            g_initial_cond_seed = strtoul(argv[++i], NULL, 10);
            g_initial_cond_seed_provided = 1;
        } else if (strcmp(argv[i], "--sidm-seed") == 0) {
            if (i + 1 >= argc || !isInteger(argv[i + 1])) {
                errorAndExit("--sidm-seed requires an integer argument", argv[i + 1], argv[0]);
            }
            g_sidm_seed = strtoul(argv[++i], NULL, 10);
            g_sidm_seed_provided = 1;
        } else if (strcmp(argv[i], "--load-seeds") == 0) {
            g_attempt_load_seeds = 1;
        } else if (strcmp(argv[i], "--profile") == 0) {
            if (i + 1 >= argc) {
                errorAndExit("--profile requires a type argument ('nfw' or 'cored')", NULL, argv[0]);
            }
            strncpy(g_profile_type_str, argv[++i], sizeof(g_profile_type_str) - 1);
            g_profile_type_str[sizeof(g_profile_type_str) - 1] = '\0'; // Ensure null termination
            if (strcmp(g_profile_type_str, "nfw") != 0 && strcmp(g_profile_type_str, "cored") != 0) {
                errorAndExit("Invalid argument for --profile. Use 'nfw' or 'cored'.", g_profile_type_str, argv[0]);
            }
            g_profile_type_str_provided = 1;
        } else if (strcmp(argv[i], "--scale-radius") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--scale-radius requires a float argument", argv[i + 1], argv[0]);
            }
            g_scale_radius_param = atof(argv[++i]);
            if (g_scale_radius_param <= 0) errorAndExit("--scale-radius must be positive", NULL, argv[0]);
            g_scale_radius_param_provided = 1;
        } else if (strcmp(argv[i], "--halo-mass") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) { // Ensure isFloat is robust for scientific notation
                errorAndExit("--halo-mass requires a float argument", argv[i + 1], argv[0]);
            }
            g_halo_mass_param = atof(argv[++i]);
            if (g_halo_mass_param <= 0) errorAndExit("--halo-mass must be positive", NULL, argv[0]);
            g_halo_mass_param_provided = 1;
        } else if (strcmp(argv[i], "--cutoff-factor") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--cutoff-factor requires a float argument", argv[i + 1], argv[0]);
            }
            g_cutoff_factor_param = atof(argv[++i]);
            if (g_cutoff_factor_param <= 0) errorAndExit("--cutoff-factor must be positive", NULL, argv[0]);
            g_cutoff_factor_param_provided = 1;
        } else if (strcmp(argv[i], "--falloff-factor") == 0) {
            if (i + 1 >= argc || !isFloat(argv[i + 1])) {
                errorAndExit("--falloff-factor requires a float argument", argv[i + 1], argv[0]);
            }
            g_falloff_factor_param = atof(argv[++i]);
            if (g_falloff_factor_param <= 0) errorAndExit("--falloff-factor must be positive", NULL, argv[0]);
            g_falloff_factor_param_provided = 1;
        }
        else if (strncmp(argv[i], "--", 2) == 0)
        {
            errorAndExit("unrecognized option", argv[i], argv[0]);
        }
        else
        {
            errorAndExit("unrecognized argument", argv[i], argv[0]);
        }
    }

    /** @note Convert user-facing method number (1-9) to internal identifier (0-8) and get description string. */
    int display_method = method_select; // Store original user input for display
    char *method_verbose_name;

    // Convert method number to internal index and set descriptive name.
    switch (method_select)
    {
    case 1:
        method_select = 5;
        method_verbose_name = "Adaptive Leapfrog with Adaptive Levi-Civita";
        break;
    case 2:
        method_select = 4;
        method_verbose_name = "Full-Step Adaptive Leapfrog + Levi-Civita";
        break;
    case 3:
        method_select = 3;
        method_verbose_name = "Full-Step Adaptive Leapfrog";
        break;
    case 4:
        method_select = 6;
        method_verbose_name = "Yoshida 4th-Order";
        break;
    case 5:
        method_select = 8;
        method_verbose_name = "Adams-Bashforth 3rd-Order";
        break;
    case 6:
        method_select = 2;
        method_verbose_name = "Leapfrog (Vel Half-Step)";
        break;
    case 7:
        method_select = 1;
        method_verbose_name = "Leapfrog (Pos Half-Step)";
        break;
    case 8:
        method_select = 7;
        method_verbose_name = "Classic RK4";
        break;
    case 9:
        method_select = 0;
        method_verbose_name = "Euler";
        break;
    default:
        method_verbose_name = "Unknown Method";
        break;
    }

    /** @note Generate internal method string identifier for filenames. */
    char method_str[32];
    switch (method_select)
    {
    case 0:
        strcpy(method_str, "euler");
        break;
    case 1:
        strcpy(method_str, "pos.leap");
        break;
    case 2:
        strcpy(method_str, "vel.leap");
        break;
    case 3:
        strcpy(method_str, "adp.leap");
        break;
    case 4:
        strcpy(method_str, "adp.leap.levi");
        break;
    case 5:
        strcpy(method_str, "adp.leap.adp.levi");
        break;
    case 6:
        strcpy(method_str, "fr4.yoshi");
        break;
    case 7:
        strcpy(method_str, "rk4");
        break;
    case 8:
        strcpy(method_str, "ab3");
        break;
    default:
        strcpy(method_str, "unknown");
        break;
    }

    // Determine active profile type (NFW is default)
    if (g_profile_type_str_provided) {
        if (strcmp(g_profile_type_str, "nfw") == 0) {
            g_use_nfw_profile = 1;
        } else if (strcmp(g_profile_type_str, "cored") == 0) {
            g_use_nfw_profile = 0;
        } else {
            // Should have been caught by parser, but as a safeguard:
            log_message("WARNING", "Unknown profile type '%s', defaulting to NFW.", g_profile_type_str);
            g_use_nfw_profile = 1;
        }
    } else {
        // Default to NFW if --profile flag was not provided
        g_use_nfw_profile = 1;
        strcpy(g_profile_type_str, "nfw"); // Update string for consistency in printouts
    }

    // Set up profile-specific parameters based on generalized flags and profile defaults
    if (g_use_nfw_profile) {
        // NFW Profile Path
        // Halo Mass for NFW
        if (g_halo_mass_param_provided) { // --halo-mass overrides NFW default
            g_nfw_profile_halo_mass = g_halo_mass_param;
        } else { // No --halo-mass, NFW uses its own default
            g_nfw_profile_halo_mass = HALO_MASS_NFW;
            g_halo_mass_param = g_nfw_profile_halo_mass; // Update general param to reflect NFW's choice
        }
        // Scale Radius for NFW
        if (g_scale_radius_param_provided) { // --scale-radius overrides NFW default
            g_nfw_profile_rc = g_scale_radius_param;
        } else { // No --scale-radius, NFW uses its own default
            g_nfw_profile_rc = RC_NFW_DEFAULT;
            g_scale_radius_param = g_nfw_profile_rc; // Update general param to reflect NFW's choice
        }
        // Cutoff Factor for NFW
        if (g_cutoff_factor_param_provided) { // --cutoff-factor overrides NFW default
            g_nfw_profile_rmax_norm_factor = g_cutoff_factor_param;
        } else { // No --cutoff-factor, NFW uses its own default
            g_nfw_profile_rmax_norm_factor = CUTOFF_FACTOR_NFW_DEFAULT;
            // g_cutoff_factor_param is NOT updated here by NFW default; it keeps its own (Cored's) default or user value.
        }
        // Falloff Factor for NFW
        if (g_falloff_factor_param_provided) { // --falloff-factor overrides NFW default
            g_nfw_profile_falloff_factor = g_falloff_factor_param;
        } else { // No --falloff-factor, NFW uses its own default
            g_nfw_profile_falloff_factor = FALLOFF_FACTOR_NFW_DEFAULT;
            // Optionally, update g_falloff_factor_param if NFW is the overall default and no flag given
            // For now, let g_falloff_factor_param keep its own default unless explicitly set by user
        }

    } else {
        // Cored Profile Path
        // Halo Mass for Cored (already defaults to HALO_MASS or takes from --halo-mass via g_halo_mass_param)
        g_cored_profile_halo_mass = g_halo_mass_param;
        // Scale Radius for Cored (already defaults to RC or takes from --scale-radius via g_scale_radius_param)
        g_cored_profile_rc = g_scale_radius_param;
        // Cutoff Factor for Cored (directly uses generalized or its (Cored's) default)
        g_cored_profile_rmax_factor = g_cutoff_factor_param;
    }

    // Set the single g_active_halo_mass for N-body forces and tdyn from the finalized g_halo_mass_param
    g_active_halo_mass = g_halo_mass_param;
    
    /** @note Display final parameter values used for the simulation run. */
    printf("Parameter values requested:\n\n");
    
    printf("  Number of Particles:          %d\n", npts);
    printf("  Number of Time Steps:         %d\n", Ntimes);
    printf("  Number of Dynamical Times:    %d\n", tfinal_factor);
    printf("  Number of Output Snapshots:   %d\n", nout);
    printf("  Steps Between Writes:         %d\n", dtwrite);
    printf("  Tidal Stripping Fraction:     %.5f\n", tidal_fraction);
    printf("  Integration Method:           %d (%s)\n", display_method, method_verbose_name);
    printf("  Sorting Algorithm:            %d (%s)\n", display_sort, get_sort_description(g_defaultSortAlg));
    /** @note Build the filename tag string based on options for display purposes. */
    char filename_tag[512] = "";

    if (custom_tag[0] != '\0')
    {
        strcat(filename_tag, custom_tag);
    }

    if (include_method_in_suffix)
    {
        if (filename_tag[0] != '\0')
        {
            strcat(filename_tag, "_");
        }
        strcat(filename_tag, method_str);
    }

    printf("  Filename Tag:                 %s\n", filename_tag[0] ? filename_tag : "[none]");
    printf("  SIDM Scattering:              %s\n", g_enable_sidm_scattering ? "Enabled via --sidm" : "Disabled (Default)");
    printf("  SIDM Execution Mode:          %s\n", g_sidm_execution_mode == 1 ? "Parallel (Default)" : "Serial");
    printf("  SIDM Opacity Kappa:           %.1f cm^2/g (Default: 50.0, User set: %s)\n", g_sidm_kappa, g_sidm_kappa_provided ? "Yes" : "No");
    
    printf("  Initial Conditions Profile:   %s\n", g_use_nfw_profile ? "NFW-like with Cutoff" : "Cored Plummer-like");
    if (g_use_nfw_profile) {
        printf("    NFW Profile Scale Radius (IC): %.3f kpc (NFW Default: %.2f, User set via --scale-radius: %s)\n", g_nfw_profile_rc, RC_NFW_DEFAULT, g_scale_radius_param_provided ? "Yes" : "No");
    } else {
        printf("    Cored Profile Scale Radius (IC): %.3f kpc (Cored Default: %.2f, User set via --scale-radius: %s)\n", g_cored_profile_rc, RC, g_scale_radius_param_provided ? "Yes" : "No");
    }
    if (g_use_nfw_profile) {
        printf("    NFW Profile Halo Mass (IC): %.3e Msun (NFW Default: %.2e, User set via --halo-mass: %s)\n", g_nfw_profile_halo_mass, HALO_MASS_NFW, g_halo_mass_param_provided ? "Yes" : "No");
    } else {
        printf("    Cored Profile Halo Mass (IC): %.3e Msun (Cored Default: %.2e, User set via --halo-mass: %s)\n", g_cored_profile_halo_mass, HALO_MASS, g_halo_mass_param_provided ? "Yes" : "No");
    }
    printf("    Profile Cutoff Factor:      %.1f (CmdLine/Default: %.1f, User set: %s)\n", g_cutoff_factor_param, (g_use_nfw_profile ? CUTOFF_FACTOR_NFW_DEFAULT : CUTOFF_FACTOR_CORED_DEFAULT), g_cutoff_factor_param_provided ? "Yes" : "No");
    
    if (g_use_nfw_profile) {
        printf("    NFW Profile Falloff Factor (C): %.1f (NFW Default: %.1f, User set via --falloff-factor: %s)\n", g_nfw_profile_falloff_factor, FALLOFF_FACTOR_NFW_DEFAULT, g_falloff_factor_param_provided ? "Yes" : "No");
    }
    // This g_active_halo_mass is now correctly set from g_halo_mass_param which reflects the chosen profile's mass
    printf("  N-body Active Halo Mass (tdyn): %.3e Msun\n", g_active_halo_mass);
    
    /** @note Display logging status based on g_enable_logging flag. */
    if (g_enable_logging)
    {
        printf("  Logging:                      Enabled (log/nsphere.log)\n\n");
        log_message("INFO", "Simulation started with %d particles, %d timesteps, %d dynamical times",
                    npts, Ntimes, tfinal_factor);
    }
    else
    {
        printf("  Logging:                      Disabled\n\n");
    }

    // Check for SIDM + parallel mode without OpenMP
    if (g_enable_sidm_scattering && g_sidm_execution_mode == 1) {
        #ifndef _OPENMP
            printf("Warning: SIDM parallel mode is enabled by default, but OpenMP is not available in this build.\n");
            printf("         SIDM will run serially. Use '--sidm-mode serial' to suppress this warning.\n\n");
            log_message("WARNING", "SIDM parallel mode requested but OpenMP not available, will run serially.");
            g_sidm_execution_mode = 0; // Force serial if no OpenMP
        #endif
    }

    // Validation occurs earlier in the argument parsing loop.

    // ... proceed with simulation ...

    /**
     * @brief Oversample initial conditions based on tidal fraction.
     * @details Calculate the number of initial particles (`npts_initial`) needed
     *          before tidal stripping to ensure `npts` particles remain afterwards.
     *          If `tidal_fraction` is 0, `npts_initial` equals `npts`.
     */
    int npts_initial;
    if (tidal_fraction > 0.0)
    {
        // Use ceiling to ensure enough particles remain after stripping
        npts_initial = ceil(npts / (1.0 - tidal_fraction));
    }
    else
    {
        npts_initial = npts;
    }

    /** @note Calculate number of snapshots (`noutsnaps`) based on desired intervals (`nout`). */
    nout = nout + 1; // nout specifies intervals, noutsnaps is number of points (intervals + 1)
    noutsnaps = nout;

    /**
     * @brief Adjust Ntimes using adjust_ntimesteps to align with output schedule.
     * @details Ensures (Ntimes - 1) is a multiple of (noutsnaps - 1) * dtwrite.
     * @see adjust_ntimesteps
     */
    int oldN = Ntimes;
    Ntimes = adjust_ntimesteps(Ntimes, noutsnaps, dtwrite); // Use noutsnaps here
    if (Ntimes != oldN)
    {
        printf("Adjusted Number of Time Steps to %d to satisfy parameter constraints.\n", Ntimes);
    }
    /** @brief Calculate total number of write events and steps between major snapshots. */
    int total_writes = ((Ntimes - 1) / dtwrite) + 1; // Total potential write points
    int stepBetweenSnaps = (int)floor(
        (double)(total_writes - 1) / (double)(noutsnaps - 1) + 0.5); // Steps between major snapshots
    int ext_Ntimes; ///< Extended time steps potentially needed for trajectory arrays bounds.
    ext_Ntimes = Ntimes + dtwrite; // Allocate trajectory arrays slightly larger

    /** @brief Initialize the global file suffix string `g_file_suffix` based on cmd line args. */
    g_file_suffix[0] = '\0';

    /** @brief Add custom tag to suffix if provided via `--tag`. */
    if (custom_tag[0] != '\0')
    {
        snprintf(g_file_suffix, sizeof(g_file_suffix), "_%s", custom_tag);
    }

    /** @brief Add method/parameter tag to suffix. */
    char temp[256];
    if (include_method_in_suffix)
    {
        snprintf(temp, sizeof(temp), "_%s_%d_%d_%d", method_str, npts, Ntimes, tfinal_factor);
    }
    else
    {
        snprintf(temp, sizeof(temp), "_%d_%d_%d", npts, Ntimes, tfinal_factor);
    }

    /** @brief Append the parameter tag to the global suffix. */
    strcat(g_file_suffix, temp); // Append temp to g_file_suffix

    /** @brief Set flag `skip_file_writes=1` if in restart mode (`g_doRestart`). */
    if (g_doRestart)
    {
        printf("Restart mode: Skipping simulation phase and file writes, proceeding directly to data product generation.\n\n");
        skip_file_writes = 1;
    }

    /** @brief Create the output 'data' directory if it doesn't exist. */
    {
        struct stat st = {0};
        if (stat("data", &st) == -1)
        {
            mkdir("data", 0755); // POSIX standard, works on most systems including MinGW/Cygwin
        }
    }

    /** @brief Create the 'init' directory if it doesn't exist. */
    {
        struct stat st_init = {0};
        if (stat("init", &st_init) == -1)
        {
            #if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
                if (mkdir("init") != 0) {
                     perror("Error creating init directory");
                     // Decide if this is fatal - perhaps not if only writing
                } else {
                     log_message("INFO", "Created init/ directory.");
                }
            #else
                if (mkdir("init", 0755) != 0) { // POSIX standard
                     perror("Error creating init directory");
                     // Decide if this is fatal
                } else {
                     log_message("INFO", "Created init/ directory.");
                }
            #endif
        }
    }

    /** @brief Write current run parameters to `data/lastparams<suffix>.dat` and create a standard link `data/lastparams.dat`. */
    {
        char file_tag[512] = "";
        if (custom_tag[0] != '\0')
        {
            strcat(file_tag, custom_tag);
            if (include_method_in_suffix)
            {
                strcat(file_tag, "_");
                strcat(file_tag, method_str);
            }
        }
        else if (include_method_in_suffix)
        {
            strcat(file_tag, method_str);
        }

        /** @note Create filename with suffix for the specific run parameters file. */
        char filename[512]; // Holds suffixed filename, e.g., data/lastparams_run1_100k_10k_5.dat
        get_suffixed_filename("data/lastparams.dat", 1, filename, sizeof(filename));
        printf("Saving parameters: %s\n", filename);

        FILE *fp_params = fopen(filename, "w"); // Text mode for regular fprintf
        if (!fp_params)
        {
            printf("Error: cannot open %s\n", filename);
            return 1;
        }

        fprintf(fp_params, "%d %d %d %s\n", npts, Ntimes, tfinal_factor, file_tag);
        fclose(fp_params);

        /** @note Create a standard-named link `data/lastparams.dat` pointing to the
                 suffixed version for compatibility with scripts. Uses copy on Windows. */
        char linkname[512] = "data/lastparams.dat"; // Standard name

        /* Platform detection using standard predefined macros */
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
        /** @note Windows: Copy file content as symlinks can be unreliable or require special privileges. */
        // Windows or Windows-like environment: create a direct file copy.
            // Ensure compatibility with various Windows environments.

        // Use lower-level file operations instead of system commands for better compatibility.
        FILE *source, *dest;
        source = fopen(filename, "rb");
        if (!source)
        {
            printf("Warning: Failed to open source file %s for copying\n", filename);
        }
        else
        {
            dest = fopen(linkname, "wb");
            if (!dest)
            {
                printf("Warning: Failed to create destination file %s\n", linkname);
                fclose(source);
            }
            else
            {
                // Copy file content.
                char buffer[4096];
                size_t bytes_read;

                while ((bytes_read = fread(buffer, 1, sizeof(buffer), source)) > 0)
                {
                    fwrite(buffer, 1, bytes_read, dest);
                }

                fclose(dest);
                fclose(source);

                // Extract the basename for display purposes
                const char *basename = strrchr(filename, '/');
                basename = basename ? basename + 1 : filename; // Skip the '/' or use full name if no '/'

                printf("Created link: %s -> %s\n\n", basename, linkname);
            }
        }
#else
        /** @note Unix: Create symbolic link from basename(filename) to 'linkname', fallback to copy. */
            // Unix-like systems (Linux, macOS, etc.) and fallback for other platforms: use symbolic links.
        char command[1024];

        // First, remove any existing link or file.
        snprintf(command, sizeof(command), "rm -f \"%s\" 2>/dev/null", linkname);
        system(command);

        // Then create the symbolic link - use the basename of the file, not the full path
        // Extract the basename from filename
        const char *basename = strrchr(filename, '/');
        basename = basename ? basename + 1 : filename; // Skip the '/' or use full name if no '/'

        snprintf(command, sizeof(command), "ln -s \"%s\" \"%s\"", basename, linkname);
        if (system(command) != 0)
        {
            // If symbolic link fails, fall back to copying the file.
            snprintf(command, sizeof(command), "cp \"%s\" \"%s\"", filename, linkname);
            if (system(command) != 0)
            {
                printf("Warning: Failed to create link or copy %s to %s\n", filename, linkname);
            }
            else
            {
                printf("Created link: %s -> %s\n\n", filename, linkname);
            }
        }
        else
        {
            printf("Created link: %s -> %s\n\n", filename, linkname);
        }
#endif
    }

    /**
     * @brief Common IC generation variables shared between profile pathways.
     * @details These variables are declared before the profile selection block
     *          and will be populated by whichever profile pathway is chosen.
     */
    double **particles = NULL;           ///< Main particle data array
    int i = 0;                          ///< Loop counter
    double result, error;               ///< GSL integration results
    double calE;                        ///< Energy value for calculations
    gsl_integration_workspace *w = NULL; ///< GSL integration workspace
    
    // Common spline objects and accelerators
    gsl_spline *splinemass = NULL;      ///< Spline for mass profile M(r)
    gsl_interp_accel *enclosedmass = NULL; ///< Accelerator for mass spline
    gsl_spline *splinePsi = NULL;       ///< Spline for potential profile Psi(r)
    gsl_interp_accel *Psiinterp = NULL; ///< Accelerator for potential spline
    gsl_spline *splinerofPsi = NULL;    ///< Spline for inverse potential r(Psi)
    gsl_interp_accel *rofPsiinterp = NULL; ///< Accelerator for r(Psi) spline
    gsl_interp *g_main_fofEinterp = NULL;  ///< Main f(E) interpolator
    gsl_interp_accel *g_main_fofEacc = NULL; ///< Accelerator for f(E)
    
    // Common data arrays
    double *radius = NULL;              ///< Radial grid points
    double *mass = NULL;                ///< Mass values at radial points
    double *Psivalues = NULL;           ///< Potential values at radial points
    double *nPsivalues = NULL;          ///< Negative potential values (for r(Psi) spline)
    double *Evalues = NULL;             ///< Energy grid points
    double *innerintegrandvalues = NULL; ///< f(E) integrand values
    double *radius_monotonic_grid_nfw = NULL; ///< Monotonic radial grid for NFW calculations
    
    // Key scalar values
    double Psimin = 0.0;                ///< Minimum potential (at rmax)
    double Psimax = 0.0;                ///< Maximum potential (at r=0)
    double rmax = 0.0;                  ///< Maximum radius for profile calculations
    int num_points = 0;                 ///< Number of points for spline interpolation
    
    // File handling
    char fname[256];                    ///< Buffer for file names
    FILE *fp;                           ///< File pointer for data output
    
    /**
     * @brief Allocate main particle data array before profile selection.
     * @details This ensures both NFW and Cored pathways use the same particles array.
     */
    particles = (double **)malloc(5 * sizeof(double *));
    if (particles == NULL) {
        fprintf(stderr, "ERROR: Memory allocation failed for particle array pointer\n");
        CLEAN_EXIT(1);
    }
    for (i = 0; i < 5; i++) {
        particles[i] = (double *)malloc(npts_initial * sizeof(double));
        if (particles[i] == NULL) {
            fprintf(stderr, "ERROR: Memory allocation failed for particles[%d]\n", i);
            CLEAN_EXIT(1);
        }
    }
    
    /**
     * @brief Initialize random number generator for particle generation.
     * @details Sets up the GSL Random Number Generator environment and allocates
     *          the global GSL RNG state used throughout the simulation.
     *          Needed for the Sample Generator if not reading ICs from file.
     */
    gsl_rng_env_setup();                           // Setup GSL RNG environment
    const gsl_rng_type * T_rng = gsl_rng_default;  // Use default RNG type
    g_rng = gsl_rng_alloc(T_rng);                  // Allocate global GSL RNG state
    if (g_rng == NULL) {                           // Check allocation success
        fprintf(stderr, "Error allocating GSL RNG.\n");
        CLEAN_EXIT(1);
    }
    
    // Seed the global g_rng (used for ICs and Serial SIDM)
    gsl_rng_set(g_rng, g_initial_cond_seed);       // Use the determined IC seed for g_rng
    log_message("INFO", "Global g_rng (intended primarily for IC generation) seeded with %lu", g_initial_cond_seed);

    // Initialize per-thread GSL RNGs if OpenMP is enabled
    #ifdef _OPENMP
        g_max_omp_threads_for_rng = omp_get_max_threads();
        if (g_max_omp_threads_for_rng <= 0) g_max_omp_threads_for_rng = 1; // Safety
    #else
        g_max_omp_threads_for_rng = 1;
    #endif

    g_rng_per_thread = (gsl_rng **)malloc(g_max_omp_threads_for_rng * sizeof(gsl_rng *));
    if (g_rng_per_thread == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for per-thread RNG array.\n");
        CLEAN_EXIT(1);
    }

    const gsl_rng_type *T_rng_thread = gsl_rng_default;
    
    for (int i_rng = 0; i_rng < g_max_omp_threads_for_rng; ++i_rng) {
        g_rng_per_thread[i_rng] = gsl_rng_alloc(T_rng_thread);
        if (g_rng_per_thread[i_rng] == NULL) {
            fprintf(stderr, "Error: Failed to allocate GSL RNG for thread %d.\n", i_rng);
            for (int k_rng = 0; k_rng < i_rng; ++k_rng) gsl_rng_free(g_rng_per_thread[k_rng]);
            free(g_rng_per_thread);
            CLEAN_EXIT(1);
        }
        gsl_rng_set(g_rng_per_thread[i_rng], g_sidm_seed + (unsigned long int)i_rng);
    }
    log_message("INFO", "Initialized %d per-thread GSL RNGs (for SIDM) using base SIDM seed %lu", g_max_omp_threads_for_rng, g_sidm_seed);
    
    if (g_use_nfw_profile) {
        log_message("INFO", "Starting IC generation using NFW-like profile pathway.");
        log_message("INFO", "Generating Initial Conditions using NFW-like profile with its specific numerics...");

        // Diagnostic loop for NFW (similar to Cored's diagnostic)
        if (g_doDebug) {
            log_message("INFO", "NFW DIAGNOSTIC LOOP: Starting convergence tests for NFW profile.");
            int diag_integration_points_array[2] = {1000, 10000};
            int diag_spline_points_array[2] = {1000, 10000};

            for (int ii_ip_nfw = 0; ii_ip_nfw < 2; ii_ip_nfw++) {
                for (int ii_sp_nfw = 0; ii_sp_nfw < 2; ii_sp_nfw++) {
                    int Nintegration_diag = diag_integration_points_array[ii_ip_nfw];
                    int Nspline_diag_base = diag_spline_points_array[ii_sp_nfw];
                    int num_points_diag = Nspline_diag_base * 10;

                    log_message("DEBUG", "Diagnostic iteration %d, spline_base=%d (points=%d)",
                                Nintegration_diag, Nspline_diag_base, num_points_diag);

                    // Use the main NFW profile parameters for this diagnostic run
                    double current_diag_rc = g_nfw_profile_rc;
                    double current_diag_halo_mass = g_nfw_profile_halo_mass;
                    double current_diag_rmax_factor = g_nfw_profile_rmax_norm_factor;
                    double current_diag_falloff_C = g_nfw_profile_falloff_factor;
                    double rmax_diag = current_diag_rmax_factor * current_diag_rc;

                    // Local GSL workspace and variables for this diagnostic iteration
                    gsl_integration_workspace *w_diag = gsl_integration_workspace_alloc(Nintegration_diag);
                    if (!w_diag) {
                        log_message("ERROR", "Failed to allocate GSL workspace for NFW diagnostic");
                        continue;
                    }
                    
                    double result_diag, error_diag;
                    double normalization_diag;

                    // Declare all splines and accelerators locally for the diagnostic loop
                    gsl_spline *splinemass_diag = NULL;
                    gsl_interp_accel *enclosedmass_diag = NULL;
                    gsl_spline *splinePsi_diag = NULL;
                    gsl_interp_accel *Psiinterp_diag = NULL;
                    gsl_spline *splinerofPsi_diag = NULL;
                    gsl_interp_accel *rofPsiinterp_diag = NULL;
                    gsl_interp *fofEinterp_diag = NULL;
                    gsl_interp_accel *fofEacc_diag = NULL;

                    double *mass_diag_arr = NULL;
                    double *radius_diag_arr = NULL;
                    double *radius_for_rofPsi_diag_arr = NULL;
                    double *Psivalues_diag_arr = NULL;
                    double *nPsivalues_diag_arr = NULL;
                    double *Evalues_diag_arr = NULL;
                    double *innerintegrandvalues_diag_arr = NULL;

                    // Prepare NFW params for this diagnostic iteration's integrands
                    double nfw_params_diag[4];
                    nfw_params_diag[0] = current_diag_rc;
                    nfw_params_diag[1] = current_diag_halo_mass;
                    nfw_params_diag[2] = 1.0; // Initial nt_nfw guess
                    nfw_params_diag[3] = current_diag_falloff_C;

                    gsl_function F_nfw_diag;
                    F_nfw_diag.function = &massintegrand_profile_nfwcutoff;
                    F_nfw_diag.params = nfw_params_diag;

                    // --- 1. Normalization for NFW Diagnostic ---
                    gsl_integration_qag(&F_nfw_diag, 0.0, rmax_diag, 1e-12, 1e-12, Nintegration_diag, GSL_INTEG_GAUSS51,
                                        w_diag, &result_diag, &error_diag);
                    normalization_diag = result_diag;
                    if (normalization_diag <= 1e-30) {
                        log_message("ERROR", "NFW diagnostic normalization too small: %e", normalization_diag);
                        gsl_integration_workspace_free(w_diag);
                        continue;
                    }
                    nfw_params_diag[2] = current_diag_halo_mass / (4.0 * M_PI * normalization_diag);

                    // --- 2. M(r) spline for NFW Diagnostic ---
                    mass_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    radius_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    if (!mass_diag_arr || !radius_diag_arr) {
                        log_message("ERROR", "Failed to allocate arrays for NFW diagnostic M(r)");
                        gsl_integration_workspace_free(w_diag);
                        free(mass_diag_arr);
                        free(radius_diag_arr);
                        continue;
                    }
                    
                    mass_diag_arr[0] = 0.0; 
                    radius_diag_arr[0] = 0.0;
                    for (int k = 1; k < num_points_diag; k++) {
                        double r_k = (double)k * rmax_diag / (num_points_diag - 1.0);
                        if (k == num_points_diag - 1) r_k = rmax_diag;
                        radius_diag_arr[k] = r_k;
                        gsl_integration_qag(&F_nfw_diag, 0.0, r_k, 1e-12, 1e-12, Nintegration_diag, GSL_INTEG_GAUSS51,
                                            w_diag, &result_diag, &error_diag);
                        mass_diag_arr[k] = 4.0 * M_PI * result_diag;
                    }
                    enclosedmass_diag = gsl_interp_accel_alloc();
                    splinemass_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    gsl_spline_init(splinemass_diag, radius_diag_arr, mass_diag_arr, num_points_diag);


                    // Write mass profile diagnostic file
                    char diag_fname_mass[256];
                    char diag_base_mass[128];
                    snprintf(diag_base_mass, sizeof(diag_base_mass), "data/massprofile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_mass, 1, diag_fname_mass, sizeof(diag_fname_mass));
                    FILE *fp_diag_mass = fopen(diag_fname_mass, "wb");
                    if (fp_diag_mass) {
                        for (double r_write_diag = 0.0; r_write_diag < radius_diag_arr[num_points_diag - 1]; r_write_diag += rmax_diag / 900.0) {
                            if (r_write_diag >= radius_diag_arr[0]) {
                                 fprintf_bin(fp_diag_mass, "%f %f\n", r_write_diag, gsl_spline_eval(splinemass_diag, r_write_diag, enclosedmass_diag));
                            }
                        }
                        fclose(fp_diag_mass);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_mass);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_mass);
                    }

                    // --- 3. Psi(r) spline for NFW Diagnostic ---
                    Psivalues_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    nPsivalues_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    radius_for_rofPsi_diag_arr = (double *)malloc(num_points_diag * sizeof(double));
                    if (!Psivalues_diag_arr || !nPsivalues_diag_arr || !radius_for_rofPsi_diag_arr) {
                        log_message("ERROR", "Failed to allocate arrays for NFW diagnostic Psi(r)");
                        goto cleanup_diag_iteration;
                    }

                    Psiintegrand_params psi_params_nfw_diag;
                    psi_params_nfw_diag.massintegrand_func = &massintegrand_profile_nfwcutoff;
                    psi_params_nfw_diag.params_for_massintegrand = nfw_params_diag;
                    gsl_function F_psi_nfw_diag;
                    F_psi_nfw_diag.function = &Psiintegrand;
                    F_psi_nfw_diag.params = &psi_params_nfw_diag;

                    for (int k = 0; k < num_points_diag; k++) {
                        double r_k = radius_diag_arr[k];
                        double r1_psi_k = fmax(r_k, current_diag_rc / 1000000.0);
                        gsl_integration_qagiu(&F_psi_nfw_diag, r1_psi_k, 1e-12, 1e-12, Nintegration_diag,
                                              w_diag, &result_diag, &error_diag);
                        double M_at_r1_k = gsl_spline_eval(splinemass_diag, r1_psi_k, enclosedmass_diag);
                        double first_term_psi_k = (r1_psi_k > 1e-9) ? (G_CONST * M_at_r1_k / r1_psi_k) : 0.0;
                        double second_term_psi_k = G_CONST * 4.0 * M_PI * result_diag;
                        Psivalues_diag_arr[k] = first_term_psi_k + second_term_psi_k;
                        nPsivalues_diag_arr[k] = -Psivalues_diag_arr[k];
                        radius_for_rofPsi_diag_arr[k] = r_k;
                    }
                    Psiinterp_diag = gsl_interp_accel_alloc();
                    splinePsi_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    gsl_spline_init(splinePsi_diag, radius_diag_arr, Psivalues_diag_arr, num_points_diag);
                    
                    
                    // Write potential profile diagnostic file
                    char diag_fname_psi[256];
                    char diag_base_psi[128];
                    snprintf(diag_base_psi, sizeof(diag_base_psi), "data/Psiprofile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_psi, 1, diag_fname_psi, sizeof(diag_fname_psi));
                    FILE *fp_diag_psi = fopen(diag_fname_psi, "wb");
                    if (fp_diag_psi) {
                        for (double r_write_diag = 0.0; r_write_diag < radius_diag_arr[num_points_diag - 1]; r_write_diag += rmax_diag / 900.0) {
                             if (r_write_diag >= radius_diag_arr[0]) {
                                fprintf_bin(fp_diag_psi, "%f %f\n", r_write_diag, evaluatespline(splinePsi_diag, Psiinterp_diag, r_write_diag));
                             }
                        }
                        fclose(fp_diag_psi);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_psi);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_psi);
                    }
                    
                    // --- 4. r(Psi) spline for NFW Diagnostic ---
                    struct RrPsiPair *temp_pairs_npsi_nfw_diag = (struct RrPsiPair *)malloc(num_points_diag * sizeof(struct RrPsiPair));
                    if(!temp_pairs_npsi_nfw_diag) {
                        log_message("ERROR", "Failed to allocate sorting pairs for NFW diagnostic r(Psi)");
                        goto cleanup_diag_iteration;
                    }
                    for(int k_sort = 0; k_sort < num_points_diag; ++k_sort) {
                        temp_pairs_npsi_nfw_diag[k_sort].rr = nPsivalues_diag_arr[k_sort];
                        temp_pairs_npsi_nfw_diag[k_sort].psi = radius_for_rofPsi_diag_arr[k_sort];
                    }
                    qsort(temp_pairs_npsi_nfw_diag, num_points_diag, sizeof(struct RrPsiPair), compare_by_rr);
                    for(int k_sort = 0; k_sort < num_points_diag; ++k_sort) {
                        nPsivalues_diag_arr[k_sort] = temp_pairs_npsi_nfw_diag[k_sort].rr;
                        radius_for_rofPsi_diag_arr[k_sort] = temp_pairs_npsi_nfw_diag[k_sort].psi;
                    }
                    free(temp_pairs_npsi_nfw_diag);

                    rofPsiinterp_diag = gsl_interp_accel_alloc();
                    splinerofPsi_diag = gsl_spline_alloc(gsl_interp_cspline, num_points_diag);
                    if(!check_strict_monotonicity(nPsivalues_diag_arr, num_points_diag, "nPsivalues_diag_arr (NFW_DIAG)")) {
                        log_message("ERROR", "NFW diagnostic nPsivalues not monotonic");
                        goto cleanup_diag_iteration;
                    }
                    gsl_spline_init(splinerofPsi_diag, nPsivalues_diag_arr, radius_for_rofPsi_diag_arr, num_points_diag);


                    // --- 5. I(E) spline (fofEinterp_diag) for NFW Diagnostic ---
                    double Psimin_diag = Psivalues_diag_arr[num_points_diag - 1];
                    double Psimax_diag = Psivalues_diag_arr[0];
                    if (Psimax_diag <= Psimin_diag) {
                        log_message("ERROR", "NFW diagnostic potential not monotonic: Psimax=%e <= Psimin=%e", Psimax_diag, Psimin_diag);
                        goto cleanup_diag_iteration;
                    }

                    Evalues_diag_arr = (double *)malloc((num_points_diag + 1) * sizeof(double));
                    innerintegrandvalues_diag_arr = (double *)malloc((num_points_diag + 1) * sizeof(double));
                    if(!Evalues_diag_arr || !innerintegrandvalues_diag_arr) {
                        log_message("ERROR", "Failed to allocate arrays for NFW diagnostic I(E)");
                        goto cleanup_diag_iteration;
                    }

                    Evalues_diag_arr[0] = Psimin_diag; 
                    innerintegrandvalues_diag_arr[0] = 0.0;
                    gsl_function F_fE_nfw_diag; 
                    F_fE_nfw_diag.function = &fEintegrand_nfw;

                    for (int k = 1; k <= num_points_diag; k++) {
                        double calE_diag = Psimin_diag + (Psimax_diag - Psimin_diag) * ((double)k) / ((double)num_points_diag);
                        if (k > 0 && calE_diag <= Evalues_diag_arr[k-1]) {
                            calE_diag = Evalues_diag_arr[k-1] + DBL_EPSILON * fabs(Evalues_diag_arr[k-1]) + DBL_MIN;
                        }
                        Evalues_diag_arr[k] = calE_diag;

                        fE_integrand_params_NFW_t params_fE_nfw_diag = {
                            calE_diag, splinerofPsi_diag, rofPsiinterp_diag,
                            splinemass_diag, enclosedmass_diag, G_CONST,
                            current_diag_rc, nfw_params_diag[2], current_diag_falloff_C,
                            Psimin_diag, Psimax_diag
                        };
                        F_fE_nfw_diag.params = &params_fE_nfw_diag;
                        double t_upper_diag = sqrt(fmax(0.0, calE_diag - Psimin_diag));
                        double t_lower_diag = (t_upper_diag > 1e-9) ? t_upper_diag / 1.0e4 : 0.0;
                        if (t_lower_diag >= t_upper_diag - 1e-12) { 
                            result_diag = 0.0; 
                        } else {
                            gsl_integration_qag(&F_fE_nfw_diag, t_lower_diag, t_upper_diag, 1e-8, 1e-8,
                                                Nintegration_diag, GSL_INTEG_GAUSS61, w_diag, &result_diag, &error_diag);
                        }
                        innerintegrandvalues_diag_arr[k] = result_diag;
                    }

                    fofEacc_diag = gsl_interp_accel_alloc();
                    fofEinterp_diag = gsl_interp_alloc(gsl_interp_linear, num_points_diag + 1);
                    if(!check_strict_monotonicity(Evalues_diag_arr, num_points_diag + 1, "Evalues_diag_arr (NFW_DIAG)")) {
                        log_message("ERROR", "NFW diagnostic Evalues not monotonic");
                        goto cleanup_diag_iteration;
                    }
                    gsl_interp_init(fofEinterp_diag, Evalues_diag_arr, innerintegrandvalues_diag_arr, num_points_diag + 1);
                    
                    // Write f(E) diagnostic file
                    char diag_fname_fofe[256];
                    char diag_base_fofe[128];
                    snprintf(diag_base_fofe, sizeof(diag_base_fofe), "data/f_of_E_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_fofe, 1, diag_fname_fofe, sizeof(diag_fname_fofe));
                    FILE *fp_diag_fofe = fopen(diag_fname_fofe, "wb");
                    if (fp_diag_fofe) {
                        for (int k = 0; k <= num_points_diag; k++) {
                            double E_diag = Evalues_diag_arr[k];
                            double deriv_diag = 0.0;
                            if (E_diag > Evalues_diag_arr[0] && E_diag < Evalues_diag_arr[num_points_diag]) {
                                 deriv_diag = gsl_interp_eval_deriv(fofEinterp_diag, Evalues_diag_arr, innerintegrandvalues_diag_arr, E_diag, fofEacc_diag);
                            }
                            double fE_val_diag = fabs(deriv_diag) / (sqrt(8.0) * PI * PI);
                            if (!isfinite(fE_val_diag)) fE_val_diag = 0.0;
                            fprintf_bin(fp_diag_fofe, "%f %f\n", E_diag, fE_val_diag);
                        }
                        fclose(fp_diag_fofe);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_fofe);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_fofe);
                    }

                    // Write NFW diagnostic integrand file
                    char diag_fname_integrand[256];
                    char diag_base_integrand[128];
                    snprintf(diag_base_integrand, sizeof(diag_base_integrand), "data/integrand_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_integrand, 1, diag_fname_integrand, sizeof(diag_fname_integrand));
                    FILE *fp_diag_int = fopen(diag_fname_integrand, "wb");
                    if (fp_diag_int) {
                        // Simple integrand convergence test (like Cored profile)
                        double calE_for_integrand = Psimax_diag;
                        fE_integrand_params_NFW_t params_int_nfw = {
                            calE_for_integrand, splinerofPsi_diag, rofPsiinterp_diag,
                            splinemass_diag, enclosedmass_diag, G_CONST,
                            current_diag_rc, nfw_params_diag[2], current_diag_falloff_C,
                            Psimin_diag, Psimax_diag
                        };
                        
                        for (int k_int = 0; k_int < num_points_diag; k_int++) {
                            double t_diag_int = sqrt(fmax(0.0, calE_for_integrand - Psimin_diag)) * ((double)k_int) / ((double)num_points_diag);
                            fprintf_bin(fp_diag_int, "%f %f\n", t_diag_int, fEintegrand_nfw(t_diag_int, &params_int_nfw));
                        }
                        
                        fclose(fp_diag_int);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_integrand);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_integrand);
                    }

                    // Write NFW diagnostic density profile
                    char diag_fname_dens[256];
                    char diag_base_dens[128];
                    snprintf(diag_base_dens, sizeof(diag_base_dens), "data/density_profile_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_dens, 1, diag_fname_dens, sizeof(diag_fname_dens));
                    FILE *fp_diag_dens = fopen(diag_fname_dens, "wb");
                    if (fp_diag_dens) {
                        for (int k = 0; k < num_points_diag; k++) {
                            double rr_k = radius_diag_arr[k];
                            double rs_k = rr_k / current_diag_rc;
                            double term_s_k = rs_k + 0.01;
                            if (term_s_k <= 1e-9) term_s_k = 1e-9;
                            double term_n_k = (1.0 + rs_k) * (1.0 + rs_k);
                            double term_c_base_k = rs_k / current_diag_falloff_C;
                            double term_c_k = 1.0 + pow(term_c_base_k, 10.0);
                            double rho_shape_k = (term_s_k < 1e-9 || term_n_k < 1e-9 || term_c_k < 1e-9) ? 0.0 : (1.0 / (term_s_k * term_n_k * term_c_k));
                            if (rr_k < 1e-6 && term_s_k < 1e-3 && rho_shape_k == 0.0) {
                               rho_shape_k = 1.0 / (term_s_k * term_n_k * term_c_k);
                            }
                            double rho_r_k = nfw_params_diag[2] * rho_shape_k;
                            fprintf_bin(fp_diag_dens, "%f %f\n", rr_k, rho_r_k);
                        }
                        fclose(fp_diag_dens);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_dens);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_dens);
                    }

                    // Write NFW diagnostic dPsi/dr
                    char diag_fname_dpsi[256];
                    char diag_base_dpsi[128];
                    snprintf(diag_base_dpsi, sizeof(diag_base_dpsi), "data/dpsi_dr_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_dpsi, 1, diag_fname_dpsi, sizeof(diag_fname_dpsi));
                    FILE *fp_diag_dpsi = fopen(diag_fname_dpsi, "wb");
                    if (fp_diag_dpsi) {
                        for (int k = 0; k < num_points_diag; k++) {
                            double rr_k = radius_diag_arr[k];
                            if (rr_k > 1e-9) {
                                double Menc_k = gsl_spline_eval(splinemass_diag, rr_k, enclosedmass_diag);
                                double dpsidr_k = -(G_CONST * Menc_k) / (rr_k * rr_k);
                                fprintf_bin(fp_diag_dpsi, "%f %f\n", rr_k, dpsidr_k);
                            }
                        }
                        fclose(fp_diag_dpsi);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_dpsi);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_dpsi);
                    }

                    // Write NFW diagnostic drho/dPsi
                    char diag_fname_drhodpsi[256];
                    char diag_base_drhodpsi[128];
                    snprintf(diag_base_drhodpsi, sizeof(diag_base_drhodpsi), "data/drho_dpsi_Ni%d_Ns%d.dat", Nintegration_diag, Nspline_diag_base);
                    get_suffixed_filename(diag_base_drhodpsi, 1, diag_fname_drhodpsi, sizeof(diag_fname_drhodpsi));
                    FILE *fp_diag_drhodpsi = fopen(diag_fname_drhodpsi, "wb");
                    if (fp_diag_drhodpsi) {
                        for (int k = 1; k < num_points_diag - 1; k++) {
                            double rr_k = radius_diag_arr[k];
                            if (rr_k <= 1e-9) continue;

                            double drhodr_val_k = drhodr_profile_nfwcutoff(rr_k, current_diag_rc, nfw_params_diag[2], current_diag_falloff_C);
                            double Menc_k = gsl_spline_eval(splinemass_diag, rr_k, enclosedmass_diag);
                            double dPsidr_mag_k = (G_CONST * Menc_k) / (rr_k * rr_k);
                            
                            if (fabs(dPsidr_mag_k) > 1e-30) {
                                double Psi_val_k = evaluatespline(splinePsi_diag, Psiinterp_diag, rr_k);
                                double drho_dPsi_val_k = drhodr_val_k / dPsidr_mag_k;
                                fprintf_bin(fp_diag_drhodpsi, "%f %f\n", Psi_val_k, drho_dPsi_val_k);
                            }
                        }
                        fclose(fp_diag_drhodpsi);
                        log_message("DEBUG", "Wrote diagnostic file: %s", diag_fname_drhodpsi);
                    } else {
                        log_message("ERROR", "Failed to open diagnostic file: %s", diag_fname_drhodpsi);
                    }
                    
                    log_message("DEBUG", "Finished diagnostic iteration %d, spline_base=%d", Nintegration_diag, Nspline_diag_base);

                    // --- Cleanup for NFW Diagnostic Iteration ---
cleanup_diag_iteration:
                    gsl_integration_workspace_free(w_diag);
                    if(splinemass_diag) gsl_spline_free(splinemass_diag);
                    if(enclosedmass_diag) gsl_interp_accel_free(enclosedmass_diag);
                    if(splinePsi_diag) gsl_spline_free(splinePsi_diag);
                    if(Psiinterp_diag) gsl_interp_accel_free(Psiinterp_diag);
                    if(splinerofPsi_diag) gsl_spline_free(splinerofPsi_diag);
                    if(rofPsiinterp_diag) gsl_interp_accel_free(rofPsiinterp_diag);
                    if(fofEinterp_diag) gsl_interp_free(fofEinterp_diag);
                    if(fofEacc_diag) gsl_interp_accel_free(fofEacc_diag);
                    free(mass_diag_arr); 
                    free(radius_diag_arr); 
                    free(radius_for_rofPsi_diag_arr);
                    free(Psivalues_diag_arr); 
                    free(nPsivalues_diag_arr);
                    free(Evalues_diag_arr); 
                    free(innerintegrandvalues_diag_arr);
                } // end Nspline_diag_base loop
            } // end Nintegration_diag loop
            log_message("INFO", "NFW DIAGNOSTIC LOOP: Completed convergence tests for NFW profile.");
        } // end if(g_doDebug) for NFW diagnostic loop

        // NFW PROFILE IC GENERATION PATHWAY
        
        /**
         * @brief NFW-specific theoretical calculation for initial conditions.
         * @details Calculates mass profile, potential, and distribution function
         *          for the NFW-like profile with power-law cutoff.
         */
        
        // Re-establish main NFW calc parameters for the main calculation
        double current_profile_rc = g_nfw_profile_rc;
        double current_profile_halo_mass = g_nfw_profile_halo_mass;
        double current_profile_rmax_norm_factor = g_nfw_profile_rmax_norm_factor;
        double current_profile_falloff_C = g_nfw_profile_falloff_factor;

        num_points = 100000; // Main NFW calculation's num_points
        int num_maxv2f = 1000; // Resolution for velocity envelope calculation only
        rmax = current_profile_rmax_norm_factor * current_profile_rc; // Main NFW rmax
        
        // Local variables for NFW pathway
        double nfw_result, nfw_error;
        double nfw_calE;
        int i_nfw;
        
        // Set NFW-specific parameters from generalized profile parameters
        g_nfw_profile_rc = g_scale_radius_param;
        g_nfw_profile_halo_mass = g_halo_mass_param;
        g_nfw_profile_rmax_norm_factor = g_cutoff_factor_param;
        
        // Update current_profile_* variables with new values
        current_profile_rc = g_nfw_profile_rc;
        current_profile_halo_mass = g_nfw_profile_halo_mass;
        current_profile_rmax_norm_factor = g_nfw_profile_rmax_norm_factor;
        current_profile_falloff_C = g_nfw_profile_falloff_factor;
        
        // Set numerical parameters for NFW
        num_points = 100000;  // Conservative default for NFW
        rmax = current_profile_rmax_norm_factor * current_profile_rc;
        
        // Allocate GSL workspace
        w = gsl_integration_workspace_alloc(1000);
        if (!w) {
            fprintf(stderr, "NFW_PATH: Failed to allocate GSL workspace\n");
            CLEAN_EXIT(1);
        }
        
        // Prepare mass integrand function
        gsl_function F_nfw_calc;
        F_nfw_calc.function = &massintegrand_profile_nfwcutoff;
        
        // Prepare parameters for NFW mass integrand: [rc, halo_mass, nt_nfw, falloff_factor]
        double nfw_params[4]; // Parameters for NFW mass integrand function
        nfw_params[0] = current_profile_rc;
        nfw_params[1] = current_profile_halo_mass; // Target total halo mass
        nfw_params[2] = 1.0; // Initial guess for nt_nfw normalization scaler  
        nfw_params[3] = current_profile_falloff_C; // Falloff transition factor C
        F_nfw_calc.params = nfw_params;
        
        // Calculate normalization for NFW profile
        int status_norm = gsl_integration_qag(&F_nfw_calc, 0.0, rmax, 1e-12, 1e-12, 1000, 
                            GSL_INTEG_GAUSS51, w, &nfw_result, &nfw_error);
        normalization = nfw_result;
        
        if (g_doDebug) {
            log_message("DEBUG", "Initial normalization integral (int r^2 * rho_guess dr):");
            log_message("DEBUG", "  rmax_for_norm_integral = %.3e kpc (factor=%.1f * rc=%.3f)", rmax, current_profile_rmax_norm_factor, current_profile_rc);
            log_message("DEBUG", "  GSL QAG status for norm_integral: %s", gsl_strerror(status_norm));
            log_message("DEBUG", "  Raw norm_integral_result (nfw_result for norm) = %.6e", nfw_result);
            log_message("DEBUG", "  Raw norm_integral_error_est = %.6e", nfw_error);
            log_message("DEBUG", "  Final 'normalization' variable = %.6e", normalization);
        }
        
        if (normalization <= 1e-30) {
            fprintf(stderr, "NFW_PATH: Normalization is zero or negative (%.3e). Exiting.\n", normalization);
            CLEAN_EXIT(1);
        }
        
        // Update nt_nfw with proper normalization
        nfw_params[2] = current_profile_halo_mass / (4.0 * M_PI * normalization);
        
        if (g_doDebug) {
            log_message("DEBUG", "Calculated nt_nfw (density scale factor):");
            log_message("DEBUG", "  current_profile_halo_mass (target M_total) = %.3e Msun", current_profile_halo_mass);
            log_message("DEBUG", "  nt_nfw = %.3e / (4pi * %.3e) = %.6e", current_profile_halo_mass, normalization, nfw_params[2]);
            if (!isfinite(nfw_params[2]) || (fabs(nfw_params[2]) < 1e-100 && fabs(nfw_params[2]) > 0)) {
                log_message("WARNING", "nt_nfw is NaN, Inf, or extremely small/large: %.6e", nfw_params[2]);
            }
        }
        
        log_message("INFO", "NFW Profile: RC=%.3f kpc, Halo Mass=%.3e Msun, Rmax_norm_calc=%.3f kpc, nt_nfw_scaler=%.6e",
               current_profile_rc, current_profile_halo_mass, rmax, nfw_params[2]);
        
        /**
         * @brief Calculate mass profile M(r) for NFW.
         */
        mass = (double *)malloc(num_points * sizeof(double));
        radius = (double *)malloc(num_points * sizeof(double));
        radius_monotonic_grid_nfw = (double *)malloc(num_points * sizeof(double));
        if (!mass || !radius || !radius_monotonic_grid_nfw) {
            fprintf(stderr, "NFW_PATH: Failed to allocate mass/radius arrays\n");
            CLEAN_EXIT(1);
        }
        
        mass[0] = 0.0;
        radius[0] = 0.0;                                  // For the y-values of r(Psi) spline later
        radius_monotonic_grid_nfw[0] = 0.0;             // For x-axes of M(r), Psi(r), maxv2f(r)
        
        
        for (i_nfw = 1; i_nfw < num_points; i_nfw++) {
            double r_current = (double)i_nfw * rmax / (num_points - 1);
            if (i_nfw == num_points - 1) r_current = rmax; // Ensure exact endpoint
            
            gsl_integration_qag(&F_nfw_calc, 0.0, r_current, 1e-12, 1e-12,
                                1000, GSL_INTEG_GAUSS51, w, &nfw_result, &nfw_error);
            mass[i_nfw] = 4.0 * M_PI * nfw_result;
            radius[i_nfw] = r_current; // This 'radius' array will be sorted with nPsivalues
            radius_monotonic_grid_nfw[i_nfw] = r_current; // This 'radius_monotonic_grid_nfw' stays sorted by r
        }
        
        if (g_doDebug) {
            log_message("DEBUG", "M(r) spline data summary (num_points=%d):", num_points);
            log_message("DEBUG", "  Target M_total for sampling = %.3e Msun", current_profile_halo_mass);
            log_message("DEBUG", "  nt_nfw used for M(r) calcs = %.3e", nfw_params[2]);
            log_message("DEBUG", "  rmax for M(r) array = %.3e kpc", rmax);
            if (num_points > 0) {
                log_message("DEBUG", "  Final Mass at rmax (radius[num_points-1]=%.3e kpc): %.3e Msun", radius[num_points-1], mass[num_points-1]);
                if (fabs(mass[num_points-1] - current_profile_halo_mass) / current_profile_halo_mass > 0.1) {
                    log_message("WARNING", "Mass at rmax (%.3e) differs significantly from target halo mass (%.3e)!", mass[num_points-1], current_profile_halo_mass);
                }
            }
            log_message("DEBUG", "End of M(r) data summary.");
        }
        
        // Create mass spline
        enclosedmass = gsl_interp_accel_alloc();
        splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!enclosedmass || !splinemass) {
            fprintf(stderr, "NFW_PATH: Failed to allocate mass spline\n");
            CLEAN_EXIT(1);
        }
        gsl_spline_init(splinemass, radius_monotonic_grid_nfw, mass, num_points);
        
        // Write generic mass profile for plotting script
        if (g_doDebug) {
            fp = fopen("data/massprofile.dat", "wb");
            if (fp) {
                for (double r_write = 0.0; r_write < radius_monotonic_grid_nfw[num_points-1]; r_write += rmax / 900.0) {
                    double mass_at_r = gsl_spline_eval(splinemass, r_write, enclosedmass);
                    fprintf(fp, "%e %e\n", r_write, mass_at_r);
                }
                fclose(fp);
            }
        }
        
        /**
         * @brief Calculate gravitational potential Psi(r) for NFW.
         */
        Psivalues = (double *)malloc(num_points * sizeof(double));
        nPsivalues = (double *)malloc(num_points * sizeof(double));
        if (!Psivalues || !nPsivalues) {
            fprintf(stderr, "NFW_PATH: Failed to allocate Psi arrays\n");
            CLEAN_EXIT(1);
        }
        
        // Prepare Psiintegrand parameters for NFW
        Psiintegrand_params psi_params_nfw;
        psi_params_nfw.massintegrand_func = &massintegrand_profile_nfwcutoff;
        psi_params_nfw.params_for_massintegrand = nfw_params;
        
        gsl_function F_for_psi_nfw;
        F_for_psi_nfw.function = &Psiintegrand;
        F_for_psi_nfw.params = &psi_params_nfw;
        
        for (i_nfw = 0; i_nfw < num_points; i_nfw++) {
            double r_current = radius[i_nfw];
            double r1_psi = fmax(r_current, current_profile_rc / 1000000.0);
            
            gsl_integration_qagiu(&F_for_psi_nfw, r1_psi, 1e-12, 1e-12,
                                  1000, w, &nfw_result, &nfw_error);
            double first_term_psi = G_CONST * gsl_spline_eval(splinemass, r1_psi, enclosedmass) / r1_psi;
            double second_term_psi = G_CONST * 4.0 * M_PI * nfw_result;
            if (r1_psi < 1e-9) first_term_psi = 0; // Avoid division by zero
            Psivalues[i_nfw] = (first_term_psi + second_term_psi);
            nPsivalues[i_nfw] = -Psivalues[i_nfw];
        }
        
        if (g_doDebug) {
            log_message("DEBUG", "Psi(r) and r(Psi) spline data summary (num_points=%d):", num_points);
            if (num_points > 1) {
                log_message("DEBUG", "  Psivalues[0] (Psimax candidate) = %.6e", Psivalues[0]);
                log_message("DEBUG", "  Psivalues[num_points-1] (Psimin candidate) = %.6e", Psivalues[num_points-1]);
                if (Psivalues[0] <= Psivalues[num_points-1]) {
                    log_message("WARNING", "Psi(r) may not be monotonic decreasing (Psivalues[0]=%.3e <= Psivalues[end]=%.3e)!", Psivalues[0], Psivalues[num_points-1]);
                }
            }
            log_message("DEBUG", "End of Psi(r) data summary.");
        }
        
        // Create Psi splines
        Psiinterp = gsl_interp_accel_alloc();
        splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!Psiinterp || !splinePsi) {
            fprintf(stderr, "NFW_PATH: Failed to allocate Psi spline\n");
            CLEAN_EXIT(1);
        }
        gsl_spline_init(splinePsi, radius_monotonic_grid_nfw, Psivalues, num_points);
        
        // Write generic Psi profile for plotting script
        if (g_doDebug) {
            fp = fopen("data/Psiprofile.dat", "wb");
            if (fp) {
                for (double r_write = 0.0; r_write < radius_monotonic_grid_nfw[num_points-1]; r_write += rmax / 900.0) {
                    double psi_at_r = gsl_spline_eval(splinePsi, r_write, Psiinterp);
                    fprintf(fp, "%e %e\n", r_write, psi_at_r);
                }
                fclose(fp);
            }
        }
        
        // Create r(Psi) spline
        rofPsiinterp = gsl_interp_accel_alloc();
        splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
        if (!rofPsiinterp || !splinerofPsi) {
            fprintf(stderr, "NFW_PATH: Failed to allocate r(Psi) spline\n");
            CLEAN_EXIT(1);
        }
        
        // Use temporary copies for r(Psi) spline to preserve the original radius grid
        double *nPsivalues_for_rPsi_spline = (double *)malloc(num_points * sizeof(double));
        double *radius_values_for_rPsi_spline = (double *)malloc(num_points * sizeof(double));

        if (!nPsivalues_for_rPsi_spline || !radius_values_for_rPsi_spline) {
            fprintf(stderr, "NFW_PATH: Failed to allocate temp arrays for r(Psi) spline data.\n");
            if (nPsivalues_for_rPsi_spline) free(nPsivalues_for_rPsi_spline);
            if (radius_values_for_rPsi_spline) free(radius_values_for_rPsi_spline);
            CLEAN_EXIT(1);
        }

        // Copy nPsivalues and the corresponding radius_monotonic_grid_nfw values
        memcpy(nPsivalues_for_rPsi_spline, nPsivalues, num_points * sizeof(double));
        memcpy(radius_values_for_rPsi_spline, radius_monotonic_grid_nfw, num_points * sizeof(double));

        // Sort nPsivalues_for_rPsi_spline and apply identical swaps to radius_values_for_rPsi_spline
        for (int k_sort = 0; k_sort < num_points - 1; k_sort++) {
            for (int j_sort = k_sort + 1; j_sort < num_points; j_sort++) {
                if (nPsivalues_for_rPsi_spline[k_sort] > nPsivalues_for_rPsi_spline[j_sort]) {
                    // Swap nPsivalues_for_rPsi_spline
                    double temp_npsi = nPsivalues_for_rPsi_spline[k_sort];
                    nPsivalues_for_rPsi_spline[k_sort] = nPsivalues_for_rPsi_spline[j_sort];
                    nPsivalues_for_rPsi_spline[j_sort] = temp_npsi;

                    // Swap corresponding radius_values_for_rPsi_spline
                    double temp_rad = radius_values_for_rPsi_spline[k_sort];
                    radius_values_for_rPsi_spline[k_sort] = radius_values_for_rPsi_spline[j_sort];
                    radius_values_for_rPsi_spline[j_sort] = temp_rad;
                }
            }
        }
        
        // Debug check for nPsivalues_for_rPsi_spline monotonicity
        if (g_doDebug) {
            int mono_violations_npsi = 0;
            for (int k_chk = 0; k_chk < num_points - 1; ++k_chk) {
                if (!(nPsivalues_for_rPsi_spline[k_chk+1] > nPsivalues_for_rPsi_spline[k_chk])) {
                    if (mono_violations_npsi < 5) log_message("DEBUG", "nPsivalues_for_rPsi_spline not strictly increasing at index %d", k_chk);
                    mono_violations_npsi++;
                }
            }
            if (mono_violations_npsi > 0) log_message("WARNING", "Total nPsivalues violations in r(Psi) spline: %d", mono_violations_npsi);
            else if (g_doDebug) log_message("DEBUG", "nPsivalues_for_rPsi_spline confirmed strictly monotonic for r(Psi) spline.");
        }

        // Initialize splinerofPsi with the sorted temporary arrays
        gsl_spline_init(splinerofPsi, nPsivalues_for_rPsi_spline, radius_values_for_rPsi_spline, num_points);
        
        // Free the temporary sorted copies
        free(nPsivalues_for_rPsi_spline);
        free(radius_values_for_rPsi_spline);
        
        /**
         * @brief Calculate f(E) distribution function for NFW using Eddington's formula.
         */
        Psimin = Psivalues[num_points - 1];
        Psimax = Psivalues[0];
        
        if (g_doDebug) {
            log_message("DEBUG", "Potential range for I(E) calculation: Psimin=%.6e, Psimax=%.6e", Psimin, Psimax);
            log_message("DEBUG", "Continuing with I(E) calculation.");
        }
        
        if (Psimax <= Psimin) {
            fprintf(stderr, "NFW_PATH: Potential not monotonic (Psimax=%.3e <= Psimin=%.3e)\n", 
                    Psimax, Psimin);
            CLEAN_EXIT(1);
        }
        
        innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
        Evalues = (double *)malloc((num_points + 1) * sizeof(double));
        if (!innerintegrandvalues || !Evalues) {
            fprintf(stderr, "NFW_PATH: Failed to allocate f(E) arrays\n");
            CLEAN_EXIT(1);
        }
        
        // NFW uses conservative tolerance for f(E) integral
        F_nfw_calc.function = &fEintegrand_nfw;
        
        innerintegrandvalues[0] = 0.0;
        Evalues[0] = Psimin;
        
        // GSL integration status tracking
        int status_fE_nfw_local;
        
        for (i_nfw = 1; i_nfw <= num_points; i_nfw++) {
            nfw_calE = Psimin + (Psimax - Psimin) * ((double)i_nfw) / ((double)num_points);
            
            // Enforce strict monotonicity for Evalues
            if (i_nfw > 0 && nfw_calE <= Evalues[i_nfw-1]) {
                // If current nfw_calE is not strictly greater than previous, add a tiny increment.
                // DBL_EPSILON for the scale of Evalues might be too small if Evalues are large.
                // A small fraction of the typical step, or a fixed small number relative to Evalues scale.
                double previous_E = Evalues[i_nfw-1];
                double ideal_step = (Psimax - Psimin) / (double)num_points;
                double increment = ideal_step * 1e-6; // Small fraction of an ideal step
                if (increment == 0.0) increment = DBL_MIN * fabs(previous_E) + DBL_MIN; // Absolute minimum if ideal step is zero
                if (increment == 0.0) increment = 1e-20; // Fallback if previous_E is zero

                nfw_calE = previous_E + increment;

                if (g_doDebug && (i_nfw <= 10 || i_nfw > num_points -10 || i_nfw % (num_points/50<1?1:num_points/50) == 0) ) { // Log adjustment sparsely
                     log_message("DEBUG","Adjusted Evalues[%d] from ideal %.17e to %.17e (prev E: %.17e)",
                            i_nfw, Psimin + (Psimax - Psimin) * ((double)i_nfw) / ((double)num_points),
                            nfw_calE, previous_E);
                }
            }
            
            // Create NFW-specific parameter structure
            
            fE_integrand_params_NFW_t params_for_fE_integrand_nfw = {
                nfw_calE,               // E_current_shell
                splinerofPsi,           // spline_r_of_Psi (this is r_of_nPsi from nPsivalues)
                rofPsiinterp,           // accel_r_of_Psi
                splinemass,             // spline_M_of_r (this is M(r) for NFW, from corrected NFMP.1)
                enclosedmass,           // accel_M_of_r
                G_CONST,                // const_G_universal
                current_profile_rc,     // profile_rc_const
                nfw_params[2],          // profile_nt_norm_const (this is the scaled nt_nfw)
                nfw_params[3],          // profile_falloff_C_const (falloff factor C)
                Psimin,                 // Psimin_global
                Psimax                  // Psimax_global
            };
            F_nfw_calc.params = &params_for_fE_integrand_nfw;
            
            double E_current_shell = nfw_calE; // E for which I(E) is being computed

            // Integration will be over t_prime = sqrt(E_shell - Psi_true)
            // As Psi_true goes from Psimin_global to E_shell, t_prime goes from sqrt(E_shell - Psimin_global) down to 0.
            // So, integrate t_prime from 0 to sqrt(E_shell - Psimin_global).
            
            double t_integration_upper_bound = sqrt(fmax(0.0, E_current_shell - Psimin));
            double t_integration_lower_bound;

            if (t_integration_upper_bound < 1e-9) { // If E_current_shell is very close to Psimin (or below)
                t_integration_lower_bound = 0.0;
                t_integration_upper_bound = 0.0; 
            } else {
                // Set a very small, but strictly positive, lower bound relative to the upper bound,
                // or an absolute small number if t_upper_bound is itself very small.
                // This helps GSL avoid evaluating exactly at t=0 if there's a 1/t or 1/sqrt(t) type issue.
                // The term 1/sqrt(E-Psi) in d(rho)/d(Psi) / sqrt(E-Psi) becomes 1/t when Psi = E-t^2.
                // Our fEintegrand_nfw is 2 * d(rho)/d(Psi), so it does not have this explicit 1/t.
                // Using t_upper_bound / 1.0e4 scaling for numerical consistency.
                t_integration_lower_bound = t_integration_upper_bound / 1.0e4; 
                // If t_integration_lower_bound becomes extremely small (e.g. < DBL_MIN), GSL might treat it as zero.
                // Ensure it's at least some representable small positive number if t_upper_bound is positive.
                if (t_integration_lower_bound == 0.0 && t_integration_upper_bound > 0.0) {
                    t_integration_lower_bound = DBL_EPSILON * t_integration_upper_bound; // Or just DBL_EPSILON if t_upper is very small
                    if (t_integration_lower_bound == 0.0) t_integration_lower_bound = 1e-20; // Absolute floor
                }
            }
            
            // Ensure lower bound is strictly less than upper bound for GSL
            if (t_integration_lower_bound >= t_integration_upper_bound - 1e-12) { // Adjusted epsilon
                t_integration_upper_bound = 0.0; // Force zero integration range
                t_integration_lower_bound = 0.0;
            }
            
            if (g_doDebug && (i_nfw <= 5 || i_nfw > num_points - 5 || i_nfw % (num_points/10 < 1 ? 1 : num_points/10) == 0) ) {
                log_message("DEBUG", "I(E) integral setup: E_shell=%.3e, Psimin=%.3e, integrating fEintegrand_nfw(t) from t_low=%.3e to t_high=%.3e",
                       E_current_shell, Psimin, t_integration_lower_bound, t_integration_upper_bound);
            }

            if (t_integration_upper_bound <= t_integration_lower_bound + 1e-10) { // If range is zero or too small
                nfw_result = 0.0;
                status_fE_nfw_local = GSL_SUCCESS; 
                 if (g_doDebug && (i_nfw <= 5 || i_nfw > num_points - 5 || i_nfw % (num_points/10 < 1 ? 1 : num_points/10) == 0) ) {
                    log_message("DEBUG", "NFEFE_INTEGRAL_SETUP: Skipping t-integration, range invalid/tiny (t_high=%.3e, t_low=%.3e)", t_integration_upper_bound, t_integration_lower_bound);
                 }
            } else {
                status_fE_nfw_local = gsl_integration_qag(&F_nfw_calc, t_integration_lower_bound, t_integration_upper_bound,
                                    1e-8, 1e-8, 1000, GSL_INTEG_GAUSS61, // Using conservative GSL tolerances
                                    w, &nfw_result, &nfw_error);
                                    
                if (g_doDebug && (i_nfw <= 5 || i_nfw > num_points - 5 || i_nfw % (num_points/10 < 1 ? 1 : num_points/10) == 0) ) {
                    log_message("DEBUG", "NFEFE_INTEGRAL_RESULT: I(E=%.3e) = %.6e, error=%.3e, status=%s",
                           E_current_shell, nfw_result, nfw_error, 
                           (status_fE_nfw_local == GSL_SUCCESS) ? "SUCCESS" : "ERROR");
                }
            }
            innerintegrandvalues[i_nfw] = nfw_result;
            Evalues[i_nfw] = nfw_calE;
        }
        
        
        // Create f(E) interpolation using cspline interpolation for NFW (smoother dI/dE)
        g_main_fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
        g_main_fofEacc = gsl_interp_accel_alloc();
        if (!g_main_fofEinterp || !g_main_fofEacc) {
            fprintf(stderr, "NFW_PATH: Failed to allocate f(E) interpolation\n");
            CLEAN_EXIT(1);
        }
        // ADD THIS BLOCK BEFORE gsl_interp_init:
        if (g_doDebug) {
            int monotonicity_violations = 0;
            log_message("DEBUG", "Checking Evalues for strict monotonicity (%d points) before I(E) spline init.", num_points + 1);
            // Evalues has num_points + 1 elements, indexed 0 to num_points.
            for (int chk_e = 0; chk_e < num_points; ++chk_e) { // Loop up to num_points-1 to check Evalues[chk_e+1] vs Evalues[chk_e]
                if (!(Evalues[chk_e+1] > Evalues[chk_e])) {
                    if (monotonicity_violations < 20) { // Print first few violations
                        fprintf(stderr, "  MONOTONICITY_VIOLATION_PRE_SPLINE: Evalues[%d]=%.17e, Evalues[%d]=%.17e (Diff: %.3e)\n",
                               chk_e, Evalues[chk_e], chk_e+1, Evalues[chk_e+1], Evalues[chk_e+1] - Evalues[chk_e]);
                    }
                    monotonicity_violations++;
                }
            }
            if (monotonicity_violations > 0) {
                fprintf(stderr, "  NFW_CRITICAL_SPLINE_INIT: Total Evalues monotonicity violations: %d. GSL interp_init will likely fail. Exiting.\n", monotonicity_violations);
                CLEAN_EXIT(1); // Add explicit exit if violations found.
            } else if (g_doDebug) { // Only log success if in debug mode
                log_message("DEBUG", "Evalues array confirmed strictly monotonic before I(E) spline init.");
            }
        }
        // END ADDED BLOCK
        
        gsl_interp_init(g_main_fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
        
        gsl_integration_workspace_free(w);
        w = NULL;
        
        log_message("INFO", "NFW theoretical calculation for IC splines complete.");
        
        /**
         * @brief NFW Sample Generator - Generate particle positions and velocities.
         * @details Uses rejection sampling with the NFW density profile and f(E) distribution.
         */
        if (doReadInit) {
            // Read initial conditions from file
            printf("NFW_PATH: Reading initial conditions from %s...\n", readInitFilename);
            read_initial_conditions(particles, npts_initial, readInitFilename);
        } else if (!g_doRestart) {
            if (tidal_fraction > 0.0) {
                log_message("INFO", "NFW IC Gen: Initial particle count before stripping: %d", npts_initial);
            }
            log_message("INFO", "NFW IC Gen: Generating %d initial particle positions and velocities...", npts_initial);
            
            // Print overall Psimin, Psimax for NFW path once
            if (npts_initial > 0) { // Avoid printing if no particles
                if (Psimax <= Psimin) {
                }
            }
            
            /**
             * @brief Allocate memory for the particle data array.
             * @details 2D array particles[5][npts_initial] where:
             *          [0] = radius, [1] = velocity, [2] = angular momentum,
             *          [3] = particle ID, [4] = orientation (mu)
             */
            particles = (double **)malloc(5 * sizeof(double *));
            if (particles == NULL) {
                fprintf(stderr, "NFW_PATH: Memory allocation failed for particle array\n");
                CLEAN_EXIT(1);
            }
            for (i = 0; i < 5; i++) {
                particles[i] = (double *)malloc(npts_initial * sizeof(double));
                if (particles[i] == NULL) {
                    fprintf(stderr, "NFW_PATH: Memory allocation failed for particles[%d]\n", i);
                    CLEAN_EXIT(1);
                }
            }
            
            /**
             * @brief Calculate maximum velocity squared at each radius for rejection sampling.
             */
            double *maxv2f_nfw = (double *)malloc(num_maxv2f * sizeof(double));
            double *radius_maxv2f_nfw = (double *)malloc(num_maxv2f * sizeof(double));
            if (!maxv2f_nfw || !radius_maxv2f_nfw) {
                fprintf(stderr, "NFW_PATH: Failed to allocate maxv2f arrays\n");
                CLEAN_EXIT(1);
            }
            
            double nfw_vel, nfw_ratio, nfw_Psir, nfw_mu, nfw_maxv, nfw_maxvalue;
            
            // Create spline for r(M) - radius as function of enclosed mass
            gsl_interp_accel *rofMaccel_nfw = gsl_interp_accel_alloc();
            gsl_spline *splinerofM_nfw = gsl_spline_alloc(gsl_interp_cspline, num_points);
            if (!rofMaccel_nfw || !splinerofM_nfw) {
                fprintf(stderr, "NFW_PATH: Failed to allocate r(M) spline\n");
                CLEAN_EXIT(1);
            }
            // ADD THIS DIAGNOSTIC BLOCK:
        if (g_doDebug) {
            int monotonicity_violations_mass_spline = 0;
            log_message("DEBUG", "Checking mass array for strict monotonicity (size: %d)", num_points);
            // 'mass' array has num_points elements. Loop up to num_points-2 to check mass[chk+1] vs mass[chk].
            if (num_points >= 2) { // Need at least 2 points to check monotonicity
                for (int chk_m = 0; chk_m < num_points - 1; ++chk_m) { 
                    if (!(mass[chk_m+1] > mass[chk_m])) {
                        if (monotonicity_violations_mass_spline < 20) { 
                            fprintf(stderr, "  Mass array monotonicity violation: mass[%d]=%.17e >= mass[%d]=%.17e (Diff: %.3e)\n",
                                   chk_m, mass[chk_m], chk_m+1, mass[chk_m+1], mass[chk_m+1] - mass[chk_m]);
                        }
                        monotonicity_violations_mass_spline++;
                    }
                }
            }
            if (monotonicity_violations_mass_spline > 0) {
                fprintf(stderr, "  NFW_CRITICAL_MONO_CHECK_ROFM: Total 'mass' array monotonicity violations: %d. gsl_spline_init for splinerofM_nfw will fail.\n", monotonicity_violations_mass_spline);
                 if (monotonicity_violations_mass_spline > 20) fprintf(stderr, "  (Further violations suppressed)\n");
            } else {
                log_message("DEBUG", "Mass array confirmed strictly monotonic.");
            }
        }
        // END ADDED DIAGNOSTIC BLOCK

        gsl_spline_init(splinerofM_nfw, mass, radius, num_points);
            
            // Calculate max v^2 * f(E) at each radius
            radius_maxv2f_nfw[0] = 0.0;
            for (int i_r_nfw = 1; i_r_nfw < num_maxv2f; i_r_nfw++) {
                nfw_maxvalue = 0.0;
                double r_maxv2f = (double)i_r_nfw * rmax / (num_maxv2f - 1);
                radius_maxv2f_nfw[i_r_nfw] = r_maxv2f;
                nfw_Psir = evaluatespline(splinePsi, Psiinterp, r_maxv2f);
                nfw_maxv = sqrt(2.0 * (nfw_Psir - Psimin));
                
                // Find maximum of v^2 * dI/dE over velocity range
                for (int j_v_nfw = 1; j_v_nfw < num_maxv2f - 2; j_v_nfw++) {
                    nfw_vel = nfw_maxv * ((double)j_v_nfw) / ((double)num_maxv2f);
                    double E_test_nfw = nfw_Psir - 0.5 * nfw_vel * nfw_vel;
                    double currentvalue_nfw = 0.0;
                    
                    if (E_test_nfw >= Psimin && E_test_nfw <= Psimax) {
                        currentvalue_nfw = nfw_vel * nfw_vel * 
                            fabs(gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, 
                                                       innerintegrandvalues, E_test_nfw, g_main_fofEacc));
                    }
                    if (isfinite(currentvalue_nfw) && currentvalue_nfw > nfw_maxvalue) {
                        nfw_maxvalue = currentvalue_nfw;
                    }
                }
                maxv2f_nfw[i_r_nfw] = nfw_maxvalue;
            }
            
            // Extrapolate for r=0
            if (num_maxv2f >= 3) {
                maxv2f_nfw[0] = 2.0 * maxv2f_nfw[1] - maxv2f_nfw[2];
                if (maxv2f_nfw[0] < 0) maxv2f_nfw[0] = 0;
            } else {
                maxv2f_nfw[0] = maxv2f_nfw[1];
            }
            
            // Create spline for max v^2 * f(E)
            gsl_interp_accel *maxv2faccel_nfw = gsl_interp_accel_alloc();
            gsl_spline *splinemaxv2f_nfw = gsl_spline_alloc(gsl_interp_cspline, num_maxv2f);
            if (!maxv2faccel_nfw || !splinemaxv2f_nfw) {
                fprintf(stderr, "NFW_PATH: Failed to allocate maxv2f spline\n");
                CLEAN_EXIT(1);
            }
            // ADD THIS DIAGNOSTIC BLOCK:
        if (g_doDebug) {
            int monotonicity_violations_rad_spline = 0;
            log_message("DEBUG", "Checking radius array for strict monotonicity (size: %d)", num_points);
            // 'radius' array has num_points elements. Loop up to num_points-2.
            if (num_points >= 2) {
                for (int chk_r = 0; chk_r < num_points - 1; ++chk_r) { 
                    if (!(radius[chk_r+1] > radius[chk_r])) {
                        if (monotonicity_violations_rad_spline < 20) {
                            fprintf(stderr, "  Radius array monotonicity violation: radius[%d]=%.17e >= radius[%d]=%.17e (Diff: %.3e)\n",
                                   chk_r, radius[chk_r], chk_r+1, radius[chk_r+1], radius[chk_r+1] - radius[chk_r]);
                        }
                        monotonicity_violations_rad_spline++;
                    }
                }
            }
            if (monotonicity_violations_rad_spline > 0) {
                fprintf(stderr, "  NFW_CRITICAL_MONO_CHECK_MAXV2F: Total 'radius' array monotonicity violations: %d. gsl_spline_init for splinemaxv2f_nfw will fail.\n", monotonicity_violations_rad_spline);
                if (monotonicity_violations_rad_spline > 20) fprintf(stderr, "  (Further violations suppressed)\n");
            } else {
                log_message("DEBUG", "Radius array confirmed strictly monotonic.");
            }
        }
        // END ADDED DIAGNOSTIC BLOCK

        gsl_spline_init(splinemaxv2f_nfw, radius_maxv2f_nfw, maxv2f_nfw, num_maxv2f);
            
            /**
             * @brief Generate particles using rejection sampling.
             */
            for (int k_nfw = 0; k_nfw < npts_initial; k_nfw++) {
                if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) { // Log for first few & periodically
                    fflush(stdout);
                }
                
                // Sample radius from mass distribution
                double mass_frac_sample_nfw = gsl_rng_uniform(g_rng) * 0.999999;
                double mass_sample_nfw = mass_frac_sample_nfw * current_profile_halo_mass;
                particles[0][k_nfw] = evaluatespline(splinerofM_nfw, rofMaccel_nfw, mass_sample_nfw);
                
                if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                }
                
                nfw_maxvalue = evaluatespline(splinemaxv2f_nfw, maxv2faccel_nfw, particles[0][k_nfw]);
                nfw_Psir = evaluatespline(splinePsi, Psiinterp, particles[0][k_nfw]);
                
                // Check for problematic values
                if (!isfinite(nfw_Psir)) {
                    if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                    }
                    particles[1][k_nfw] = 0.0; // Assign zero velocity
                    nfw_mu = (2.0 * gsl_rng_uniform(g_rng) - 1.0);
                    particles[2][k_nfw] = 0.0; // L = 0 since v = 0
                    particles[4][k_nfw] = nfw_mu;
                    particles[3][k_nfw] = (double)k_nfw;
                    continue;
                }
                
                if (nfw_Psir <= Psimin + 1e-9 * fabs(Psimin)) { // Check if Psir is too close to Psimin
                    if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                    }
                    particles[1][k_nfw] = 0.0; // No kinetic energy possible
                } else {
                    // Sample velocity using rejection method
                    nfw_maxv = sqrt(fmax(0.0, 2.0 * (nfw_Psir - Psimin)));
                    if (!isfinite(nfw_maxv) || nfw_maxv < 1e-9) {
                        particles[1][k_nfw] = 0.0;
                    } else {
                        
                        if (!isfinite(nfw_maxvalue) || nfw_maxvalue <= 1e-30) { // If envelope is effectively zero
                            particles[1][k_nfw] = 0.0;
                        } else {
                            // Velocity Rejection Sampling Loop
                            int vflag_nfw = 0;
                            int v_trials_nfw = 0;
                    
                    while (vflag_nfw == 0 && v_trials_nfw < 20000) {
                        v_trials_nfw++;
                        nfw_vel = gsl_rng_uniform(g_rng) * nfw_maxv;
                        double E_test_nfw = nfw_Psir - 0.5 * nfw_vel * nfw_vel;
                        double target_func_val_nfw = 0.0;
                        
                        double deriv_val_dIdE = 0.0;
                        if (E_test_nfw >= Psimin - 1e-9*fabs(Psimin) && E_test_nfw <= Psimax + 1e-9*fabs(Psimax)) { // Looser check for spline domain
                            deriv_val_dIdE = gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, 
                                                                   innerintegrandvalues, E_test_nfw, g_main_fofEacc);
                        }
                        
                        // Add diagnostic for dI/dE values
                        if (g_doDebug && v_trials_nfw <= 2 && k_nfw < 5) { // Only for very first few trials of first few particles
                        }
                        
                        target_func_val_nfw = nfw_vel * nfw_vel * fabs(deriv_val_dIdE);
                        if (!isfinite(target_func_val_nfw) || target_func_val_nfw < 0) target_func_val_nfw = 0.0; // Ensure non-negative
                        
                        nfw_ratio = target_func_val_nfw / nfw_maxvalue; // maxvalue should be >0 here
                        if (nfw_ratio < 0) nfw_ratio = 0;
                        if (nfw_ratio > 1.001) { // If ratio is slightly > 1 due to numerics
                            nfw_ratio = 1.0;
                        }
                        
                        if ((k_nfw < 2 && v_trials_nfw < 5) || (v_trials_nfw % 5000 == 0 && v_trials_nfw > 0) ) {
                        }
                        
                        // Enhanced high trial count diagnostics
                        if (g_doDebug && (v_trials_nfw % 4000 == 0 && v_trials_nfw > 0)) {
                        }
                        
                        // Diagnostic for zero dI/dE in valid energy range
                        if (g_doDebug && fabs(deriv_val_dIdE) < 1e-20 && (E_test_nfw > Psimin + 1e-6*fabs(Psimin) && E_test_nfw < Psimax - 1e-6*fabs(Psimax)) && (v_trials_nfw % 100 == 0) && v_trials_nfw > 0 && k_nfw < 100) {
                        }
                        
                        if (gsl_rng_uniform(g_rng) < nfw_ratio) {
                            particles[1][k_nfw] = nfw_vel;
                            vflag_nfw = 1;
                        }
                    }
                    if (!vflag_nfw) {
                        particles[1][k_nfw] = 0.0; // Failed to find velocity
                    }
                        } // End else (maxvalue_envelope is finite and positive)
                    } // End else (maxv is finite and positive)
                } // End else (Psir > Psimin)
                
                // Sample angular momentum direction
                nfw_mu = 2.0 * gsl_rng_uniform(g_rng) - 1.0;
                // Ensure L is non-negative and well-defined even if particles[1][k_nfw] (velocity magnitude) is 0
                double L_val_nfw = 0.0;
                if (particles[1][k_nfw] > 1e-9) { // If velocity is non-zero
                    L_val_nfw = particles[1][k_nfw] * particles[0][k_nfw] * sqrt(fmax(0.0, 1.0 - nfw_mu * nfw_mu));
                }
                particles[2][k_nfw] = L_val_nfw;
                particles[3][k_nfw] = (double)k_nfw; // Particle ID
                particles[4][k_nfw] = nfw_mu;        // Orientation
                
                if (k_nfw < 5 || k_nfw % (npts_initial / 10 < 1 ? 1 : npts_initial/10) == 0) {
                }
            }
            
            // Clean up NFW sample generator allocations
            gsl_spline_free(splinerofM_nfw);
            gsl_interp_accel_free(rofMaccel_nfw);
            gsl_spline_free(splinemaxv2f_nfw);
            gsl_interp_accel_free(maxv2faccel_nfw);
            free(maxv2f_nfw);
            free(radius_maxv2f_nfw);
            
            log_message("INFO", "NFW IC Gen: Successfully generated %d particles.", npts_initial);
        } // End NFW sample generator


    } else { // Default: Use Cored Plummer-like Profile (Original Pathway)
        log_message("INFO", "Starting IC generation using Cored Plummer-like profile pathway.");
        log_message("INFO", "Generating Initial Conditions using Cored Plummer-like profile (original method)...");


        // ORIGINAL CORED PLUMMER-LIKE IC GENERATION PATHWAY
        // This is the entire block from nsphere.c.main.may20_1554.txt starting with
        // its "Theoretical Calculation Loop (Diagnostic)" down to the end of its
        // "SAMPLE GENERATOR" block.
        // It uses RC, HALO_MASS macros, its original massintegrand/drhodr,
        // and its original GSL settings.

        /** @brief Arrays defining integration and spline point counts for theoretical calculations. */
        int integration_points_array[2] = {1000, 10000};
        int spline_points_array[2] = {1000, 10000};

    /**
     * @brief Theoretical Calculation Loop (Eddington's Formula - Multiple Params).
     * @details Calculates theoretical profiles (mass, potential, f(E), density) based on the
     *          assumed initial density profile using Eddington's formula and GSL integration/splines.
     *          This loop iterates through different numbers of integration points (`Nintegration`)
     *          and spline points (`Nspline`) from the arrays above to generate reference files
     *          (e.g., massprofile_NiX_NsY.dat). These files are generated for comparison/validation
     *          but are *not* directly used in the main simulation timestepping or IC generation.
     *          They demonstrate the calculation process with varying numerical precision settings.
     */
    {
        for (int ii_ip = 0; ii_ip < 2; ii_ip++) // Loop over Nintegration values
        {
            for (int ii_sp = 0; ii_sp < 2; ii_sp++) // Loop over Nspline values
            {
                int Nintegration = integration_points_array[ii_ip];
                int Nspline = spline_points_array[ii_sp];

                double result, error, r;
                double calE;
                gsl_integration_workspace *w = gsl_integration_workspace_alloc(Nintegration);
                int i;

                gsl_function F;
                F.function = &massintegrand;
                F.params = NULL;

                double rmax = g_cored_profile_rmax_factor * g_cored_profile_rc;
                /** @note Calculate mass normalization factor for these params. */
                gsl_integration_qag(&F, 0.0, rmax, 0, 1.0e-12, Nintegration, 5, w, &result, &error);
                normalization = result;

                /** @note Calculate M(r) and create mass spline for these params. */
                int num_points = Nspline * 10; // Use more points for spline data generation than for integration
                double *mass = (double *)malloc(num_points * sizeof(double));
                double *radius = (double *)malloc(num_points * sizeof(double));

                for (int i = 0; i < num_points; i++)
                {
                    double r = (double)i * rmax / (num_points);
                    gsl_integration_qag(&F, 0.0, r, 0, 1.0e-12, Nintegration, 5, w, &result, &error);
                    mass[i] = result * g_cored_profile_halo_mass / normalization;
                    radius[i] = r;
                }

                gsl_interp_accel *enclosedmass = gsl_interp_accel_alloc();
                gsl_spline *splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinemass, radius, mass, num_points);
                double rlow = radius[0];
                double rhigh = radius[num_points - 1];

                /** @note Write mass profile file for these params (e.g., data/massprofile_Ni1k_Ns1k.dat). */
                char fname[256];
                FILE *fp;
                char base_filename_massprofile[256];
                snprintf(base_filename_massprofile, sizeof(base_filename_massprofile), "data/massprofile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_massprofile, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (r = 0.0; r < rhigh; r += rmax / 900.0)
                {
                    if (r < rlow || r > rhigh)
                    {
                        printf("r out of range\n");
                        CLEAN_EXIT(1);
                    }
                    fprintf_bin(fp, "%f %f\n", r, gsl_spline_eval(splinemass, r, enclosedmass));
                }
                fclose(fp);

                /** @note Calculate Psi(r) and create potential spline for these params. */
                double *Psivalues = (double *)malloc(num_points * sizeof(double));
                double *nPsivalues = (double *)malloc(num_points * sizeof(double)); // For inverse spline r(Psi)
                // Prepare Psiintegrand parameters for diagnostic loop
                Psiintegrand_params psi_params_diag;
                psi_params_diag.massintegrand_func = &massintegrand;
                psi_params_diag.params_for_massintegrand = NULL;
                
                gsl_function F_for_psi_diag;
                F_for_psi_diag.function = &Psiintegrand;
                F_for_psi_diag.params = &psi_params_diag;
                
                for (i = 0; i < num_points; i++)
                {
                    double r = (double)i * rmax / ((double)num_points);
                    double r1 = fmax(r, g_cored_profile_rc / 1000000.0);
                    gsl_integration_qagiu(&F_for_psi_diag, r1, 0, 1e-12, Nintegration, w, &result, &error);
                    double M_at_r1 = gsl_spline_eval(splinemass, r1, enclosedmass);
                    double first_term = G_CONST * M_at_r1 / r1;
                    double second_term = G_CONST * result * g_cored_profile_halo_mass / normalization;
                    Psivalues[i] = (first_term + second_term);
                    nPsivalues[i] = -Psivalues[i];
                }


                gsl_interp_accel *Psiinterp = gsl_interp_accel_alloc();
                gsl_spline *splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinePsi, radius, Psivalues, num_points);

                /** @note Write potential profile file for these params (e.g., data/Psiprofile_Ni1k_Ns1k.dat). */
                char base_filename_psiprofile[256];
                snprintf(base_filename_psiprofile, sizeof(base_filename_psiprofile), "data/Psiprofile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_psiprofile, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (r = 0.0; r < ((double)num_points - 1.0) / ((double)num_points) * rmax; r += rmax / 900.0)
                {
                    if (r < rlow || r > rhigh)
                    {
                        printf("r out of range\n");
                        CLEAN_EXIT(1);
                    }
                    fprintf_bin(fp, "%f %f\n", r, evaluatespline(splinePsi, Psiinterp, r));
                }
                fclose(fp);

                /** @note Create inverse spline r(Psi) for these params. */
                gsl_interp_accel *rofPsiinterp = gsl_interp_accel_alloc();
                gsl_spline *splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinerofPsi, nPsivalues, radius, num_points);

                /** @note Calculate inner integral for f(E) and create f(E) spline for these params. */
                double *innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
                double *Evalues = (double *)malloc((num_points + 1) * sizeof(double));
                double Psimin = Psivalues[num_points - 1];
                double Psimax = Psivalues[0];

                char base_filename_integrand[256];
                snprintf(base_filename_integrand, sizeof(base_filename_integrand), "data/integrand_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_integrand, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                calE = Psivalues[0];
                F.function = &fEintegrand;
                fEintegrand_params params = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass};
                F.params = &params;
                for (i = 0; i < num_points; i++)
                {
                    double t = sqrt(calE - Psimin) * ((double)i) / ((double)num_points);
                    fprintf_bin(fp, "%f %f\n", t, fEintegrand(t, &params));
                }
                fclose(fp);

                if (Psimax <= Psimin) { // Check for diagnostic loop
                    if (g_doDebug) log_message("DEBUG", "Diagnostic: Psimax (%.6e) <= Psimin (%.6e) in diagnostic loop", Psimax, Psimin);
                    // Continue with diagnostic but note the issue
                }

                innerintegrandvalues[0] = 0.0;
                Evalues[0] = Psimin;
                for (i = 1; i <= num_points; i++)
                {
                    calE = Psimin + (Psimax - Psimin) * ((double)i) / ((double)num_points);
                    if (i > 0 && calE <= Evalues[i-1]) { // Adjust if not strictly increasing
                        double prev_E_diag = Evalues[i-1];
                        double ideal_step_diag = (Psimax - Psimin) / (double)num_points;
                        double incr_diag = ideal_step_diag * 1e-6;
                        if(incr_diag == 0.0) incr_diag = DBL_MIN * fabs(prev_E_diag) + DBL_MIN;
                        if(incr_diag == 0.0) incr_diag = 1e-20;
                        calE = prev_E_diag + incr_diag;
                        if (g_doDebug && (i <= 3 || i > num_points - 3)) {
                            log_message("DEBUG", "Evalues[%d] adjusted to %.6e", i, calE);
                        }
                    }
                    Evalues[i] = calE;

                    fEintegrand_params params2 = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass};
                    F.params = &params2;
                    // Ensure sqrt argument is non-negative
                    double sqrt_arg_diag = calE - Psimin;
                    if (sqrt_arg_diag < 0) sqrt_arg_diag = 0.0;
                    gsl_integration_qag(&F, sqrt(sqrt_arg_diag) / 1.0e4, sqrt(sqrt_arg_diag), 1.0e-12, 1.0e-12, Nintegration, 6, w, &result, &error);
                    innerintegrandvalues[i] = result;
                }

                gsl_interp *fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
                gsl_interp_init(fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
                gsl_interp_accel *fofEacc = gsl_interp_accel_alloc();

                /** @note Write theoretical density profile file for these params (e.g., data/density_profile_NiX_NsY.dat). */
                char base_filename[256];
                snprintf(base_filename, sizeof(base_filename), "data/density_profile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i < num_points; i++)
                {
                    double rr = radius[i];
                    double rho_r = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(rr / g_cored_profile_rc)));
                    fprintf_bin(fp, "%f %f\n", rr, rho_r);
                }
                fclose(fp);

                /** @note Write dPsi/dr file (data/dpsi_dr<suffix>.dat) (overwrites previous if suffix same). */
                get_suffixed_filename("data/dpsi_dr.dat", 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i < num_points; i++)
                {
                    double rr = radius[i];
                    if (rr > 0.0)
                    {
                        double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                        double dpsidr = -(G_CONST * Menc) / (rr * rr);
                        fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                    }
                }
                fclose(fp);

                /** @note Write drho/dPsi file (data/drho_dpsi<suffix>.dat) (overwrites previous if suffix same). */
                get_suffixed_filename("data/drho_dpsi.dat", 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 1; i < num_points - 1; i++)
                {
                    double rr = radius[i];
                    double rho_left = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i - 1] / g_cored_profile_rc)));
                    double rho_right = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i + 1] / g_cored_profile_rc)));
                    double drho_dr_num = (rho_right - rho_left) / (radius[i + 1] - radius[i - 1]);
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                    double dPsidr = -(G_CONST * Menc) / (rr * rr);

                    if (dPsidr != 0.0)
                    {
                        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                        fprintf_bin(fp, "%f %f\n", Psi_val, drho_dr_num / dPsidr);
                    }
                }
                fclose(fp);

                /** @note Write f(E) = dI/dE / const file for these params (e.g., data/f_of_E_NiX_NsY.dat). */
                char base_filename_fofe[256];
                snprintf(base_filename_fofe, sizeof(base_filename_fofe), "data/f_of_E_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename_fofe, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i <= num_points; i++)
                {
                    double E = Evalues[i];
                    double deriv = 0.0;
                    if (i > 0 && i < num_points + 1)
                    {
                        if (i > 0 && i < num_points)
                        {
                            deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i - 1]) / (Evalues[i + 1] - Evalues[i - 1]);
                        }
                        else if (i == 0)
                        {
                            deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i]) / (Evalues[i + 1] - Evalues[i]);
                        }
                        else if (i == num_points)
                        {
                            deriv = (innerintegrandvalues[i] - innerintegrandvalues[i - 1]) / (Evalues[i] - Evalues[i - 1]);
                        }
                    }
                    double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                    if (E == 0.0 || !isfinite(fE))
                        fE = 0.0;
                    fprintf_bin(fp, "%f %f\n", E, fE);
                }
                fclose(fp);

                get_suffixed_filename("data/df_fixed_radius.dat", 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                {
                    double r_fixed = 200.0;
                    double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_fixed);
                    Psi_rf *= VEL_CONV_SQ;
                    int vsteps = 100;
                    for (int vv = 0; vv <= vsteps; vv++)
                    {
                        double vtest = (double)vv * (sqrt(2.0 * Psi_rf) / (vsteps));
                        double Etest = Psi_rf - 0.5 * vtest * vtest;
                        double fEval = 0.0;
                        if (Etest > Psimin && Etest < Psimax)
                        {
                            double dval = gsl_interp_eval_deriv(fofEinterp, Evalues, innerintegrandvalues, Etest, fofEacc);
                            fEval = dval / (sqrt(8.0) * PI * PI) * vtest * vtest * r_fixed * r_fixed;
                        }
                        fprintf_bin(fp, "%f %f\n", vtest, fEval);
                    }
                }
                fclose(fp);

                gsl_spline_free(splinePsi);
                gsl_spline_free(splinerofPsi);
                gsl_interp_accel_free(Psiinterp);
                gsl_interp_accel_free(rofPsiinterp);
                gsl_spline_free(splinemass);
                gsl_interp_accel_free(enclosedmass);
                gsl_interp_free(fofEinterp);
                gsl_interp_accel_free(fofEacc);
                free(mass);
                free(radius);
                if (radius_monotonic_grid_nfw != NULL) {
                    free(radius_monotonic_grid_nfw);
                    radius_monotonic_grid_nfw = NULL;
                }
                free(Psivalues);
                free(nPsivalues);
                free(innerintegrandvalues);
                free(Evalues);
                gsl_integration_workspace_free(w);
                w = NULL;
            }
        }
    }

    /**
     * @brief Main Theoretical Calculation (using default Nintegration=1000, Nspline=10000).
     * @details Repeats the theoretical profile calculations using fixed, default parameters
     *          (Nintegration=1000, Nspline=10000). The results from *this* block (splines:
     *          `splinemass`, `splinePsi`, `splinerofPsi`, `fofEinterp` and accelerators)
     *          are the ones used for generating the initial particle distribution and potentially
     *          for comparison during the simulation (e.g., debug energy calculation).
     *          Generates primary output files like `massprofile<suffix>.dat`, `Psiprofile<suffix>.dat`, `f_of_E<suffix>.dat`.
     */
    double r;
    
    /** @brief Allocate workspace for GSL integration operations. */
    w = gsl_integration_workspace_alloc(1000);

    gsl_function F;
    F.function = &massintegrand;
    F.params = NULL;

    rmax = g_cored_profile_rmax_factor * g_cored_profile_rc;
    gsl_integration_qag(&F, 0.0, rmax, 0, 1.0e-12, 1000, 5, w, &result, &error);
    normalization = result;

    num_points = 10000;
    /** @brief Allocate arrays for mass profile calculation. */
    mass = (double *)malloc(num_points * sizeof(double));
    radius = (double *)malloc(num_points * sizeof(double));

    for (int i = 0; i < num_points; i++)
    {
        double r = (double)i * rmax / (num_points);
        gsl_integration_qag(&F, 0.0, r, 0, 1.0e-12, 1000, 5, w, &result, &error);
        mass[i] = result * g_cored_profile_halo_mass / normalization;
        radius[i] = r;
    }

    /** @brief Create mass interpolation spline for M(r). */
    enclosedmass = gsl_interp_accel_alloc();
    splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
    if (!check_strict_monotonicity(radius, num_points, "radius (main splinemass)")) { 
        fprintf(stderr, "CRITICAL: radius array not monotonic for main splinemass\n");
        fflush(stderr);
        CLEAN_EXIT(1); 
    }
    gsl_spline_init(splinemass, radius, mass, num_points);
    double rlow = radius[0];
    double rhigh = radius[num_points - 1];

    /** @brief Create 'data' directory again (harmless if exists). */
    {
        struct stat st = {0};
        if (stat("data", &st) == -1)
        {
            mkdir("data", 0755);
        }
    }

    /** @brief Write main mass profile file (data/massprofile<suffix>.dat). */
    get_suffixed_filename("data/massprofile.dat", 1, fname, sizeof(fname));
    fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
    for (r = 0.0; r < rhigh; r += rmax / 900.0)
    {
        if (r < rlow || r > rhigh)
        {
            printf("r out of range\n");
            CLEAN_EXIT(1);
        }
        fprintf_bin(fp, "%f %f\n", r, gsl_spline_eval(splinemass, r, enclosedmass));
    }
    fclose(fp);

    /** @brief Calculate Psi(r) array (num_points=10000) and create primary `splinePsi`. */
    Psivalues = (double *)malloc(num_points * sizeof(double));
    nPsivalues = (double *)malloc(num_points * sizeof(double));
    // Prepare Psiintegrand parameters for Cored profile
    Psiintegrand_params psi_params_cored;
    psi_params_cored.massintegrand_func = &massintegrand;
    psi_params_cored.params_for_massintegrand = NULL;
    
    gsl_function F_for_psi_cored;
    F_for_psi_cored.function = &Psiintegrand;
    F_for_psi_cored.params = &psi_params_cored;
    
    for (i = 0; i < num_points; i++)
    {
        double r = (double)i * rmax / ((double)num_points);
        double r1 = fmax(r, g_cored_profile_rc / 1000000.0);
        gsl_integration_qagiu(&F_for_psi_cored, r1, 0, 1e-12, 1000, w, &result, &error);
        double first_term = G_CONST * gsl_spline_eval(splinemass, r1, enclosedmass) / r1;
        double second_term = G_CONST * result * g_cored_profile_halo_mass / normalization;
        Psivalues[i] = (first_term + second_term);
        nPsivalues[i] = -Psivalues[i];
    }

    /** @brief Create potential interpolation spline for Psi(r). */
    Psiinterp = gsl_interp_accel_alloc();
    splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
    if (!check_strict_monotonicity(radius, num_points, "radius (main splinePsi)")) { 
        fprintf(stderr, "CRITICAL: radius array not monotonic for main splinePsi\n");
        fflush(stderr);
        CLEAN_EXIT(1); 
    }
    gsl_spline_init(splinePsi, radius, Psivalues, num_points);


    /** @brief Write main potential profile file (data/Psiprofile<suffix>.dat). */
    get_suffixed_filename("data/Psiprofile.dat", 1, fname, sizeof(fname));
    fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
    for (r = 0.0; r < ((double)num_points - 1.0) / ((double)num_points) * rmax; r += rmax / 900.0)
    {
        if (r < rlow || r > rhigh)
        {
            printf("r out of range\n");
            CLEAN_EXIT(1);
        }
        fprintf_bin(fp, "%f %f\n", r, evaluatespline(splinePsi, Psiinterp, r), evaluatespline(splinemass, enclosedmass, r));
    }
    fclose(fp);

    /** @brief Create inverse spline r(Psi) for radius lookup from potential. */
    rofPsiinterp = gsl_interp_accel_alloc();
    splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
    
    gsl_spline_init(splinerofPsi, nPsivalues, radius, num_points);

    /** @brief Calculate distribution function f(E) using Eddington's formula. */
    innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
    Evalues = (double *)malloc((num_points + 1) * sizeof(double));
    Psimin = Psivalues[num_points - 1];
    Psimax = Psivalues[0];
    get_suffixed_filename("data/integrand.dat", 1, fname, sizeof(fname));
    fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
    calE = Psivalues[0];
    F.function = &fEintegrand;
    fEintegrand_params params = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass};
    F.params = &params;
    for (i = 0; i < num_points; i++)
    {
        double t = sqrt(calE - Psimin) * ((double)i) / ((double)num_points);
        fprintf_bin(fp, "%f %f\n", t, fEintegrand(t, &params));
    }
    fclose(fp);


    innerintegrandvalues[0] = 0.0;
    Evalues[0] = Psimin;

    for (i = 1; i <= num_points; i++)
    {
        calE = Psimin + (Psimax - Psimin) * ((double)i) / ((double)num_points);
        fEintegrand_params params2 = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass};
        F.params = &params2;
        gsl_integration_qag(&F, sqrt(calE - Psimin) / 1.0e4, sqrt(calE - Psimin), 1.0e-12, 1.0e-12, 1000, 6, w, &result, &error);
        innerintegrandvalues[i] = result;
        Evalues[i] = calE;
    }

    /** @brief Create f(E) interpolation for particle generation. */

    g_main_fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
    gsl_interp_init(g_main_fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
    g_main_fofEacc = gsl_interp_accel_alloc();

    /**
     * @brief Initialize particle arrays with default values (0.0, ID=index).
     * @details Uses the already allocated `particles` array from outer scope.
     *          Component indices: 0=radius, 1=velocity magnitude (initially), 2=ang. mom.,
     *          3=ID (initial index 0..npts_initial-1), 4=orientation(mu).
     */
    for (i = 0; i < npts_initial; i++)
    {
        particles[0][i] = 0.0;
        particles[1][i] = 0.0;
        particles[2][i] = 0.0;
        particles[3][i] = (double)i; // Initial ID is the index
        particles[4][i] = 0.0;
    }

    /**
     * Seed determination logic:
     * 1. If a specific seed (`--initial-cond-seed` or `--sidm-seed`) is provided, use it.
     * 2. Else if `--master-seed` is provided, derive specific seeds from it.
     * 3. Else if `--load-seeds` is specified (or by default if files exist and seeds not given), try to load from last_X_seed_{suffix}.dat.
     * 4. Else (no seeds provided, no load requested/possible), generate new seeds from time/pid.
     * Finally, save the seeds actually used to last_X_seed_{suffix}.dat and link last_X_seed.dat.
     */

    unsigned long int current_time_pid_seed = (unsigned long int)time(NULL) ^ (unsigned long int)getpid();
    char seed_filepath[512];
    FILE *fp_seed;

    // Determine Initial Conditions Seed
    if (!g_initial_cond_seed_provided) {
        if (g_master_seed_provided) {
            g_initial_cond_seed = g_master_seed + 1; // Deterministic offset
        } else if (g_attempt_load_seeds) {
            get_suffixed_filename(g_initial_cond_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
            fp_seed = fopen(seed_filepath, "r");
            if (fp_seed) {
                if (fscanf(fp_seed, "%lu", &g_initial_cond_seed) == 1) {
                    log_message("INFO", "Loaded initial conditions seed %lu from %s", g_initial_cond_seed, seed_filepath);
                } else {
                    g_initial_cond_seed = current_time_pid_seed + 100; // Fallback if read fails
                    log_message("WARNING", "Failed to read IC seed from %s, generating new: %lu", seed_filepath, g_initial_cond_seed);
                }
                fclose(fp_seed);
            } else {
                g_initial_cond_seed = current_time_pid_seed + 100; // File not found, generate
                log_message("INFO", "No IC seed file found, generating new: %lu", g_initial_cond_seed);
            }
        } else {
            g_initial_cond_seed = current_time_pid_seed + 100; // Default generation
            log_message("INFO", "Generating new IC seed: %lu", g_initial_cond_seed);
        }
    } else {
        log_message("INFO", "Using user-provided IC seed: %lu", g_initial_cond_seed);
    }

    // Determine SIDM Seed
    if (!g_sidm_seed_provided) {
        if (g_master_seed_provided) {
            g_sidm_seed = g_master_seed + 2; // Deterministic offset, different from IC seed
        } else if (g_attempt_load_seeds) {
            get_suffixed_filename(g_sidm_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
            fp_seed = fopen(seed_filepath, "r");
            if (fp_seed) {
                if (fscanf(fp_seed, "%lu", &g_sidm_seed) == 1) {
                    log_message("INFO", "Loaded SIDM seed %lu from %s", g_sidm_seed, seed_filepath);
                } else {
                    g_sidm_seed = current_time_pid_seed + 200; // Fallback
                    log_message("WARNING", "Failed to read SIDM seed from %s, generating new: %lu", seed_filepath, g_sidm_seed);
                }
                fclose(fp_seed);
            } else {
                g_sidm_seed = current_time_pid_seed + 200; // File not found, generate
                log_message("INFO", "No SIDM seed file found, generating new: %lu", g_sidm_seed);
            }
        } else {
            g_sidm_seed = current_time_pid_seed + 200; // Default generation
            log_message("INFO", "Generating new SIDM seed: %lu", g_sidm_seed);
        }
    } else {
        log_message("INFO", "Using user-provided SIDM seed: %lu", g_sidm_seed);
    }
    
    // If --readinit is used, we should generally use a specified/loaded SIDM seed
    // or a newly generated one, NOT one derived from IC seed, as ICs are fixed.
    if (doReadInit && !g_sidm_seed_provided && !g_master_seed_provided && !g_attempt_load_seeds) {
        // If reading ICs and no SIDM/master seed is given and not told to load, ensure SIDM seed is fresh.
        // This case might have already generated g_sidm_seed from current_time_pid_seed, which is fine.
        // If it was derived from g_initial_cond_seed (which itself might have been from time), it's also fine.
        // The logic above should correctly make g_sidm_seed independent of g_initial_cond_seed
        // if g_master_seed_provided is false.
    }

    // Save the seeds that will actually be used
    // Save Initial Conditions Seed
    get_suffixed_filename(g_initial_cond_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
    fp_seed = fopen(seed_filepath, "w");
    if (fp_seed) {
        fprintf(fp_seed, "%lu\n", g_initial_cond_seed);
        fclose(fp_seed);
        log_message("INFO", "Saved initial conditions seed %lu to %s", g_initial_cond_seed, seed_filepath);
        // Create link/copy
        char linkname_ic[512];
        snprintf(linkname_ic, sizeof(linkname_ic), "%s.dat", g_initial_cond_seed_filename_base); // e.g. data/last_initial_seed.dat
        
        // Platform-dependent link/copy code (similar to lastparams.dat)
        #if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
            // Windows: use copy
            char copy_cmd[1024];
            sprintf(copy_cmd, "copy \"%s\" \"%s\"", seed_filepath, linkname_ic);
            if (system(copy_cmd) != 0) {
                log_message("WARNING", "Failed to copy IC seed file from %s to %s", seed_filepath, linkname_ic);
            }
        #else
            // Unix/Linux/macOS: use symbolic link
            unlink(linkname_ic); // Remove existing link if present
            // Extract basename for relative symlink within data directory
            const char *basename_ic = strrchr(seed_filepath, '/');
            basename_ic = basename_ic ? basename_ic + 1 : seed_filepath; // Skip the '/' or use full name if no '/'
            if (symlink(basename_ic, linkname_ic) != 0) {
                log_message("WARNING", "Failed to create symbolic link from %s to %s", basename_ic, linkname_ic);
            }
        #endif
    } else {
        log_message("ERROR", "Failed to save IC seed to %s", seed_filepath);
    }

    // Save SIDM Seed
    get_suffixed_filename(g_sidm_seed_filename_base, 1, seed_filepath, sizeof(seed_filepath));
    fp_seed = fopen(seed_filepath, "w");
    if (fp_seed) {
        fprintf(fp_seed, "%lu\n", g_sidm_seed);
        fclose(fp_seed);
        log_message("INFO", "Saved SIDM seed %lu to %s", g_sidm_seed, seed_filepath);
        // Create link/copy
        char linkname_sidm[512];
        snprintf(linkname_sidm, sizeof(linkname_sidm), "%s.dat", g_sidm_seed_filename_base); // e.g. data/last_sidm_seed.dat
        
        // Platform-dependent link/copy code
        #if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
            // Windows: use copy
            char copy_cmd[1024];
            sprintf(copy_cmd, "copy \"%s\" \"%s\"", seed_filepath, linkname_sidm);
            if (system(copy_cmd) != 0) {
                log_message("WARNING", "Failed to copy SIDM seed file from %s to %s", seed_filepath, linkname_sidm);
            }
        #else
            // Unix/Linux/macOS: use symbolic link
            unlink(linkname_sidm); // Remove existing link if present
            // Extract basename for relative symlink within data directory
            const char *basename_sidm = strrchr(seed_filepath, '/');
            basename_sidm = basename_sidm ? basename_sidm + 1 : seed_filepath; // Skip the '/' or use full name if no '/'
            if (symlink(basename_sidm, linkname_sidm) != 0) {
                log_message("WARNING", "Failed to create symbolic link from %s to %s", basename_sidm, linkname_sidm);
            }
        #endif
    } else {
        log_message("ERROR", "Failed to save SIDM seed to %s", seed_filepath);
    }

    /**
     * @brief INITIAL CONDITION HANDLING block.
     * @details Determines whether to generate new initial conditions or load existing ones,
     *          or skip if restarting.
     *          - If `doReadInit` is true: Loads from `readInitFilename` via `read_initial_conditions`.
     *          - If `g_doRestart` is true: Skips generation (assumes simulation data exists or will be checked).
     *          - Otherwise: Generates new particles using the Sample Generator.
     */

    if (doReadInit)
    {
        /** @brief Load existing initial conditions from specified file. */
        printf("Reading initial conditions from %s...\n", readInitFilename);
        read_initial_conditions(particles, npts_initial, readInitFilename);
    }
    else if (!g_doRestart) // Only generate if NOT reading init file and NOT restarting
    {
        /**
         * @brief SAMPLE GENERATOR block.
         * @details Generates the initial particle distribution (radius, velocity magnitude, orientation)
         *          for `npts_initial` particles, based on the theoretical equilibrium distribution
         *          function `f(E)` derived via Eddington's formula (using primary splines).
         *          Uses inverse transform sampling for radius (via M(r) spline) and rejection
         *          sampling for velocity magnitude (using `f(E)` derivative spline).
         */
        if (tidal_fraction > 0.0) log_message("INFO", "Cored IC Gen: Initial particle count before stripping: %d", npts_initial);
        log_message("INFO", "Cored IC Gen: Generating %d initial particle positions and velocities...", npts_initial);

        double vel, ratio, Psir, mu, maxv, maxvalue;

        /** @brief Set up GSL spline for radius as a function of enclosed mass: r(M).
         *         Used for inverse transform sampling of radius. */
        gsl_interp_accel *rofMaccel = gsl_interp_accel_alloc();
        gsl_spline *splinerofM = gsl_spline_alloc(gsl_interp_cspline, num_points);
        // Add checks for allocation failure
        if (!rofMaccel || !splinerofM) { /* Handle error */ CLEAN_EXIT(1); }
        gsl_spline_init(splinerofM, mass, radius, num_points);

        /** @brief Set up GSL spline for the maximum of `v^2 * f(E)` envelope at each radius `r`.
         *         Used for rejection sampling efficiency. f(E) proportional to dI/dE. */
        double *maxv2f = (double *)malloc(num_points * sizeof(double));
        if (!maxv2f) { /* Handle error */ CLEAN_EXIT(1); }

        for (int i_r = 1; i_r < num_points; i_r++)
        {
            maxvalue = 0.0;
            Psir = evaluatespline(splinePsi, Psiinterp, radius[i_r]);
            maxv = sqrt(2.0 * (Psir - Psimin));
            for (int j_v = 1; j_v < num_points - 2; j_v++)
            {
                vel = maxv * ((double)j_v) / ((double)num_points);
                double currentvalue = vel * vel * gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, innerintegrandvalues, Psir - (0.5) * vel * vel, g_main_fofEacc);
                if (currentvalue > maxvalue) maxvalue = currentvalue;
            }
            maxv2f[i_r] = maxvalue;
        }
        // Extrapolate for r=0 assuming linear behavior near origin based on points 1 and 2
        if (num_points >= 3) {
            maxv2f[0] = 2.0 * maxv2f[1] - maxv2f[2];
            if (maxv2f[0] < 0.0) maxv2f[0] = 0.0; // Ensure non-negative
        } else if (num_points == 2) {
             maxv2f[0] = maxv2f[1]; // Simple fallback
        } else {
             maxv2f[0] = 0.0; // Fallback for very few points
        }

        gsl_interp_accel *maxv2faccel = gsl_interp_accel_alloc();
        gsl_spline *splinemaxv2f = gsl_spline_alloc(gsl_interp_cspline, num_points);
        // Add checks for allocation failure
        if (!maxv2faccel || !splinemaxv2f) { /* Handle error */ free(maxv2f); CLEAN_EXIT(1); }
        gsl_spline_init(splinemaxv2f, radius, maxv2f, num_points);

        /**
         * @brief Generate `npts_initial` particle samples using sampling methods.
         * @note Uses GSL random number generation. For thread-safety in parallel code,
         *       thread-specific RNG states should be used instead of the global state.
         */
        for (i = 0; i < npts_initial; i++) // Loop over particles to generate
        {
            /** @note 1. Choose radius `r` using inverse transform sampling on M(r). */
            double mass_frac_sample = gsl_rng_uniform(g_rng) * 0.999999; // Avoid sampling exactly M_total
            double mass_sample = mass_frac_sample * g_cored_profile_halo_mass;
            particles[0][i] = evaluatespline(splinerofM, rofMaccel, mass_sample);

            /** @note 2. Find velocity magnitude `v` using rejection sampling against `max(v^2*f(E))`. */
            maxvalue = evaluatespline(splinemaxv2f, maxv2faccel, particles[0][i]);
            Psir = evaluatespline(splinePsi, Psiinterp, particles[0][i]);
            maxv = sqrt(2.0 * (Psir - Psimin));
            int vflag = 0;
            while (vflag == 0)
            {
                vel = gsl_rng_uniform(g_rng) * maxv;
                // Evaluate target function (proportional to v^2 * f(E))
                double target_func_val = vel * vel * gsl_interp_eval_deriv(g_main_fofEinterp, Evalues, innerintegrandvalues, Psir - (0.5) * vel * vel, g_main_fofEacc);
                ratio = (maxvalue > 1e-15) ? (target_func_val / maxvalue) : 0.0; // Avoid division by zero
                if (gsl_rng_uniform(g_rng) < ratio)
                {
                    particles[1][i] = vel;
                    vflag = 1; // Accept
                }
            } // End rejection loop

            /** @note 3. Generate random velocity orientation `mu = cos(theta)`. */
            mu = (2.0 * gsl_rng_uniform(g_rng) - 1.0); // Uniform distribution in [-1, 1] for isotropy

            /** @note 4. Calculate angular momentum `L = r * v_tangential = r * v * sqrt(1-mu^2)`. */
            particles[2][i] = particles[0][i] * particles[1][i] * sqrt(1.0 - mu * mu);

            /** @note 5. Store orientation parameter `mu` (needed later for v_radial). */
            particles[4][i] = mu;

            /** @note 6. Store initial index as particle ID. */
            particles[3][i] = (double)i;
        } // End particle generation loop

        /** @brief Cleanup Sample Generator resources (splines, accelerators, temp arrays). */
        gsl_spline_free(splinerofM);
        gsl_interp_accel_free(rofMaccel);
        gsl_spline_free(splinemaxv2f);
        gsl_interp_accel_free(maxv2faccel);
        free(maxv2f);

    } // End Sample Generator block (if !doReadInit && !g_doRestart)

    } // End if/else for profile selection for IC generation

    /**
     * @brief Save generated initial conditions to file if requested (`--writeinit`).
     * @details Saves the initial state if `doWriteInit` is true and not in restart mode.
     *          The state saved is *before* tidal stripping and unit conversion, using `write_initial_conditions`.
     * @see write_initial_conditions
     */
    if (doWriteInit && !g_doRestart)
    {
        printf("Saving initial conditions to %s...\n", writeInitFilename);
        write_initial_conditions(particles, npts_initial, writeInitFilename);
    }

    /**
     * @brief TIDAL STRIPPING IMPLEMENTATION block.
     * @details If `tidal_fraction` > 0 and not in restart mode, simulates tidal stripping by removing the outermost
     *          fraction of particles based on radius. It first sorts the `npts_initial`
     *          particles by radius, then keeps only the innermost `npts` particles.
     *          It reallocates the `particles` array to the final size `npts` and remaps
     *          the original indices stored in `particles[3]` to ranks [0, npts-1] using
     *          `reassign_orig_ids_with_rank`.
     * @see reassign_orig_ids_with_rank
     * @see sort_particles_with_alg
     */
    if (!g_doRestart) // Skip stripping if restarting
    {
        /** @note Only show stripping message if `--ftidal` was used. */
        if (tidal_fraction > 0.0) printf("Tidal stripping: sorting and retaining inner %.1f%% of particles...\n", (1.0 - tidal_fraction) * 100.0);

        /** @note Sort all `npts_initial` particles by radius using basic quadsort. */
        sort_particles_with_alg(particles, npts_initial, "quadsort"); // Sorts by particles[0]

        /** @note Allocate new smaller arrays (`final_particles`) for the `npts` particles to keep. */
        double **final_particles = (double **)malloc(5 * sizeof(double *));
        if (final_particles == NULL)
        {
            fprintf(stderr, "Memory allocation failed for final_particles\n");
            CLEAN_EXIT(1);
        }

        /** @brief Copy innermost `npts` particles to final arrays and replace `particles` pointers. */
        for (int i = 0; i < 5; i++) // Loop over components
        {
            final_particles[i] = (double *)malloc(npts * sizeof(double));
            if (final_particles[i] == NULL)
            {
                fprintf(stderr, "Memory allocation failed for final_particles[%d]\n", i);
                CLEAN_EXIT(1);
            }
            /** @note Copy only the first `npts` elements (innermost after sort). */
            memcpy(final_particles[i], particles[i], npts * sizeof(double));
            
            /** @note Free original oversized array and update `particles[i]` pointer. */
            free(particles[i]);                // Free the original oversized array
            particles[i] = final_particles[i]; // particles[i] now points to the smaller array
        }
        free(final_particles); // Free the temporary ** structure, not the data arrays

        /** @note Only show completion message if `--ftidal` was used. */
        if (tidal_fraction > 0.0)
        {
            printf("Tidal stripping complete: %d particles retained.\n\n", npts);
        }

        /**
         * @brief Remap original IDs (now in `particles[3]` for the kept particles) to ranks [0, npts-1].
         * @details Ensures `particles[3][i]` holds the final rank ID (0 to npts-1)
         *          for the particle currently at index `i` after stripping and sorting.
         */
        reassign_orig_ids_with_rank(particles[3], npts);
    } // End tidal stripping block (!g_doRestart)

    /**
     * @brief VELOCITY UNIT CONVERSION and ORIENTATION block.
     * @details Converts particle velocity magnitude (`particles[1]`) and angular momentum
     *          (`particles[2]`) from simulation generation units (implicitly km/s from `f(E)`)
     *          to physical units used in timestepping (kpc/Myr). It also applies the
     *          orientation parameter `mu = v_radial / v_total` (stored in `particles[4]`)
     *          to `particles[1]` to get the actual radial velocity component for integration.
     *          The original velocity magnitude in `particles[1]` is overwritten.
     *          Skipped in restart mode (`g_doRestart`).
     * @see kmsec_to_kpcmyr
     */
    if (!g_doRestart)
    {
        for (i = 0; i < npts; i++) // Loop over final npts particles
        {
            // particles[1] holds velocity magnitude 'v' from Sample Generator
            // particles[4] holds orientation 'mu' from Sample Generator
            particles[1][i] *= particles[4][i]; ///< Apply orientation: v_rad = v * mu
            particles[1][i] *= kmsec_to_kpcmyr; ///< Convert v_rad [km/s] to [kpc/Myr]
            // particles[2] holds angular momentum L = r*v*sqrt(1-mu^2)
            particles[2][i] *= kmsec_to_kpcmyr; ///< Convert L [kpc*km/s] to [kpc^2/Myr]
            // particles[4] (mu) is no longer needed after this step.
        }
    }

    /**
     * @brief PARTICLE DATA EXPORT block (Initial State for Simulation).
     * @details Writes the initial state of all `npts` particles (after potential stripping,
     *          unit conversion, orientation application, and ID remapping) to a file
     *          named `data/particles<suffix>.dat`. This file represents the state at t=0
     *          entering the simulation loop.
     *          Format (binary via fprintf_bin):
     *          radius(kpc, float) v_radial(kpc/Myr, float) ang_mom(kpc^2/Myr, float) final_rank_id(float? check fprintf_bin)
     *          Skipped if `skip_file_writes` is true (restart mode).
     * @see fprintf_bin
     */
    if (!skip_file_writes)
    {
        char filename[256];
        get_suffixed_filename("data/particles.dat", 1, filename, sizeof(filename));
        FILE *fpp = fopen(filename, "wb"); // Binary mode for fprintf_bin output
        if (fpp == NULL)
        {
            fprintf(stderr, "Error opening file %s for writing initial particles\n", filename);
            exit(1); // Use CLEAN_EXIT?
        }

        /** @brief Write each particle's state (using fprintf_bin). */
        for (int i = 0; i < npts; i++) // i is current index (0..npts-1)
        {
            fprintf_bin(fpp, "%f %f %f %f\n", // Format string likely ignored by fprintf_bin beyond types
                        particles[0][i], // Radius (kpc)
                        particles[1][i], // Radial velocity (kpc/Myr)
                        particles[2][i], // Angular momentum (kpc²/Myr)
                        particles[3][i]);// Particle ID (final rank, written as float)
        }
        fclose(fpp);
        printf("Wrote initial particle state to %s\n", filename);
    } // End skip_file_writes block for particles.dat

    /**
     * @brief SIMULATION TIMESTEP CALCULATION block.
     * @details Calculates the characteristic dynamical time (`tdyn`) based on core radius (`RC`)
     *          and total mass (`HALO_MASS`). Uses this to determine the total simulation
     *          duration (`totaltime = tfinal_factor * tdyn`) and the individual timestep
     *          size (`dt = totaltime / Ntimes`) used in the integration loop.
     * @see tdyn
     * @see totaltime
     * @see dt
     */
    double characteristic_radius_for_tdyn;
    if (g_use_nfw_profile) {
        characteristic_radius_for_tdyn = g_nfw_profile_rc; // Set from g_scale_radius_param
    } else {
        characteristic_radius_for_tdyn = g_cored_profile_rc; // Set from g_scale_radius_param
    }
    double tdyn = 1.0 / sqrt((VEL_CONV_SQ * G_CONST) * g_active_halo_mass / cube(characteristic_radius_for_tdyn));
    double totaltime = (double)tfinal_factor * tdyn; ///< Total simulation time (Myr)
    double dt = totaltime / ((double)Ntimes);        ///< Individual timestep size (Myr)
    printf("Dynamical time tdyn = %.4f Myr\n", tdyn);
    printf("Total simulation time = %.4f Myr (%.1f tdyn)\n", totaltime, (double)tfinal_factor);
    printf("Timestep dt = %.6f Myr\n\n", dt);

    /** @brief Initialize simulation time tracking and progress reporting. */
    double time = 0.0;                   ///< Current simulation time (Myr)
    double start_time = omp_get_wtime(); ///< Wall-clock start time for timing
    /** @brief Setup progress reporting steps (array `print_steps` holding step numbers for 0%, 5%, ..., 100%). */
    int print_steps[21];
    for (int k = 0; k <= 20; k++) print_steps[k] = (int)floor(k * 0.05 * Ntimes); // Calculate steps for progress output

    /** @brief Flag to determine if simulation phase can be skipped. */
    int skip_simulation = 0; 

    /** @brief Set up simulation tracking variables. */

    /**
     * @brief TRAJECTORY TRACKING SETUP block.
     * @details Allocates memory arrays (`trajectories`, `energies`, `velocities_arr`, etc.)
     *          for tracking the evolution of a small number (`num_traj_particles`, max 10)
     *          of selected particles over time. These particles are identified by their
     *          *final rank ID* (0 to `upper_npts_num_traj - 1`). Uses `ext_Ntimes` for array
     *          size to accommodate potential loop overruns or post-loop access.
     * @note The specific particles tracked are those with final rank IDs 0, 1, ..., 9 (or fewer if npts < 10).
     */
    /** @brief Determine number of particles to track (max 10 or npts). */
    int num_traj_particles = 10; ///< Max number of low-ID particles to track
    int upper_npts_num_traj = (num_traj_particles < npts) ? num_traj_particles : npts; ///< Actual number tracked

    /** @brief Allocate trajectory arrays [tracked_particle_index][time_step]. */
    // Index `p` corresponds to the particle with final_rank_id `p`.
    double **trajectories = (double **)malloc(upper_npts_num_traj * sizeof(double *)); // Radius (kpc)
    // Add checks for allocation failure
    for (i = 0; i < upper_npts_num_traj; i++) trajectories[i] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
    double **energies = (double **)malloc(upper_npts_num_traj * sizeof(double *)); // Relative Energy E_rel (per unit mass)
    // Add checks for allocation failure
    for (i = 0; i < upper_npts_num_traj; i++) energies[i] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure

    /** @brief Allocate arrays for additional tracked properties [tracked_particle_index][time_step]. */
    double **velocities_arr = (double **)malloc(upper_npts_num_traj * sizeof(double *)); // Radial velocity (kpc/Myr)
    // Add checks for allocation failure
    double **mu_arr = (double **)malloc(upper_npts_num_traj * sizeof(double *));         // Radial direction cosine (v_rad / v_tot)
    // Add checks for allocation failure
    // Note: E_arr seems redundant with 'energies' array tracking E_rel. Verify necessity.
    double **E_arr = (double **)malloc(upper_npts_num_traj * sizeof(double *));          // Total relative energy E_rel
    // Add checks for allocation failure
    double **L_arr = (double **)malloc(upper_npts_num_traj * sizeof(double *));          // Angular momentum (kpc^2/Myr)
    // Add checks for allocation failure

    // Allocate memory for each particle's timestep history
    for (i = 0; i < upper_npts_num_traj; i++)
    {
        velocities_arr[i] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
        mu_arr[i] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
        E_arr[i] = (double *)malloc(ext_Ntimes * sizeof(double)); // If kept, needs allocation check
        L_arr[i] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
    }

    /**
     * @brief ENERGY AND ANGULAR MOMENTUM INITIALIZATION block.
     * @details Calculates the initial relative energy `E_rel = Psi - KE` and angular
     *          momentum `L` for *all* particles based on their initial state (after
     *          stripping/conversion/remapping). Stores these initial values in `E_i_arr` and `L_i_arr`,
     *          indexed by the particle's final rank ID (`particles[3]`). Uses the
     *          theoretical potential spline `splinePsi` for the potential energy term.
     *          These arrays `E_i_arr`, `L_i_arr` store the *initial* values for later comparison.
     * @see E_i_arr
     * @see L_i_arr
     */
    /** @brief Allocate arrays for initial E and L, indexed by final rank ID. */
    double *E_i_arr = (double *)malloc(npts * sizeof(double)); // Stores initial E_rel[final_rank_id]
    // Add checks for allocation failure
    double *L_i_arr = (double *)malloc(npts * sizeof(double)); // Stores initial L[final_rank_id]
    // Add checks for allocation failure

    /** @brief Calculate initial E_rel and L for each particle and store by final rank ID. */
    for (i = 0; i < npts; i++) // Loop through particles 0..npts-1 (current index 'i')
    {
        double rr = particles[0][i];            // Radius (at current index i)
        double vrad = particles[1][i];          // Radial velocity (at current index i)
        double ell = particles[2][i];           // Angular momentum (at current index i)
        int final_rank_id = (int)particles[3][i]; // Final rank ID (stored at current index i)

        // Ensure final_rank_id is within bounds [0, npts-1]
        if (final_rank_id < 0 || final_rank_id >= npts)
        {
            log_message("ERROR", "Invalid remapped_id %d encountered at index %d during initial E/L calculation.", final_rank_id, i);
            // Handle error appropriately, maybe skip or exit
            continue;
        }

        /** @brief Calculate theoretical potential Psi(r) using initial theoretical spline. */
        // Uses Psiinterp accelerator associated with theoretical splinePsi
        double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
        Psi_val *= VEL_CONV_SQ; // Convert to physical units (kpc/Myr)^2

        /** @brief Calculate relative energy E = Psi - (1/2)(v_r^2 + L^2/r^2). */
        double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

        /** @brief Store initial E_rel and L using the final rank ID as index. */
        E_i_arr[final_rank_id] = E_rel;
        L_i_arr[final_rank_id] = ell;
    } // End initial E/L calculation loop

    /**
     * @brief LOW ANGULAR MOMENTUM PARTICLE SELECTION block.
     * @details Identifies the `nlowest` particles with either the lowest initial absolute angular
     *          momentum (`use_closest_to_Lcompare` = 0) or initial angular momentum closest
     *          to a reference value `Lcompare` (`use_closest_to_Lcompare` = 1), based on
     *          the values stored in `L_i_arr` (indexed by final_rank_id).
     *          Stores the final rank IDs of these selected particles in the `chosen` array.
     *          Allocates tracking arrays (`lowestL_r`, `lowestL_E`, `lowestL_L`) for these particles.
     * @see LAndIndex
     * @see cmp_LAI
     * @see chosen
     * @see lowestL_r
     * @see lowestL_E
     * @see lowestL_L
     */
    {
        /** @brief Create temporary LAndIndex array (size npts) to facilitate sorting by L. */
        LAndIndex *LAI = (LAndIndex *)malloc(npts * sizeof(LAndIndex));
        if (!LAI) { fprintf(stderr, "Error: Failed to allocate LAI array\n"); CLEAN_EXIT(1); }
        for (int i = 0; i < npts; i++) // 'i' here is the final_rank_id
        {
            if (use_closest_to_Lcompare) // Mode 1: Closest to Lcompare
            {
                double L_initial = L_i_arr[i];
                LAI[i].L = (L_initial - Lcompare) * (L_initial - Lcompare); // Store squared difference
                // Store sign of L_initial, used later for reconstruction or potentially unused
                LAI[i].sign = (L_initial >= 0.0) ? 1 : -1;
            }
            else // Mode 0: Lowest absolute L (using signed L for now based on original code)
            {
                LAI[i].L = L_i_arr[i]; // Store initial L value (signed)
                LAI[i].sign = 0;       // Sign field not used for sorting in this mode
            }
            LAI[i].idx = i; // Store the final_rank_id associated with this L value
        }

        /** @brief Sort LAI array by L member (ascending L or ascending L_diff^2). */
        qsort(LAI, npts, sizeof(LAndIndex), cmp_LAI);

        /** @brief If using Mode 1 (closest to Lcompare), reconstruct actual L values in LAI[].L.
         *         Note: This is primarily for the debug log output below; the primary goal is selecting indices. */
        if (use_closest_to_Lcompare) {
            for (int i = 0; i < npts; i++) {
                // Reconstruct L = Lcompare +/- sqrt(diff^2). The stored sign determines +/-.
                // Assumes LAI[i].sign stored the sign of (L_initial - Lcompare) or similar intention.
                // If sign was sign(L_initial), this reconstruction isn't quite right but matches original apparent logic.
                LAI[i].L = Lcompare + LAI[i].sign * sqrt(LAI[i].L);
            }
        }

        /** @brief Log the lowest L particles (L value and final_rank_id) if not restarting. */
        if (!g_doRestart) {
             log_message("DEBUG", "Selecting %d lowest L particles (mode=%d):", nlowest, use_closest_to_Lcompare);
             for (int i = 0; i < nlowest; i++) {
                 log_message("DEBUG", "  Rank %d: L=%.6f ID=%d", i, LAI[i].L, LAI[i].idx);
             }
        }

        /** @brief Allocate `chosen` array to store the final_rank_ids of the selected particles. */
        chosen = (int *)malloc(nlowest * sizeof(int));
        if (!chosen) { fprintf(stderr, "Error: Failed to allocate chosen array\n"); CLEAN_EXIT(1); }

        /** @brief Store the final rank IDs of the nlowest L particles into `chosen`. */
        for (int i = 0; i < nlowest; i++) chosen[i] = LAI[i].idx;

        free(LAI);

        /** @brief Allocate tracking arrays for chosen low-L particles [chosen_index][time_step]. */
        // Index 'p' corresponds to the p-th particle in the 'chosen' array.
        lowestL_r = (double **)malloc(nlowest * sizeof(double *)); // Radius
        // Add checks for allocation failure
        lowestL_E = (double **)malloc(nlowest * sizeof(double *)); // Energy
        // Add checks for allocation failure
        lowestL_L = (double **)malloc(nlowest * sizeof(double *)); // Angular Momentum
        // Add checks for allocation failure

        // Allocate memory for each selected particle's time history
        for (int p = 0; p < nlowest; p++) {
            lowestL_r[p] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
            lowestL_E[p] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
            lowestL_L[p] = (double *)malloc(ext_Ntimes * sizeof(double)); // Add checks for allocation failure
        }
    } // End Low-L selection block

    /**
     * @brief SIMULATION DATA TRACKING SETUP block.
     * @details Allocates memory for tracking particle data during the simulation.
     *          Includes the `inverse_map` array (mapping final_rank_id to current array index after sorting),
     *          calculates `deltaM` (mass per particle), and allocates block storage arrays
     *          (`L_block`, `Rank_block`, `R_block`, `Vrad_block`) if `g_doAllParticleData` is enabled.
     *          These blocks store data for `block_size` timesteps before being written to disk.
     * @see inverse_map
     * @see deltaM
     * @see L_block
     * @see Rank_block
     * @see R_block
     * @see Vrad_block
     */
    /** @brief Allocate index map: `inverse_map[final_rank_id]` will store the current index `i` after sorting. */
    int *inverse_map = (int *)malloc(npts * sizeof(int));
    if (!inverse_map) { fprintf(stderr, "Error: Failed to allocate inverse_map\n"); CLEAN_EXIT(1); }

    /** @brief Allocate particle scatter state array for AB3 history management after SIDM scattering. */
    g_particle_scatter_state = (int *)calloc(npts, sizeof(int)); // Use calloc to initialize all to 0
    if (!g_particle_scatter_state) {
        fprintf(stderr, "Error: Failed to allocate g_particle_scatter_state array\n");
        CLEAN_EXIT(1);
    }

    /** @brief Calculate mass per particle based on *initial* particle count before stripping. Used for M(rank). */
    double deltaM = g_active_halo_mass / (double)npts_initial;

    /** @brief Define block storage size (number of timesteps per block written to all_particle_data.dat). */
    int block_size = 100;



    /** @brief Create filename for the main all-particle data output file. */
    char apd_filename[256]; // Filename for data/all_particle_data<suffix>.dat
    get_suffixed_filename("data/all_particle_data.dat", 1, apd_filename, sizeof(apd_filename));

    /** @brief Allocate primary block storage arrays (float/int for memory efficiency).
     *         These store data for `block_size` steps, indexed [step_in_block * npts + final_rank_id].
     *         Freed later via `cleanup_all_particle_data()`. */
    L_block = (float *)malloc((size_t)npts * block_size * sizeof(float));    // Angular momentum
    Rank_block = (int *)malloc((size_t)npts * block_size * sizeof(int));     // Particle rank (sorted index) at that step
    R_block = (float *)malloc((size_t)npts * block_size * sizeof(float));    // Radius
    Vrad_block = (float *)malloc((size_t)npts * block_size * sizeof(float)); // Radial velocity
    // Check allocation results
    if (!L_block || !Rank_block || !R_block || !Vrad_block) {
        fprintf(stderr, "Error: Failed to allocate block storage arrays.\n");
        free(L_block); free(Rank_block); free(R_block); free(Vrad_block); // Free any that were allocated
        L_block = NULL; Rank_block = NULL; R_block = NULL; Vrad_block = NULL; // Prevent double free in cleanup
        CLEAN_EXIT(1);
    }

    /**
     * @brief RESTART MODE DATA VERIFICATION block (`all_particle_data.dat` check).
     * @details If running in restart mode (`g_doRestart`), this block checks if the main particle
     *          data output file (`all_particle_data<suffix>.dat`) already exists and is non-empty.
     *          If it exists and contains data, the flag `skip_simulation` is set to 1. This flag
     *          will cause the main timestepping loop to be bypassed entirely, allowing the program
     *          to proceed directly to the post-simulation snapshot analysis phase.
     * @note This check happens *before* the main loop. A more detailed check using
     *       `find_last_processed_snapshot` occurs *later* if snapshot analysis needs restarting.
     * @see skip_simulation
     * @see find_last_processed_snapshot
     */
    if (g_doRestart)
    {
        /** @brief Check for existence and size of the particle data file. */
        FILE *check_file = fopen(apd_filename, "rb");
        if (check_file)
        {
            /** @brief Determine file size by seeking to end and getting position. */
            fseek(check_file, 0, SEEK_END);
            long file_size = ftell(check_file);
            fclose(check_file);

            if (file_size > 0)
            {
                /** @brief Valid data file found - enable simulation phase bypass. */
                char human_size[32];
                format_file_size(file_size, human_size, sizeof(human_size));
                printf("Restart mode: Found existing all_particle_data file '%s' (%s).\n",
                       apd_filename, human_size);
                skip_simulation = 1;
                
                /** @brief Display appropriate message based on particle count comparison. */
                if (npts == npts_initial) {
                    printf("Skipping simulation phase and proceeding to post-processing. "
                           "Initial condition generation already performed, skipping these steps.\n\n");
                } else {
                    printf("Skipping simulation phase and proceeding to post-processing. "
                           "Initial condition generation and tidal stripping already performed, "
                           "skipping these steps.\n\n");
                }
            }
            else {
                /** @brief File exists but contains no data - simulation required. */
                printf("Restart mode: Found empty all_particle_data file '%s'. Will run simulation.\n", apd_filename);
            }
        }
        else {
            /** @brief File not found - simulation required. */
            printf("Restart mode: all_particle_data file '%s' not found. Will run simulation.\n", apd_filename);
        }
    } // End restart check block

    /**
     * @brief OUTPUT FILE INITIALIZATION block (`all_particle_data.dat`).
     * @details Creates (or overwrites) an empty binary file `all_particle_data<suffix>.dat`
     *          if saving all particle data (`g_doAllParticleData` is true) AND the simulation
     *          is *not* being skipped (`skip_simulation` is false). This file will be appended to
     *          incrementally during the simulation timestepping loop via block writes.
     * @see append_all_particle_data_chunk_to_file
     * @see apd_filename
     * @see g_doAllParticleData
     * @see skip_simulation
     */
    if (g_doAllParticleData && !skip_simulation)
    {
        // Create/truncate the output file in binary write mode.
        FILE *fapd = fopen(apd_filename, "wb");
        if (!fapd) {
            fprintf(stderr, "Error: cannot create all_particle_data output file %s\n", apd_filename);
            CLEAN_EXIT(1);
        }
        fclose(fapd); // Close immediately, file is now ready for appending.
        printf("Initialized empty file for all particle data: %s\n", apd_filename);
        
        // Calculate and display expected file size
        long long expected_size = (long long)total_writes * (long long)npts * 16LL; // 16 bytes per particle record
        double size_gb = expected_size / (1024.0 * 1024.0 * 1024.0);
        double size_mb = expected_size / (1024.0 * 1024.0);
        double size_kb = expected_size / 1024.0;
        
        if (size_gb >= 1.0) {
            printf("All particle data file requires: %.1f GB (%lld bytes)\n", size_gb, expected_size);
        } else if (size_mb >= 1.0) {
            printf("All particle data file requires: %.1f MB (%lld bytes)\n", size_mb, expected_size);
        } else if (size_kb >= 1.0) {
            printf("All particle data file requires: %.1f KB (%lld bytes)\n", size_kb, expected_size);
        } else {
            printf("All particle data file requires: %lld bytes\n", expected_size);
        }
        
        // Calculate and display expected snapshot file sizes
        // Each snapshot has 2 files: unsorted (28 bytes/particle) and sorted (32 bytes/particle)
        long long snapshot_size = (long long)npts * (28LL + 32LL); // Total per snapshot pair
        long long total_snapshot_size = snapshot_size * (long long)noutsnaps;
        double snap_size_gb = total_snapshot_size / (1024.0 * 1024.0 * 1024.0);
        double snap_size_mb = total_snapshot_size / (1024.0 * 1024.0);
        double snap_size_kb = total_snapshot_size / 1024.0;
        
        printf("%d time snapshot files will require: ", noutsnaps);
        if (snap_size_gb >= 1.0) {
            printf("%.1f GB (%lld bytes)\n", snap_size_gb, total_snapshot_size);
        } else if (snap_size_mb >= 1.0) {
            printf("%.1f MB (%lld bytes)\n", snap_size_mb, total_snapshot_size);
        } else if (snap_size_kb >= 1.0) {
            printf("%.1f KB (%lld bytes)\n", snap_size_kb, total_snapshot_size);
        } else {
            printf("%lld bytes\n", total_snapshot_size);
        }
        
        // Calculate and display total disk space
        long long total_disk_space = expected_size + total_snapshot_size;
        double total_gb = total_disk_space / (1024.0 * 1024.0 * 1024.0);
        double total_mb = total_disk_space / (1024.0 * 1024.0);
        double total_kb = total_disk_space / 1024.0;
        
        printf("Total disk space required: ");
        if (total_gb >= 1.0) {
            printf("%.1f GB (%lld bytes)\n", total_gb, total_disk_space);
        } else if (total_mb >= 1.0) {
            printf("%.1f MB (%lld bytes)\n", total_mb, total_disk_space);
        } else if (total_kb >= 1.0) {
            printf("%.1f KB (%lld bytes)\n", total_kb, total_disk_space);
        } else {
            printf("%lld bytes\n", total_disk_space);
        }
        
        // Check available disk space
        long long available_space = get_available_disk_space("data/");
        if (available_space > 0) {
            double avail_gb = available_space / (1024.0 * 1024.0 * 1024.0);
            double avail_mb = available_space / (1024.0 * 1024.0);
            double avail_kb = available_space / 1024.0;
            
            printf("Available disk space: ");
            if (avail_gb >= 1.0) {
                printf("%.1f GB (%lld bytes)\n", avail_gb, available_space);
            } else if (avail_mb >= 1.0) {
                printf("%.1f MB (%lld bytes)\n", avail_mb, available_space);
            } else if (avail_kb >= 1.0) {
                printf("%.1f KB (%lld bytes)\n", avail_kb, available_space);
            } else {
                printf("%lld bytes\n", available_space);
            }
            
            // Check if we're within 5% of total available or insufficient
            double usage_after = (double)(available_space - total_disk_space) / (double)available_space;
            
            if (available_space < total_disk_space) {
                // Insufficient space
                fprintf(stderr, "\nError: Insufficient disk space!\n");
                fprintf(stderr, "Required: %.1f GB\n", total_gb);
                fprintf(stderr, "Available: %.1f GB\n", avail_gb);
                fprintf(stderr, "Shortfall: %.1f GB\n", total_gb - avail_gb);
                CLEAN_EXIT(1);
            } else if (usage_after < 0.05) {
                // Within 5% of capacity after simulation
                printf("\nWarning: Simulation will use %.1f%% of available disk space!\n", 
                       (100.0 * total_disk_space / available_space));
                printf("After simulation: %.1f GB free (%.1f%% remaining)\n", 
                       (available_space - total_disk_space) / (1024.0 * 1024.0 * 1024.0),
                       usage_after * 100.0);
                
                if (!prompt_yes_no("Continue")) {
                    printf("Aborting simulation.\n");
                    CLEAN_EXIT(0);
                }
            }
        } else {
            fprintf(stderr, "Warning: Could not determine available disk space.\n");
            if (!prompt_yes_no("Continue without disk space check")) {
                printf("Aborting simulation.\n");
                CLEAN_EXIT(0);
            }
        }
        
        printf("\n");
        
        /** @brief Display initial simulation progress. */
        printf("0%% complete, timestep 0/%d, time=0.0000 Myr, elapsed=0.00 s\n", Ntimes);
    }

    /**
     * @brief BLOCK STORAGE INITIALIZATION block (Initial L).
     * @details Copies initial angular momentum values (`L_i_arr`, indexed by final_rank_id)
     *          into the first timestep slot (index 0 implicitly) of the `L_block` storage array.
     *          Converts from double precision (`L_i_arr`) to single precision (`L_block`).
     *          This prepares the block storage for the first chunk of simulation data.
     * @note Assumes L_block is indexed [step_in_block * npts + final_rank_id].
     */
    for (i = 0; i < npts; i++) // Loop over final_rank_id 'i'
    {
        double l_val = L_i_arr[i];
        float lf = (float)l_val;
        // Store initial L in the slot for step 0 for this final_rank_id
        L_block[i] = lf; // Index 'i' corresponds to final_rank_id for the first block slot (step 0)
    }

    /**
     * @brief Store initial (t=0) approximate energy for the debug particle.
     * @details If `g_doDebug` is enabled, this block calculates the theoretical
     *          approximate energy `E = Psi - KE` for the particle with final rank ID
     *          `DEBUG_PARTICLE_ID` based on its *initial* state (before any
     *          timesteps) and stores it using `store_debug_approxE` at snapshot index 0.
     * @note Assumes `DEBUG_PARTICLE_ID` refers to the *final rank ID* after potential
     *       stripping and remapping. Retrieves the initial state using this ID as the index
     *       into the `particles` array *before* the first sort in the main loop.
     */
    if (g_doDebug)
    {
        int debug_id = DEBUG_PARTICLE_ID;
        if (debug_id >= 0 && debug_id < npts) {
            // Retrieve initial state using final_rank_id as index into initial particles array
            double r0 = particles[0][debug_id];
            double v0 = particles[1][debug_id]; // This is v_rad
            double l0 = particles[2][debug_id];

            /** @brief Evaluate theoretical Psi(r) at initial radius using initial spline. */
            double psi_val = evaluatespline(splinePsi, Psiinterp, r0) * VEL_CONV_SQ;

            /** @brief Calculate initial approximate energy E = Psi - KE. */
            double E_approx0 = psi_val - 0.5 * (v0 * v0 + (l0 * l0) / (r0 * r0));
            double time0 = 0.0;

            /** @brief Store initial energy in debug arrays at snapshot index 0. */
            store_debug_approxE(0, E_approx0, time0);
        } else {
             fprintf(stderr, "Warning: Invalid DEBUG_PARTICLE_ID %d (must be 0 <= ID < %d)\n",
                     debug_id, npts);
        }
    }

    static int nwrite_total = 0; ///< Counter for number of dtwrite-interval writes performed.

    /**
     * @brief Perform initial sort of particles by radius before time integration.
     * @details Uses `quadsort` for this initial sort. Executed only by the master thread
     *          via `#pragma omp single`. After this sort, the index `i` of `particles[:][i]`
     *          corresponds to its rank based on radius.
     */
    #pragma omp single
    {
        sort_particles_with_alg(particles, npts, "quadsort");
    }

    /**
     * @brief Main Simulation Timestepping Loop.
     * @details This loop iterates from j = 0 to Ntimes + dtwrite - 1.
     *          In each iteration `j` (representing timestep from \f$t_j\f$ to \f$t_{j+1}\f$):
     *          1. The appropriate N-body integration method (selected by `method_select`)
     *             is called to advance all particle positions and velocities over `dt` due
     *             to gravitational forces. This typically involves sorting particles by radius
     *             for rank-based force calculation and updating an inverse map from original
     *             particle ID to current sorted rank.
     *          2. If SIDM is enabled (`g_enable_sidm_scattering`), the `handle_sidm_step()`
     *             function is called to apply stochastic SIDM scattering to particle pairs,
     *             further modifying their velocities and angular momenta.
     *          3. Simulation time `time` is incremented by `dt`.
     *          4. Trajectory and energy data for selected particles (low-ID and low-L) are recorded.
     *          5. If it's a `dtwrite` interval, a block of full particle data (rank, R, Vrad, L)
     *             is stored and potentially appended to `all_particle_data.dat`. Debug energy
     *             for `DEBUG_PARTICLE_ID` is also calculated and stored if it's a snapshot step.
     *          6. Progress is printed to the console at 5% intervals.
     *          This entire loop is skipped if `skip_simulation` is true (e.g., in restart mode
     *          where only post-processing of existing `all_particle_data.dat` is required).
     */
    if (!skip_simulation)
    {
        for (int j = 0; j < Ntimes + dtwrite; j++) // Main time loop
        {
            int current_step;

            // Method_select = 1; Flag now.
            if (method_select == 0)
            {
/****************************/
// EULER METHOD
/****************************/
#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier

#pragma omp parallel for default(shared) schedule(static)
                {
                    for (int idx = 0; idx < npts; idx++)
                    {
                        int orig_id = (int)particles[3][idx];
                        inverse_map[orig_id] = idx;
                    }
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];
                    double drdt = vrad;

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[0][i] += drdt * dt;
                    particles[1][i] += dvdt * dt;
                }

                // SIDM scattering: profile-aware scale radius selection and execution mode handling
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;
                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    double l_current = ell;
                    trajectories[p][j] = rr;
                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = l_current;
                }
            }
            else if (method_select == 1)
            {
                /****************************/
                // LEAPFROG METHOD (POSITION HALF STEP)
                /****************************/

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    particles[0][i] += particles[1][i] * (dt / 2.0);
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] += dvdt * dt;
                    particles[0][i] += particles[1][i] * (dt / 2.0);
                }

                // SIDM scattering after leapfrog drift completion
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;
                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    double l_current = ell;
                    trajectories[p][j] = rr;
                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = l_current;
                }
            }
            else if (method_select == 2)
            {

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] = vrad + 0.5 * dvdt * dt;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * dt;
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }
#pragma omp barrier

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] = vrad + 0.5 * dvdt * dt;
                }

                // SIDM scattering after velocity half-step completion
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;
                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    double l_current = ell;
                    trajectories[p][j] = rr;
                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = l_current;
                }
            }
            else if (method_select == 3)
            {
                /**
                 * @brief Full-step adaptive leapfrog integration.
                 * @details Performs single step integration from (r_n, v_n) to (r_{n+1}, v_{n+1})
                 *          using adaptive timestep control and the doAdaptiveFullLeap function.
                 */

                double velocity_tol = 1.0e-5;
                double radius_tol = 1.0e-5;
                // Int max_subdiv = 8383608;
                int max_subdiv = 1;
                int out_type = 0;

                /**
                 * @brief Radial sorting phase - organize particles by radius.
                 * @details Sorting improves cache locality and force calculation efficiency.
                 */
#pragma omp single
                {
                    sort_particles(particles, npts);
                }
                /**
                 * @brief Main integration loop - adaptive leapfrog update for each particle.
                 * @details Each particle is advanced independently using adaptive timestepping.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double v = particles[1][i];
                    double ell = particles[2][i];

                    // One full step => h = dt.
                    double r_new, v_new;
                    doAdaptiveFullLeap(
                        i, npts,
                        r, v,
                        ell,
                        dt,
                        radius_tol, velocity_tol,
                        max_subdiv,
                        G_CONST,
                        out_type,
                        &r_new, &v_new);

                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

                /**
                 * Particle sorting after integration (commented out)
                 *
                 * Optional re-sorting could be performed after each full step
                 * Currently disabled for performance reasons
                 */
                // #pragma omp single.
                // Sort_particles(particles, npts);.

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }
                // }

                // SIDM scattering after adaptive leapfrog completion
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

                /**
                 * @brief Timestep wrap-up phase - update time and record particle states.
                 * @details Updates global time counter and records trajectory information
                 *          for selected particles.
                 */
#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];

                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;

                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;

                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = ell;
                }
            }
            else if (method_select == 4)
            {
                /**
                 * @brief Hybrid integration with adaptive method selection.
                 * @details Uses Levi-Civita regularization for close encounters (r < r_crit)
                 *          and standard leapfrog otherwise. Radius threshold r_crit is 
                 *          dynamically calculated for each particle.
                 */

                double velocity_tol = 1.0e-8;
                double radius_tol = 1.0e-8;
                int max_subdiv = 4096 * 4096;
                int out_type = 2;
                int N_taumin = 1000;

                double alpha_param = 0.05;

#pragma omp single
                {
                    sort_particles(particles, npts);
                }
                /**
                 * @brief Main integration loop - method selection based on orbital parameters.
                 * @details Dynamically selects between standard leapfrog and Levi-Civita
                 *          regularization based on particle's radius and angular momentum.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double v = particles[1][i];
                    double ell = particles[2][i];

                    // Define critical radius r_crit for switching integration method:
                    // r_crit = alpha_param * (ell^2) / (G * M(r))
                    double r_crit = 0.0;
                    if (ell != 0.0)
                    {
                        double M_enc = ((double)i / (double)npts) * g_active_halo_mass;
                        double gravPart = (VEL_CONV_SQ * G_CONST) * M_enc;
                        r_crit = (ell * ell) * (alpha_param) / (gravPart);
                    }
                    else
                    {
                        r_crit = 0.0;
                    }

                    double r_new, v_new;

                    if ((r > 1.0e-30) && (r < r_crit))
                    {
                        doLeviCivitaLeapfrog(
                            i, npts,
                            r, v,
                            ell,
                            dt,
                            N_taumin,
                            G_CONST,
                            &r_new, &v_new);
                    }
                    else
                    {
                        doAdaptiveFullLeap(
                            i, npts,
                            r, v,
                            ell,
                            dt,
                            radius_tol, velocity_tol,
                            max_subdiv,
                            G_CONST,
                            out_type,
                            &r_new, &v_new);
                    }

                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }
                // }

                // SIDM scattering after hybrid integrator completion (Levi-Civita/adaptive leapfrog)
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];

                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;

                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;

                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = ell;
                }
            }
            else if (method_select == 5)
            {
                double velocity_tol = 1.0e-7;
                double radius_tol = 1.0e-7;
                int max_subdiv = 4096 * 4096 * 16;
                int out_type = 2;
                int N_taumin = 10;
                double alpha_param = 0.05;

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }

                // SIDM scattering before adaptive orbital integration with Levi-Civita regularization
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp parallel for default(shared) schedule(static)
                for (int i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double v = particles[1][i];
                    double ell = particles[2][i];

                    // Calculate r_crit, using special handling for particle i=0 (M_enc=0) to avoid division by zero.
                    double r_crit = 0.0;
                    if (fabs(ell) > 1.0e-30)
                    {
                        double M_enc;
                        if (i == 0)
                        {
                            // Special handling for i=0 to avoid zero mass in denominator
                            M_enc = 0.1 * (1.0 / (double)npts) * g_active_halo_mass;
                        }
                        else
                        {
                            M_enc = ((double)i / (double)npts) * g_active_halo_mass;
                        }
                        double gravPart = (VEL_CONV_SQ * G_CONST) * M_enc;
                        r_crit = (ell * ell) * alpha_param / gravPart;
                    }

                    double r_new, v_new;
                    if (r > 1.0e-30 && r < r_crit) // Switch based on critical radius
                    {
                        doAdaptiveFullLeviCivita(
                            i, npts, r, v, ell, dt, N_taumin,
                            radius_tol, velocity_tol, max_subdiv,
                            G_CONST, out_type, &r_new, &v_new);
                    }
                    else
                    {
                        doAdaptiveFullLeap(
                            i, npts, r, v, ell, dt,
                            radius_tol, velocity_tol, max_subdiv,
                            G_CONST, out_type, &r_new, &v_new);
                    }
                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)
                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vr = particles[1][idx];
                    double ell = particles[2][idx];

                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;

                    double vtot = sqrt(vr * vr + (ell * ell) / (rr * rr));
                    double mu_val = vr / vtot;
                    double E_rel = Psi_val - 0.5 * (vr * vr + (ell * ell) / (rr * rr));

                    trajectories[p][j] = rr;
                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vr;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = ell;
                }
            }
            else if (method_select == 6)
            {

                // Coefficients for 4th-order Forest-Ruth-Yoshida integrator (c1=c3).
                // Derived from: c1 = 1 / (2 - 2^(1/3)), c2 = 1 - 2*c1
                double c1 = 0.6756035959798289;
                double c2 = -0.3512071919596578; // = 1.0 - 2.0 * c1
                double c3 = c1;

                /**
                 * @brief STEP 1: Kick by (c1 * dt/2).
                 * @details Velocity update using the old position.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);
                    particles[1][i] = vrad + 0.5 * c1 * dt * dvdt;
                }

                /**
                 * @brief STEP 2: Drift by (c1 * dt).
                 * @details Position update using the intermediate velocity v^*.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * (c1 * dt);
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }
                // }

                /**
                 * @brief STEP 3: Kick by ((c1 + c2) * dt/2).
                 * @details Velocity update using the new position after the first drift.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);
                    double coeff = 0.5 * (c1 + c2);
                    particles[1][i] = vrad + coeff * dt * dvdt;
                }

                /**
                 * @brief STEP 4: Drift by (c2 * dt).
                 * @details Position update using the intermediate velocity.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * (c2 * dt);
                }

                /**
                 * @brief STEP 5: Kick by ((c2 + c3) * dt/2).
                 * @details Velocity update using the new position after the second drift.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    // Recompute acceleration at new position.
                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    // Combination of the remaining half of c2 and half of c3.
                    double coeff = 0.5 * (c2 + c3);
                    particles[1][i] = vrad + coeff * dt * dvdt;
                }

                /**
                 * @brief STEP 6: Drift by (c3 * dt).
                 * @details Final position update in this integration step.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double vrad = particles[1][i];
                    particles[0][i] += vrad * (c3 * dt);
                }

                /**
                 * @brief STEP 7: Kick by (c3 * dt/2).
                 * @details Final velocity update (half-kick) to complete the integration step.
                 */
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);
                    particles[1][i] = vrad + 0.5 * c3 * dt * dvdt;
                }

                // SIDM scattering after 4th-order Yoshida symplectic integration
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp single
                {
                    current_step = j + 1;
                    time += dt;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;
                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    double l_current = ell;
                    trajectories[p][j] = rr;
                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = l_current;
                }
            }
            else if (method_select == 7)
            {
                /****************************/
                // RK4 METHOD
                // In similar style as
                // the Euler code above
                /****************************/

                double *r_orig_by_id = (double *)malloc(npts * sizeof(double));
                double *v_orig_by_id = (double *)malloc(npts * sizeof(double));
                double *k1r_by_id = (double *)malloc(npts * sizeof(double));
                double *k1v_by_id = (double *)malloc(npts * sizeof(double));
                double *k2r_by_id = (double *)malloc(npts * sizeof(double));
                double *k2v_by_id = (double *)malloc(npts * sizeof(double));
                double *k3r_by_id = (double *)malloc(npts * sizeof(double));
                double *k3v_by_id = (double *)malloc(npts * sizeof(double));
                double *k4r_by_id = (double *)malloc(npts * sizeof(double));
                double *k4v_by_id = (double *)malloc(npts * sizeof(double));
                double h = dt;

// Store original state by orig_id.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    r_orig_by_id[orig_id] = particles[0][i];
                    v_orig_by_id[orig_id] = particles[1][i];
                }

// K1 calculation.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r = particles[0][i];
                    double vrad = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = vrad;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r, ell);

                    k1r_by_id[orig_id] = drdt;
                    k1v_by_id[orig_id] = dvdt;
                }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r_mid = particles[0][i];
                    double v_mid = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = v_mid;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r_mid, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r_mid, ell);

                    k2r_by_id[orig_id] = drdt;
                    k2v_by_id[orig_id] = dvdt;
                }

#pragma omp single
                {
                    sort_particles(particles, npts);
                }

#pragma omp barrier
#pragma omp parallel for default(shared) schedule(static)
                for (int idx = 0; idx < npts; idx++)
                {
                    int orig_id = (int)particles[3][idx];
                    inverse_map[orig_id] = idx;
                }
                // }

#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r_mid = particles[0][i];
                    double v_mid = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = v_mid;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r_mid, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r_mid, ell);

                    k3r_by_id[orig_id] = drdt;
                    k3v_by_id[orig_id] = dvdt;
                }

// K4 calculation.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];
                    double r_end = particles[0][i];
                    double v_end = particles[1][i];
                    double ell = particles[2][i];

                    double drdt = v_end;
                    // Use the gravitational_force and effective_angular_force functions.
                    double force = gravitational_force(r_end, i, npts, G_CONST, g_active_halo_mass);
                    double dvdt = force + effective_angular_force(r_end, ell);

                    k4r_by_id[orig_id] = drdt;
                    k4v_by_id[orig_id] = dvdt;
                }

// Final RK4 combination.
#pragma omp parallel for default(shared) schedule(static)
                for (i = 0; i < npts; i++)
                {
                    int orig_id = (int)particles[3][i];

                    double r_new = r_orig_by_id[orig_id] + (h / 6.0) * (k1r_by_id[orig_id] + 2.0 * k2r_by_id[orig_id] + 2.0 * k3r_by_id[orig_id] + k4r_by_id[orig_id]);
                    double v_new = v_orig_by_id[orig_id] + (h / 6.0) * (k1v_by_id[orig_id] + 2.0 * k2v_by_id[orig_id] + 2.0 * k3v_by_id[orig_id] + k4v_by_id[orig_id]);

                    particles[0][i] = r_new;
                    particles[1][i] = v_new;
                }

                // SIDM scattering after RK4 state update completion
                double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, 0);

#pragma omp single
                {
                    current_step = j + 1;
                    time += h;
                }
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

                for (int p = 0; p < upper_npts_num_traj; p++)
                {
                    int idx = inverse_map[p];
                    double rr = particles[0][idx];
                    double vrad = particles[1][idx];
                    double ell = particles[2][idx];
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    Psi_val *= VEL_CONV_SQ;
                    double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                    double mu_val = vrad / vtot;
                    double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                    double l_current = ell;
                    trajectories[p][j] = rr;
                    energies[p][j] = E_rel;
                    velocities_arr[p][j] = vrad;
                    mu_arr[p][j] = mu_val;
                    E_arr[p][j] = E_rel;
                    L_arr[p][j] = l_current;
                }

                // Free RK4 arrays.
                free(r_orig_by_id);
                free(v_orig_by_id);
                free(k1r_by_id);
                free(k1v_by_id);
                free(k2r_by_id);
                free(k2v_by_id);
                free(k3r_by_id);
                free(k3v_by_id);
                free(k4r_by_id);
                free(k4v_by_id);
            }
            else if (method_select == 8)
            {
                // Static variables for Adams-Bashforth 3rd Order (AB3) method
                static int ab3_bootstrap_done = 0;           ///< Flag indicating if the AB3 bootstrap phase has been completed (0=no, 1=yes).
                static double **f_ab3_r = NULL;               ///< History array for \f$dr/dt\f$ derivatives. Indexed by `[history_slot (0..2)][original_particle_id]`. Slot 2 is most recent (\f$f_n\f$).
                static double **f_ab3_v = NULL;               ///< History array for \f$dv_{rad}/dt\f$ derivatives. Indexed by `[history_slot (0..2)][original_particle_id]`. Slot 2 is most recent (\f$f_n\f$).
                static double h_ab3_bootstrap_step;         ///< Timestep size (`dt`) used during the Euler steps of the bootstrap phase.

                // Adams-Bashforth coefficients for different orders
                static int ab3_num[3] = {23, -16, 5};        ///< Numerator coefficients for the AB3 formula: \f$y_{n+1} = y_n + (h/12) \sum (\text{ab3_num}_i \cdot f_{n-i})\f$.
                static int ab2_num[2] = {18, -6};           ///< Numerator coefficients for the AB2 formula (for comparison or fallback).
                static int ab3_den = 12;                     ///< Common denominator for the AB3 formula coefficients.

                static int bootstrap_euler_steps_needed = 2; ///< Number of full Euler steps (each of size \f$h_{bootstrap}\f$) required to generate the initial 3 derivative history points (\f$f_0, f_1, f_2\f$).

                // Allocate AB3 history arrays once.
                if (f_ab3_r == NULL)
                {
                    f_ab3_r = (double **)malloc(3 * sizeof(double *));
                    f_ab3_v = (double **)malloc(3 * sizeof(double *));
                    for (int hh = 0; hh < 3; hh++)
                    {
                        f_ab3_r[hh] = (double *)malloc(npts * sizeof(double));
                        f_ab3_v[hh] = (double *)malloc(npts * sizeof(double));
                    }
                    h_ab3_bootstrap_step = dt; // Step size equals dt.
                }

                // If bootstrap not done yet, do it ONCE to fill the 3-step derivative history.
                if (!ab3_bootstrap_done)
                {

                    // Bootstrap for AB3: Need to compute f0, f1, f2.
                    // This requires 2 Euler steps to get states y1, y2.
                    // sub_step = 0: calc f0 (from y0), store f_ab3_x[0]. Euler y0->y1. state is y1.
                    // sub_step = 1: calc f1 (from y1), store f_ab3_x[1]. Euler y1->y2. state is y2.
                    // sub_step = 2: calc f2 (from y2), store f_ab3_x[2]. NO Euler update. state is y2.
#pragma omp single
                    {
                        for (int sub_step = 0; sub_step <= bootstrap_euler_steps_needed; sub_step++)
                        {
                            sort_particles(particles, npts);
#pragma omp parallel for default(shared) schedule(static)
                            for (int idx = 0; idx < npts; idx++)
                            {
                                int orig_id = (int)particles[3][idx];
                                inverse_map[orig_id] = idx;
                            }
                            // Compute derivatives => store in f_ab3_r[sub_step], f_ab3_v[sub_step].
#pragma omp parallel for default(shared) schedule(static)
                            for (int i_eval = 0; i_eval < npts; i_eval++)
                            {
                                int orig_id = (int)particles[3][i_eval];
                                double rr = particles[0][i_eval];
                                double vrad = particles[1][i_eval];
                                double ell = particles[2][i_eval];

                                double drdt = vrad;
                                // Use the gravitational_force and effective_angular_force functions.
                                double force = gravitational_force(rr, i_eval, npts, G_CONST, g_active_halo_mass);
                                double dvdt = force + effective_angular_force(rr, ell);

                                f_ab3_r[sub_step][orig_id] = drdt;
                                f_ab3_v[sub_step][orig_id] = dvdt;
                            }

                            // If sub_step < bootstrap_euler_steps_needed (i.e., for sub_step 0 and 1),
                            // perform mini-substeps to advance particles to the next state with higher accuracy.
                            if (sub_step < bootstrap_euler_steps_needed)
                            {
                                double dt_mini = h_ab3_bootstrap_step / (double)NUM_MINI_SUBSTEPS_BOOTSTRAP;

#pragma omp parallel for default(shared) schedule(static)
                                for (int i_part = 0; i_part < npts; i_part++)
                                {
                                    // Each particle is evolved independently over NUM_MINI_SUBSTEPS_BOOTSTRAP
                                    double current_r_mini = particles[0][i_part];
                                    double current_vrad_mini = particles[1][i_part];
                                    double current_ell_mini = particles[2][i_part]; // Angular momentum (constant)
                                    
                                    // Perform NUM_MINI_SUBSTEPS_BOOTSTRAP mini-steps
                                    for (int m = 0; m < NUM_MINI_SUBSTEPS_BOOTSTRAP; m++)
                                    {
                                        // Calculate derivatives based on current mini-step state
                                        double drdt_m = current_vrad_mini;
                                        double force_m = gravitational_force(current_r_mini, i_part, npts, G_CONST, g_active_halo_mass);
                                        double dvdt_m = force_m + effective_angular_force(current_r_mini, current_ell_mini);
                                        
                                        // Euler update for this mini-step
                                        current_r_mini += dt_mini * drdt_m;
                                        current_vrad_mini += dt_mini * dvdt_m;
                                    }
                                    
                                    // After all mini-steps, update the main particles array
                                    particles[0][i_part] = current_r_mini;
                                    particles[1][i_part] = current_vrad_mini;
                                }
                            }
                        }
                    }

#pragma omp single
                    {
                        // Mark bootstrap done.
                        ab3_bootstrap_done = 1;
                    }
                }
                else
                {
                    /**
                     * Normal AB3 step each iteration
                     */

#pragma omp single
                    {
                        sort_particles(particles, npts);
                    }

#pragma omp barrier
// #pragma omp single.
// {
#pragma omp parallel for default(shared) schedule(static)
                    for (int idx = 0; idx < npts; idx++)
                    {
                        int orig_id = (int)particles[3][idx];
                        inverse_map[orig_id] = idx;
                    }
                    // }

#pragma omp parallel for default(shared) schedule(static)
                    for (int i = 0; i < npts; i++)
                    {
                        // Adams-Bashforth 8th order integration step.
                        int orig_id = (int)particles[3][i];
                        double rr = particles[0][i];
                        double vrad = particles[1][i];

                        double sum_r = 0.0;
                        double sum_v = 0.0;
                        int particle_state = g_particle_scatter_state[orig_id];

                        // AB3 coefficients: b0=23/12, b1=-16/12, b2=5/12. Denom ab3_den=12.
                        // History: f_ab3_[r/v][2] is f_n (latest), [1] is f_{n-1}, [0] is f_{n-2}

                        if (particle_state == 1) { // Just scattered: Use AB1 (Euler-like)
                            // sum = 12 * f_n
                            sum_r = 12.0 * f_ab3_r[2][orig_id];
                            sum_v = 12.0 * f_ab3_v[2][orig_id];
                            if (g_doDebug && i < 5) { // Extremely sparse debug
                                 log_message("DEBUG", "AB3_RESET: Particle %d (orig_id) using AB1 step (state 1)", orig_id);
                            }
                        } else if (particle_state == 2) { // One step after scatter: Use AB2
                            // sum = ab2_num[0] * f_n + ab2_num[1] * f_{n-1}
                            sum_r = ab2_num[0] * f_ab3_r[2][orig_id] + ab2_num[1] * f_ab3_r[1][orig_id];
                            sum_v = ab2_num[0] * f_ab3_v[2][orig_id] + ab2_num[1] * f_ab3_v[1][orig_id];
                            if (g_doDebug && i < 5) {
                                 log_message("DEBUG", "AB3_RESET: Particle %d (orig_id) using AB2 step (state 2)", orig_id);
                            }
                        } else { // Normal AB3 step
                            sum_r = ab3_num[0] * f_ab3_r[2][orig_id] + ab3_num[1] * f_ab3_r[1][orig_id] + ab3_num[2] * f_ab3_r[0][orig_id];
                            sum_v = ab3_num[0] * f_ab3_v[2][orig_id] + ab3_num[1] * f_ab3_v[1][orig_id] + ab3_num[2] * f_ab3_v[0][orig_id];
                        }

                        double r_next = rr + (dt / (double)ab3_den) * sum_r;
                        double v_next = vrad + (dt / (double)ab3_den) * sum_v;

                        particles[0][i] = r_next;
                        particles[1][i] = v_next;
                    }

                    // SIDM scattering after Adams-Bashforth update (skip during bootstrap)
                    double current_active_rc_for_sidm = g_use_nfw_profile ? g_nfw_profile_rc : g_cored_profile_rc;
                    handle_sidm_step(particles, npts, dt, time, current_active_rc_for_sidm, display_method, !ab3_bootstrap_done);

                    // We re-sort & compute new derivatives to shift the AB3 history.
#pragma omp single
                    {
                        sort_particles(particles, npts);
                    }
#pragma omp barrier
                    // #pragma omp single.
                    // {

#pragma omp parallel for default(shared) schedule(static)
                    for (int idx = 0; idx < npts; idx++)
                    {
                        int orig_id = (int)particles[3][idx];
                        inverse_map[orig_id] = idx;
                    }

                    // Recompute the derivatives for the new time => goes into f_ab3_r[2], f_ab3_v[2].
                    double **f_new_r = (double **)malloc(sizeof(double *));
                    double **f_new_v = (double **)malloc(sizeof(double *));
                    f_new_r[0] = (double *)malloc(npts * sizeof(double));
                    f_new_v[0] = (double *)malloc(npts * sizeof(double));

#pragma omp parallel for default(shared) schedule(static)
                    for (int i_dbg = 0; i_dbg < npts; i_dbg++)
                    {
                        int orig_id = (int)particles[3][i_dbg];
                        double rr = particles[0][i_dbg];
                        double vrad = particles[1][i_dbg];
                        double ell = particles[2][i_dbg];

                        double drdt = vrad;
                        // Use the gravitational_force and effective_angular_force functions.
                        double force = gravitational_force(rr, i_dbg, npts, G_CONST, g_active_halo_mass);
                        double dvdt = force + effective_angular_force(rr, ell);

                        f_new_r[0][orig_id] = drdt;
                        f_new_v[0][orig_id] = dvdt;
                    }

#pragma omp single
                    {
                        // SHIFT AB3 HISTORY: f0 <- f1, f1 <- f2
                        for (int i_s = 0; i_s < npts; i_s++)
                        {
                            f_ab3_r[0][i_s] = f_ab3_r[1][i_s]; // f_{n-2} becomes old f_{n-1}
                            f_ab3_v[0][i_s] = f_ab3_v[1][i_s];

                            f_ab3_r[1][i_s] = f_ab3_r[2][i_s]; // f_{n-1} becomes old f_n
                            f_ab3_v[1][i_s] = f_ab3_v[2][i_s];
                        }
                        // Put the new derivative (f_n for the just-completed step) in slot #2
                        for (int i_s = 0; i_s < npts; i_s++)
                        {
                            f_ab3_r[2][i_s] = f_new_r[0][i_s]; // f_n (latest)
                            f_ab3_v[2][i_s] = f_new_v[0][i_s];
                        }

                        free(f_new_r[0]);
                        free(f_new_v[0]);
                        free(f_new_r);
                        free(f_new_v);

                        current_step = j + 1;
                        time += dt;

                        // Advance particle scatter states for next AB step
                        if (ab3_bootstrap_done) { // Only advance state if AB is active and past bootstrap
                            for (int k_pstate = 0; k_pstate < npts; k_pstate++) {
                                // k_pstate here is the original_id since g_particle_scatter_state is indexed by orig_id
                                if (g_particle_scatter_state[k_pstate] == 2) {
                                    g_particle_scatter_state[k_pstate] = 0; // Transition from AB2 to full AB3
                                } else if (g_particle_scatter_state[k_pstate] == 1) {
                                    g_particle_scatter_state[k_pstate] = 2; // Transition from AB1 to AB2
                                }
                                // If state is 0, it remains 0 unless SIDM sets it to 1 in the next call to handle_sidm_step
                            }
                        }
                    }
                } // End of the "else" block for normal AB3.
            }
            // Record trajectory data for selected low-ID particles
#pragma omp parallel for if (upper_npts_num_traj > 1000) schedule(static)

            for (int p = 0; p < upper_npts_num_traj; p++)
            {
                int idx = inverse_map[p];
                double rr = particles[0][idx];
                double vrad = particles[1][idx];
                double ell = particles[2][idx];
                double Psi_val = evaluatespline(splinePsi, Psiinterp, rr) * VEL_CONV_SQ;
                double vtot = sqrt(vrad * vrad + (ell * ell) / (rr * rr));
                double mu_val = vrad / vtot;

                double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));
                trajectories[p][j] = rr;
                energies[p][j] = E_rel;
                velocities_arr[p][j] = vrad;
                mu_arr[p][j] = mu_val;
                E_arr[p][j] = E_rel;
                L_arr[p][j] = ell;
            }

            // Record trajectory data for selected low-L particles
            int max_threads = omp_get_max_threads();
            gsl_interp_accel **thread_accel = (gsl_interp_accel **)malloc(max_threads * sizeof(gsl_interp_accel *));
            for (int i = 0; i < max_threads; i++)
            {
                thread_accel[i] = gsl_interp_accel_alloc();
            }

#pragma omp parallel for if (nlowest > 1000) schedule(static)
            for (int p = 0; p < nlowest; p++)
            {
                int thread_id = omp_get_thread_num();
                gsl_interp_accel *thread_safe_accel = thread_accel[thread_id];

                int idx_lowest = inverse_map[chosen[p]];

                double rr = particles[0][idx_lowest];
                double vrad = particles[1][idx_lowest];
                double ell = particles[2][idx_lowest];

                if (rr < 0.0)
                {
                    rr = 0.0000000001;
                }
                if (rr > rmax)
                {
                    rr = rmax;
                }

                // Use thread-safe accelerator instead of shared one.
                double Psi_val = evaluatespline(splinePsi, thread_safe_accel, rr) * VEL_CONV_SQ;
                double E_rel = Psi_val - 0.5 * (vrad * vrad + (ell * ell) / (rr * rr));

                lowestL_r[p][j] = rr;    // Store R.
                lowestL_E[p][j] = E_rel; // Store E.
                lowestL_L[p][j] = ell;   // Store L.
            }

            // Clean up thread-local accelerators.
            for (int i = 0; i < max_threads; i++)
            {
                gsl_interp_accel_free(thread_accel[i]);
            }
            free(thread_accel);
            // }

#pragma omp single
            {
                if ((current_step % dtwrite) == 0)
                {
                    double elapsed = omp_get_wtime() - start_time;
                    printf("Write data at timestep %d after %.2f s.\n",
                           current_step, elapsed);

                    // Calculate the write index (0-based) corresponding to this timestep
                    int nwrite = current_step / dtwrite - 1;
                    if (g_doAllParticleData)
                    {
                        // Calculate index within the current block (0 to block_size-1)
                        int block_index_apd = nwrite % block_size;
                        for (int pi = 0; pi < npts; pi++)
                        {
                            int orig_id = pi;
                            int rank = inverse_map[orig_id];
                            double par_r = particles[0][rank];
                            double par_vrad = particles[1][rank];
                            double par_ell = particles[2][rank];

                            Rank_block[block_index_apd * npts + orig_id] = rank;
                            R_block[block_index_apd * npts + orig_id] = (float)par_r;
                            Vrad_block[block_index_apd * npts + orig_id] = (float)par_vrad;
                            L_block[block_index_apd * npts + orig_id] = (float)par_ell;

                            if (g_doDebug)
                            {
                                // Check if this write corresponds to a desired snapshot output time
                                if (nwrite % stepBetweenSnaps == 0)
                                {
                                    int snapIndex = nwrite / stepBetweenSnaps;
                                    if (snapIndex < noutsnaps) // Ensure snapIndex is valid
                                    {
                                        int debug_id = DEBUG_PARTICLE_ID;
                                        
                                        // Retrieve current state for debug particle from block arrays
                                        float r_valF = R_block[block_index_apd * npts + debug_id];
                                        float v_valF = Vrad_block[block_index_apd * npts + debug_id];
                                        float l_valF = L_block[block_index_apd * npts + debug_id];
                                        double r_val = (double)r_valF;
                                        double v_val = (double)v_valF;
                                        double l_val = (double)l_valF;
                                        
                                        // Evaluate theoretical potential using original spline
                                        double psi_val = 0.0;
                                        if (r_val >= 0.0 && r_val <= rmax)
                                        {
                                            psi_val = evaluatespline(splinePsi, Psiinterp, r_val) * VEL_CONV_SQ;
                                        }
                                        
                                        // Calculate approximate energy E = Psi - KE
                                        double E_approx = psi_val - 0.5 * (v_val * v_val + (l_val * l_val) / (r_val * r_val));
                                        double sim_time = time;
                                        
                                        // Store the approximate energy for comparison
                                        store_debug_approxE(snapIndex, E_approx, sim_time);
                                    }
                                }
                            }
                        }
                    }
                    if (g_doAllParticleData)
                    {
                        // Append block to file if block is full
                        if (((nwrite + 1) % block_size) == 0 && nwrite > 0)
                        {
                            append_all_particle_data_chunk_to_file(apd_filename,
                                                                   npts,
                                                                   block_size,
                                                                   L_block,
                                                                   Rank_block,
                                                                   R_block,
                                                                   Vrad_block);
                            printf("Appended block ending write index %d to %s\n", nwrite, apd_filename);
                        }
                    }
                    // Increment count of dtwrite-based writes
                    nwrite_total++;
                }
            }

            for (int k = 0; k <= 20; k++)
            {
                if (current_step == print_steps[k])
                {
                    double elapsed = omp_get_wtime() - start_time;
                    int percent = k * 5;
                    printf("%d%% complete, timestep %d/%d, time=%f Myr, elapsed=%.2f s\n",
                           percent, current_step, Ntimes, time, elapsed);
                    break;
                }
            }
        }
        // After the main simulation loop, flush any remaining data in the last partial block.
        {
            if (g_doAllParticleData)
            {
                int leftover = nwrite_total % block_size;
                if (leftover > 0)
                {
                    printf("Flushing leftover %d steps from block storage...\n", leftover);
                    append_all_particle_data_chunk_to_file(apd_filename,
                                                           npts,
                                                           leftover, // Number of steps in this partial block
                                                           L_block,
                                                           Rank_block,
                                                           R_block,
                                                           Vrad_block);
                }
            }
        }
    } // Close the skip_simulation if block.

    // Write final particle state if simulation was run
    if (!skip_file_writes)
    {
        char suffixed_filename[256];
        get_suffixed_filename("data/particlesfinal.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        if (fp == NULL)
        {
            fprintf(stderr, "Error opening file %s for writing\n", suffixed_filename);
            exit(1);
        }

        for (i = 0; i < npts; i++)
        {
            fprintf_bin(fp, "%f %f %f  %f\n", particles[0][i], particles[1][i], particles[2][i], particles[3][i]);
        }
        fclose(fp);
    }

    // Write theoretical profiles (profile-specific formulas)
    char suffixed_filename[256];
    
    if (g_use_nfw_profile) {
        /**
         * @brief Write final theoretical NFW profile characteristics to .dat files.
         * @details This block outputs several files (massprofile, Psiprofile, density_profile,
         *          dpsi_dr, drho_dpsi, f_of_E, df_fixed_radius) using the splines
         *          (e.g., splinemass, splinePsi, g_main_fofEinterp) and parameters
         *          (e.g., num_points, radius, normalization, g_nfw_profile_rc, etc.)
         *          that were established during the main NFW initial condition generation phase.
         *          Analytical formulas for NFW density and its derivatives are used where appropriate.
         */
        log_message("INFO", "Writing NFW theoretical profiles to final .dat files...");

        // Write NFW theoretical mass profile
        get_suffixed_filename("data/massprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                    fprintf_bin(fp, "%f %f\n", r_plot, gsl_spline_eval(splinemass, r_plot, enclosedmass));
                }
            }
            if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], gsl_spline_eval(splinemass, radius[num_points-1], enclosedmass));
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW mass profile", suffixed_filename); 
        }

        // Write NFW theoretical potential profile
        get_suffixed_filename("data/Psiprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                     fprintf_bin(fp, "%f %f\n", r_plot, evaluatespline(splinePsi, Psiinterp, r_plot));
                }
            }
             if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], evaluatespline(splinePsi, Psiinterp, radius[num_points-1]));
             }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW Psi profile", suffixed_filename); 
        }

        // Write NFW theoretical density profile
        get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            double nt_nfw_scaler_final = g_nfw_profile_halo_mass / (4.0 * M_PI * normalization);
            for (i = 0; i < num_points; i++) {
                double rr = radius[i]; 
                double rs_k = rr / g_nfw_profile_rc;
                double term_s_k = rs_k + 0.01; 
                if (term_s_k <= 1e-9) term_s_k = 1e-9;
                double term_n_k = (1.0 + rs_k) * (1.0 + rs_k);
                double term_c_base_k = rs_k / g_nfw_profile_falloff_factor;
                double term_c_k = 1.0 + pow(term_c_base_k, 10.0);
                double rho_shape_k = (term_s_k < 1e-9 || term_n_k < 1e-9 || term_c_k < 1e-9) ? 0.0 : (1.0 / (term_s_k * term_n_k * term_c_k));
                if (rr < 1e-6 && term_s_k < 1e-3 && rho_shape_k == 0.0) { 
                   rho_shape_k = 1.0 / (term_s_k * term_n_k * term_c_k);
                }
                double rho_r_k = nt_nfw_scaler_final * rho_shape_k;
                fprintf_bin(fp, "%f %f\n", rr, rho_r_k);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW density profile", suffixed_filename); 
        }

        // Write NFW theoretical dPsi/dr profile
        get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                if (rr > 0.0) {
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass); 
                    double dpsidr = -(G_CONST * Menc) / (rr * rr);
                    fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                }
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW dpsi/dr profile", suffixed_filename); 
        }

        // Write NFW theoretical drho/dPsi profile
        get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            double nt_nfw_scaler_final = g_nfw_profile_halo_mass / (4.0 * M_PI * normalization);
            for (i = 1; i < num_points - 1; i++) {
                double rr = radius[i];
                if (rr <= 1e-9) continue;

                // Use NFW derivative function
                double drhodr_val_k = drhodr_profile_nfwcutoff(rr, g_nfw_profile_rc, nt_nfw_scaler_final, g_nfw_profile_falloff_factor);
                
                double Menc_k = gsl_spline_eval(splinemass, rr, enclosedmass);
                double dPsidr_mag_k = (G_CONST * Menc_k) / (rr * rr); 
                
                if (fabs(dPsidr_mag_k) > 1e-30) {
                    double Psi_val_k = evaluatespline(splinePsi, Psiinterp, rr);
                    double drho_dPsi_val_k = drhodr_val_k / dPsidr_mag_k; 
                    fprintf_bin(fp, "%f %f\n", Psi_val_k, drho_dPsi_val_k);
                }
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW drho/dpsi profile", suffixed_filename); 
        }

        // Write NFW theoretical f(E) profile
        get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i <= num_points; i++) {
                double E = Evalues[i];
                double deriv = 0.0;
                if (i > 0 && i < num_points + 1) {
                    if (i > 0 && i < num_points) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i - 1]) / (Evalues[i + 1] - Evalues[i - 1]);
                    }
                    else if (i == 0) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i]) / (Evalues[i + 1] - Evalues[i]);
                    }
                    else if (i == num_points) {
                        deriv = (innerintegrandvalues[i] - innerintegrandvalues[i - 1]) / (Evalues[i] - Evalues[i - 1]);
                    }
                }
                double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                if (E == 0.0 || !isfinite(fE))
                    fE = 0.0;
                fprintf_bin(fp, "%f %f\n", E, fE);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final NFW f(E) profile", suffixed_filename); 
        }

        // Write NFW distribution function at a fixed radius if simulation was run
        if (!skip_file_writes) {
            get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            fp = fopen(suffixed_filename, "wb");
            if (fp) {
                double r_F = 2.0 * g_nfw_profile_rc;  // r_F = 2 × NFW scale radius
                double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_F);
                Psi_rf *= VEL_CONV_SQ;
                double Psimin_test = VEL_CONV_SQ * Psimin; // Convert to (km/s)² for velocity calculation

                int vsteps = 10000;
                int reduce_vsteps = 300;
                for (int vv = 0; vv <= vsteps - reduce_vsteps; vv++) {
                    double sqrt_arg_v = Psi_rf - Psimin_test;
                    if (sqrt_arg_v < 0) sqrt_arg_v = 0;
                    double vtest = (double)vv * (sqrt(2.0 * sqrt_arg_v) / (vsteps));
                    double Etest = Psi_rf - 0.5 * vtest * vtest;
                    Etest = Etest / VEL_CONV_SQ; // Convert back to code units for bounds check
                    double fEval = 0.0;
                    if (Etest >= Psimin && Etest <= Psimax) { // Bounds check in code units
                        double derivative;
                        int status = gsl_interp_eval_deriv_e(g_main_fofEinterp, Evalues, innerintegrandvalues, Etest, g_main_fofEacc, &derivative);
                        if (status == GSL_SUCCESS) {
                            fEval = derivative / (sqrt(8.0) * PI * PI) * vtest * vtest * r_F * r_F;
                        }
                    }
                    if (!isfinite(fEval)) fEval = 0.0;
                    fprintf_bin(fp, "%f %f\n", vtest, fEval);
                }
                fclose(fp);
            } else { 
                log_message("ERROR", "Failed to open %s for final NFW df_fixed_radius", suffixed_filename); 
            }
        }

    } else {
        /**
         * @brief Write final theoretical Cored Plummer-like profile characteristics to .dat files.
         * @details This block outputs several files (massprofile, Psiprofile, density_profile,
         *          dpsi_dr, drho_dpsi, f_of_E, df_fixed_radius) using the splines
         *          (e.g., splinemass, splinePsi, g_main_fofEinterp) and parameters
         *          (e.g., num_points, radius, normalization, g_cored_profile_rc, etc.)
         *          that were established during the main Cored Plummer initial condition generation phase.
         *          Analytical formulas for the Cored density and its derivatives are used where appropriate.
         */
        log_message("INFO", "Writing Cored theoretical profiles to final .dat files...");

        // Write Cored theoretical mass profile
        get_suffixed_filename("data/massprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                    fprintf_bin(fp, "%f %f\n", r_plot, gsl_spline_eval(splinemass, r_plot, enclosedmass));
                }
            }
            if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], gsl_spline_eval(splinemass, radius[num_points-1], enclosedmass));
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final cored mass profile", suffixed_filename); 
        }

        // Write Cored theoretical potential profile
        get_suffixed_filename("data/Psiprofile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (double r_plot = 0.0; r_plot < radius[num_points - 1]; r_plot += (radius[num_points - 1] / 900.0)) {
                if (r_plot >= radius[0]) {
                     fprintf_bin(fp, "%f %f\n", r_plot, evaluatespline(splinePsi, Psiinterp, r_plot));
                }
            }
             if (num_points > 0) {
                 fprintf_bin(fp, "%f %f\n", radius[num_points-1], evaluatespline(splinePsi, Psiinterp, radius[num_points-1]));
             }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final cored Psi profile", suffixed_filename); 
        }

        // Write Cored theoretical density profile
        get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                double rho_r = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(rr / g_cored_profile_rc)));
                fprintf_bin(fp, "%f %f\n", rr, rho_r);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final cored density profile", suffixed_filename); 
        }

        // Write Cored theoretical dPsi/dr profile
        get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i < num_points; i++) {
                double rr = radius[i];
                if (rr > 0.0) {
                    double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                    double dpsidr = -(G_CONST * Menc) / (rr * rr);
                    fprintf_bin(fp, "%f %f\n", rr, dpsidr);
                }
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final cored dpsi/dr profile", suffixed_filename); 
        }

        // Write Cored theoretical drho/dPsi profile
        get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 1; i < num_points - 1; i++) {
                double rr = radius[i];
                double rho_left = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i - 1] / g_cored_profile_rc)));
                double rho_right = g_cored_profile_halo_mass / normalization * (1.0 / cube(1.0 + sqr(radius[i + 1] / g_cored_profile_rc)));
                double drho_dr_num = (rho_right - rho_left) / (radius[i + 1] - radius[i - 1]);
                double Menc = gsl_spline_eval(splinemass, rr, enclosedmass);
                double dPsidr = -(G_CONST * Menc) / (rr * rr);
                if (dPsidr != 0.0) {
                    double Psi_val = evaluatespline(splinePsi, Psiinterp, rr);
                    fprintf_bin(fp, "%f %f\n", Psi_val, drho_dr_num / dPsidr);
                }
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final cored drho/dpsi profile", suffixed_filename); 
        }

        // Write theoretical f(E) profile
        get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb");
        if (fp) {
            for (i = 0; i <= num_points; i++) {
                double E = Evalues[i];
                double deriv = 0.0;
                if (i > 0 && i < num_points + 1) {
                    if (i > 0 && i < num_points) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i - 1]) / (Evalues[i + 1] - Evalues[i - 1]);
                    }
                    else if (i == 0) {
                        deriv = (innerintegrandvalues[i + 1] - innerintegrandvalues[i]) / (Evalues[i + 1] - Evalues[i]);
                    }
                    else if (i == num_points) {
                        deriv = (innerintegrandvalues[i] - innerintegrandvalues[i - 1]) / (Evalues[i] - Evalues[i - 1]);
                    }
                }
                double fE = fabs(deriv) / (sqrt(8.0) * PI * PI);
                if (E == 0.0 || !isfinite(fE))
                    fE = 0.0;
                fprintf_bin(fp, "%f %f\n", E, fE);
            }
            fclose(fp);
        } else { 
            log_message("ERROR", "Failed to open %s for final f(E) profile (%s)", suffixed_filename, g_use_nfw_profile ? "NFW" : "Cored"); 
        }

        // Write distribution function at a fixed radius if simulation was run
        if (!skip_file_writes) {
            get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            fp = fopen(suffixed_filename, "wb");
            if (fp) {
                double r_F = 2.0 * g_cored_profile_rc;  // r_F = 2 × Cored scale radius
                double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_F);
                Psi_rf *= VEL_CONV_SQ;
                double Psimin_test = VEL_CONV_SQ * Psimin; // Convert to (km/s)² for velocity calculation

                int vsteps = 10000;
                int reduce_vsteps = 300;
                for (int vv = 0; vv <= vsteps - reduce_vsteps; vv++) {
                    double sqrt_arg_v = Psi_rf - Psimin_test;
                    if (sqrt_arg_v < 0) sqrt_arg_v = 0;
                    double vtest = (double)vv * (sqrt(2.0 * sqrt_arg_v) / (vsteps));
                    double Etest = Psi_rf - 0.5 * vtest * vtest;
                    Etest = Etest / VEL_CONV_SQ; // Convert back to code units for bounds check
                    double fEval = 0.0;
                    if (Etest >= Psimin && Etest <= Psimax) { // Bounds check in code units
                        double derivative;
                        int status = gsl_interp_eval_deriv_e(g_main_fofEinterp, Evalues, innerintegrandvalues, Etest, g_main_fofEacc, &derivative);
                        if (status == GSL_SUCCESS) {
                            fEval = derivative / (sqrt(8.0) * PI * PI) * vtest * vtest * r_F * r_F;
                        }
                    }
                    if (!isfinite(fEval)) fEval = 0.0;
                    fprintf_bin(fp, "%f %f\n", vtest, fEval);
                }
                fclose(fp);
            } else { 
                log_message("ERROR", "Failed to open %s for final df_fixed_radius (%s)", suffixed_filename, g_use_nfw_profile ? "NFW" : "Cored"); 
            }
        }
    }

    char filename[256];
    get_suffixed_filename("data/particles.dat", 1, filename, sizeof(filename));
    FILE *finit = fopen(filename, "rb"); // Binary mode for fscanf_bin
    if (!finit)
    {
        printf("Error: can't open %s\n", filename);
        CLEAN_EXIT(1);
    }
    double *r_initial = (double *)malloc(npts * sizeof(double));
    double *rv_initial = (double *)malloc(npts * sizeof(double));
    double *v_initial = (double *)malloc(npts * sizeof(double));
    double *l_initial = (double *)malloc(npts * sizeof(double));
    double rank_id;
    for (i = 0; i < npts; i++)
    {
        fscanf_bin(finit, "%f %f %f %f\n", &r_initial[i], &rv_initial[i], &l_initial[i], &rank_id);
    }
    fclose(finit);

    for (i = 0; i < npts; i++)
    {
        v_initial[i] = sqrt(rv_initial[i] * rv_initial[i] + l_initial[i] * l_initial[i] / (r_initial[i] * r_initial[i]));
    }

    double *r_final = (double *)malloc(npts * sizeof(double));
    double *v_final = (double *)malloc(npts * sizeof(double));
    for (i = 0; i < npts; i++)
    {
        r_final[i] = particles[0][i];
        v_final[i] = sqrt(particles[1][i] * particles[1][i] + particles[2][i] * particles[2][i] / (particles[0][i] * particles[0][i]));
    }

    /**
     * @brief Calculate percentile-based ranges for dynamic histogram binning.
     * @details Combines initial and final particle distributions to determine
     * appropriate histogram ranges using the 99th percentile with 1.2x padding.
     * This ensures virtually all particles are captured while avoiding outliers.
     */
    double *r_all_sorted = (double *)malloc(2 * npts * sizeof(double));
    double *v_all_sorted = (double *)malloc(2 * npts * sizeof(double));

    // Combine initial and final data for percentile calculation
    for (i = 0; i < npts; i++) {
        r_all_sorted[i] = r_initial[i];
        r_all_sorted[npts + i] = r_final[i];
        v_all_sorted[i] = fabs(v_initial[i]) * (1.0 / kmsec_to_kpcmyr); // Convert to km/s
        v_all_sorted[npts + i] = fabs(v_final[i]) * (1.0 / kmsec_to_kpcmyr); // Convert to km/s
    }

    // Sort arrays to find percentiles
    qsort(r_all_sorted, 2 * npts, sizeof(double), double_cmp);
    qsort(v_all_sorted, 2 * npts, sizeof(double), double_cmp);

    // Calculate 99th percentile with 1.2x multiplier for padding
    int p99_index = (int)(0.99 * (2 * npts - 1));
    double max_r_all = r_all_sorted[p99_index] * 1.2;
    double max_v_all = v_all_sorted[p99_index] * 1.2;

    // Ensure minimum ranges for very concentrated distributions
    if (max_r_all < 50.0) max_r_all = 50.0;   // Minimum 50 kpc
    if (max_v_all < 50.0) max_v_all = 50.0;   // Minimum 50 km/s

    // Log the calculated ranges
    if (g_enable_logging) {
        log_message("INFO", "2D Histogram dynamic ranges: r=[0, %.1f] kpc, v=[0, %.1f] km/s", 
                    max_r_all, max_v_all);
    }

    free(r_all_sorted);
    free(v_all_sorted);

    /**
     * @brief Define histogram parameters with 400x400 bins for higher resolution.
     * @details Bin widths are calculated dynamically based on the percentile ranges
     * to ensure optimal coverage of the particle distribution.
     */
    #define HIST_NBINS 400

    double rbin_width = max_r_all / HIST_NBINS;
    double vbin_width = max_v_all / HIST_NBINS;

    double bin_width = max_r_all / HIST_NBINS;  // For 1D histogram

    /**
     * @brief Generate and write 2D phase-space histograms (radius vs. velocity magnitude).
     * @details Calculates 2D histograms of particle counts in (r, |v|) bins for both the
     *          initial (t=0, after IC generation and any stripping/conversion) and final
     *          (end of simulation) particle distributions. The bin ranges are dynamically
     *          determined based on the 99th percentile of the combined initial and final
     *          radial positions and velocity magnitudes, with a 1.2x padding factor.
     *          Outputs `2d_hist_initial.dat` and `2d_hist_final.dat`.
     */
    int hist_initial[HIST_NBINS][HIST_NBINS];
    memset(hist_initial, 0, sizeof(hist_initial));
    int hist_final[HIST_NBINS][HIST_NBINS];
    memset(hist_final, 0, sizeof(hist_final));

    for (i = 0; i < npts; i++)
    {
        int rbini = (int)(r_initial[i] / rbin_width);
        int vbini = (int)(fabs(v_initial[i]) * (1.0 / kmsec_to_kpcmyr) / vbin_width);
        if (rbini < HIST_NBINS && vbini < HIST_NBINS && rbini >= 0 && vbini >= 0)
            hist_initial[rbini][vbini]++;

        rbini = (int)(r_final[i] / rbin_width);
        vbini = (int)(fabs(v_final[i]) * (1.0 / kmsec_to_kpcmyr) / vbin_width);
        if (rbini < HIST_NBINS && vbini < HIST_NBINS && rbini >= 0 && vbini >= 0)
            hist_final[rbini][vbini]++;
    }

    // Write initial 2D histogram if simulation was run
    if (!skip_file_writes)
    {
        char suffixed_filename[256];
        get_suffixed_filename("data/2d_hist_initial.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int rr = 0; rr < HIST_NBINS; rr++)
        {
            for (int vv = 0; vv < HIST_NBINS; vv++)
            {
                fprintf_bin(fp, "%f %f %d\n", (rr + 0.5) * rbin_width, (vv + 0.5) * vbin_width, hist_initial[rr][vv]);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    // Write final 2D histogram if simulation was run
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/2d_hist_final.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int rr = 0; rr < HIST_NBINS; rr++)
        {
            for (int vv = 0; vv < HIST_NBINS; vv++)
            {
                fprintf_bin(fp, "%f %f %d\n", (rr + 0.5) * rbin_width, (vv + 0.5) * vbin_width, hist_final[rr][vv]);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    /**
     * @brief Generate and write 1D radial distribution histograms.
     * @details Calculates 1D histograms of particle counts in radial bins for both the
     *          initial and final particle distributions. Uses the same dynamically determined
     *          radial binning as the 2D histograms.
     *          Outputs `combined_histogram.dat` with columns: r_bin_center, count_initial, count_final.
     */
    int hist_i[HIST_NBINS];
    memset(hist_i, 0, sizeof(hist_i));
    int hist_f[HIST_NBINS];
    memset(hist_f, 0, sizeof(hist_f));

    for (i = 0; i < npts; i++)
    {
        int b = (int)(r_initial[i] / bin_width);
        if (b < HIST_NBINS && b >= 0)
            hist_i[b]++;
        b = (int)(r_final[i] / bin_width);
        if (b < HIST_NBINS && b >= 0)
            hist_f[b]++;
    }

    // Write combined 1D radius histogram if simulation was run
    if (!skip_file_writes)
    {
        char filename[256];
        get_suffixed_filename("data/combined_histogram.dat", 1, filename, sizeof(filename));
        fp = fopen(filename, "wb"); // Binary mode for fprintf_bin
        for (i = 0; i < HIST_NBINS; i++)
        {
            double bin_center = (i + 0.5) * bin_width;
            fprintf_bin(fp, "%f %d %d\n", bin_center, hist_i[i], hist_f[i]);
        }
        fclose(fp);
    } // End skip_file_writes block.

    /**
     * @brief Write trajectory data for a selection of low-original-ID particles.
     * @details Outputs the time evolution of radius, radial velocity, and the radial
     *          direction cosine (mu = v_rad / v_total) for the first `num_traj_particles`
     *          (typically 10, or fewer if npts < 10) particles, identified by their
     *          final rank ID after any stripping and remapping.
     *          The data is written to `trajectories.dat`.
     */
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/trajectories.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int step = 0; step < Ntimes; step++)
        {
            fprintf_bin(fp, "%f", step * dt);
            for (int p = 0; p < num_traj_particles; p++)
            {
                fprintf_bin(fp, " %f %f %f", trajectories[p][step], velocities_arr[p][step], mu_arr[p][step]);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    /**
     * @brief Write trajectory data for the particle with final rank ID 0.
     * @details Outputs the time evolution of radius, radial velocity, and mu for the
     *          particle that ended up with rank ID 0. This is a subset of the data
     *          in `trajectories.dat`. Written to `single_trajectory.dat`.
     */
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/single_trajectory.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int step = 0; step < Ntimes; step++)
        {
            fprintf_bin(fp, "%f %f %f %f\n", step * dt, trajectories[0][step], velocities_arr[0][step], mu_arr[0][step]);
        }
        fclose(fp);
    } // End skip_file_writes block.

    /**
     * @brief Write energy and angular momentum evolution for low-original-ID particles.
     * @details Outputs the time evolution of the current relative energy (E_cur), initial
     *          relative energy (E_i), current angular momentum (L_cur), and initial angular
     *          momentum (L_i) for the same set of `num_traj_particles` tracked for `trajectories.dat`.
     *          Written to `energy_and_angular_momentum_vs_time.dat`.
     */
    // Write energy/angular momentum evolution for low-ID particles if simulation was run
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/energy_and_angular_momentum_vs_time.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int step = 0; step < Ntimes; step++)
        {
            fprintf_bin(fp, "%f", step * dt);
            for (int p = 0; p < num_traj_particles; p++)
            {
                double E_i = E_i_arr[p];
                double l_i = L_i_arr[p];
                double Ecur = E_arr[p][step];
                double lcur = L_arr[p][step];
                fprintf_bin(fp, " %f %f %f %f", Ecur, E_i, lcur, l_i);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    {
        /**
         * @brief Write trajectory data (radius, energy, angular momentum) for selected low-L particles.
         * @details Outputs the time evolution of radius, relative energy, and angular momentum
         *          for `nlowest` particles selected based on their initial angular momentum
         *          (either lowest absolute L or closest to a reference L, per `use_closest_to_Lcompare`).
         *          The specific particles are stored in the `chosen` array (by their original IDs).
         *          Written to `lowest_l_trajectories.dat`.
         */
        // Write trajectories for selected lowest-L particles if simulation was run
        if (!skip_file_writes)
        {
            char suffixed_filename[256];
            get_suffixed_filename("data/lowest_l_trajectories.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            FILE *fp_lowest = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
            if (!fp_lowest)
            {
                fprintf(stderr, "Error: cannot open data/lowest_l_trajectories.dat\n");
                CLEAN_EXIT(1);
            }

            // Write data for Ntimes steps:
            for (int step = 0; step < Ntimes; step++)
            {
                double tval = step * dt;
                fprintf_bin(fp_lowest, "%f", tval);
                for (int p = 0; p < nlowest; p++)
                {
                    double rr = lowestL_r[p][step];
                    double Ecur = lowestL_E[p][step];
                    double lcur = lowestL_L[p][step];
                    fprintf_bin(fp_lowest, " %f %f %f", rr, Ecur, lcur);
                }
                fprintf_bin(fp_lowest, "\n");
            }
            fclose(fp_lowest);
        } // Close if (.skip_file_writes)
    }

    // Free lowest-L tracking arrays
    for (int p = 0; p < nlowest; p++)
    {
        free(lowestL_r[p]);
        free(lowestL_E[p]);
        free(lowestL_L[p]);
    }
    free(lowestL_r);
    free(lowestL_E);
    free(lowestL_L);



    int snapshot_steps[noutsnaps];
    // Determine the timesteps corresponding to the desired snapshot outputs
    for (int s = 0; s < noutsnaps; s++)
    {
        snapshot_steps[s] = (int)floor(
            s * (total_writes - 1) / (double)(noutsnaps - 1));
        log_message("DEBUG", "snapshot_steps[%d] = %d", s, snapshot_steps[s]);
    }

    if (g_doAllParticleData)
    {
        size_t total_outsnaps_size = (size_t)npts * noutsnaps; // Use size_t for large allocations
        log_message("INFO", "Allocating arrays for %zu particles across %d snapshots (total: %zu elements)",
                    (size_t)npts, noutsnaps, total_outsnaps_size);

        int *Rank_partdata_outsnaps = (int *)malloc(total_outsnaps_size * sizeof(int));
        float *R_partdata_outsnaps = (float *)malloc(total_outsnaps_size * sizeof(float));
        float *Vrad_partdata_outsnaps = (float *)malloc(total_outsnaps_size * sizeof(float));

        if (!Rank_partdata_outsnaps || !R_partdata_outsnaps || !Vrad_partdata_outsnaps)
        {
            printf("[ERROR] Failed to allocate memory for output arrays. Aborting.\n");
            fflush(stdout);
            CLEAN_EXIT(1);
        }

        char apd_filename_for_read[256];
        get_suffixed_filename("data/all_particle_data.dat", 1, apd_filename_for_read, sizeof(apd_filename_for_read));

        double fixed_bin_width = 0.0; // Initialized, set during first snapshot processing

        int start_index = 0; // Default start index for snapshot processing loop
        if (g_doRestart)
        {
            start_index = find_last_processed_snapshot(snapshot_steps, noutsnaps);
            if (start_index == -2) // Indicates all snapshots already processed
            {
                log_message("INFO", "All Rank files already exist for suffix '%s'. Skipping snapshot regeneration.", g_file_suffix);
                goto cleanup_partdata_outsnaps; // Jump past the processing loop
            }
            else if (start_index < 0) // Indicates error or no files found
            {
                start_index = 0; // Default to starting from the beginning
            }
            // Otherwise, start_index is the index of the first snapshot *to be* processed
            log_message("INFO", "Restart: Starting snapshot processing from index %d.", start_index);
        }

        printf("Starting parallel processing of %d snapshots (from %d to %d) using %d threads\n\n",
               noutsnaps - start_index, start_index, noutsnaps - 1, omp_get_max_threads());
        log_message("INFO", "Starting parallel processing with %d threads for %d snapshots (from index %d to %d)",
                    omp_get_max_threads(), noutsnaps - start_index, start_index, noutsnaps - 1);

#pragma omp parallel for schedule(static, 1) ordered
        for (int s = start_index; s < noutsnaps; s++)
        {
            int snap = snapshot_steps[s];

            float *tmpL_partdata_snap = (float *)malloc(npts * sizeof(float));
            int *tmpRank_partdata_snap = (int *)malloc(npts * sizeof(int));
            float *tmpR_partdata_snap = (float *)malloc(npts * sizeof(float));
            float *tmpV_partdata_snap = (float *)malloc(npts * sizeof(float));

            if (!tmpL_partdata_snap || !tmpRank_partdata_snap ||
                !tmpR_partdata_snap || !tmpV_partdata_snap)
            {
                fprintf(stderr, "Error: out of memory in parallel loop.\n");
                CLEAN_EXIT(1);
            }

            retrieve_all_particle_snapshot(
                apd_filename_for_read,
                snap,       // Which snap to read.
                npts,       // #particles
                block_size, // Block size.
                tmpL_partdata_snap,
                tmpRank_partdata_snap,
                tmpR_partdata_snap,
                tmpV_partdata_snap);

/**
 * Process snapshot data in order.
 * The #pragma omp ordered block ensures the copying into output arrays
 * (Rank_partdata_outsnaps, etc.) and status messages happen sequentially.
 */
#pragma omp ordered
            {
                // Display status message sequentially.
                printf("Processing snapshot %d...\n", snap);

                // Check array validity before copying
                if (!Rank_partdata_outsnaps || !R_partdata_outsnaps || !Vrad_partdata_outsnaps)
                {
                    log_message("ERROR", "Thread %d: Output arrays not properly allocated for snapshot %d",
                                omp_get_thread_num(), snap);
                }
                else if (!tmpRank_partdata_snap || !tmpR_partdata_snap || !tmpV_partdata_snap)
                {
                    log_message("ERROR", "Thread %d: Input arrays not properly allocated for snapshot %d",
                                omp_get_thread_num(), snap);
                }
                else
                {
                    // Copy data from thread-local buffers to the large shared output arrays.
                    for (int ii = 0; ii < npts; ii++)
                    {
                        if ((size_t)(s * npts + ii) < total_outsnaps_size)
                        {
                            Rank_partdata_outsnaps[s * npts + ii] = tmpRank_partdata_snap[ii];
                            R_partdata_outsnaps[s * npts + ii] = tmpR_partdata_snap[ii];
                            Vrad_partdata_outsnaps[s * npts + ii] = tmpV_partdata_snap[ii];
                        }
                        else
                        {
                            log_message("ERROR", "Thread %d: Index %zu out of bounds (%zu) in ordered copy for snapshot %d", 
                                        omp_get_thread_num(), (size_t)(s * npts + ii), total_outsnaps_size, snap);
                            break;
                        }
                    }
                }
            }

            /** @brief Build "unsorted" arrays (Rank, Mass, R, Vrad, L) from local snapshot data. */
            log_message("INFO", "Thread %d: Allocating arrays for snapshot %d",
                        omp_get_thread_num(), snap);

            int *Rank_unsorted = (int *)malloc(npts * sizeof(int));
            double *Mass_unsorted = (double *)malloc(npts * sizeof(double));
            double *R_unsorted = (double *)malloc(npts * sizeof(double));
            double *Vrad_unsorted = (double *)malloc(npts * sizeof(double));
            double *L_unsorted = (double *)malloc(npts * sizeof(double));

            if (!Rank_unsorted || !Mass_unsorted || !R_unsorted || !Vrad_unsorted || !L_unsorted)
            {
                printf("[ERROR] Thread %d: Memory allocation failed for snapshot %d\n",
                       omp_get_thread_num(), snap);
                // Free any memory that was allocated.
                if (Rank_unsorted)
                    free(Rank_unsorted);
                if (Mass_unsorted)
                    free(Mass_unsorted);
                if (R_unsorted)
                    free(R_unsorted);
                if (Vrad_unsorted)
                    free(Vrad_unsorted);
                if (L_unsorted)
                    free(L_unsorted);
                continue; // Skip to next snapshot.
            }

            for (int ii = 0; ii < npts; ii++)
            {
                int rankval = tmpRank_partdata_snap[ii];
                double massv = (rankval + 1) * deltaM;

                Rank_unsorted[ii] = rankval;
                Mass_unsorted[ii] = massv;
                R_unsorted[ii] = (double)tmpR_partdata_snap[ii];
                Vrad_unsorted[ii] = (double)tmpV_partdata_snap[ii];
                L_unsorted[ii] = (double)tmpL_partdata_snap[ii];
            }

            // Sort by Rank
            // --- Prepare data structure for sorting by radius ---
            /** @brief Allocate `partarr` (array of struct PartData) for sorting snapshot data. */
            struct PartData *partarr = malloc(npts * sizeof(struct PartData));
            if (!partarr)
            {
                log_message("ERROR", "Thread %d: Failed to allocate partarr for snapshot %d",
                            omp_get_thread_num(), snap);
                // Free allocated memory.
                free(Rank_unsorted);
                free(Mass_unsorted);
                free(R_unsorted);
                free(Vrad_unsorted);
                free(L_unsorted);
                continue;
            }

            struct PartData *local_partarr = (struct PartData *)malloc(npts * sizeof(struct PartData));
            if (!local_partarr)
            {
                log_message("ERROR", "Thread %d: Failed to allocate local_partarr for snapshot %d",
                            omp_get_thread_num(), snap);
                free(partarr);
                continue;
            }

            for (int ii = 0; ii < npts; ii++)
            {
                // Check for invalid values to prevent sort issues.
                float rad_val = tmpR_partdata_snap[ii];
                float vrad_val = tmpV_partdata_snap[ii];
                float angmom_val = tmpL_partdata_snap[ii];

                // Handle potential NaN values before assigning to struct
                if (rad_val != rad_val) { // Check for NaN (safe with fast-math)
                    log_message("WARNING", "Thread %d: NaN radius at index %d for snapshot %d, replaced with 0", 
                                omp_get_thread_num(), ii, snap);
                    rad_val = 0.0f;
                }
                if (vrad_val != vrad_val) vrad_val = 0.0f; // Check for NaN (safe with fast-math)
                if (angmom_val != angmom_val) angmom_val = 0.0f; // Check for NaN (safe with fast-math)

                local_partarr[ii].rank = tmpRank_partdata_snap[ii];
                local_partarr[ii].rad = rad_val;
                local_partarr[ii].vrad = vrad_val;
                local_partarr[ii].angmom = angmom_val;
                local_partarr[ii].original_index = ii; // Store original index before sort
            }

            memcpy(partarr, local_partarr, npts * sizeof(struct PartData));
            free(local_partarr);

            log_message("INFO", "Thread %d: Sorting partarr by radius for snapshot %d", 
                        omp_get_thread_num(), snap);

            // Sort particle data by radius for snapshot processing
            sort_by_rad(partarr, npts);

            // If processing the first snapshot (index start_index), write lowest radius IDs
            if (s == start_index)
            {
                // Skip writing if file writes are disabled (e.g., restart post-processing only)
                if (!skip_file_writes)
                {
                    char fname_lowest_ids[256];
                    get_suffixed_filename("data/lowest_radius_ids.dat", 1, fname_lowest_ids, sizeof(fname_lowest_ids));
                    log_message("INFO", "Thread %d: Writing lowest radius IDs to %s",
                                omp_get_thread_num(), fname_lowest_ids);

                    FILE *id_file = fopen(fname_lowest_ids, "wb"); // Binary mode for fprintf_bin
                    if (id_file)
                    {
                        fprintf_bin(id_file, "# ID  Initial_Radius\n"); // Header (binary float for radius)

                        int num_tracked = (npts < 1000) ? npts : 1000; // Track up to 1000 particles

                        for (int i = 0; i < num_tracked; i++)
                        {
                            int original_id = partarr[i].original_index;
                            float initial_radius = partarr[i].rad;
                            fprintf_bin(id_file, "%d %f\n", original_id, initial_radius);
                        }
                        fclose(id_file);
                        log_message("INFO", "Thread %d: Successfully wrote %d lowest-radius particle IDs to %s",
                                    omp_get_thread_num(), num_tracked, fname_lowest_ids);
                    }
                    else
                    {
                        log_message("ERROR", "Thread %d: Failed to open %s for writing",
                                    omp_get_thread_num(), fname_lowest_ids);
                    }
                }
                else
                {
                    log_message("INFO", "Thread %d: Skipping lowest radius ID file creation in restart mode",
                                omp_get_thread_num());
                }
            }

            // Allocate arrays to store the data sorted by radius
            int *Rank_sorted = (int *)malloc(npts * sizeof(int));
            double *Mass_sorted = (double *)malloc(npts * sizeof(double));
            double *R_sorted = (double *)malloc(npts * sizeof(double));
            double *Vrad_sorted = (double *)malloc(npts * sizeof(double));
            double *L_sorted = (double *)malloc(npts * sizeof(double));

            if (!Rank_sorted || !Mass_sorted || !R_sorted || !Vrad_sorted || !L_sorted)
            {
                log_message("ERROR", "Thread %d: Failed to allocate sorted arrays for snapshot %d",
                            omp_get_thread_num(), snap);
                // Free allocated memory.
                if (Rank_sorted)
                    free(Rank_sorted);
                if (Mass_sorted)
                    free(Mass_sorted);
                if (R_sorted)
                    free(R_sorted);
                if (Vrad_sorted)
                    free(Vrad_sorted);
                if (L_sorted)
                    free(L_sorted);
                free(partarr);
                free(Rank_unsorted);
                free(Mass_unsorted);
                free(R_unsorted);
                free(Vrad_unsorted);
                free(L_unsorted);
                continue;
            }

            // Populate the sorted arrays using the sorted partarr
            for (int ii = 0; ii < npts; ii++)
            {
                partarr[ii].rank = ii; // Update rank based on sorted position
                int r_val = partarr[ii].rank; // Rank is now simply the index 'ii'
                Rank_sorted[ii] = r_val;
                Mass_sorted[ii] = (r_val + 1) * deltaM; // Mass enclosed up to this rank
                R_sorted[ii] = (double)partarr[ii].rad;
                Vrad_sorted[ii] = (double)partarr[ii].vrad;
                // Retrieve L from the unsorted array using the original index stored in partarr
                L_sorted[ii] = L_unsorted[partarr[ii].original_index];
            }

            /** @brief Set fixed bin width for density calculation using the first processed snapshot. */
            log_message("INFO", "Thread %d: Checking fixed bin width for snapshot %d", 
                        omp_get_thread_num(), snap);
            if (s == start_index) // Only calculate on the first snapshot processed in this run
            {
                // Determine number of bins based on particle count (e.g., proportional to npts^(1/3))
                int num_bins_fixed = ceil(100 * pow(npts / 10000.0, 0.3333333333));
                if (num_bins_fixed < 1) num_bins_fixed = 1;
                // Use the maximum radius found in this first snapshot
                double r_max_first = R_sorted[npts - 1];
                if (r_max_first <= 0 || num_bins_fixed <= 0) {
                    log_message("ERROR", "Thread %d: Invalid r_max (%f) or bins (%d) for fixed_bin_width calc", 
                                omp_get_thread_num(), r_max_first, num_bins_fixed);
                    // Set a default or handle error
                    fixed_bin_width = 1.0;
                } else {
                    fixed_bin_width = r_max_first / num_bins_fixed;
                }
                log_message("INFO", "Thread %d: Calculated fixed_bin_width=%f for snapshot %d (r_max=%f, bins=%d)",
                            omp_get_thread_num(), fixed_bin_width, snap, r_max_first, num_bins_fixed);
            }

            double *density_sorted = NULL;

            log_message("INFO", "Thread %d: Starting density calculation for snapshot %d",
                        omp_get_thread_num(), snap);
            {
                // Free previous allocation if it exists (safety measure)
                if (density_sorted != NULL)
                {
                    free(density_sorted);
                    density_sorted = NULL; // Avoid dangling pointer
                    log_message("INFO", "Thread %d: Freed previous density_sorted for snapshot %d",
                                omp_get_thread_num(), snap);
                }

                log_message("INFO", "Thread %d: Allocating density_sorted for snapshot %d",
                            omp_get_thread_num(), snap);
                density_sorted = malloc(npts * sizeof(double));
                if (!density_sorted)
                {
                    fprintf(stderr, "Error: Failed to allocate memory for density_sorted\n");
                    // Cleanup memory allocated within this snapshot's loop iteration
                    free(tmpL_partdata_snap); free(tmpRank_partdata_snap); free(tmpR_partdata_snap); free(tmpV_partdata_snap);
                    free(Rank_unsorted); free(Mass_unsorted); free(R_unsorted); free(Vrad_unsorted); free(L_unsorted);
                    free(partarr);
                    free(Rank_sorted); free(Mass_sorted); free(R_sorted); free(Vrad_sorted); free(L_sorted);
                    continue; // Proceed to the next snapshot
                }

                double bandwidth_factor = 0.08;
                log_message("INFO", "Thread %d: Checking R_sorted monotonicity for snapshot %d",
                            omp_get_thread_num(), snap);
                int r_violations = 0;
                for (int i = 1; i < npts; i++)
                {
                    if (R_sorted[i] <= R_sorted[i - 1])
                        r_violations++;
                }
                log_message("INFO", "Thread %d: Found %d radius violations for snapshot %d",
                            omp_get_thread_num(), r_violations, snap);

                double *R_filtered = NULL;
                double *Mass_filtered = NULL;
                int filtered_count = 0;

                if (r_violations > 0)
                {
                    // Allocate arrays to hold strictly monotonic radius/mass data
                    R_filtered = malloc(npts * sizeof(double));
                    Mass_filtered = malloc(npts * sizeof(double));
                    if (!R_filtered || !Mass_filtered) { 
                        log_message("ERROR", "Thread %d: Failed to allocate filtered arrays for snapshot %d", 
                            omp_get_thread_num(), snap);
                        continue; 
                    }

                    // Copy only the points that maintain strict monotonicity
                    R_filtered[0] = R_sorted[0];
                    Mass_filtered[0] = Mass_sorted[0];
                    filtered_count = 1;

                    for (int i = 1; i < npts; i++)
                    {
                        if (R_sorted[i] > R_filtered[filtered_count - 1])
                        {
                            R_filtered[filtered_count] = R_sorted[i];
                            Mass_filtered[filtered_count] = Mass_sorted[i];
                            filtered_count++;
                        }
                    }
                }
                else
                {
                    // If no violations, point directly to the sorted data (no copy needed)
                    R_filtered = R_sorted;
                    Mass_filtered = Mass_sorted;
                    filtered_count = npts;
                }

                // Decimate the filtered data for spline interpolation efficiency
                int decimated_size = imin((int)ceil(npts / 100.0), 30000); // Limit max size
                if (decimated_size <= 0) decimated_size = 1; // Ensure at least one point
                double *R_decimated = malloc(decimated_size * sizeof(double));
                double *Mass_decimated = malloc(decimated_size * sizeof(double));
                if (!R_decimated || !Mass_decimated) { 
                    log_message("ERROR", "Thread %d: Failed to allocate decimated arrays for snapshot %d", 
                        omp_get_thread_num(), snap);
                    if (r_violations > 0) { free(R_filtered); free(Mass_filtered); }
                    continue; 
                }

                if (filtered_count <= decimated_size)
                {
                    // Use all filtered points if fewer than desired decimated size
                    decimated_size = filtered_count;
                    memcpy(R_decimated, R_filtered, filtered_count * sizeof(double));
                    memcpy(Mass_decimated, Mass_filtered, filtered_count * sizeof(double));
                }
                else
                {
                    // Sample evenly from the filtered data
                    double step = (double)(filtered_count - 1) / (decimated_size - 1);
                    for (int i = 0; i < decimated_size; i++)
                    {
                        int idx = (int)round(i * step); // Use round for potentially better sampling
                        if (idx >= filtered_count)
                            idx = filtered_count - 1;
                        R_decimated[i] = R_filtered[idx];
                        Mass_decimated[i] = Mass_filtered[idx];
                    }
                }

                // Validate the range and finiteness of the decimated radius data for spline
                double min_r_check = (decimated_size > 0) ? R_decimated[0] : -1.0;
                double max_r_check = (decimated_size > 0) ? R_decimated[decimated_size - 1] : -1.0;
                // Check for invalid size, non-positive min, inverted range, or non-finite values (NaN or Inf)
                if (decimated_size <= 0 || min_r_check <= 0.0 || max_r_check <= min_r_check ||
                    ((min_r_check != min_r_check) || !(fabs(min_r_check) <= DBL_MAX)) || // Check !isfinite(min_r_check)
                    ((max_r_check != max_r_check) || !(fabs(max_r_check) <= DBL_MAX))    // Check !isfinite(max_r_check)
                   ) {
                    log_message("ERROR", "Thread %d: Invalid radius range [%f, %f] with %d points after decimation for snapshot %d", 
                                omp_get_thread_num(), min_r_check, max_r_check, decimated_size, snap);
                    free(R_decimated); free(Mass_decimated);
                    if (r_violations > 0) { free(R_filtered); free(Mass_filtered); } // Free if allocated
                    free(density_sorted); density_sorted = NULL;
                    continue;
                }

                // Define the range and parameters for the uniform log-spaced grid
                double min_r = R_decimated[0];
                double max_r = R_decimated[decimated_size - 1] * 1.04; // Extend range slightly
                double log_min_r = log10(min_r);
                double log_max_r = log10(max_r);

                // Define size for the uniform grid (used for convolution)
                int grid_size = 131072; // Power of 2 often good for FFT, but direct used here
                if (grid_size <= 1) grid_size = 2; // Ensure at least 2 points
                double dlog = (log_max_r - log_min_r) / (grid_size - 1);

                // Allocate arrays for the uniform log-spaced grid
                double *r_grid = malloc(grid_size * sizeof(double));
                double *log_r_grid = malloc(grid_size * sizeof(double));
                double *mass_grid = malloc(grid_size * sizeof(double));
                double *density_grid = malloc(grid_size * sizeof(double));
                if (!r_grid || !log_r_grid || !mass_grid || !density_grid) { 
                    log_message("ERROR", "Thread %d: Failed to allocate grid arrays for snapshot %d", 
                        omp_get_thread_num(), snap);
                    // Free previously allocated resources
                    free(R_decimated); free(Mass_decimated);
                    if (r_violations > 0) { free(R_filtered); free(Mass_filtered); }
                    if (r_grid) free(r_grid);
                    if (log_r_grid) free(log_r_grid);
                    if (mass_grid) free(mass_grid);
                    if (density_grid) free(density_grid);
                    free(density_sorted); density_sorted = NULL;
                    continue; 
                }

                // Populate the log-spaced grid coordinates
                for (int i = 0; i < grid_size; i++)
                {
                    log_r_grid[i] = log_min_r + i * dlog;
                    r_grid[i] = pow(10.0, log_r_grid[i]);
                }

                /** @brief Ensure r_grid is strictly monotonic for GSL spline init. */
                int rgrid_corrections_made = 0;
                for (int i = 1; i < grid_size; i++) {
                    if (r_grid[i] <= r_grid[i - 1]) {
                        // Ensure strict monotonicity with small absolute increment
                        r_grid[i] = r_grid[i - 1] + 1e-12;
                        rgrid_corrections_made++;
                    }
                }
                if (rgrid_corrections_made > 0) {
                     log_message("WARNING", "Thread %d made %d corrections to r_grid for monotonicity in snapshot %d", omp_get_thread_num(), rgrid_corrections_made, snap);
                }

                // Interpolate mass from decimated data onto the uniform log-spaced grid
                gsl_interp_accel *acc = NULL;
                gsl_spline *mass_spline = NULL;

#pragma omp critical(gsl_accel)
                {
                    acc = gsl_interp_accel_alloc();
                }

                if (!acc)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate GSL interp accel for snapshot %d",
                                omp_get_thread_num(), snap);
                    continue;
                }

                /** @brief Ensure R_decimated is strictly monotonic for GSL spline init. */
                int corrections_made = 0;
                for (int i = 1; i < decimated_size; i++) {
                    if (R_decimated[i] <= R_decimated[i - 1]) {
                        // Ensure a strictly larger value using a small absolute increment
                        R_decimated[i] = R_decimated[i - 1] + 1e-12; // Add a small absolute value
                        corrections_made++;
                    }
                }
                if (corrections_made > 0) {
                     log_message("WARNING", "Thread %d made %d corrections to R_decimated for monotonicity in snapshot %d", omp_get_thread_num(), corrections_made, snap);
                }

#pragma omp critical(gsl_mass_spline)
                {
                    mass_spline = gsl_spline_alloc(gsl_interp_cspline, decimated_size);
                    if (mass_spline)
                    {
                        gsl_spline_init(mass_spline, R_decimated, Mass_decimated, decimated_size);
                    }
                }

                if (!mass_spline)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate or initialize mass spline for snapshot %d",
                                omp_get_thread_num(), snap);
                    gsl_interp_accel_free(acc);
                    continue;
                }

                for (int i = 0; i < grid_size; i++)
                {
                    if (r_grid[i] < R_decimated[0])
                    {
                        mass_grid[i] = Mass_decimated[0]; // Extrapolate using first point
                    }
                    else if (r_grid[i] > R_decimated[decimated_size - 1])
                    {
                        mass_grid[i] = Mass_decimated[decimated_size - 1]; // Extrapolate using last point
                    }
                    else
                    {
#pragma omp critical(gsl_mass_eval)
                        {
                            // Use _e version for error checking if needed
                            mass_grid[i] = gsl_spline_eval(mass_spline, r_grid[i], acc);
                        }
                    }
                }

                // Calculate density = dM/dr on the uniform grid using central differences
                for (int i = 1; i < grid_size - 1; i++)
                {
                    // Avoid division by zero if grid points coincide
                    double dr = r_grid[i + 1] - r_grid[i - 1];
                    if (dr > 1e-15) {
                        double dM = mass_grid[i + 1] - mass_grid[i - 1];
                        density_grid[i] = dM / dr;
                    } else {
                        // Fallback for coincident points (e.g., use forward/backward difference or average)
                        density_grid[i] = (i > 1) ? density_grid[i-1] : 0.0; // Simple fallback
                    }

                    // Ensure non-negative density, apply floor
                    if (density_grid[i] < 1e-10)
                    {
                        density_grid[i] = 1e-10;
                    }
                }

                // Handle endpoints using one-sided difference or extrapolation
                if (grid_size >= 2) {
                   density_grid[0] = density_grid[1]; // Simple extrapolation
                   density_grid[grid_size - 1] = density_grid[grid_size - 2]; // Simple extrapolation
                } else if (grid_size == 1) {
                   density_grid[0] = 0.0; // Or some default
                }

                log_message("INFO", "Thread %d: Allocating memory for density smoothing arrays for snapshot %d",
                            omp_get_thread_num(), snap);
                double *density_smoothed_direct = malloc(npts * sizeof(double));
                double *direct_grid_result = malloc(grid_size * sizeof(double));

                if (!density_smoothed_direct || !direct_grid_result)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate memory for smoothing arrays for snapshot %d",
                                omp_get_thread_num(), snap);
                    if (density_smoothed_direct)
                        free(density_smoothed_direct);
                    if (direct_grid_result)
                        free(direct_grid_result);
                    continue;
                }

                // Calculate smoothing kernel width (sigma) based on particle count
                double sigma = bandwidth_factor * pow(npts / 10000.0, -0.40);
                double sigma_log = sigma; // Apply smoothing in log-space

                // Perform Gaussian convolution on the uniform density grid
                if (!density_grid || !log_r_grid || !direct_grid_result)
                {
                    log_message("ERROR", "Thread %d: NULL arrays detected before gaussian_convolution for snapshot %d", 
                                omp_get_thread_num(), snap);
                    // Clean up resources
                    if (density_grid) free(density_grid);
                    if (log_r_grid) free(log_r_grid);
                    if (direct_grid_result) free(direct_grid_result);
                    continue;
                }

                // Use critical section for FFTW thread safety if FFT is used internally
#pragma omp critical(gaussian_convolution)
                {
                    gaussian_convolution(density_grid, grid_size, log_r_grid, sigma_log, direct_grid_result);
                }

                // Create a GSL spline from the smoothed density on the uniform grid
                gsl_spline *density_spline = NULL;

#pragma omp critical(gsl_alloc)
                {
                    density_spline = gsl_spline_alloc(gsl_interp_cspline, grid_size);
                }

                if (!density_spline)
                {
                    log_message("ERROR", "Thread %d: Failed to allocate GSL spline for snapshot %d",
                                omp_get_thread_num(), snap);
                    // Clean up and skip.
                    free(density_smoothed_direct);
                    free(direct_grid_result);
                    continue;
                }

#pragma omp critical(gsl_init)
                {
                    gsl_spline_init(density_spline, r_grid, direct_grid_result, grid_size);
                }

                // Temporarily disable GSL default error handler to manage errors locally
                gsl_error_handler_t *old_handler = gsl_set_error_handler_off();

                // Interpolate smoothed density back onto the original sorted particle radii (R_sorted)
                for (int i = 0; i < npts; i++)
                {
                    double r_val = R_sorted[i]; // Target radius

                    if (r_val <= r_grid[0])
                    {
                        density_smoothed_direct[i] = direct_grid_result[0]; // Extrapolate below range
                    }
                    else if (r_val >= r_grid[grid_size - 1])
                    {
                        density_smoothed_direct[i] = direct_grid_result[grid_size - 1]; // Extrapolate above range
                    }
                    else
                    {
                        // For values within range, use spline interpolation with error checking.
                        int status = 0;
                        double result = 0.0;

// Try to evaluate with error checking.
#pragma omp critical(gsl_eval)
                        {
                            status = gsl_spline_eval_e(density_spline, r_val, acc, &result);
                        }

                        if (status != GSL_SUCCESS)
                        {
                            // Interp_errors++; // Commented out since variable is not used.

                            // Fall back to linear interpolation.
                            int idx_low = 0;

                            // Binary search to find the lower index.
                            int left = 0;
                            int right = grid_size - 1;

                            while (left <= right)
                            {
                                int mid = left + (right - left) / 2;

                                if (r_grid[mid] <= r_val && (mid == grid_size - 1 || r_grid[mid + 1] > r_val))
                                {
                                    idx_low = mid;
                                    break;
                                }
                                else if (r_grid[mid] > r_val)
                                {
                                    right = mid - 1;
                                }
                                else
                                {
                                    left = mid + 1;
                                }
                            }

                            int idx_high = idx_low + 1;

                            // Safety check.
                            if (idx_high >= grid_size)
                            {
                                idx_high = grid_size - 1;
                                idx_low = idx_high - 1;
                            }

                            // Linear interpolation.
                            double t = (r_val - r_grid[idx_low]) / (r_grid[idx_high] - r_grid[idx_low]);
                            result = direct_grid_result[idx_low] * (1.0 - t) + direct_grid_result[idx_high] * t;
                        }

                        density_smoothed_direct[i] = result;
                    }
                }

                // Restore the original error handler.
                gsl_set_error_handler(old_handler);

                // Copy the smoothed density values to the output array.
                memcpy(density_sorted, density_smoothed_direct, npts * sizeof(double));

                // Free resources.
                gsl_spline_free(mass_spline);
                gsl_spline_free(density_spline);
                gsl_interp_accel_free(acc);
                free(r_grid);
                free(log_r_grid);
                free(mass_grid);
                free(density_grid);
                free(direct_grid_result);
                free(density_smoothed_direct);
                free(R_decimated);
                free(Mass_decimated);

                // Free filtered arrays if they were allocated.
                if (r_violations > 0)
                {
                    free(R_filtered);
                    free(Mass_filtered);
                }
            }
            // =========================================================================
            // END DENSITY CALCULATION
            // =========================================================================

            // Density calculation complete using fixed binning from R_sorted min to max with (0,0) anchor point

            if (g_doDynPsi)
            {

                double *rrA = (double *)malloc(npts * sizeof(double));
                double *M_rA = (double *)malloc(npts * sizeof(double));

                for (int ii = 0; ii < npts; ii++)
                {
                    M_rA[ii] = Mass_sorted[ii];
                    rrA[ii] = R_sorted[ii];
                }

                double *inv_rA = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    inv_rA[ii] = 1.0 / rrA[ii];
                }
                double *suffix_sum_invrA = (double *)malloc(npts * sizeof(double));
                suffix_sum_invrA[npts - 1] = inv_rA[npts - 1];
                for (int jj = npts - 2; jj >= 0; jj--)
                {
                    suffix_sum_invrA[jj] = suffix_sum_invrA[jj + 1] + inv_rA[jj];
                }

                double *psiAarr = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = rrA[ii];
                    double M_r = M_rA[ii];
                    double outer_sum = 0.0;
                    if (ii < npts - 1)
                        outer_sum = (deltaM)*suffix_sum_invrA[ii + 1];
                    double psiA = G_CONST * ((M_r / rr) + outer_sum);
                    psiA *= VEL_CONV_SQ;
                    psiAarr[ii] = psiA;
                }

                // Compute PsiA(0).
                double sum_invr = 0.0;
                for (int ii = 0; ii < npts; ii++)
                {
                    sum_invr += inv_rA[ii];
                }
                double PsiA0 = G_CONST * deltaM * sum_invr * VEL_CONV_SQ; // PsiA at r=0.

                // No offset needed for PsiA, we keep psiAarr as is.
                // If Psi_theory_0 or offset is mentioned, we do nothing for PsiA.
                // For PsiB we do what original code says (it mentioned offset?).

                // Write PsiA and PsiB to files (as originally)
                char fname_PsiA[256];
                char base_filename[256];
                snprintf(base_filename, sizeof(base_filename), "data/Psi_methodA_t%05d.dat", snap);
                get_suffixed_filename(base_filename, 1, fname_PsiA, sizeof(fname_PsiA));

                FILE *fA = fopen(fname_PsiA, "wb"); // Binary mode for fprintf_bin
                // Write the zero point first.
                double rr_zero = 0.0;
                fprintf_bin(fA, "%f %f\n", rr_zero, PsiA0);
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = rrA[ii];
                    fprintf_bin(fA, "%f %f\n", rr, psiAarr[ii]);
                }
                fclose(fA);

                // Now create a new array with npts+1 points for PsiA to include (0,PsiA0) for spline
                double *rrA_spline = (double *)malloc((npts + 1) * sizeof(double));
                double *psiAarr_spline = (double *)malloc((npts + 1) * sizeof(double));
                rrA_spline[0] = 0.0;
                psiAarr_spline[0] = PsiA0;
                for (int ii = 0; ii < npts; ii++)
                {
                    rrA_spline[ii + 1] = rrA[ii];
                    psiAarr_spline[ii + 1] = psiAarr[ii];
                }

                // Start sketchy sorting hack.

                // Ensure strictly increasing values for spline initialization:
                for (int i = 1; i <= npts; i++)
                {
                    if (rrA_spline[i] <= rrA_spline[i - 1])
                    {
                        rrA_spline[i] = rrA_spline[i - 1] + 1e-5;
                    }
                }

                // End sketchy sorting hack.

                // Create a spline for PsiA(r) including the (0,PsiA0) point
                gsl_interp_accel *PsiAinterp = gsl_interp_accel_alloc();
                gsl_spline *splinePsiA = gsl_spline_alloc(gsl_interp_linear, npts + 1);

                gsl_spline_init(splinePsiA, rrA_spline, psiAarr_spline, npts + 1);

                // Compute PsiA for all unsorted and sorted radii using this spline
                double *PsiA_unsorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr_uf = R_unsorted[ii];
                    PsiA_unsorted[ii] = gsl_spline_eval(splinePsiA, rr_uf, PsiAinterp);
                }

                double *PsiA_sorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr_sf = R_sorted[ii];
                    PsiA_sorted[ii] = gsl_spline_eval(splinePsiA, rr_sf, PsiAinterp);
                }

                // =========================================================================
                // ENERGY CALCULATION FOR PARTICLE DISTRIBUTIONS
                // =========================================================================
                //
                // Calculate total energy for each particle using the formula:
                // E = Ψ - L²/(2r²) - v²/2
                //
                // Components:
                // - Ψ: Gravitational potential (from spline interpolation)
                // - L: Angular momentum
                // - r: Radial position
                // - v: Radial velocity
                //
                // Computed separately for both unsorted and sorted distributions
                double *E_unsorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = R_unsorted[ii];      // Radius
                    double vrad = Vrad_unsorted[ii]; // Radial velocity
                    double l = L_unsorted[ii];       // Angular momentum

                    // Calculate total energy using the orbital energy equation
                    E_unsorted[ii] = PsiA_unsorted[ii]           // Potential energy (Ψ)
                                     - (l * l / (2.0 * rr * rr)) // Rotational energy (L²/2r²)
                                     - 0.5 * vrad * vrad;        // Kinetic energy (v²/2)
                }

                double *E_sorted = (double *)malloc(npts * sizeof(double));
                for (int ii = 0; ii < npts; ii++)
                {
                    double rr = R_sorted[ii];      // Radius
                    double vrad = Vrad_sorted[ii]; // Radial velocity
                    double l = L_sorted[ii];       // Angular momentum

                    // Same energy formula applied to sorted distribution
                    E_sorted[ii] = PsiA_sorted[ii]             // Potential energy (Ψ)
                                   - (l * l / (2.0 * rr * rr)) // Rotational energy (L²/2r²)
                                   - 0.5 * vrad * vrad;        // Kinetic energy (v²/2)
                }

                if (g_doDebug)
                {
                    // =========================================================================
                    // DEBUG ENERGY COMPUTATION - DYNAMIC ANALYSIS
                    // =========================================================================
                    //
                    // Computes dynamic energy for tracked particle before rewriting files.
                    // Uses PsiA_unsorted to calculate energy for the specific debug ID.
                    // This information is used to validate energy conservation during simulation.

                    {
                        /**
                         * TRACKED PARTICLE ENERGY ANALYSIS
                         *
                         * Extract and compute dynamic energy components for a single tracked particle
                         * identified by DEBUG_PARTICLE_ID. Used to validate energy conservation
                         * and compare with theoretical predictions. The components are stored for
                         * post-simulation analysis and visualization.
                         */

                        // Retrieve the particle ID to track from the global constant
                        int debug_id = DEBUG_PARTICLE_ID;

                        // Extract physical properties from the unsorted arrays (original ordering)
                        double r_val = R_unsorted[debug_id];       // Radius
                        double v_val = Vrad_unsorted[debug_id];    // Radial velocity
                        double l_val = L_unsorted[debug_id];       // Angular momentum
                        double psiA_val = PsiA_unsorted[debug_id]; // Potential (newly computed)

                        // Calculate energy components:
                        // Total energy = potential - kinetic
                        double E_dyn = psiA_val - 0.5 * (v_val * v_val + (l_val * l_val) / (r_val * r_val));

                        // Kinetic energy = radial + rotational components
                        double K_dyn = 0.5 * (v_val * v_val + (l_val * l_val) / (r_val * r_val));

                        // Calculate simulation time corresponding to this snapshot
                        // Ensure consistent time basis with approximate energy calculation
                        double sim_time = (double)(snapshot_steps[s]) * dtwrite * dt;

                        // Store energy components in global arrays for later analysis
                        store_debug_dynE_components(
                            s,        // Snapshot index
                            E_dyn,    // Total energy
                            K_dyn,    // Kinetic energy
                            psiA_val, // Potential energy
                            sim_time, // Simulation time
                            r_val     // Radius
                        );

                        // =========================================================================
                        // END DEBUG ENERGY COMPUTATION
                        // =========================================================================
                    }
                }

                log_message("INFO", "Thread %d: Starting to write rank files for snapshot %d",
                            omp_get_thread_num(), snap);

                if (g_doDynRank)
                {
                    log_message("INFO", "Thread %d: About to write unsorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                    // Unsorted file write
                    char fname_unsorted[256];
                    char base_filename[256];
                    snprintf(base_filename, sizeof(base_filename), "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
                    get_suffixed_filename(base_filename, 1, fname_unsorted, sizeof(fname_unsorted));

                    log_message("INFO", "Thread %d: Opening %s for writing",
                                omp_get_thread_num(), fname_unsorted);

                    FILE *fun_final = fopen(fname_unsorted, "wb"); // Binary mode for fprintf_bin

                    if (!fun_final)
                    {
                        log_message("ERROR", "Thread %d: Failed to open %s for writing",
                                    omp_get_thread_num(), fname_unsorted);
                        // Skip this file but continue with other operations.
                        goto skip_unsorted_write;
                    }
                    for (int ii = 0; ii < npts; ii++)
                    {
                        fprintf_bin(fun_final, "%d %f %f %f %f %f %f\n",
                                    Rank_unsorted[ii],
                                    Mass_unsorted[ii],
                                    R_unsorted[ii],
                                    Vrad_unsorted[ii],
                                    PsiA_unsorted[ii],
                                    E_unsorted[ii],
                                    L_unsorted[ii]);
                    }
                    fclose(fun_final);
                    log_message("INFO", "Thread %d: Successfully wrote unsorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                skip_unsorted_write:

                    // Sorted file write
                    log_message("INFO", "Thread %d: About to write sorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                    char fname_sorted[256];
                    snprintf(base_filename, sizeof(base_filename), "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
                    get_suffixed_filename(base_filename, 1, fname_sorted, sizeof(fname_sorted));

                    log_message("INFO", "Thread %d: Opening %s for writing",
                                omp_get_thread_num(), fname_sorted);

                    FILE *fsort_final = fopen(fname_sorted, "wb"); // Binary mode for fprintf_bin

                    if (!fsort_final)
                    {
                        log_message("ERROR", "Thread %d: Failed to open %s for writing",
                                    omp_get_thread_num(), fname_sorted);
                        // Skip this file but continue.
                        goto skip_sorted_write;
                    }

                    for (int ii = 0; ii < npts; ii++)
                    {
                        fprintf_bin(fsort_final, "%d %f %f %f %f %f %f %f\n",
                                    Rank_sorted[ii],
                                    Mass_sorted[ii],
                                    R_sorted[ii],
                                    Vrad_sorted[ii],
                                    PsiA_sorted[ii],
                                    E_sorted[ii],
                                    L_sorted[ii],
                                    density_sorted[ii]);
                    }
                    fclose(fsort_final);
                    log_message("INFO", "Thread %d: Successfully wrote sorted Rank file for snapshot %d",
                                omp_get_thread_num(), snap);

                skip_sorted_write:; // Empty statement needed after label.
                } // END if (g_doDynRank).

                log_message("INFO", "Thread %d: Completed writing rank files for snapshot %d, freeing memory",
                            omp_get_thread_num(), snap);

                free(inv_rA);
                free(suffix_sum_invrA);
                free(rrA);
                free(M_rA);
                free(psiAarr);
                free(rrA_spline);
                free(psiAarr_spline);
                free(PsiA_unsorted);
                free(PsiA_sorted);
                free(E_unsorted);
                free(E_sorted);
                gsl_spline_free(splinePsiA);
                gsl_interp_accel_free(PsiAinterp);
            }
            // Free temporary arrays

            // =========================================================================
            // CLEANUP SECTION - PARTICLE DATA PROCESSING COMPLETE
            // =========================================================================
            //
            // Release all memory allocated during particle data processing phase.
            // Memory is freed in the reverse order of allocation to prevent memory leaks.
            // This includes thread-local arrays and shared memory structures.
            // Free thread-local arrays
            // Free all local arrays
            free(Rank_unsorted);
            free(Mass_unsorted);
            free(R_unsorted);
            free(Vrad_unsorted);
            free(L_unsorted);

            free(Rank_sorted);
            free(Mass_sorted);
            free(R_sorted);
            free(Vrad_sorted);
            free(L_sorted);

            free(partarr);

            // Free density_sorted array which was allocated during density calculation
            if (density_sorted != NULL) {
                free(density_sorted);
                log_message("INFO", "Thread %d: Freed density_sorted for snapshot %d",
                           omp_get_thread_num(), snap);
            }

            // Free the initial local buffers
            log_message("INFO", "Thread %d: FINAL cleanup for snapshot %d",
                        omp_get_thread_num(), snap);

            free(tmpL_partdata_snap);
            free(tmpRank_partdata_snap);
            free(tmpR_partdata_snap);
            free(tmpV_partdata_snap);

            log_message("INFO", "Thread %d COMPLETED processing of snapshot index %d (snapshot number %d)",
                        omp_get_thread_num(), s, snap);
        }

        log_message("INFO", "All snapshot processing completed. Cleaning up.");

    cleanup_partdata_outsnaps:
        // Free the memory for particle data arrays
        free(Rank_partdata_outsnaps);
        free(R_partdata_outsnaps);
        free(Vrad_partdata_outsnaps);
    }
    {
        log_message("INFO", "Final operations using file suffix: %s", g_file_suffix);

        for (int s = 0; s < noutsnaps; s++)
        {
            snapshot_steps[s] = (int)floor(
                s * (total_writes - 1) / (double)(noutsnaps - 1));
        }
    }

    free(inverse_map);

    for (i = 0; i < num_traj_particles; i++)
    {
        free(trajectories[i]);
        free(energies[i]);
        free(velocities_arr[i]);
        free(mu_arr[i]);
        free(E_arr[i]);
        free(L_arr[i]);
    }
    free(trajectories);
    free(energies);
    free(velocities_arr);
    free(mu_arr);
    free(E_arr);
    free(L_arr);

    free(E_i_arr);
    free(L_i_arr);

    free(r_initial);
    free(rv_initial);
    free(v_initial);
    free(l_initial);
    free(r_final);
    free(v_final);


    free(mass);
    free(radius);
    if (radius_monotonic_grid_nfw != NULL) {
        free(radius_monotonic_grid_nfw);
        radius_monotonic_grid_nfw = NULL;
    }
    free(Psivalues);
    free(nPsivalues);
    free(innerintegrandvalues);
    free(Evalues);
    if (w != NULL) {
        gsl_integration_workspace_free(w);
    }
    // Free GSL RNG resources
    if (g_rng != NULL) {
        gsl_rng_free(g_rng);
        g_rng = NULL;
    }

    // End method_select == 3 block.

    // Move the freeing of particles *outside* the if-block.
    for (i = 0; i < 5; i++)
    {
        free(particles[i]);
    }
    free(particles);

    if (g_doDebug)
    {
        finalize_debug_energy_output(); // Ensures all data is collected first
    }

    gsl_spline_free(splinemass);
    gsl_spline_free(splinePsi);
    gsl_spline_free(splinerofPsi);
    gsl_interp_accel_free(enclosedmass);
    gsl_interp_accel_free(Psiinterp);
    gsl_interp_accel_free(rofPsiinterp);
    gsl_interp_free(g_main_fofEinterp);
    gsl_interp_accel_free(g_main_fofEacc);

    free_local_snap_arrays();
    cleanup_all_particle_data();

#ifdef _OPENMP
    // Clean up FFTW threads only if they were initialized.
    fftw_cleanup_threads();

    if (g_enable_sidm_scattering) {
        printf("Total SIDM scattering events during simulation: %lld\n", g_total_sidm_scatters);
        log_message("INFO", "Total SIDM scattering events: %lld", g_total_sidm_scatters);
    }
#endif

    // Free per-thread GSL RNG resources
    if (g_rng_per_thread != NULL) {
        for (int i_rng = 0; i_rng < g_max_omp_threads_for_rng; ++i_rng) {
            if (g_rng_per_thread[i_rng] != NULL) {
                gsl_rng_free(g_rng_per_thread[i_rng]);
            }
        }
        free(g_rng_per_thread);
        g_rng_per_thread = NULL;
        log_message("INFO", "Freed per-thread GSL RNGs.");
    }

    // Cleanup for the conditionally declared persistent sort buffer.
    if (g_sort_columns_buffer != NULL) {
        for (int i = 0; i < g_sort_columns_buffer_npts; i++) {
            if (g_sort_columns_buffer[i]) free(g_sort_columns_buffer[i]);
        }
        free(g_sort_columns_buffer);
        g_sort_columns_buffer = NULL; // Mark as freed.
        g_sort_columns_buffer_npts = 0; // Reset size.
    }

    // Free global NFW spline resources if they were allocated for NFW profile
    if (g_nfw_splinemass_for_force) {
        gsl_spline_free(g_nfw_splinemass_for_force);
        g_nfw_splinemass_for_force = NULL;
        log_message("INFO", "Freed global NFW mass spline for force calculation.");
    }
    if (g_nfw_enclosedmass_accel_for_force) {
        gsl_interp_accel_free(g_nfw_enclosedmass_accel_for_force);
        g_nfw_enclosedmass_accel_for_force = NULL;
        log_message("INFO", "Freed global NFW mass spline accelerator.");
    }

    // Free particle scatter state array
    free(g_particle_scatter_state);

    return 0;
} // End main function.

/**
 * @brief GSL integrand \f$r^2 \rho_{shape}(r)\f$ for Cored Plummer-like profile mass calculation.
 * @details Computes the term \f$r^2 \rho_{shape}(r)\f$ for the Cored Plummer-like density profile,
 *          where \f$\rho_{shape}(r) = (1 + (r/RC)^2)^{-3}\f$. This integrand is used in
 *          GSL numerical integration routines (e.g., `gsl_integration_qag`) to calculate
 *          the normalization factor or the enclosed mass \f$M(<r) = 4\pi \int_0^r r'^2 \rho_{physical}(r') dr'\f$.
 *          It directly uses the `RC` macro for the scale radius.
 *
 * @param r      [in] Radial coordinate \f$r\f$ (kpc).
 * @param params [in] Void pointer to parameters (unused in this version, hence `__attribute__((unused))`).
 * @return double The value of the mass integrand \f$r^2 \rho_{shape}(r)\f$ at radius `r`.
 */
double massintegrand(double r, void *params __attribute__((unused)))
{
    double startingprofile = 1.0 / cube((1.0 + sqr(r / g_cored_profile_rc)));
    return r * r * startingprofile;
}

/**
 * @brief Calculates \f$d\rho/dr\f$ for the Cored Plummer-like density profile.
 * @details The density profile is \f$\rho(r) \propto (1 + (r/RC)^2)^{-3}\f$.
 *          This function computes its analytical derivative with respect to \f$r\f$.
 *          It directly uses the `RC` macro for the scale radius.
 *
 * @param r [in] Radial coordinate (kpc) at which to evaluate the derivative.
 * @return double The value of \f$d\rho/dr\f$ at radius `r`.
 */
double drhodr(double r)
{
    return -6.0 * r / (g_cored_profile_rc * g_cored_profile_rc) / pow(1.0 + sqr(r / g_cored_profile_rc), 4.0);
}

/**
 * @brief GSL integrand \f$r^2 \rho(r)\f$ for NFW-like profile mass calculation.
 * @details Computes \f$r^2 \rho(r)\f$ where \f$\rho(r)\f$ is an NFW-like density profile
 *          with an inner softening term and an outer power-law cutoff.
 *          The density \f$\rho(r) = p[2] \times [ (r_s + \epsilon)(1+r_s)^2 (1 + (r_s/C)^N) ]^{-1}\f$,
 *          where \f$r_s = r/p[0]\f$, \f$\epsilon=0.01\f$, \f$C=p[3]\f$, \f$N=10.0\f$.
 *          This integrand is used in GSL routines to calculate the normalization factor
 *          or enclosed mass for the NFW-like profile.
 *
 * @param r      [in] Radial coordinate \f$r\f$ (kpc).
 * @param params [in] Void pointer to a `double` array `p` of size 4:
 *                    - `p[0]` (rc_param): Scale radius (kpc).
 *                    - `p[1]` (halo_mass): Target total halo mass (Msun) - used to derive nt_nfw_scaler.
 *                    - `p[2]` (nt_nfw_scaler): Density normalization constant \f$nt_{NFW}\f$.
 *                    - `p[3]` (falloff_C_param): Falloff transition factor \f$C\f$.
 * @return double The value of the mass integrand \f$r^2 \rho(r)\f$. Returns 0.0 if `rc_param` (p[0]) is non-positive.
 */
double massintegrand_profile_nfwcutoff(double r, void *params) {
    double *p = (double *)params;
    double rc_param = p[0];                 // Scale radius RC from parameters
    // p[1] is current_profile_halo_mass, not used directly in density formula
    double nt_nfw_scaler = p[2];            // Density scaling factor nt_nfw

    // Parameters for the NFW-like profile shape
    const double epsilon_softening = 0.01;  // Softening parameter for r/rc term
    double C_cutoff_factor = p[3];         // Falloff factor from params
    if (C_cutoff_factor <= 0) C_cutoff_factor = 19.0; // Safety default if param is bad
    const double N_cutoff_power = 10.0;     // Power for the cutoff term

    if (rc_param <= 0) { // Avoid division by zero if rc is invalid
        return 0.0;
    }
    
    double rs = r / rc_param; // r normalized by scale radius

    // Calculate the structural part of the density profile (unscaled by nt_nfw)
    double term_softening = rs + epsilon_softening;
    if (term_softening <= 1e-9) term_softening = 1e-9; // Avoid division by zero from softening

    double term_nfw_slope = (1.0 + rs) * (1.0 + rs); // (1 + r/rc)^2

    double cutoff_rs = rs / C_cutoff_factor;
    double term_cutoff = 1.0 + pow(cutoff_rs, N_cutoff_power);

    double density_shape;
    if (term_softening < 1e-9 || term_nfw_slope < 1e-9 || term_cutoff < 1e-9) { // Denominator terms too small
         density_shape = 0.0; // Or handle as very large if r is very small
         if (r < 1e-6 && term_softening < 1e-3) { // special handling for very small r to match NFW cusp
             density_shape = 1.0 / (term_softening * term_nfw_slope * term_cutoff); // Let it be large
         }
    } else {
         density_shape = 1.0 / (term_softening * term_nfw_slope * term_cutoff);
    }

    // Apply the overall density scaling factor
    double physical_density = nt_nfw_scaler * density_shape;
    
    return r * r * physical_density;
}

/**
 * @brief Calculates \f$d\rho/dr\f$ for the NFW-like profile with a power-law cutoff.
 * @details The NFW-like density profile used is:
 *          \f$\rho(r) = \text{nt_nfw_scaler} \times [ (r_s + \epsilon)(1+r_s)^2 (1 + (r_s/C)^N) ]^{-1}\f$
 *          where \f$r_s = r / \text{rc_param}\f$, \f$\epsilon\f$ is a softening parameter (0.01),
 *          \f$C\f$ is the `falloff_C_param`, and \f$N\f$ is a power-law index (10.0).
 *          This function computes the analytical derivative of this \f$\rho(r)\f$ with respect to \f$r\f$.
 *
 * @param r               [in] Radial coordinate (kpc) at which to evaluate the derivative.
 * @param rc_param        [in] Scale radius (RC) of the NFW-like profile (kpc).
 * @param nt_nfw_scaler   [in] Density normalization constant (nt_nfw) for the profile.
 * @param falloff_C_param [in] Falloff transition factor \f$C\f$ for the power-law cutoff.
 * @return double The value of \f$d\rho/dr\f$ at radius `r`. Returns 0.0 if `rc_param` is non-positive.
 */
double drhodr_profile_nfwcutoff(double r, double rc_param, double nt_nfw_scaler, double falloff_C_param) { // Added falloff_C_param

    // Parameters for the NFW-like profile shape
    const double epsilon_softening = 0.01;
    double C_cutoff_factor = falloff_C_param;
    if (C_cutoff_factor <= 0) C_cutoff_factor = 19.0; // Safety default
    const double N_cutoff_power = 10.0;

    if (rc_param <= 0) return 0.0;
    
    double rs = r / rc_param;

    //rho_shape(rs) = 1.0 / ( (rs+eps) * (1+rs)^2 * (1+(rs/C)^N) )
    double term_s = rs + epsilon_softening;
    if (term_s <= 1e-9) term_s = 1e-9; // Avoid division by zero for denominator
    double term_n = (1.0 + rs) * (1.0 + rs);
    double term_c_base = rs / C_cutoff_factor;
    double term_c_pow_N = pow(term_c_base, N_cutoff_power);
    double term_c = 1.0 + term_c_pow_N;

    double density_shape_val;
    if (term_s < 1e-9 || term_n < 1e-9 || term_c < 1e-9) {
        density_shape_val = 0.0; // Or very large if r is tiny
        if (r < 1e-6 && term_s < 1e-3) {
            density_shape_val = 1.0 / (term_s * term_n * term_c);
        }
    } else {
        density_shape_val = 1.0 / (term_s * term_n * term_c);
    }

    // Derivative of each term in the denominator w.r.t rs (d/d(rs)):
    // d/drs (rs+eps) = 1
    // d/drs (1+rs)^2 = 2*(1+rs)
    // d/drs (1+(rs/C)^N) = N * (rs/C)^(N-1) * (1/C)
    
    double d_log_term_s_d_rs = 1.0 / term_s;
    double d_log_term_n_d_rs = 2.0 / (1.0 + rs);
    double d_log_term_c_d_rs = (N_cutoff_power / C_cutoff_factor) * pow(term_c_base, N_cutoff_power - 1.0) / term_c;
    if (!isfinite(term_c_base) || (term_c_base < 1e-9 && N_cutoff_power -1 < 0)) { // Avoid pow(small_negative_base)
         d_log_term_c_d_rs = 0; // If rs/C is zero and N-1 is negative
    }

    // d(rho_shape)/dr = d(rho_shape)/d(rs) * d(rs)/dr = d(rho_shape)/d(rs) * (1/rc_param)
    // d(log(rho_shape))/d(rs) = - (d_log_term_s_d_rs + d_log_term_n_d_rs + d_log_term_c_d_rs)
    // d(rho_shape)/d(rs) = rho_shape * d(log(rho_shape))/d(rs)
    
    double d_rho_shape_d_rs = -density_shape_val * (d_log_term_s_d_rs + d_log_term_n_d_rs + d_log_term_c_d_rs);
    double drho_dr = nt_nfw_scaler * d_rho_shape_d_rs / rc_param;

    
    return drho_dr;
}

/**
 * @brief GSL integrand for calculating the gravitational potential \f$\Psi(r)\f$.
 * @details This function computes the integrand \f$M_{enc}(r')/r'\f$ or \f$r' \rho_{shape}(r')\f$
 *          (depending on the exact formulation of \f$\Psi\f$) needed for the integral part of
 *          the potential calculation: \f$\Psi(r) = G M(<r)/r + G \int_r^{\infty} \text{integrand_val } dr'\f$.
 *          It uses a function pointer (`p_psi->massintegrand_func`) passed via the `params`
 *          argument (a `Psiintegrand_params` struct) to call the profile-specific
 *          mass integrand (which itself returns \f$r'^2 \rho_{shape}(r')\f$ or similar).
 *          The function then typically divides by `rp` to get \f$r' \rho_{shape}(r')\f$ if `massintegrand_func`
 *          returned \f$r'^2 \rho_{shape}(r')\f$.
 *          It includes a safety check to exit if `rp` (radius prime) is non-positive.
 *
 * @param rp     [in] The radial integration variable \f$r'\f$ (kpc).
 * @param params [in] Void pointer to a `Psiintegrand_params` struct. This struct contains
 *                    a function pointer to the profile-specific mass integrand and its parameters.
 * @return double The value of the potential integrand at `rp`.
 *
 * @see Psiintegrand_params
 * @see massintegrand
 * @see massintegrand_profile_nfwcutoff
 */
double Psiintegrand(double rp, void *params)
{
    Psiintegrand_params *p_psi = (Psiintegrand_params *)params;
    if (rp <= 0.0)
    {
        printf("rp out of range\n");
        CLEAN_EXIT(1);
    }
    // Call the profile-specific mass integrand function
    return p_psi->massintegrand_func(rp, p_psi->params_for_massintegrand) / rp;
}


/**
 * @brief Safely evaluates a GSL spline at a given value with robust bounds checking.
 * @details This function evaluates the provided GSL spline at the specified `value`.
 *          It includes critical safety checks:
 *          1. It verifies that the `spline` and accelerator `acc` pointers are not NULL.
 *          2. It checks if the `value` is outside the defined range of the spline's x-values.
 *             If `value` is out of bounds, it clamps `value` to the nearest valid boundary
 *             (plus/minus a small MARGIN) before evaluation to prevent GSL domain errors.
 *          This robust approach ensures that spline evaluations do not cause crashes due to
 *          out-of-range inputs, which can occur due to floating-point inaccuracies or
 *          unexpected data.
 *
 * @param spline [in] Pointer to the initialized GSL spline object.
 * @param acc    [in] Pointer to the GSL interpolation accelerator associated with the spline.
 * @param value  [in] The x-coordinate at which to evaluate the spline.
 * @return double The interpolated y-value from the spline. Returns the boundary spline value
 *                if `value` was clamped. Returns 0.0 if `spline` or `acc` is NULL (error logged).
 */
double evaluatespline(gsl_spline *spline, gsl_interp_accel *acc, double value)
{
    // NULL pointer safety check.
    if (spline == NULL || acc == NULL)
    {
        fprintf(stderr, "Error: NULL pointer passed to evaluatespline (spline=%p, acc=%p)\n",
                (void *)spline, (void *)acc);
        return 0.0; // Return a default value instead of crashing.
    }

    // Get the actual min and max ranges of the spline from its data directly.
    double x_min = spline->x[0];
    double x_max = spline->x[spline->size - 1];

    // Ensure the value is within the valid interpolation range with a small safety margin.
    const double MARGIN = 1e-10; // Small safety margin.

    if (value < x_min)
    {
#ifdef DEBUG_SPLINE
        fprintf(stderr, "Warning: Spline interpolation value %g below minimum %g, clamping\n",
                value, x_min);
#endif
        // Clamp to minimum with a tiny margin to stay inside the valid range.
        return gsl_spline_eval(spline, x_min + MARGIN, acc);
    }
    else if (value > x_max)
    {
#ifdef DEBUG_SPLINE
        fprintf(stderr, "Warning: Spline interpolation value %g above maximum %g, clamping\n",
                value, x_max);
#endif
        // Clamp to maximum with a tiny margin to stay inside the valid range.
        return gsl_spline_eval(spline, x_max - MARGIN, acc);
    }

    // Normal case - value is within range.
    return gsl_spline_eval(spline, value, acc);
}

/**
 * Calculates the integrand for the distribution function calculation.
 *
 * Uses splines to evaluate the potential and mass at a given energy and radius,
 * and computes the distribution function integrand according to Eddington's formula.
 *
 * @param t Integration variable
 * @param params Structure containing necessary splines and energy value
 * @return Value of the distribution function integrand at the specified point
 */
double fEintegrand(double t, void *params)
{
    fEintegrand_params *p = (fEintegrand_params *)params;
    gsl_spline *splinePsi = p->splinePsi;
    gsl_spline *splinemass = p->splinemass;
    gsl_interp_accel *rofPsiarray = p->rofPsiarray;
    gsl_interp_accel *massarray = p->massarray;
    double E = p->E;

    double Psi = E - t * t;
    double r = evaluatespline(splinePsi, rofPsiarray, -Psi);
    double drhodr(double r1);
    double drhodpsi = -(g_cored_profile_halo_mass / normalization) * drhodr(r) / (G_CONST * evaluatespline(splinemass, massarray, r) / (r * r));
    return 2.0 * drhodpsi;
}

/**
 * @brief Integrand for calculating the I(E) component of the NFW distribution function.
 * @details This function is integrated with respect to t_integration_var = sqrt(E_shell - Psi_true_at_r).
 *          It computes -2 * (d(rho)/d(Psi_true)) to ensure a positive integrand,
 *          leading to a monotonically increasing I(E). Psi_true_at_r is the potential
 *          corresponding to the radius r reached when energy E_shell has been reduced by t_integration_var^2.
 *          The radius r is determined from Psi_true_at_r using a spline.
 *          Derivatives d(rho)/dr and d(Psi_true)/dr are then calculated at this r.
 *
 * @param t_integration_var [in] The integration variable, t_prime = sqrt(E_shell - Psi_true_at_r).
 * @param params            [in] Void pointer to a `fE_integrand_params_NFW_t` structure. This struct
 *                               contains the current energy shell E_shell, splines for r(Psi) and M(r),
 *                               physical constants (G), NFW profile-specific parameters (scale radius,
 *                               density normalization, falloff factor C), and global Psimin/Psimax
 *                               for physical range validation.
 * @return double The value of the integrand -2 * (d(rho)/d(Psi_true)). Returns 0.0 if Psi_true_at_r
 *                is outside the physical range, if the derived radius is non-physical,
 *                if dPsi/dr is too small (or zero), or if the result is non-finite.
 */
double fEintegrand_nfw(double t_integration_var, void *params) {
    fE_integrand_params_NFW_t *p_nfw = (fE_integrand_params_NFW_t *)params;

    double E_shell = p_nfw->E_current_shell; // Energy of the current shell for I(E)

    // Calculate Psi_true from t_integration_var: Psi_true = E_shell - t_integration_var^2
    // This means as t_integration_var goes from 0 to sqrt(E_shell - Psimin_global),
    // Psi_true goes from E_shell down to Psimin_global.
    double Psi_true_at_r = E_shell - t_integration_var * t_integration_var;


    // Ensure Psi_true_at_r is within the valid physical range [Psimin_global, Psimax_global]
    if (Psi_true_at_r < p_nfw->Psimin_global - 1e-7*fabs(p_nfw->Psimin_global) || Psi_true_at_r > p_nfw->Psimax_global + 1e-7*fabs(p_nfw->Psimax_global)) {
         return 0.0;
    }
    
    // Get radius r from Psi_true_at_r. Spline p->spline_r_of_Psi expects -Psi_true as input.
    double r_val;
    double spline_x_input_rPsi = -Psi_true_at_r; // Input for r_of_Psi spline
    double spline_rPsi_x_min = p_nfw->spline_r_of_Psi->x[0];
    double spline_rPsi_x_max = p_nfw->spline_r_of_Psi->x[p_nfw->spline_r_of_Psi->size - 1];

    if (spline_x_input_rPsi < spline_rPsi_x_min) spline_x_input_rPsi = spline_rPsi_x_min;
    if (spline_x_input_rPsi > spline_rPsi_x_max) spline_x_input_rPsi = spline_rPsi_x_max;
        
    r_val = gsl_spline_eval(p_nfw->spline_r_of_Psi, spline_x_input_rPsi, p_nfw->accel_r_of_Psi); 


    // If radius is non-physical (negative or zero), the integrand is ill-defined or zero.
    if (r_val <= 1e-10) { // Using a slightly larger epsilon than machine precision for safety
        return 0.0; 
    }
    // No more flooring of r_val here; use it as is if positive, or return 0 if not.

    // Calculate drho/dr at r_val
    double drho_dr_val = drhodr_profile_nfwcutoff(r_val, p_nfw->profile_rc_const, p_nfw->profile_nt_norm_const, p_nfw->profile_falloff_C_const);

    // Calculate dPsi_true/dr = G * M(r_val) / r_val^2 (magnitude)
    double M_at_r_val = gsl_spline_eval(p_nfw->spline_M_of_r, r_val, p_nfw->accel_M_of_r);
    if (M_at_r_val < 0) M_at_r_val = 0; // Mass must be non-negative
    
    // Calculate dPsi_true/dr = G * M(r_val) / r_val^2 (magnitude)
    // Since r_val > 1e-10, division by r_val^2 is safe
    double dPsi_dr_mag = p_nfw->const_G_universal * M_at_r_val / (r_val * r_val);

    if (fabs(dPsi_dr_mag) < 1e-30) { // If dPsi/dr is effectively zero (e.g. M(r)=0 at r=0)
        return 0.0; // drho/dPsi would be undefined or infinite
    }

    // drho/dPsi = (drho/dr) / (dPsi/dr)
    // Sign convention: Assume Psi is defined such that dPsi/dr is positive (potential less negative further out).
    // drho/dr is negative. So drho/dPsi is negative.
    // The quantity 2 * drho/dPsi is typically what appears in one form of Eddington's.
    double drho_dPsi_val = drho_dr_val / dPsi_dr_mag; 
    // As per ANFIS.1 and subsequent findings, for I(E) to be increasing, 
    // the integrand 2*d(rho)/d(Psi_true) needs to be positive.
    // Since drho_dPsi_val = (drho/dr) / (dPsi/dr_mag) is (negative/positive) = negative,
    // we need to flip the sign.
    double integrand_value = -2.0 * drho_dPsi_val;

    
    if (!isfinite(integrand_value)) {
        if (g_doDebug) fprintf(stderr, "Warning: NFW fEintegrand (refactored) returning non-finite value for t_in=%.3e, E_shell=%.3e\n", t_integration_var, E_shell);
        return 0.0; // Return 0 for non-finite cases
    }

    return integrand_value;
}

// =========================================================================
// PARALLEL SORTING ALGORITHM FUNCTIONS
// =========================================================================
//
// Functions implementing parallel sorting algorithms.
// The sorting implementation uses a parallel chunk-based approach with overlapping
// regions between chunks to ensure correct ordering at chunk boundaries.
// Constants controlling behavior are defined at the top of this file.

/**
 * @brief Compares two particle entries for sorting based on the first column value.
 *
 * @details Used as a comparison function for qsort, quadsort, and other sorting algorithms.
 *          Particles with smaller values in column 0 will be sorted before those with larger values.
 *
 * @param a Pointer to the first particle entry (as void pointer, expected double**).
 * @param b Pointer to the second particle entry (as void pointer, expected double**).
 * @return -1 if a<b, 1 if a>b, 0 if equal based on the first column value (radius).
 *
 * @note Assumes the particle data is structured as [particle_index][component_index]
 *       when passed via `columns` array in sorting functions, and compares `columns[i][0]`.
 */
int compare_particles(const void *a, const void *b)
{
    double *col_a = *(double **)a;
    double *col_b = *(double **)b;
    if (col_a[0] < col_b[0])
        return -1;
    if (col_a[0] > col_b[0])
        return 1;
    return 0;
}

/**
 * @brief Implementation of classic insertion sort algorithm for particle data.
 *
 * @details Sorts an array of particle data using the insertion sort algorithm.
 *          While not the fastest algorithm for large datasets, it is stable and works well
 *          for small arrays or nearly sorted data.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
static void insertion_sort(double **columns, int n)
{
    for (int i = 1; i < n; i++)
    {
        double *temp = columns[i];
        int j = i - 1;
        while (j >= 0 && compare_particles(&columns[j], &temp) > 0)
        {
            columns[j + 1] = columns[j];
            j--;
        }
        columns[j + 1] = temp;
    }
}

/**
 * @brief Wrapper for the standard C library quicksort function.
 *
 * @details Provides a consistent interface to the standard library `qsort` function
 *          using the `compare_particles` function as the comparison callback.
 *
 * @param columns 2D array of particle data to be sorted (passed as `double**`).
 * @param n Number of elements (particles) to sort.
 */
static void stdlib_qsort_wrapper(double **columns, int n)
{
    qsort(columns, n, sizeof(double *), compare_particles);
}

/**
 * Wrapper for the external quadsort algorithm.
 *
 * Provides a consistent interface to the external quadsort function,
 * which is typically faster than standard quicksort for many distributions.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
static void quadsort_wrapper(double **columns, int n)
{
    quadsort(columns, n, sizeof(double *), compare_particles);
}

/**
 * @brief Helper function that performs insertion sort on a subarray of particle data.
 *
 * @details Similar to `insertion_sort` but operates on a specific range `[start..end]` inclusive.
 *          Used by the parallel sorting algorithms to sort individual chunks and seam regions.
 *
 * @param columns 2D array of particle data containing the target subarray (`double**`).
 * @param start Starting index of the subarray (inclusive).
 * @param end Ending index of the subarray (inclusive).
 */
static void insertion_sort_sub(double **columns, int start, int end)
{
    for (int i = start + 1; i <= end; i++)
    {
        double *temp = columns[i];
        int j = i - 1;
        while (j >= start && compare_particles(&columns[j], &temp) > 0)
        {
            columns[j + 1] = columns[j];
            j--;
        }
        columns[j + 1] = temp;
    }
}

/**
 * Parallel insertion sort implementation using chunk-based approach with overlap.
 *
 * Divides the data into num_sort_sections chunks, sorts each chunk in parallel,
 * then fixes the boundaries between chunks by re-sorting overlap regions.
 * This approach balances parallelism with the need to ensure properly sorted output.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
/**
 * @brief Parallel implementation of insertion sort algorithm optimized for particle sorting.
 *
 * @details Uses multiple OpenMP threads to sort sections of particle data in parallel, 
 * followed by seam-fixing operations to ensure global ordering. Includes dynamic section
 * calculation and optimized overlap sizing based on chunk characteristics.
 *
 * @param columns Column-major data array [particle][component]
 * @param n Number of particles to sort
 */
void insertion_parallel_sort(double **columns, int n)
{
    // Dynamically determine number of sections based on runtime threads and constants.
    int active_num_sort_sections;
    #ifdef _OPENMP
        int n_runtime_threads = omp_get_max_threads(); 
        if (n_runtime_threads <= 0) n_runtime_threads = 1;
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD; 
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD;
        if (active_num_sort_sections <= 0) active_num_sort_sections = PARALLEL_SORT_DEFAULT_SECTIONS;
    #else
        active_num_sort_sections = 1; // Force serial behavior if OpenMP is not compiled in
    #endif

    // Ensure a reasonable number of sections
    if (active_num_sort_sections < 1) active_num_sort_sections = 1;
    if (n > 0 && active_num_sort_sections > n) active_num_sort_sections = n; 
    // Optional: Add a hard cap for maximum sections if desired, e.g.:
    // if (active_num_sort_sections > 96) active_num_sort_sections = 96;

    // Fallback to serial sort for small N or if chunks would be too small
    int estimated_avg_chunk_size = (n > 0 && active_num_sort_sections > 0) ? (n / active_num_sort_sections) : n;
    if (n < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD || \
        active_num_sort_sections <= 1 || \
        estimated_avg_chunk_size < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD) {
        // If PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD is set carefully (e.g. >= 2 * PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP),
        // this also helps ensure chunks are large enough for meaningful overlap.
        insertion_sort(columns, n); // Call serial insertion sort
        return;
    }

    // Determine chunk boundaries, distributing remainder
    int base_chunk_size = n / active_num_sort_sections;
    int remainder = n % active_num_sort_sections;
    int *startIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    int *endIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    if (!startIdx || !endIdx) { /* Handle error */ CLEAN_EXIT(1); }

    int offset = 0;
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        int size_c = base_chunk_size + (c < remainder ? 1 : 0);
        startIdx[c] = offset;
        endIdx[c] = offset + size_c - 1;
        offset += size_c;
    }

    // Sort each chunk in parallel using insertion sort
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        insertion_sort_sub(columns, startIdx[c], endIdx[c]);
    }

    // Calculate minChunkSize based on actual chunk distribution using active_num_sort_sections
    int minChunkSize = n; 
    if (active_num_sort_sections > 0 && n > 0 && endIdx && startIdx) { // Check endIdx/startIdx validity
        minChunkSize = (endIdx[0] - startIdx[0] + 1);
        for (int c = 1; c < active_num_sort_sections; c++) {
            int csize = endIdx[c] - startIdx[c] + 1;
            if (csize < minChunkSize) minChunkSize = csize;
        }
    }
    if (minChunkSize <= 0 && n > 0) minChunkSize = 1; // Safety for valid n

    int overlapSize;
    if (n <= 1 || active_num_sort_sections <= 1 || minChunkSize <= 0) {
        overlapSize = 0; 
    } else {
        int proportional_overlap = minChunkSize / PARALLEL_SORT_OVERLAP_DIVISOR;
        if (proportional_overlap == 0 && minChunkSize > 0) {
            proportional_overlap = 1; 
        }

        // Ensure overlap is at least the minimum required for correctness,
        // but only if that minimum isn't itself making the overlap too large for the chunk.
        if (PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP > 0 && proportional_overlap < PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP) {
            overlapSize = PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP;
        } else {
            overlapSize = proportional_overlap;
        }

        // Cap the overlap: It should not be an excessive fraction of the smallest chunk.
        // This also handles cases where MIN_CORRECTNESS_OVERLAP might be too large for a small chunk.
        int max_permissible_relative_overlap = minChunkSize / 2; // Example: Cap at 50% of chunk
        if (max_permissible_relative_overlap < 1 && minChunkSize > 0) max_permissible_relative_overlap = 1; // Ensure cap is at least 1 if chunk exists

        if (overlapSize > max_permissible_relative_overlap && minChunkSize > 1) {
            overlapSize = max_permissible_relative_overlap;
        }
        
        // If, after all logic, overlap is 0 but we have multiple sections and data, ensure minimal overlap.
        if (overlapSize == 0 && minChunkSize > 0 && active_num_sort_sections > 1) {
             overlapSize = 1; 
        }
    }
    if (overlapSize < 0) overlapSize = 0; // Final safety check
    // Additional absolute cap based on total N, mostly for sanity with very few sections.
    if (n > 1 && overlapSize > n / 2) overlapSize = n / 2;


    // Optional debug print (controlled by -DDEBUG_SORT_PARAMS compile flag)
    #ifdef DEBUG_SORT_PARAMS
    #ifdef _OPENMP
    if (omp_get_thread_num() == 0) // Print only from one thread
    #endif
    {
        printf("[NSPHERE_IS_PARALLEL_DEBUG] N=%d, Sections=%d, MinChunkSz=%d, OverlapSize=%d (Using DIV:%d, MIN_CORRECT:%d)\n",
               n, active_num_sort_sections, minChunkSize, overlapSize, 
               PARALLEL_SORT_OVERLAP_DIVISOR, PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP);
        fflush(stdout);
    }
    #endif

    // Merge/fix the seams in parallel
    int nSeams = active_num_sort_sections - 1;
    if (overlapSize > 0 && nSeams > 0) {
#pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nSeams; s++)
        {
            int c_left = s;
            int c_right = s + 1;

            // Robust boundary calculations for seam sorting
            int seam_sort_start = endIdx[c_left] - overlapSize + 1;
            if (seam_sort_start < startIdx[c_left]) seam_sort_start = startIdx[c_left];

            int seam_sort_end = startIdx[c_right] + overlapSize - 1;
            if (seam_sort_end > endIdx[c_right]) seam_sort_end = endIdx[c_right];

            // Sort the combined overlap region
            if (seam_sort_start <= seam_sort_end)
            {
                insertion_sort_sub(columns, seam_sort_start, seam_sort_end);
            }
        }
    }

    free(startIdx);
    free(endIdx);
}

/**
 * Parallel quadsort implementation using chunk-based approach with overlap.
 *
 * Similar to insertion_parallel_sort but uses the quadsort algorithm for both
 * the initial chunk sorting and the seam fixing. Quadsort is generally faster
 * than insertion sort for larger datasets while maintaining stability.
 *
 * @param columns 2D array of particle data to be sorted
 * @param n Number of elements to sort
 */
/**
 * @brief Parallel implementation of quadsort algorithm optimized for particle sorting.
 *
 * @details Uses multiple OpenMP threads to sort sections of particle data in parallel, 
 * followed by seam-fixing operations to ensure global ordering. Includes dynamic section
 * calculation and optimized overlap sizing based on chunk characteristics.
 *
 * @param columns Column-major data array [particle][component]
 * @param n Number of particles to sort
 */
void quadsort_parallel_sort(double **columns, int n)
{
    // Dynamically determine number of sections based on runtime threads and constants.
    int active_num_sort_sections;
    #ifdef _OPENMP
        int n_runtime_threads = omp_get_max_threads(); 
        if (n_runtime_threads <= 0) n_runtime_threads = 1;
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD; 
        active_num_sort_sections = n_runtime_threads * PARALLEL_SORT_SECTIONS_PER_THREAD;
        if (active_num_sort_sections <= 0) active_num_sort_sections = PARALLEL_SORT_DEFAULT_SECTIONS;
    #else
        active_num_sort_sections = 1; // Force serial behavior if OpenMP is not compiled in
    #endif

    // Ensure a reasonable number of sections
    if (active_num_sort_sections < 1) active_num_sort_sections = 1;
    if (n > 0 && active_num_sort_sections > n) active_num_sort_sections = n; 
    // Optional: Add a hard cap for maximum sections if desired, e.g.:
    // if (active_num_sort_sections > 96) active_num_sort_sections = 96;

    // Fallback to serial sort for small N or if chunks would be too small
    int estimated_avg_chunk_size = (n > 0 && active_num_sort_sections > 0) ? (n / active_num_sort_sections) : n;
    if (n < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD || \
        active_num_sort_sections <= 1 || \
        estimated_avg_chunk_size < PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD) {
        // If PARALLEL_SORT_MIN_CHUNK_SIZE_THRESHOLD is set carefully (e.g. >= 2 * PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP),
        // this also helps ensure chunks are large enough for meaningful overlap.
        quadsort_wrapper(columns, n); // Call serial quadsort wrapper
        return;
    }

    // Determine chunk boundaries
    int base_chunk_size = n / active_num_sort_sections;
    int remainder = n % active_num_sort_sections;
    int *startIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    int *endIdx = (int *)malloc(active_num_sort_sections * sizeof(int));
    if (!startIdx || !endIdx) { /* Handle error */ CLEAN_EXIT(1); }

    int offset = 0;
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        int size_c = base_chunk_size + (c < remainder ? 1 : 0);
        startIdx[c] = offset;
        endIdx[c] = offset + size_c - 1;
        offset += size_c;
    }

    // Sort each chunk in parallel using quadsort
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < active_num_sort_sections; c++)
    {
        quadsort(&columns[startIdx[c]], endIdx[c] - startIdx[c] + 1, sizeof(double *), compare_particles);
    }

    // Calculate minChunkSize based on actual chunk distribution using active_num_sort_sections
    int minChunkSize = n; 
    if (active_num_sort_sections > 0 && n > 0 && endIdx && startIdx) { // Check endIdx/startIdx validity
        minChunkSize = (endIdx[0] - startIdx[0] + 1);
        for (int c = 1; c < active_num_sort_sections; c++) {
            int csize = endIdx[c] - startIdx[c] + 1;
            if (csize < minChunkSize) minChunkSize = csize;
        }
    }
    if (minChunkSize <= 0 && n > 0) minChunkSize = 1; // Safety for valid n

    int overlapSize;
    if (n <= 1 || active_num_sort_sections <= 1 || minChunkSize <= 0) {
        overlapSize = 0; 
    } else {
        int proportional_overlap = minChunkSize / PARALLEL_SORT_OVERLAP_DIVISOR;
        if (proportional_overlap == 0 && minChunkSize > 0) {
            proportional_overlap = 1; 
        }

        // Ensure overlap is at least the minimum required for correctness,
        // but only if that minimum isn't itself making the overlap too large for the chunk.
        if (PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP > 0 && proportional_overlap < PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP) {
            overlapSize = PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP;
        } else {
            overlapSize = proportional_overlap;
        }

        // Cap the overlap: It should not be an excessive fraction of the smallest chunk.
        // This also handles cases where MIN_CORRECTNESS_OVERLAP might be too large for a small chunk.
        int max_permissible_relative_overlap = minChunkSize / 2; // Example: Cap at 50% of chunk
        if (max_permissible_relative_overlap < 1 && minChunkSize > 0) max_permissible_relative_overlap = 1; // Ensure cap is at least 1 if chunk exists

        if (overlapSize > max_permissible_relative_overlap && minChunkSize > 1) {
            overlapSize = max_permissible_relative_overlap;
        }
        
        // If, after all logic, overlap is 0 but we have multiple sections and data, ensure minimal overlap.
        if (overlapSize == 0 && minChunkSize > 0 && active_num_sort_sections > 1) {
             overlapSize = 1; 
        }
    }
    if (overlapSize < 0) overlapSize = 0; // Final safety check
    // Additional absolute cap based on total N, mostly for sanity with very few sections.
    if (n > 1 && overlapSize > n / 2) overlapSize = n / 2;


    // Optional debug print (controlled by -DDEBUG_SORT_PARAMS compile flag)
    #ifdef DEBUG_SORT_PARAMS
    #ifdef _OPENMP
    if (omp_get_thread_num() == 0) // Print only from one thread
    #endif
    {
        printf("[NSPHERE_QS_PARALLEL_DEBUG] N=%d, Sections=%d, MinChunkSz=%d, OverlapSize=%d (Using DIV:%d, MIN_CORRECT:%d)\n",
               n, active_num_sort_sections, minChunkSize, overlapSize, 
               PARALLEL_SORT_OVERLAP_DIVISOR, PARALLEL_SORT_MIN_CORRECTNESS_OVERLAP);
        fflush(stdout);
    }
    #endif

    // Merge/fix the seams in parallel using quadsort
    int nSeams = active_num_sort_sections - 1;
    if (overlapSize > 0 && nSeams > 0) {
#pragma omp parallel for schedule(dynamic)
        for (int s = 0; s < nSeams; s++)
        {
            int c_left = s;
            int c_right = s + 1;

            // Robust boundary calculations for seam sorting
            int seam_sort_start = endIdx[c_left] - overlapSize + 1;
            if (seam_sort_start < startIdx[c_left]) seam_sort_start = startIdx[c_left];

            int seam_sort_end = startIdx[c_right] + overlapSize - 1;
            if (seam_sort_end > endIdx[c_right]) seam_sort_end = endIdx[c_right];

            // Sort the combined overlap region
            if (seam_sort_start <= seam_sort_end)
            {
                int seam_len = seam_sort_end - seam_sort_start + 1;
                if (seam_len > 1) {  // Only sort if there's more than one element
                    quadsort(&columns[seam_sort_start], seam_len, sizeof(double *), compare_particles);
                }
            }
        }
    }

    free(startIdx);
    free(endIdx);
}

/**
 * Validates sorting results by comparing with standard qsort.
 *
 * Creates a copy of the input array, sorts it with standard qsort,
 * then compares the results element by element to verify that the
 * sorting algorithm produced the expected ordering.
 *
 * @param columns 2D array of particle data that has been sorted
 * @param n Number of elements in the array
 * @param label Name of the sorting algorithm for diagnostic output
 */
void verify_sort_results(double **columns, int n, const char *label)
{
    // Create a temporary array of pointers to the columns
    double **tempCopy = (double **)malloc(n * sizeof(double *));
    if (!tempCopy) { fprintf(stderr, "Malloc failed in verify_sort_results\n"); return; }
    for (int i = 0; i < n; i++)
    {
        tempCopy[i] = columns[i];
    }

    // Sort the temporary pointer array using standard qsort
    stdlib_qsort_wrapper(tempCopy, n);

    // Compare the original sorted array with the qsort-ed copy
    long mismatches = 0;
    for (int i = 0; i < n; i++)
    {
        // Compare based on the actual data pointed to
        if (compare_particles(&columns[i], &tempCopy[i]) != 0)
        {
            mismatches++;
        }
    }

    if (mismatches == 0)
    {
        fprintf(stderr, "[DEBUG] SortAlg='%s': Verified => Results match standard qsort.\n", label);
    }
    else
    {
        fprintf(stderr, "[DEBUG] SortAlg='%s': *** MISMATCH *** => %ld rows differ from qsort.\n",
                label, mismatches);
    }

    free(tempCopy);
}

/**
 * Main function for sorting particle data using a specified algorithm.
 *
 * Transposes the data format from particles[component][particle] to
 * columns[particle][component], applies the specified sorting algorithm,
 * and then transposes back to the original format.
 *
 * @param particles 2D array of particle data to be sorted [component][particle]
 * @param npts Number of particles to sort
 * @param sortAlg String identifier of the sorting algorithm to use ("quadsort", "quadsort_parallel", or default to "insertion_parallel")
 */
/**
 * @brief Sorts particle data using the specified sorting algorithm.
 * @details Performs a three-phase particle sorting operation:
 *   1. Memory allocation and data transposition to column-major format 
 *   2. Application of the selected sorting algorithm
 *   3. Reverse transposition of sorted data and memory cleanup
 * 
 * The function uses per-call local buffer allocation for transposing data.
 *
 * @param particles 2D array of particle data to be sorted [component][particle]
 * @param npts Number of particles to sort
 * @param sortAlg Sorting algorithm to use ("quadsort", "quadsort_parallel", 
 *               "insertion", or "insertion_parallel")
 */
void sort_particles_with_alg(double **particles, int npts, const char *sortAlg)
{

    /**
     * Phase 1: Memory allocation and data transposition
     * Prepares the column-major data format required for efficient sorting.
     */

    double **columns_to_sort_on; // Will point to the buffer used for sorting

    /**
     * Persistent buffer allocation strategy.
     * Uses a global buffer to reduce allocation overhead across multiple sort operations.
     */
    
    // Allocate or reallocate only if needed
    if (g_sort_columns_buffer == NULL || g_sort_columns_buffer_npts != npts) {
        // Free existing buffer if size has changed
        if (g_sort_columns_buffer != NULL) {
            for (int i = 0; i < g_sort_columns_buffer_npts; i++) {
                if (g_sort_columns_buffer[i]) free(g_sort_columns_buffer[i]);
            }
            free(g_sort_columns_buffer);
        }

        // Allocate new buffer with the required size
        g_sort_columns_buffer = (double **)malloc(npts * sizeof(double *));
        if (!g_sort_columns_buffer) {
            fprintf(stderr, "ERROR: Malloc failed for g_sort_columns_buffer in sort_particles_with_alg\n");
            CLEAN_EXIT(1);
        }
        
        // Allocate sub-arrays for each particle's components
        for (int i = 0; i < npts; i++) {
            g_sort_columns_buffer[i] = (double *)malloc(5 * sizeof(double));
            if (!g_sort_columns_buffer[i]) { 
                fprintf(stderr, "ERROR: Malloc failed for g_sort_columns_buffer[%d] in sort_particles_with_alg\n", i);
                // Clean up partial allocation
                for(int k=0; k<i; ++k) free(g_sort_columns_buffer[k]);
                free(g_sort_columns_buffer);
                g_sort_columns_buffer = NULL;
                CLEAN_EXIT(1);
            }
        }
        g_sort_columns_buffer_npts = npts;
    }

    // Transpose data from row-major (particles) to column-major (g_sort_columns_buffer)
    #pragma omp parallel for
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < 5; j++) {
            g_sort_columns_buffer[i][j] = particles[j][i];
        }
    }
    columns_to_sort_on = g_sort_columns_buffer;


    /**
     * Phase 2: Apply the selected sorting algorithm
     * Uses one of several sorting algorithms based on input parameter or default.
     * Available algorithms include quadsort (sequential), quadsort_parallel,
     * insertion sort (sequential), and insertion_parallel (default).
     */

    // Select and apply the sorting algorithm
    const char *method = (sortAlg ? sortAlg : "insertion_parallel"); // Default if NULL

    if (strcmp(method, "quadsort") == 0) {
        quadsort_wrapper(columns_to_sort_on, npts);
    }
    else if (strcmp(method, "quadsort_parallel") == 0) {
        quadsort_parallel_sort(columns_to_sort_on, npts);
    }
    else if (strcmp(method, "insertion") == 0) {
        insertion_sort(columns_to_sort_on, npts);
    }
    else { // Default to parallel insertion sort
        insertion_parallel_sort(columns_to_sort_on, npts);
    }


    /**
     * Phase 3: Data transposition and memory cleanup
     * Restores sorted data to original format and performs appropriate cleanup.
     */

    // Transpose sorted data back to the original format
    #pragma omp parallel for
    for (int i = 0; i < npts; i++) {
        for (int j = 0; j < 5; j++) {
            particles[j][i] = columns_to_sort_on[i][j];
        }
    }
    // Note: The persistent buffer g_sort_columns_buffer is NOT freed here.
    // It will be reused for subsequent sort operations and freed at program exit.


}

/**
 * @brief Convenience wrapper function for sorting particles with the default algorithm.
 * @details Calls sort_particles_with_alg using the default sorting algorithm specified in g_defaultSortAlg.
 *
 * Parameters
 * ----------
 * particles : double**
 *     2D array of particle data to be sorted [component][particle]
 * npts : int
 *     Number of particles to sort
 */
void sort_particles(double **particles, int npts)
{
    sort_particles_with_alg(particles, npts, g_defaultSortAlg);
}

/**
 * @brief Comparison function for qsort to sort PartData structures by radius.
 * @details Compares two PartData structures based on their `rad` (radius) member
 *          for sorting in ascending order. Handles NaN values by placing them
 *          consistently (e.g., at the beginning or end, behavior might depend on qsort NaN handling).
 *
 * @param a Pointer to the first PartData structure.
 * @param b Pointer to the second PartData structure.
 * @return int -1 if pa->rad < pb->rad, 1 if pa->rad > pb->rad, 0 otherwise.
 *             Specific return for NaNs ensures consistent ordering.
 */
int compare_partdata_by_rad(const void *a, const void *b)
{
    if (!a && !b) return 0;
    if (!a) return -1;
    if (!b) return 1;

    const struct PartData *pa = (const struct PartData *)a;
    const struct PartData *pb = (const struct PartData *)b;

    int pa_is_nan = (pa->rad != pa->rad); // Check for NaN (safe with fast-math)
    int pb_is_nan = (pb->rad != pb->rad); // Check for NaN (safe with fast-math)

    if (pa_is_nan && pb_is_nan) return 0;
    if (pa_is_nan) return -1;
    if (pb_is_nan) return 1;

    if (pa->rad < pb->rad) return -1;
    if (pa->rad > pb->rad) return 1;
    return 0;
}

/**
 * @brief Sorts an array of PartData structures by their radial position (`rad`).
 * @details Uses the standard library `qsort` function with `compare_partdata_by_rad`
 *          as the comparison function. Includes basic safety checks for NULL array
 *          or non-positive `npts`. The sort is performed in-place.
 *
 * @param array [in,out] Array of PartData structures to be sorted.
 * @param npts  [in] Number of elements in the array.
 */
void sort_by_rad(struct PartData *array, int npts)
{
    if (!array)
    {
        fprintf(stderr, "ERROR: sort_by_rad called with NULL array\n");
        return;
    }
    if (npts <= 0)
    {
        // Sorting an empty or negatively sized array is meaningless or an error.
        // fprintf(stderr, "Warning: sort_by_rad called with npts <= 0: %d\n", npts);
        return; // Nothing to sort
    }

    qsort(array, (size_t)npts, sizeof(struct PartData), compare_partdata_by_rad);
}

// =========================================================================
// SPLINE DATA SORTING UTILITIES
// =========================================================================
//
// Utility functions and structures for sorting spline data arrays.
// Provides mechanisms to sort arrays used for GSL spline creation (like radius `r`
// and potential `Psi`) while maintaining the correct correspondence between
// paired values after sorting based on one of the arrays (typically radius).


/**
 * @brief Comparison function for qsort to sort RrPsiPair structures by the 'rr' member.
 * @details Used to sort an array of RrPsiPair structures in ascending order
 *          based on their radial (`rr`) component. This is primarily used when
 *          preparing data for splines where the x-axis (e.g., radius or -Psi)
 *          must be strictly monotonic.
 *
 * @param a Pointer to the first RrPsiPair structure.
 * @param b Pointer to the second RrPsiPair structure.
 * @return int -1 if pa->rr < pb->rr, 1 if pa->rr > pb->rr, 0 otherwise.
 */
int compare_by_rr(const void *a, const void *b)
{
    const struct RrPsiPair *pa = (const struct RrPsiPair *)a;
    const struct RrPsiPair *pb = (const struct RrPsiPair *)b;
    if (pa->rr < pb->rr)
        return -1;
    if (pa->rr > pb->rr)
        return 1;
    return 0;
}

/**
 * @brief Sorts radius and potential arrays in tandem, maintaining their correspondence.
 * @details This function takes an array of radial coordinates (`rrA_spline`) and an
 *          array of corresponding potential values (`psiAarr_spline`). It sorts
 *          `rrA_spline` in ascending order and applies the identical swaps to
 *          `psiAarr_spline`, ensuring that `psiAarr_spline[i]` still corresponds to
 *          `rrA_spline[i]` after sorting. This is crucial for creating GSL splines
 *          where the x-array must be strictly monotonic and the y-array must maintain
 *          its pairing with the x-values. The arrays are assumed to have `npts + 1` elements,
 *          indexed from 0 to `npts`.
 *
 * @param rrA_spline    [in,out] Array of radial coordinates to be sorted. Modified in-place.
 * @param psiAarr_spline [in,out] Array of corresponding Psi values. Modified in-place in tandem with `rrA_spline`.
 * @param npts          The number of points, typically meaning arrays are of size `npts + 1`.
 */
void sort_rr_psi_arrays(double *rrA_spline, double *psiAarr_spline, int npts)
{
    // Allocate temporary array of pairs
    struct RrPsiPair *pairs = (struct RrPsiPair *)malloc((npts + 1) * sizeof(struct RrPsiPair));
    if (!pairs)
    {
        perror("malloc failed in sort_rr_psi_arrays");
        CLEAN_EXIT(EXIT_FAILURE);
    }

    // Populate the pairs array
    for (int i = 0; i <= npts; i++)
    {
        pairs[i].rr = rrA_spline[i];
        pairs[i].psi = psiAarr_spline[i];
    }

    // Sort the pairs based on the radius value
    qsort(pairs, npts + 1, sizeof(struct RrPsiPair), compare_by_rr);

    // Copy the sorted data back into the original arrays
    for (int i = 0; i <= npts; i++)
    {
        rrA_spline[i] = pairs[i].rr;
        psiAarr_spline[i] = pairs[i].psi;
    }

    free(pairs);
}

/**
 * @brief Constructs a three-dimensional vector from its Cartesian components.
 * @details This utility function initializes a `threevector` structure with the
 *          provided x, y, and z components. It serves as a convenient constructor.
 *
 * @param x [in] The x-component of the vector.
 * @param y [in] The y-component of the vector.
 * @param z [in] The z-component of the vector.
 * @return threevector An initialized `threevector` structure.
 */
threevector make_threevector(double x, double y, double z) {
    return (threevector){x, y, z};
}

/**
 * @brief Computes the scalar dot product of two three-dimensional vectors.
 * @details Calculates \f$X \cdot Y = X_x Y_x + X_y Y_y + X_z Y_z\f$.
 *          The dot product is a measure of the projection of one vector onto another
 *          and is used in various physics calculations, such as determining the
 *          magnitude squared of a vector (\f$V \cdot V = |V|^2\f$) or the angle between vectors.
 *
 * @param X [in] The first threevector operand.
 * @param Y [in] The second threevector operand.
 * @return double The scalar result of the dot product \f$X \cdot Y\f$.
 */
double dotproduct(threevector X, threevector Y) {
  return X.x * Y.x + X.y * Y.y + X.z * Y.z;
}

/**
 * @brief Computes the vector cross product of two three-dimensional vectors.
 * @details Calculates \f$Z = X \times Y\f$, where \f$X = (X_x, X_y, X_z)\f$ and \f$Y = (Y_x, Y_y, Y_z)\f$.
 *          The components of the resulting vector \f$Z\f$ are determined by:
 *          \f$Z_x = X_y Y_z - X_z Y_y\f$
 *          \f$Z_y = X_z Y_x - X_x Y_z\f$
 *          \f$Z_z = X_x Y_y - X_y Y_x\f$
 *          This follows the standard right-hand rule for vector cross products.
 *
 * @param X [in] The first threevector operand.
 * @param Y [in] The second threevector operand.
 * @return threevector The resulting vector \f$Z = X \times Y\f$.
 */
threevector crossproduct(threevector X, threevector Y) {
    threevector Z;
    Z.x = X.y * Y.z - X.z * Y.y;
    Z.y = X.z * Y.x - X.x * Y.z;
    Z.z = X.x * Y.y - X.y * Y.x;
    return Z;
}

/**
 * @brief Calculates self-interacting dark matter (SIDM) scattering cross-section.
 * @details Implements a velocity-independent cross-section model where the opacity
 *          \f$\sigma/m = \kappa\f$ is constant. The default value for \f$\kappa\f$ is
 *          taken from the global variable `g_sidm_kappa`. The total cross-section
 *          \f$\sigma\f$ then scales with the individual particle mass, \f$m_{particle}\f$, which is
 *          derived from the total `halo_mass_for_calc` and the number of particles `npts`.
 *          The function includes necessary unit conversions to return the cross-section
 *          in simulation units (kpc²).
 *
 *          Unit Conversion Detail:
 *          \f$\kappa\f$ (cm²/g) \f$\times m_{particle}\f$ (Msun) \f$\rightarrow \sigma\f$ (kpc²)
 *          The conversion factor used is \f$2.089 \times 10^{-10} \text{ (kpc}^2 \text{ Msun}^{-1}) / (\text{cm}^2 \text{ g}^{-1})\f$,
 *          derived from \f$(1.989 \times 10^{33} \text{ g/Msun}) / (3.086 \times 10^{21} \text{ cm/kpc})^2\f$.
 *
 * @param vrel                 [in] Relative velocity between particles (kpc/Myr). Currently unused as
 *                             the implemented model is velocity-independent. Marked with `__attribute__((unused))`.
 * @param npts                 [in] Total number of simulation particles, used for calculating \f$m_{particle}\f$.
 * @param halo_mass_for_calc [in] Total halo mass (Msun) for the active profile, used for \f$m_{particle}\f$.
 * @param rc_for_calc          [in] Scale radius (kpc) of the active profile. Currently unused in this
 *                             velocity-independent cross-section model. Marked with `__attribute__((unused))`.
 * @return double Total scattering cross-section \f$\sigma\f$ in kpc². Returns 0.0 if `npts` or
 *                the calculated `particle_mass_Msun` is non-positive to prevent errors.
 */
double sigmatotal(double vrel __attribute__((unused)), int npts, double halo_mass_for_calc, double rc_for_calc __attribute__((unused))) {
  double kappa = g_sidm_kappa; // Self-interaction opacity parameter (cm²/g)
  // Ensure npts is positive to prevent division by zero or negative particle mass
  if (npts <= 0) {
    return 0.0;
  }
  double particle_mass_Msun = halo_mass_for_calc / ((double)npts);
  if (particle_mass_Msun <= 0) {
    return 0.0;
  }
  return 2.089e-10 * kappa * particle_mass_Msun; // Cross-section (kpc²)
}

/**
 * @brief Executes self-interacting dark matter (SIDM) scattering for one simulation timestep (Serial version).
 * @details Implements a serial SIDM scattering algorithm. For each particle `i` (the primary scatterer),
 *          it considers up to `nscat` (typically 10) subsequent particles in the array as potential
 *          scattering partners. (The `particles` array is assumed to be sorted by radius, so these
 *          are spatially nearby neighbors).
 *          The algorithm proceeds as follows for each primary particle `i`:
 *          1. Constructs 3D velocity for particle `i`, assigning a random azimuthal angle to its perpendicular component.
 *          2. For each of the `nscat` potential partners `m`:
 *             a. Constructs 3D velocity for partner `m` (assuming a fixed azimuthal orientation for simplicity).
 *             b. Calculates relative velocity \f$v_{rel}\f$ and interaction rate \f$\Gamma_m = \sigma(v_{rel}) v_{rel}\f$.
 *             c. Accumulates total interaction rate \f$\Gamma_{tot} = \sum \Gamma_m\f$.
 *          3. Estimates local particle number density using a shell volume. The shell is defined by particle `i`
 *             and an outer radius typically determined by particle `i + nscat + 1`.
 *          4. Calculates the total scattering probability for particle `i` in timestep `dt`: \f$P_{scatter} \approx \Gamma_{tot} \times (0.5 \times dt / \text{Volume}_{shell})\f$.
 *             The 0.5 factor accounts for double counting pairs.
 *          5. A random number is drawn. If it's less than \f$P_{scatter}\f$, a scatter occurs:
 *             a. One partner `m_scatter` is stochastically chosen from the `nscat` candidates, weighted by their individual \f$\Gamma_m\f$.
 *             b. Isotropic scattering is performed in the center-of-mass frame of the pair \f$(i, m_{scatter})\f$.
 *             c. The 3D velocities of both particles are updated.
 *             d. New radial velocities and angular momenta are calculated from the updated 3D velocities
 *                and stored back into the `particles` array, modifying it in-place.
 *             e. The `g_particle_scatter_state` flags are set for the scattered particles.
 *          The total number of scattering events in this timestep is accumulated.
 *
 * @param particles             [in,out] Main particle data array: `particles[component][current_sorted_index]`.
 *                              Modified in-place with post-scattering velocities/angular momenta.
 * @param npts                  [in] Total number of simulation particles.
 * @param dt                    [in] Integration timestep (Myr).
 * @param current_time          [in] Current simulation time (Myr). Marked `unused` but available.
 * @param rng                   [in] GSL random number generator instance for all stochastic processes.
 * @param Nscatter_total_step   [out] Pointer to a long long to accumulate total scattering events this timestep.
 * @param halo_mass_for_sidm    [in] Total halo mass (Msun) for the active profile, passed to `sigmatotal`.
 * @param rc_for_sidm           [in] Scale radius (kpc) for the active profile, passed to `sigmatotal`.
 */
void perform_sidm_scattering_serial(double **particles, int npts, double dt, double current_time __attribute__((unused)), gsl_rng *rng, long long *Nscatter_total_step, double halo_mass_for_sidm __attribute__((unused)), double rc_for_sidm __attribute__((unused))) {
    long long Nscatters_this_call = 0;
    int i;

    // Iterate through each particle as potential scatterer
    for (i = 0; i < npts - 1; i++) {
        int nscat = 10; // Consider 10 nearest neighbors as scattering candidates
        if (npts - 1 - i < nscat) {
            nscat = npts - 1 - i; // Limit to available particles
        }
        if (nscat <= 0) continue;

        double partialprobability[nscat + 1]; // Interaction rates for each candidate
        double probability_sum_term = 0.0;   // Total interaction rate sum

        // Construct 3D velocity vector for primary particle
        // Random azimuthal orientation for transverse velocity component
        double phii = 2.0 * PI * gsl_rng_uniform(rng);
        double Viperp = particles[2][i] / particles[0][i]; // v_perp = L/r
        threevector Vi = make_threevector(Viperp * cos(phii), Viperp * sin(phii), particles[1][i]);

        // Calculate interaction rates with neighboring particles
        for (int m = 1; m <= nscat; m++) {
            int partner_idx = i + m;

            // Construct 3D velocity for scattering partner
            // Assumes fixed azimuthal alignment for partner particle
            double Vmperp = particles[2][partner_idx] / particles[0][partner_idx];
            threevector Vm = make_threevector(Vmperp, 0.0, particles[1][partner_idx]);

            threevector Vrel_vec = make_threevector(Vi.x - Vm.x, Vi.y - Vm.y, Vi.z - Vm.z);
            double vrel_val = sqrt(dotproduct(Vrel_vec, Vrel_vec));

            // Calculate interaction rate: σ × v_rel
            partialprobability[m] = sigmatotal(vrel_val, npts, halo_mass_for_sidm, rc_for_sidm) * vrel_val;
            probability_sum_term += partialprobability[m];
        }

        // Determine the outer radius of the shell containing these nscat neighbors
        // Option 1 (Consistent): Shell defined by the nscat-th summed neighbor
        // Option 2 (Current): Shell defined by the (nscat+1)-th particle
        int use_nscat_plus_1_for_shell = 1; // Set to 1 for current method, 0 for alternative
        int outer_shell_particle_idx_for_vol;

        if (use_nscat_plus_1_for_shell) {
            outer_shell_particle_idx_for_vol = i + nscat + 1;
        } else {
            outer_shell_particle_idx_for_vol = i + nscat;
        }

        // Ensure the chosen outer index is within bounds
        if (outer_shell_particle_idx_for_vol >= npts) {
            // If out of bounds, try to use the last available particle as the boundary
            if (i + nscat < npts) {
                outer_shell_particle_idx_for_vol = i + nscat;
            } else {
                // No valid shell can be formed
                probability_sum_term = 0.0; // Force no scatter, skip probability calculation
            }
        }
        
        double radius_diff = 0.0;
        // Calculate radius_diff only if there's a chance to scatter and a valid shell
        if (probability_sum_term > 1e-30 && (outer_shell_particle_idx_for_vol > i)) {
            radius_diff = particles[0][outer_shell_particle_idx_for_vol] - particles[0][i];
        } else {
            probability_sum_term = 0.0; // Ensure no scatter if shell is invalid
        }

        double probability = 0.0;
        if (radius_diff > 1e-15 && particles[0][i] > 1e-15 && probability_sum_term > 1e-30) {
            // Calculate scattering probability using shell volume approximation
            probability = probability_sum_term * (0.5) * dt / (4.0 * PI * sqr(particles[0][i]) * radius_diff);
        }

        // Stochastic scattering determination
        if (gsl_rng_uniform(rng) < probability) {
            Nscatters_this_call++;
            int m_scatter = 1; // Default to first neighbor

            // Weighted selection among multiple neighbors
            if (nscat > 1 && probability_sum_term > 1e-15) {
                double cumulative_prob[nscat + 1];
                cumulative_prob[0] = 0.0;
                for (int k = 1; k <= nscat; k++) {
                    cumulative_prob[k] = (k > 1 ? cumulative_prob[k - 1] : 0.0) + partialprobability[k] / probability_sum_term;
                }
                if (nscat > 0) cumulative_prob[nscat] = 1.0;

                double random_select = gsl_rng_uniform(rng);
                m_scatter = 1;
                // Select partner based on cumulative probability distribution
                while (m_scatter < nscat && random_select > cumulative_prob[m_scatter]) {
                    m_scatter++;
                }
            }

            int actual_partner_idx = i + m_scatter;
            if (actual_partner_idx >= npts) {
                Nscatters_this_call--;
                continue;
            }

            // Reconstruct velocities for selected scattering pair
            double Vmperp_scatter = particles[2][actual_partner_idx] / particles[0][actual_partner_idx];
            threevector Vm_scatter = make_threevector(Vmperp_scatter, 0.0, particles[1][actual_partner_idx]);
            threevector Vrel_scatter_vec = make_threevector(Vi.x - Vm_scatter.x, Vi.y - Vm_scatter.y, Vi.z - Vm_scatter.z);
            double vrel_scatter_val = sqrt(dotproduct(Vrel_scatter_vec, Vrel_scatter_vec));

            if (vrel_scatter_val < 1e-15) {
                Nscatters_this_call--;
                continue;
            }

            // Generate isotropic scattering angles in center-of-mass frame
            double costheta = 2.0 * gsl_rng_uniform(rng) - 1.0;
            double sintheta = sqrt(fmax(0.0, 1.0 - costheta * costheta));
            double phif_scatter = 2.0 * PI * gsl_rng_uniform(rng);
            double cf = cos(phif_scatter);
            double sf = sin(phif_scatter);

            // Construct orthonormal coordinate system for scattering transformation
            threevector nhat0, nhat1, nhat2, nhatref;
            nhat0 = make_threevector(Vrel_scatter_vec.x / vrel_scatter_val, Vrel_scatter_vec.y / vrel_scatter_val, Vrel_scatter_vec.z / vrel_scatter_val);

            if (fabs(nhat0.z) < 0.999) {
                nhatref = make_threevector(0.0, 0.0, 1.0);
            } else {
                nhatref = make_threevector(1.0, 0.0, 0.0);
            }

            nhat1 = crossproduct(nhat0, nhatref);
            double normnhat1 = sqrt(dotproduct(nhat1, nhat1));
            if (normnhat1 < 1e-15) {
                // Fallback for parallel vectors
                if (fabs(nhat0.x) < 0.999) {
                    nhatref = make_threevector(1.0, 0.0, 0.0);
                } else {
                    nhatref = make_threevector(0.0, 1.0, 0.0);
                }
                nhat1 = crossproduct(nhat0, nhatref);
                normnhat1 = sqrt(dotproduct(nhat1, nhat1));
                if (normnhat1 < 1e-15) {
                     Nscatters_this_call--; continue;
                }
            }
            nhat1 = make_threevector(nhat1.x / normnhat1, nhat1.y / normnhat1, nhat1.z / normnhat1);
            nhat2 = crossproduct(nhat0, nhat1);

            // Transform scattered velocities from CM frame to lab frame
            threevector nhat_perp_rotated = make_threevector(nhat1.x * cf + nhat2.x * sf, nhat1.y * cf + nhat2.y * sf, nhat1.z * cf + nhat2.z * sf);
            threevector V_rel_final_half = make_threevector(
                (vrel_scatter_val / 2.0) * (costheta * nhat0.x + sintheta * nhat_perp_rotated.x),
                (vrel_scatter_val / 2.0) * (costheta * nhat0.y + sintheta * nhat_perp_rotated.y),
                (vrel_scatter_val / 2.0) * (costheta * nhat0.z + sintheta * nhat_perp_rotated.z)
            );
            threevector V_cm = make_threevector((Vi.x + Vm_scatter.x) / 2.0, (Vi.y + Vm_scatter.y) / 2.0, (Vi.z + Vm_scatter.z) / 2.0);
            
            threevector Vifinal_vec = make_threevector(V_cm.x + V_rel_final_half.x, V_cm.y + V_rel_final_half.y, V_cm.z + V_rel_final_half.z);
            threevector Vmfinal_vec = make_threevector(V_cm.x - V_rel_final_half.x, V_cm.y - V_rel_final_half.y, V_cm.z - V_rel_final_half.z);

            // Apply velocity changes to particle data arrays
            particles[1][i] = Vifinal_vec.z;
            particles[1][actual_partner_idx] = Vmfinal_vec.z;

            double Vperp_i_final_mag = sqrt(sqr(Vifinal_vec.x) + sqr(Vifinal_vec.y));
            double Vperp_m_final_mag = sqrt(sqr(Vmfinal_vec.x) + sqr(Vmfinal_vec.y));

            particles[2][i] = particles[0][i] * Vperp_i_final_mag; // Update angular momentum
            particles[2][actual_partner_idx] = particles[0][actual_partner_idx] * Vperp_m_final_mag;

            // Mark both scattered particles in case it is needed elsewhere
            int orig_id1 = (int)particles[3][i];
            int orig_id2 = (int)particles[3][actual_partner_idx];
            if (orig_id1 >= 0 && orig_id1 < npts) g_particle_scatter_state[orig_id1] = 1;
            if (orig_id2 >= 0 && orig_id2 < npts) g_particle_scatter_state[orig_id2] = 1;
        }
    }

    *Nscatter_total_step = Nscatters_this_call;
}

/**
 * @brief Comparison function for `qsort` to order `ScatterEvent` structures.
 * @details Sorts an array of `ScatterEvent` structures primarily by the first particle's
 *          original index (`i`) in ascending order. If two events have the same primary
 *          particle index `i`, they are then secondarily sorted by the partner's offset
 *          (`m_offset`) in ascending order. This ensures a deterministic (and efficient
 *          for potential cache effects) order when applying buffered scatter updates to
 *          the main particle array, preventing race conditions or non-deterministic outcomes
 *          if multiple scatters involve the same primary particle.
 *
 * @param a [in] Pointer to the first `ScatterEvent` structure.
 * @param b [in] Pointer to the second `ScatterEvent` structure.
 * @return int - An integer less than, equal to, or greater than zero if the first
 *               argument is considered to be respectively less than, equal to,
 *               or greater than the second.
 */
static int compare_scatter_events(const void *a, const void *b) {
    const ScatterEvent *event_a = (const ScatterEvent *)a;
    const ScatterEvent *event_b = (const ScatterEvent *)b;

    if (event_a->i < event_b->i) return -1;
    if (event_a->i > event_b->i) return 1;
    // If i is the same, sort by m_offset
    if (event_a->m_offset < event_b->m_offset) return -1;
    if (event_a->m_offset > event_b->m_offset) return 1;
    return 0;
}

/**
 * @brief Performs SIDM scattering calculations for one timestep using OpenMP for parallelism.
 * @details This function implements a two-phase parallel algorithm for SIDM scattering:
 *          Phase 1 (Parallel Particle Evaluation):
 *            - The main particle loop (over `i`) is parallelized using OpenMP.
 *            - Each thread processes its assigned subset of primary particles (`i`).
 *            - For each particle `i`, it considers `nscat` neighbors (by sorted rank) as potential scattering partners.
 *            - Interaction rates and total scattering probability for particle `i` with its neighbors
 *              are calculated using a shell volume approximation for local density.
 *            - A per-thread GSL RNG (`local_rng` from `rng_per_thread_list`) is used for all
 *              stochastic decisions (scatter occurrence, partner selection, scattering angles).
 *            - If a scatter occurs for particle `i` with a chosen partner `i+m_scatter`, the
 *              resulting final 3D velocities for both particles are computed.
 *            - These outcomes (indices `i`, `m_offset`, and final velocities) are stored in a
 *              `ScatterEvent` structure and added to a dynamically resizing global buffer
 *              (`global_scatter_results`) under an OpenMP critical section to ensure thread-safe appending.
 *          Phase 2 (Serial Update from Buffered Results):
 *            - After the parallel loop completes, a single thread sorts the `global_scatter_results`
 *              (by primary particle index, then partner offset) to ensure deterministic application order.
 *            - It then iterates through the sorted scatter events and updates the main `particles`
 *              array (radial velocity `particles[1]` and angular momentum `particles[2]`) with the
 *              final post-scatter velocities.
 *            - Updates the `g_particle_scatter_state` flags for particles involved in scattering.
 *          The `particles` array is assumed to be sorted by radius prior to calling this function.
 *
 * @param particles             [in,out] Main particle data array: `particles[component][current_sorted_index]`.
 *                              Modified in-place with post-scattering velocities/angular momenta.
 * @param npts                  [in] Total number of particles.
 * @param dt                    [in] Simulation timestep (Myr), used in probability calculation.
 * @param current_time          [in] Current simulation time (Myr). Marked `unused` but available for future use.
 * @param rng_per_thread_list   [in] Array of GSL RNG states, one for each OpenMP thread.
 * @param num_threads_for_rng   [in] The number of allocated RNGs in `rng_per_thread_list` (should match max threads).
 * @param Nscatter_total_step   [out] Pointer to a long long to accumulate the total number of scatter events
 *                              that occurred in this timestep.
 * @param halo_mass_for_sidm    [in] Total halo mass (Msun) for the active profile, passed to `sigmatotal`.
 * @param rc_for_sidm           [in] Scale radius (kpc) for the active profile, passed to `sigmatotal`.
 */
void perform_sidm_scattering_parallel(double **particles, int npts, double dt, double current_time __attribute__((unused)), gsl_rng **rng_per_thread_list, int num_threads_for_rng, long long *Nscatter_total_step, double halo_mass_for_sidm, double rc_for_sidm) {
    long long Nscatters_this_call_atomic = 0; // Accumulated in parallel reduction

    // Buffer for storing scattering event outcomes from all threads
    ScatterEvent *global_scatter_results = NULL;
    size_t global_results_count = 0;
    size_t global_results_capacity = 0;
    // Initial capacity can be a small fraction of npts, e.g., npts/100 or a fixed moderate number
    // Adjust if typical scatter rates are known.
    size_t initial_capacity = (npts > 1000) ? (npts / 100) : 100;
    if (initial_capacity == 0) initial_capacity = 10; // Ensure non-zero for very small npts

    global_scatter_results = (ScatterEvent *)malloc(initial_capacity * sizeof(ScatterEvent));
    if (global_scatter_results == NULL) {
        fprintf(stderr, "Error: Failed to allocate initial global_scatter_results buffer.\n");
        // Don't CLEAN_EXIT here, try to proceed without SIDM for this step or log error
        *Nscatter_total_step = 0;
        return;
    }
    global_results_capacity = initial_capacity;

    #pragma omp parallel reduction(+:Nscatters_this_call_atomic)
    {
        gsl_rng *local_rng = NULL; // Initialize to NULL
        int thread_id_for_rng = 0;
        #ifdef _OPENMP
            thread_id_for_rng = omp_get_thread_num();
        #endif

        if (rng_per_thread_list != NULL && thread_id_for_rng < num_threads_for_rng && rng_per_thread_list[thread_id_for_rng] != NULL) {
            local_rng = rng_per_thread_list[thread_id_for_rng];
        } else {
            // Critical issue: Per-thread RNG not available for an active thread.
            // This should not happen if g_rng_per_thread is sized to omp_get_max_threads()
            // and num_threads_for_rng passed to this function matches that.
            // Proceeding with a shared g_rng would be unsafe and non-reproducible.
            // For now, this thread will not perform scattering.
            #pragma omp critical (rng_error_sidm_parallel)
            {
                fprintf(stderr, "CRITICAL SIDM WARNING: Thread %d has no valid per-thread RNG (num_threads_for_rng=%d). This thread will skip SIDM calculations.\n", thread_id_for_rng, num_threads_for_rng);
                log_message("ERROR", "CRITICAL SIDM: Thread %d missing per-thread RNG.", thread_id_for_rng);
            }
            // To make this thread skip its iterations of the omp for loop:
            // One way is to jump past the loop content for this thread.
            // A cleaner way is to check local_rng before using it inside the loop.
        }


        /**
         * Using schedule(static,1) to ensure deterministic assignment of particles
         * to threads, which is crucial for run-to-run reproducibility of the
         * parallel SIDM simulation when using per-thread RNGs seeded identically
         * across runs (for a fixed number of threads).
         */
        #pragma omp for schedule(static,1)
        for (int i = 0; i < npts - 1; i++) {
            // Check if this thread has a valid RNG before proceeding
            if (local_rng == NULL) {
                continue; // This thread skips its assigned SIDM work
            }
            
            int nscat = 10;
            if (npts - 1 - i < nscat) nscat = npts - 1 - i;
            if (nscat <= 0) continue;

            double partialprobability[nscat + 1]; // Max nscat=10, stack is fine
            double probability_sum_term = 0.0;

            double phii = 2.0 * PI * gsl_rng_uniform(local_rng);
            double Viperp = particles[2][i] / particles[0][i];
            threevector Vi = make_threevector(Viperp * cos(phii), Viperp * sin(phii), particles[1][i]);

            for (int m = 1; m <= nscat; m++) {
                int partner_idx = i + m;
                double Vmperp = particles[2][partner_idx] / particles[0][partner_idx];
                threevector Vm = make_threevector(Vmperp, 0.0, particles[1][partner_idx]);
                threevector Vrel_vec = make_threevector(Vi.x - Vm.x, Vi.y - Vm.y, Vi.z - Vm.z);
                double vrel_val = sqrt(dotproduct(Vrel_vec, Vrel_vec));
                partialprobability[m] = sigmatotal(vrel_val, npts, halo_mass_for_sidm, rc_for_sidm) * vrel_val;
                probability_sum_term += partialprobability[m];
            }

            int use_nscat_plus_1_for_shell_par = 1; // Consistent with serial for now
            int outer_shell_particle_idx_for_vol_par;
            if (use_nscat_plus_1_for_shell_par) {
                outer_shell_particle_idx_for_vol_par = i + nscat + 1;
            } else {
                outer_shell_particle_idx_for_vol_par = i + nscat;
            }
            if (outer_shell_particle_idx_for_vol_par >= npts) {
                if (i + nscat < npts) outer_shell_particle_idx_for_vol_par = i + nscat;
                else probability_sum_term = 0.0;
            }

            double radius_diff_par = 0.0;
            if (probability_sum_term > 1e-30 && (outer_shell_particle_idx_for_vol_par > i) ) {
                radius_diff_par = particles[0][outer_shell_particle_idx_for_vol_par] - particles[0][i];
            } else {
                probability_sum_term = 0.0;
            }

            double probability_par = 0.0;
            if (radius_diff_par > 1e-15 && particles[0][i] > 1e-15 && probability_sum_term > 1e-30) {
                probability_par = probability_sum_term * (0.5) * dt / (4.0 * PI * sqr(particles[0][i]) * radius_diff_par);
            }

            if (gsl_rng_uniform(local_rng) < probability_par) {
                Nscatters_this_call_atomic++; // Atomically increment shared counter
                int m_scatter = 1;
                if (nscat > 1 && probability_sum_term > 1e-15) {
                    double cumulative_prob[nscat + 1];
                    cumulative_prob[0] = 0.0;
                    for (int k_cs = 1; k_cs <= nscat; k_cs++) {
                        cumulative_prob[k_cs] = (k_cs > 1 ? cumulative_prob[k_cs - 1] : 0.0) + partialprobability[k_cs] / probability_sum_term;
                    }
                    if (nscat > 0) cumulative_prob[nscat] = 1.0;
                    double random_select = gsl_rng_uniform(local_rng);
                    while (m_scatter < nscat && random_select > cumulative_prob[m_scatter]) {
                        m_scatter++;
                    }
                }
                int actual_partner_idx = i + m_scatter;
                if (actual_partner_idx >= npts) continue; // Should be rare with nscat logic

                double Vmperp_scatter = particles[2][actual_partner_idx] / particles[0][actual_partner_idx];
                threevector Vm_scatter = make_threevector(Vmperp_scatter, 0.0, particles[1][actual_partner_idx]);
                threevector Vrel_scatter_vec = make_threevector(Vi.x - Vm_scatter.x, Vi.y - Vm_scatter.y, Vi.z - Vm_scatter.z);
                double vrel_scatter_val = sqrt(dotproduct(Vrel_scatter_vec, Vrel_scatter_vec));
                if (vrel_scatter_val < 1e-15) continue;

                double costheta = 2.0 * gsl_rng_uniform(local_rng) - 1.0;
                double sintheta = sqrt(fmax(0.0, 1.0 - costheta*costheta));
                double phif_scatter = 2.0 * PI * gsl_rng_uniform(local_rng);
                double cf = cos(phif_scatter); double sf = sin(phif_scatter);
                threevector nhat0, nhat1, nhat2, nhatref; // Orthonormal basis construction (as in serial)
                nhat0 = make_threevector(Vrel_scatter_vec.x/vrel_scatter_val, Vrel_scatter_vec.y/vrel_scatter_val, Vrel_scatter_vec.z/vrel_scatter_val);
                if (fabs(nhat0.z) < 0.999) nhatref = make_threevector(0.0,0.0,1.0); else nhatref = make_threevector(1.0,0.0,0.0);
                nhat1 = crossproduct(nhat0,nhatref); double normnhat1 = sqrt(dotproduct(nhat1,nhat1));
                if (normnhat1 < 1e-15) { if (fabs(nhat0.x) < 0.999) nhatref = make_threevector(1.0,0.0,0.0); else nhatref = make_threevector(0.0,1.0,0.0);
                    nhat1 = crossproduct(nhat0,nhatref); normnhat1 = sqrt(dotproduct(nhat1,nhat1)); if (normnhat1 < 1e-15) continue; }
                nhat1 = make_threevector(nhat1.x/normnhat1, nhat1.y/normnhat1, nhat1.z/normnhat1);
                nhat2 = crossproduct(nhat0,nhat1);

                threevector nhat_perp_rotated = make_threevector(nhat1.x*cf+nhat2.x*sf, nhat1.y*cf+nhat2.y*sf, nhat1.z*cf+nhat2.z*sf);
                threevector V_rel_final_half = make_threevector( (vrel_scatter_val/2.0)*(costheta*nhat0.x+sintheta*nhat_perp_rotated.x), (vrel_scatter_val/2.0)*(costheta*nhat0.y+sintheta*nhat_perp_rotated.y), (vrel_scatter_val/2.0)*(costheta*nhat0.z+sintheta*nhat_perp_rotated.z) );
                threevector V_cm = make_threevector( (Vi.x+Vm_scatter.x)/2.0, (Vi.y+Vm_scatter.y)/2.0, (Vi.z+Vm_scatter.z)/2.0 );
                
                ScatterEvent current_event;
                current_event.i = i;
                current_event.m_offset = m_scatter; // Store offset, not absolute index
                current_event.Vifinal = make_threevector( V_cm.x+V_rel_final_half.x, V_cm.y+V_rel_final_half.y, V_cm.z+V_rel_final_half.z );
                current_event.Vmfinal = make_threevector( V_cm.x-V_rel_final_half.x, V_cm.y-V_rel_final_half.y, V_cm.z-V_rel_final_half.z );

                #pragma omp critical (add_scatter_result_sidm)
                {
                    if (global_results_count >= global_results_capacity) {
                        size_t new_capacity = (global_results_capacity == 0) ? initial_capacity : global_results_capacity * 2;
                         // Cap growth to avoid excessive memory if many scatters happen (unlikely but safe)
                        if (new_capacity > (size_t)npts && global_results_capacity < (size_t)npts) new_capacity = (size_t)npts;
                        
                        ScatterEvent *new_results_buffer = (ScatterEvent *)realloc(global_scatter_results, new_capacity * sizeof(ScatterEvent));
                        if (!new_results_buffer) {
                            // This is a critical error if realloc fails.
                            // For now, we'll just stop adding results, but ideally, log and potentially terminate.
                             fprintf(stderr, "CRITICAL ERROR: Failed to reallocate global_scatter_results buffer in thread %d.\n", thread_id_for_rng);
                            // To prevent further issues, we could try to signal other threads or exit.
                            // This error means we are likely out of memory.
                        } else {
                            global_scatter_results = new_results_buffer;
                            global_results_capacity = new_capacity;
                        }
                    }
                    // Only add if capacity is sufficient (realloc might have failed)
                    if (global_results_count < global_results_capacity) {
                         global_scatter_results[global_results_count++] = current_event;
                    }
                } // end critical section
            } // end if scatter occurs
        } // end omp for loop over particles i
    } // end parallel region

    // Phase 2: Serial Update - Apply buffered scatter results
    // Sort the collected scatter events to ensure deterministic application order
    if (global_results_count > 1) {
        qsort(global_scatter_results, global_results_count, sizeof(ScatterEvent), compare_scatter_events);
    }
    
    // This part is done by a single thread after the parallel computation.
    for (size_t k = 0; k < global_results_count; k++) {
        int p_i = global_scatter_results[k].i;
        int p_m_offset = global_scatter_results[k].m_offset;
        int p_partner_idx = p_i + p_m_offset;

        // Redundant check, but good for safety, especially if realloc failed silently for some threads
        if (p_i < 0 || p_i >= npts || p_partner_idx < 0 || p_partner_idx >= npts || p_m_offset <= 0) {
            // log_message("WARNING", "Skipping invalid scatter event from buffer: i=%d, partner_idx=%d, m_offset=%d", p_i, p_partner_idx, p_m_offset);
            continue;
        }

        threevector Vifinal_upd = global_scatter_results[k].Vifinal;
        threevector Vmfinal_upd = global_scatter_results[k].Vmfinal;

        particles[1][p_i] = Vifinal_upd.z; // Update radial velocity for particle i
        particles[1][p_partner_idx] = Vmfinal_upd.z; // Update radial velocity for partner

        double Vperp_i_final_mag_upd = sqrt(sqr(Vifinal_upd.x) + sqr(Vifinal_upd.y));
        double Vperp_m_final_mag_upd = sqrt(sqr(Vmfinal_upd.x) + sqr(Vmfinal_upd.y));

        particles[2][p_i] = particles[0][p_i] * Vperp_i_final_mag_upd; // Update L for particle i
        particles[2][p_partner_idx] = particles[0][p_partner_idx] * Vperp_m_final_mag_upd; // Update L for partner

        // Mark both scattered particles in case it is needed elsewhere
        int orig_id1 = (int)particles[3][p_i];
        int orig_id2 = (int)particles[3][p_partner_idx];
        if (orig_id1 >= 0 && orig_id1 < npts) g_particle_scatter_state[orig_id1] = 1;
        if (orig_id2 >= 0 && orig_id2 < npts) g_particle_scatter_state[orig_id2] = 1;
    }

    if (global_scatter_results != NULL) {
        free(global_scatter_results);
    }

    *Nscatter_total_step = Nscatters_this_call_atomic;
}
