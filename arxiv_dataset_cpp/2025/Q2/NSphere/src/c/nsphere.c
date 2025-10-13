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

/**
 * =========================================================================
 * @brief Windows‑compatibility shims
 * =========================================================================
 * Provide POSIX‑style helpers for MinGW/Clang:
 *   • mkdir(path,mode)   → _mkdir(path)
 *   • drand48 / srand48  → wrappers around ANSI rand
 */
#if defined(_WIN32) || defined(_WIN64) || defined(__CYGWIN__)
  #include <direct.h>
  #include <stdlib.h>

  /** Accept 1‑ or 2‑argument forms of mkdir on Windows. */
  #define mkdir(path, ...) _mkdir(path)

  #define NSPHERE_WINDOWS_SHIMS_DONE 1
#endif
/* ========================================================================= */

static char g_file_suffix[256] = ""; ///< Global file suffix string.
static gsl_rng *g_rng = NULL; ///< GSL Random Number Generator state.

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

/**
 * =========================================================================
 * GLOBAL CONFIGURATION AND MACROS
 * =========================================================================
 */

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
int g_doDebug = 0;           ///< Enable detailed debug output.
int g_doDynPsi = 0;          ///< Enable dynamic potential recalculation.
int g_doDynRank = 0;         ///< Enable dynamic rank calculation per step.
int g_doAllParticleData = 0; ///< Save complete particle evolution history.
int g_doRestart = 0;         ///< Enable simulation restart from checkpoint.
int skip_file_writes = 0;    ///< Skip file writes during simulation restart.
int g_enable_logging = 0;    ///< Enable logging to file (controlled by --enable-log flag).

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

/**
 * =========================================================================
 * PHYSICAL CONSTANTS AND ASTROPHYSICAL PARAMETERS
 * =========================================================================
 *
 * Core constants and unit conversion factors for astrophysical calculations
 */
#define PI 3.14159265358979323846 ///< Mathematical constant Pi.
#define G_CONST 4.3e-6           ///< Newton's gravitational constant in kpc (km/sec)^2/Msun.
#define RC 100.0                 ///< Core radius in kpc.
/** @def sqr(x) Calculates the square of a value. */
#define sqr(x) ((x) * (x))
/** @def cube(x) Calculates the cube of a value. */
#define cube(x) ((x) * (x) * (x))
#define kmsec_to_kpcmyr 1.02271e-3 ///< Conversion factor: km/s to kpc/Myr.
#define HALO_MASS 1.0e12           ///< Default halo mass in solar masses (Msun).
#define VEL_CONV_SQ (kmsec_to_kpcmyr * kmsec_to_kpcmyr) ///< Velocity conversion squared (kpc/Myr)^2 per (km/s)^2.

/**
 * @brief Angular momentum selection configuration for particle filtering.
 * @details Mode 0: Select particles with the 5 lowest L values.
 *          Mode 1: Select particles with L values closest to Lcompare.
 */
static int use_closest_to_Lcompare = 1; ///< Mode selector (0 or 1).
static double Lcompare = 0.05;          ///< Reference L value for closest-match mode (Mode 1).

/**
 * =========================================================================
 * GLOBAL PARTICLE DATA ARRAYS
 * =========================================================================
 */

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

/**
 * =========================================================================
 * Energy Debugging and Validation Subsystem
 * =========================================================================
 */

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

/**
 * =========================================================================
 * ENERGY CALCULATION AND INTEGRATION STRUCTURES
 * =========================================================================
 */

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
 * @brief Comparison function for sorting double values (ascending order).
 * @details Used with qsort and other standard library sorting functions.
 *
 * Parameters
 * ----------
 * a : const void*
 *     Pointer to the first double value.
 * b : const void*
 *     Pointer to the second double value.
 *
 * Returns
 * -------
 * int
 *     -1 if a < b, 1 if a > b, 0 if a == b.
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

/**
 * =========================================================================
 * PARTICLE SORTING AND ORDERING OPERATIONS
 * =========================================================================
 */

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

/**
 * =========================================================================
 * PHYSICS CALCULATION FUNCTIONS
 * =========================================================================
 */

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
 * @brief Checks if a string represents a valid integer.
 * @details Allows an optional leading '+' or '-' sign. Checks if all subsequent
 *          characters are digits. Returns false for empty strings or strings
 *          containing only a sign.
 *
 * Parameters
 * ----------
 * str : const char*
 *     The input string to check.
 *
 * Returns
 * -------
 * int
 *     1 if the string is a valid integer, 0 otherwise.
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
 * Checks if a string is a valid floating point number.
 *
 * @param str The string to check
 * @return 1 if string is a valid float, 0 otherwise
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

    while (*str)
    {
        if (isdigit((unsigned char)*str))
        {
            has_digit = 1;
        }
        else if (*str == '.' && !has_decimal)
        {
            has_decimal = 1;
        }
        else
        {
            return 0;
        }
        str++;
    }

    return has_digit;
}

/**
 * =========================================================================
 * COMMAND LINE ARGUMENT PROCESSING
 * =========================================================================
 */

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
 * @note Called when the user specifies --help or when errors occur during
 *       argument parsing.
 */
static void printUsage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --help              Show this usage message.\n"
            "  --restart           [Default Off] Restart processing from the last successful snapshot\n"
            "  --nparticles <int>  [Default 100000] Number of particles\n"
            "  --ntimesteps <int>  [Default 10000] Requested total timesteps\n"
            "                         Note: Ntimes will be adjusted to the minimum value that\n"
            "                         satisfies the constraint (Ntimes - 1) = k*(dtwrite)*(nout)\n"
            "  --nout <int>        [Default 100] Number of output times\n"
            "  --dtwrite <int>     [Default 100] Write interval in timesteps\n"
            "  --tag <string>      [Default Off] Add a custom tag to output filenames (e.g. \"run1\")\n"
            "  --method <int>      Integration method (1..9):\n"
            "                      [Default] 1   Adaptive Leapfrog with Adaptive Levi-Civita\n"
            "                                2   Full-step adaptive Leapfrog + Levi-Civita\n"
            "                                3   Full-step adaptive Leapfrog\n"
            "                                4   Yoshida 4th-order\n"
            "                                5   Adams-Bashforth 3\n"
            "                                6   Leapfrog (vel half-step)\n"
            "                                7   Leapfrog (pos half-step)\n"
            "                                8   Classic RK4\n"
            "                                9   Euler\n"
            "  --methodtag         [Default Off] Include method string in output filenames\n"
            "  --sort <int>        Sorting algorithm (1..4):\n"
            "                      [Default] 1   Parallel Quadsort\n"
            "                                  2   Sequential Quadsort\n"
            "                                  3   Parallel Insertion Sort\n"
            "                                  4   Sequential Insertion Sort\n"
            "  --readinit <file>   [Default Off] Read initial conditions from <file> (binary)\n"
            "  --writeinit <file>  [Default Off] Write initial conditions to <file> (binary)\n"
            "  --tfinal <int>      [Default 5] Factor controlling # of dynamical times\n"
            "  --ftidal <float>    [Default 0.0] Set tidal fraction outer stripping value (0.0 to 1.0)\n"
            "\n"
            "  --save <subargs>    Enable various data-saving modes (may combine any):\n"
            "                                raw-data      => only raw particle data\n"
            "                                psi-snaps     => plus Psi snapshots\n"
            "                                full-snaps    => plus full data snapshots\n"
            "                                debug-energy  => plus energy diagnostics\n"
            "                      [Default] all      => output everything\n"
            "                         If multiple subargs are given, the highest priority\n"
            "                         one overrides the lower: raw-data < psi-snaps <\n"
            "                         full-snaps < all/debug-energy.\n"
            "  --enable-log        [Default Off] Enable writing detailed logs to log/nsphere.log\n"
            "\n"
            "Example:\n"
            "  %s --nparticles 50000 --ntimesteps 20000 --tfinal 5 \\\n"
            "     --nout 100 --dtwrite 10 --method 6 --sort 2 --readinit initial_conditions.bin\n",
            prog, prog);
}

/**
 * @brief Displays an error message, suggests --help, and terminates the program.
 * @details Formats and prints an error message to stderr, includes a suggestion
 *          to use the --help flag for usage information, performs necessary
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
 *     The program name to display in the --help usage suggestion.
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

/**
 * =========================================================================
 * PARTICLE DATA STRUCTURES AND OPERATIONS
 * =========================================================================
 */

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
 * Appends a block of particle data to an output file.
 *
 * Writes particle data (rank, radius, velocity, angular momentum) in step-major order,
 * where all particles for a single time step are stored contiguously. This layout enables
 * efficient seeking to specific snapshots during post-processing and visualization phases.
 *
 * @param filename   Path to the output file
 * @param npts       Number of particles
 * @param block_size Number of time steps in this block
 * @param L_block    Angular momentum data block (npts * block_size)
 * @param Rank_block Particle rank data block (npts * block_size)
 * @param R_block    Radial position data block (npts * block_size)
 * @param Vrad_block Radial velocity data block (npts * block_size)
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
 * Retrieves particle data for a specific snapshot from a binary file
 *
 * This function reads particle data (rank, radius, velocity, angular momentum)
 * for a specific snapshot number from a binary file (`all_particle_data.dat`).
 * It allocates temporary thread-local buffers to avoid race conditions when called
 * in parallel. File I/O (seeking and reading) is performed within an OpenMP critical
 * section (`file_access`) to ensure thread safety. After reading, the data is copied
 * from the local buffers to the output arrays provided by the caller.
 *
 * @param filename    Path to the binary data file
 * @param snap        Snapshot number to retrieve
 * @param npts        Number of particles
 * @param block_size  Number of snapshots per block in the file
 * @param L_out       Output buffer for angular momentum values
 * @param Rank_out    Output buffer for particle ranks
 * @param R_out       Output buffer for radial positions
 * @param Vrad_out    Output buffer for radial velocities
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
 * Adjusts the number of timesteps to ensure proper alignment with output intervals
 *
 * This function calculates an adjusted number of timesteps (`N'`) such that it is
 * at least the requested initial value (`N`) and satisfies the constraint:
 * (N' - 1) = (M - 1) * k * p for some integer k >= 1, where M (`nout`) is the
 * number of snapshots, and p (`dtwrite`) is the write interval.
 *
 * This ensures that the simulation will produce exactly M output snapshots
 * at intervals of approximately k*p timesteps.
 *
 * @param Ntimes_initial  Initially requested number of timesteps (N)
 * @param nout            Number of output snapshots needed (M)
 * @param dtwrite         Write interval in timesteps (p)
 * @return                Adjusted number of timesteps (N') satisfying the constraint
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
 * Comparison function for sorting LAndIndex structures by angular momentum.
 * Used with qsort for ordering particles by L values.
 *
 * @param a  Pointer to first LAndIndex structure
 * @param b  Pointer to second LAndIndex structure
 * @return   -1 if a<b, 1 if a>b, 0 if equal
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

/**
 * =========================================================================
 * ADAPTIVE FULL LEAPFROG STEP: r(n), v(n) --> r(n+1), v(n+1)
 * =========================================================================
 *
 * Physics-based time integration method with adaptive step refinement:
 * - Subdivide the time step h = ΔT in powers-of-2 "micro-steps"
 * - Compare a (2N+1)-step "coarse" vs. a (4N+1)-step "fine" integration
 * - If within tolerance, return one of {coarse, fine, rich (Richardson extrapolation)}
 * - Otherwise, double N and repeat until convergence or max subdivision reached
 */
/**
 * Performs a micro-subdivided leapfrog step.
 *
 * @details This helper function implements the leapfrog integration using a fixed
 * number of micro-steps (`subSteps`, either 2N+1 for coarse or 4N+1 for fine).
 * It takes the current radius and velocity (`r_in`, `v_in`) and applies a
 * Kick-Drift-Kick sequence:
 * 1. Initial half-kick using `halfKick` timestep.
 * 2. A sequence of `(subSteps - 1) / 2 - 1` full Drift-Kick pairs using `midStep` timestep.
 * 3. Final full Drift using `midStep`.
 * 4. Final half-kick using `halfKick`.
 * The timestep sizes `halfKick` and `midStep` depend on whether it's a coarse
 * (`h/(2N)` and `h/N`) or fine (`h/(4N)` and `h/(2N)`) integration sequence.
 * The function returns the new radius and velocity (`r_out`, `v_out`).
 *
 * @param i        Particle index (used for rank in gravitational force).
 * @param npts     Total number of particles.
 * @param r_in     Input radius.
 * @param v_in     Input velocity.
 * @param ell      Angular momentum.
 * @param h        Total timestep size for the full adaptive step (ΔT).
 * @param N        Base subdivision factor for this micro-step sequence.
 * @param subSteps Number of substeps (either 2N+1 or 4N+1).
 * @param grav     Gravitational constant value.
 * @param r_out    Pointer to store the output radius.
 * @param v_out    Pointer to store the output velocity.
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
    double force = gravitational_force(r_curr, i, npts, grav, HALO_MASS);
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
        force = gravitational_force(r_curr, i, npts, grav, HALO_MASS);
        dvdt = force + effective_angular_force(r_curr, ell);
        v_curr += midStep * dvdt;
    }

    // Final full drift
    r_curr += midStep * v_curr;

    // Final half-kick
    force = gravitational_force(r_curr, i, npts, grav, HALO_MASS);
    dvdt = force + effective_angular_force(r_curr, ell);
    v_curr += halfKick * dvdt;

    *r_out = r_curr;
    *v_out = v_curr;
}

/**
 * Performs an adaptive full leapfrog step with error control.
 *
 * @details This function implements the adaptive leapfrog algorithm to advance a
 * particle's state (r, v) over a full timestep `h` (ΔT). It iteratively refines
 * the integration by comparing coarse (2N+1 substeps) and fine (4N+1 substeps)
 * integrations performed via `doMicroLeapfrog`.
 * The process starts with subdivision factor N=1. In each iteration:
 * 1. A "coarse" pass using 2N+1 micro-steps is performed.
 * 2. A "fine" pass using 4N+1 micro-steps is performed.
 * 3. The relative differences between the final radius and velocity from the
 *    coarse and fine passes are compared against `radius_tol` and `velocity_tol`.
 * 4. If both tolerances are met, the integration has converged. The final state
 *    (`r_out`, `v_out`) is determined based on `out_type`:
 *    - 0: Use the coarse result.
 *    - 1: Use the fine result.
 *    - 2: Use Richardson extrapolation (`4*fine - 3*coarse`) for a potentially
 *         higher-order result.
 * 5. If tolerances are not met, the subdivision factor N is doubled (`N *= 2`),
 *    and the process repeats, up to a maximum subdivision factor `max_subdiv`.
 * If convergence is not achieved within `max_subdiv`, the function returns the
 * result from the highest N iteration based on `out_type`.
 *
 * @param i              Particle index (for rank in force computation).
 * @param npts           Total number of particles.
 * @param r_in           Input radius at the start of the step.
 * @param v_in           Input velocity at the start of the step.
 * @param ell            Angular momentum (conserved).
 * @param h              Full physical timestep size ΔT.
 * @param radius_tol     Relative convergence tolerance for radius.
 * @param velocity_tol   Relative convergence tolerance for velocity.
 * @param max_subdiv     Maximum subdivision factor N allowed.
 * @param grav           Gravitational constant value.
 * @param out_type       Result selection: 0=coarse, 1=fine, 2=Richardson extrapolation.
 * @param r_out          Pointer to store the final radius.
 * @param v_out          Pointer to store the final velocity.
 *
 * @see doMicroLeapfrog
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

/**
 * =========================================================================
 * LEVI-CIVITA REGULARIZATION
 * =========================================================================
 *
 * Physics-based regularized time integration method:
 * - Transforms coordinates (r -> ρ = √r) to handle close encounters
 * - Uses fictitious time τ to integrate equations of motion
 * - Maps back to physical coordinates and time after integration
 * - Provides enhanced stability for high-eccentricity orbits
 */

/**
 * Calculates the derivative of rho with respect to the fictitious time tau.
 * Part of the Levi-Civita regularization transformation.
 *
 * @param rhoVal The transformed radial coordinate ρ = √r
 * @param vVal   The velocity in transformed coordinates
 * @return       dρ/dτ = 0.5 * ρ * v
 */
static inline double dRhoDtaufun(double rhoVal, double vVal)
{
    // dρ/dτ = 0.5 * ρ * v
    return 0.5 * rhoVal * vVal;
}

/**
 * Calculates the total force acting on a particle in Levi-Civita transformed coordinates
 *
 * This function combines the gravitational and angular components of force
 * in the ρ (rho) coordinate system which is the square root transformation of radius.
 *
 * @param i          Particle index for force computation
 * @param npts       Total number of particles
 * @param totalmass  Total mass of the system including dark matter halo
 * @param grav       Gravitational constant
 * @param ell        Angular momentum
 * @param rhoVal     Current value of ρ = √r in Levi-Civita coordinates
 * @return           Total force in transformed coordinates
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
 * Performs integration of particle motion using Levi-Civita regularization.
 *
 * @details This function implements the Levi-Civita leapfrog integration algorithm
 * to advance a particle's state over a physical time step `dt`.
 * 1. Transforms coordinates from physical `r_in` to regularized `rho = sqrt(r_in)`.
 * 2. Estimates an initial fictitious time step `deltaTau` aiming for roughly
 *    `N_taumin` steps within the physical interval `dt`.
 * 3. Integrates the equations of motion in `(rho, v_rad, t_phys)` using a
 *    leapfrog scheme in fictitious time `tau`:
 *    - Compute force `fval` at `rho_cur`.
 *    - Half-kick velocity: `v_half = v_cur + 0.5 * deltaTau * fval`.
 *    - Drift rho: `rho_next = rho_cur + deltaTau * dRhoDtaufun(rho_cur, v_half)`.
 *    - Compute force `fval2` at `rho_next`.
 *    - Second half-kick velocity: `v_next = v_half + 0.5 * deltaTau * fval2`.
 *    - Update physical time `t_phys` using the midpoint rho: `t_next = t_phys + deltaTau * rho_mid^2`.
 * 4. The loop continues until `t_phys >= dt`.
 * 5. If a step overshoots `dt` (`t_next > dt`), the final state `(rho_f, v_f)`
 *    at exactly `t_phys = dt` is found using linear interpolation between the
 *    state before and after the overshoot step.
 * 6. Transforms the final regularized state `(rho_f, v_f)` back to physical
 *    coordinates `r_out = rho_f^2` and `v_out = v_f`.
 * Includes a maximum step count to prevent infinite loops.
 *
 * @param i         Particle index (for rank in force calculation).
 * @param npts      Total number of particles.
 * @param r_in      Initial physical radius at time t_i.
 * @param v_in      Initial physical radial velocity at time t_i.
 * @param ell       Angular momentum for the force law.
 * @param dt        Physical time step ΔT to advance.
 * @param N_taumin  Target number of tau-steps (influences `deltaTau` guess).
 * @param grav      Gravitational constant value.
 * @param r_out     Pointer to store the final physical radius at time t_i + ΔT.
 * @param v_out     Pointer to store the final physical velocity at time t_i + ΔT.
 *
 * @see forceLCfun
 * @see dRhoDtaufun
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
        double fval = forceLCfun(i, npts, HALO_MASS, grav, ell, rho_cur);

        // Leapfrog step 2: First half-kick for velocity
        double v_half = v_cur + 0.5 * deltaTau * fval;

        // Leapfrog step 3: Full drift for position
        double rho_next = rho_cur + deltaTau * dRhoDtaufun(rho_cur, v_half);

        // Leapfrog step 4: Second half-kick with force at new position
        double fval2 = forceLCfun(i, npts, HALO_MASS, grav, ell, rho_next);
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

/**
 * =========================================================================
 * ADAPTIVE FULL LEVI-CIVITA REGULARIZATION
 * =========================================================================
 *
 * Enhanced regularization scheme that combines adaptive step sizing with
 * Levi-Civita coordinate transformation for optimal performance near
 * the coordinate origin.
 */

/**
 * @brief Performs Levi-Civita regularized integration for a fixed number of micro-steps.
 * @details Integrates the equations of motion in (rho, v, t) using fictitious time tau
 *          for a specified number of `subSteps`. This is a helper function used by
 *          `doSingleTauStepAdaptiveLeviCivita`.
 *
 * Parameters
 * ----------
 * i : int
 *     Particle index (for rank in force calculation).
 * npts : int
 *     Total number of particles in the simulation.
 * rho_in : double
 *     Initial rho = sqrt(r) at the start of the tau interval `h_tau`.
 * v_in : double
 *     Initial radial velocity at the start of the tau interval.
 * t_in : double
 *     Initial physical time at the start of the tau interval.
 * subSteps : int
 *     Number of micro-steps to perform (e.g., 2N+1 or 4N+1).
 * h_tau : double
 *     Total fictitious time interval (Delta tau) for this integration sequence.
 * grav : double
 *     Gravitational constant value (e.g., G_CONST).
 * ell : double
 *     Angular momentum per unit mass (conserved).
 * rho_out : double*
 *     Pointer to store the final rho value after `h_tau`.
 * v_out : double*
 *     Pointer to store the final velocity after `h_tau`.
 * t_out : double*
 *     Pointer to store the final physical time after `h_tau`.
 *
 * Returns
 * -------
 * None (results are returned via rho_out, v_out, t_out pointers).
 *
 * @see doSingleTauStepAdaptiveLeviCivita
 * @see forceLCfun
 * @see dRhoDtaufun
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
        double fLC = forceLCfun(i, npts, HALO_MASS, grav, ell, rho_curr);
        double v_half = v_curr + 0.5 * dtau * fLC;

        // Drift for rho.
        double rho_next = rho_curr + dtau * (0.5 * rho_curr * v_half);

        // Second half-kick.
        double fLC2 = forceLCfun(i, npts, HALO_MASS, grav, ell, rho_next);
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
 * @brief Performs a single adaptive step in Levi-Civita coordinates with error control.
 * @details Integrates over a proposed fictitious time step `h_guess` (Δτ) using adaptive
 *          refinement. It compares coarse (2N+1 substeps) and fine (4N+1 substeps)
 *          integrations via `doMicroLeviCivita`.
 *          The process starts with N=1. In each iteration:
 *          1. Perform coarse integration (2N+1 substeps) to get `(rhoC, vC, tC)`.
 *          2. Perform fine integration (4N+1 substeps) to get `(rhoF, vF, tF)`.
 *          3. Compare relative differences in `rho` and `v` against tolerances.
 *          4. If converged: Return result `(rho_out, v_out, t_out)` based on `out_type`.
 *             - `out_type=0`: Coarse result `(rhoC, vC, tC)`.
 *             - `out_type=1`: Fine result `(rhoF, vF, tF)`.
 *             - `out_type=2`: Richardson extrapolation result (`4*fine - 3*coarse`) applied
 *                             to rho, v, and t for a higher-order estimate. A floor of 0.0
 *                             is applied to the extrapolated rho.
 *          5. If not converged: Double N (`N *= 2`) and repeat, up to `max_subdiv`.
 *          If convergence is not achieved, the result from the highest `N` fine step
 *          (`4*N+1` substeps) is calculated and returned.
 *
 * @param i         Particle index (for rank in force calculation).
 * @param npts      Total number of particles.
 * @param rho_in    Initial rho = sqrt(r) at the start of this adaptive tau step.
 * @param v_in      Initial radial velocity at the start of this step.
 * @param t_in      Initial physical time at the start of this step.
 * @param h_guess   Proposed fictitious time step (Δτ guess) for this adaptive step.
 * @param radius_tol  Relative convergence tolerance for rho comparison.
 * @param velocity_tol Relative convergence tolerance for velocity comparison.
 * @param max_subdiv Maximum allowed value for the subdivision factor `N`.
 * @param grav      Gravitational constant value (e.g., G_CONST).
 * @param ell       Angular momentum per unit mass (conserved).
 * @param out_type  Result selection mode: 0=coarse, 1=fine, 2=Richardson extrapolation.
 * @param rho_out   Pointer to store the final rho value after the adaptive step.
 * @param v_out     Pointer to store the final velocity after the adaptive step.
 * @param t_out     Pointer to store the final physical time after the adaptive step.
 *
 * @return None (results are returned via rho_out, v_out, t_out pointers).
 *
 * @see doMicroLeviCivita
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
 * @brief Performs an adaptive full Levi-Civita regularized leapfrog step over physical time dt.
 * @details Integrates particle motion over a physical time step `dt` (ΔT) using the adaptive
 *          Levi-Civita scheme. It takes multiple adaptive `tau` steps (using
 *          `doSingleTauStepAdaptiveLeviCivita`) until the accumulated physical time `t_cur`
 *          reaches or exceeds `dt`.
 *          1. Handles the edge case of near-zero initial radius `r_in` by returning immediately.
 *          2. Transforms initial physical state `(r_in, v_in)` to regularized `(rho_current, v_current)`.
 *          3. Estimates an initial fictitious time step `deltaTau` based on `r_in` and `N_taumin`.
 *          4. Enters a loop that continues as long as `t_cur < dt`:
 *             a. Calls `doSingleTauStepAdaptiveLeviCivita` to perform one adaptive τ-step,
 *                advancing the state to `(rho_next, v_next, t_next)`.
 *             b. Checks if the step overshot the target time (`t_next > dt`).
 *             c. If overshoot: Performs linear interpolation between the state before
 *                (`rho_current`, `v_current`, `t_cur`) and after (`rho_next`, `v_next`, `t_next`)
 *                the step to find the state `(rho_final, v_final)` exactly at `t = dt`.
 *                Transforms back to physical `(r_out, v_out)` and returns.
 *             d. If no overshoot: Updates the current state (`rho_current = rho_next`, etc.)
 *                and continues the loop.
 *          5. Includes a maximum step count (`stepMax`) as a safety measure against infinite loops.
 *          6. If the loop completes normally (e.g., `t_cur` exactly equals `dt` on step acceptance,
 *             or if the overshoot check somehow fails), it transforms the final state
 *             `(rho_current, v_current)` back to physical coordinates `(r_out, v_out)`.
 *
 * @param i         Particle index (for rank in force calculation).
 * @param npts      Total number of particles.
 * @param r_in      Initial physical radius at the start of the physical step `dt`.
 * @param v_in      Initial physical radial velocity at the start of the step `dt`.
 * @param ell       Angular momentum per unit mass (conserved).
 * @param dt        Physical time step (ΔT) to advance.
 * @param N_taumin  Target number of tau-steps (influences `deltaTau` guess).
 * @param radius_tol  Relative convergence tolerance for rho comparison (passed to substep function).
 * @param velocity_tol Relative convergence tolerance for velocity comparison (passed to substep function).
 * @param max_subdiv Maximum allowed value for the subdivision factor `N` (passed to substep function).
 * @param grav      Gravitational constant value (e.g., G_CONST).
 * @param out_type  Result selection mode for substeps: 0=coarse, 1=fine, 2=Richardson.
 * @param r_out     Pointer to store the final physical radius after time `dt`.
 * @param v_out     Pointer to store the final physical velocity after time `dt`.
 *
 * @return None (results are returned via r_out, v_out pointers).
 *
 * @see doSingleTauStepAdaptiveLeviCivita
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

/**
 * =========================================================================
 * FILE I/O AND DATA MANAGEMENT SUBSYSTEM
 * =========================================================================
 *
 * Functions for saving, loading, and managing simulation data including:
 * - Initial condition generation and I/O
 * - Snapshot file management
 * - Binary file format utilities
 */

static int doReadInit = 0;            ///< Flag indicating whether to read initial conditions from file (1=yes, 0=no).
static int doWriteInit = 0;           ///< Flag indicating whether to write initial conditions to file (1=yes, 0=no).
static const char *readInitFilename = NULL; ///< Filename to read initial conditions from (if doReadInit=1).
static const char *writeInitFilename = NULL; ///< Filename to write initial conditions to (if doWriteInit=1).

/**
 * =========================================================================
 * INITIAL CONDITION FILE I/O FUNCTIONS
 * =========================================================================
 */

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

/**
 * =========================================================================
 * SORTING ALGORITHM CONFIGURATION
 * =========================================================================
 */
///< Default sorting algorithm identifier string. Set based on command-line options.
static const char *g_defaultSortAlg = "quadsort_parallel";

/**
 * @brief Returns a human-readable description for a sort algorithm identifier string.
 * @details Maps internal sort algorithm identifiers to user-friendly descriptive names.
 *          Supported algorithms: quadsort_parallel, quadsort, insertion_parallel, insertion.
 *
 * @param sort_alg The internal algorithm identifier string (e.g., "quadsort_parallel")
 * @return A descriptive name for the algorithm (e.g., "Parallel Quadsort"),
 *         or the input string itself if no match is found
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

/**
 * =========================================================================
 * BINARY FILE I/O UTILITIES
 * =========================================================================
 */

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
 * @brief Parses sub-arguments for the "--save" command-line option.
 * @details Reads subsequent arguments after "--save" (until another option starting
 *          with '-' is encountered) and sets global data output flags (`g_doDebug`,
 *          `g_doDynPsi`, `g_doDynRank`, `g_doAllParticleData`) based on the
 *          highest priority sub-argument found.
 *          Valid sub-arguments: "raw-data", "psi-snaps", "full-snaps", "debug-energy", "all".
 *
 * Parameters
 * ----------
 * argc : int
 *     Total argument count from main().
 * argv : char**
 *     Argument array from main().
 * pIndex : int*
 *     Pointer to the current index in `argv` (pointing to "--save").
 *     This index is updated by the function to skip the consumed sub-arguments.
 *
 * Returns
 * -------
 * None (modifies global flags and `*pIndex`).
 *
 * @note Priority order: raw-data < psi-snaps < full-snaps < all/debug-energy.
 * @warning Exits(1) if an unknown sub-argument is encountered.
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
 * @brief Remaps original particle IDs to their rank (0...n-1) among the final set.
 * @details Takes an array of potentially non-contiguous original particle IDs
 *          (stored as doubles) and replaces each ID with its zero-based rank
 *          within the sorted sequence of those IDs. This effectively transforms
 *          arbitrary ID values into a contiguous sequence [0, n-1]. Used after
 *          tidal stripping.
 *
 * Parameters
 * ----------
 * orig_ids : double*
 *     Array of original particle IDs (stored as doubles). Modified in-place.
 * n : int
 *     Number of elements in the `orig_ids` array (the final number of particles, `npts`).
 *
 * Returns
 * -------
 * None (modifies the `orig_ids` array in-place).
 *
 * @note Uses qsort and binary search internally. Allocates temporary memory.
 * @warning Exits(1) if temporary memory allocation fails.
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

/**
 * =========================================================================
 * SIGNAL PROCESSING AND FILTERING UTILITIES
 * =========================================================================
 *
 * Advanced numerical processing utilities for density field handling including:
 * - FFT-based convolution for density smoothing
 * - Direct Gaussian convolution for smaller datasets
 * - Signal filtering and processing functions
 */
/**
 * =========================================================================
 * FFT METHODS AND CONVOLUTION IMPLEMENTATIONS
 * =========================================================================
 */
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

/**
 * =========================================================================
 * RESTART AND RECOVERY MANAGEMENT
 * =========================================================================
 */

/**
 * @brief Finds the index of the last successfully processed snapshot in restart mode.
 * @details Checks for the existence and basic validity (non-zero size, consistent size
 *          compared to snapshot 0) of expected output data files (unsorted and sorted
 *          Rank/Mass/Rad/Vrad files) for each snapshot step defined in `snapshot_steps`.
 *          Used to determine where to resume processing when the `--restart` flag is enabled.
 *
 * Parameters
 * ----------
 * snapshot_steps : int*
 *     Array containing the step numbers (indices) of the snapshots that were *supposed*
 *     to be written during the simulation.
 * noutsnaps : int
 *     Total number of snapshot indices in the `snapshot_steps` array.
 *
 * Returns
 * -------
 * int
 *     - The index `s` in `snapshot_steps` corresponding to the *next* snapshot
 *       that needs processing (i.e., `last_valid_index + 1`).
 *     - Returns -1 if no valid snapshot files are found (start from beginning).
 *     - Returns -2 if *all* expected snapshot files exist and appear valid (nothing to do).
 *
 * @note Compares file sizes against the first snapshot's files as a reference, allowing
 *       for a +/- 5% tolerance. Assumes `g_file_suffix` is set correctly.
 * @warning Prints status messages and warnings/errors to stdout/log.
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
        sprintf(base_filename_1, "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_1, 1, fname_unsorted, sizeof(fname_unsorted));
        char base_filename_2[256];
        sprintf(base_filename_2, "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
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
        sprintf(base_filename_3, "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
        get_suffixed_filename(base_filename_3, 1, fname_unsorted, sizeof(fname_unsorted));

        char base_filename_4[256];
        sprintf(base_filename_4, "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
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

    int method_select = 1;            // Default: option 1 (Adaptive Leapfrog with Adaptive Levi-Civita)
    int display_sort = 1;             // Default: option 1 (Parallel Quadsort)
    int include_method_in_suffix = 0; // Default: exclude method from filenames.
    char custom_tag[256] = {0};       // Default: no custom tag.

    /** @note Check for the --help argument first before parsing other options. */
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
                sprintf(buf, "method must be in [1..9]");
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
                sprintf(buf, "sort must be in [1..4]");
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

            /** @warning Check for incompatibility with --restart and --writeinit. */
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

            /** @warning Check for incompatibility with --restart and --readinit. */
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
            /** @warning Check for incompatibility with --readinit and --writeinit. */
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
                sprintf(buf, "tidal_fraction must be in [0.0..1.0]");
                errorAndExit(buf, NULL, argv[0]);
            }
        }
        else if (strcmp(argv[i], "--methodtag") == 0)
        {
            /** @note Flag to include integration method string in output filename suffix. */
            include_method_in_suffix = 1;
        }
        else if (strcmp(argv[i], "--enable-log") == 0)
        {
            /** @note Flag to enable logging to log/nsphere.log. */
            g_enable_logging = 1;
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
        sprintf(g_file_suffix, "_%s", custom_tag);
    }

    /** @brief Add method/parameter tag to suffix. */
    char temp[256];
    if (include_method_in_suffix)
    {
        sprintf(temp, "_%s_%d_%d_%d", method_str, npts, Ntimes, tfinal_factor);
    }
    else
    {
        sprintf(temp, "_%d_%d_%d", npts, Ntimes, tfinal_factor);
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
         *       suffixed version for compatibility with scripts. Uses copy on Windows. */
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
        sprintf(command, "rm -f \"%s\" 2>/dev/null", linkname);
        system(command);

        // Then create the symbolic link - use the basename of the file, not the full path
        // Extract the basename from filename
        const char *basename = strrchr(filename, '/');
        basename = basename ? basename + 1 : filename; // Skip the '/' or use full name if no '/'

        sprintf(command, "ln -s \"%s\" \"%s\"", basename, linkname);
        if (system(command) != 0)
        {
            // If symbolic link fails, fall back to copying the file.
            sprintf(command, "cp \"%s\" \"%s\"", filename, linkname);
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
                double alpha = 1.0;
                double calE;
                gsl_integration_workspace *w = gsl_integration_workspace_alloc(Nintegration);
                int i;

                gsl_function F;
                F.function = &massintegrand;
                F.params = &alpha;

                double rmax = 10.0 * RC;
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
                    mass[i] = result * HALO_MASS / normalization;
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
                sprintf(base_filename_massprofile, "data/massprofile_Ni%d_Ns%d.dat", Nintegration, Nspline);
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
                for (i = 0; i < num_points; i++)
                {
                    double r = (double)i * rmax / ((double)num_points);
                    double r1 = fmax(r, RC / 1000000.0);
                    F.function = &Psiintegrand;
                    gsl_integration_qagiu(&F, r1, 0, 1e-12, Nintegration, w, &result, &error);
                    double first_term = G_CONST * gsl_spline_eval(splinemass, r1, enclosedmass) / r1;
                    double second_term = G_CONST * result * HALO_MASS / normalization;
                    Psivalues[i] = (first_term + second_term);
                    nPsivalues[i] = -Psivalues[i];
                }

                gsl_interp_accel *Psiinterp = gsl_interp_accel_alloc();
                gsl_spline *splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
                gsl_spline_init(splinePsi, radius, Psivalues, num_points);

                /** @note Write potential profile file for these params (e.g., data/Psiprofile_Ni1k_Ns1k.dat). */
                char base_filename_psiprofile[256];
                sprintf(base_filename_psiprofile, "data/Psiprofile_Ni%d_Ns%d.dat", Nintegration, Nspline);
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
                sprintf(base_filename_integrand, "data/integrand_Ni%d_Ns%d.dat", Nintegration, Nspline);
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

                innerintegrandvalues[0] = 0.0;
                Evalues[0] = Psimin;
                for (i = 1; i <= num_points; i++)
                {
                    calE = Psimin + (Psimax - Psimin) * ((double)i) / ((double)num_points);
                    fEintegrand_params params2 = {calE, splinerofPsi, splinemass, rofPsiinterp, enclosedmass};
                    F.params = &params2;
                    gsl_integration_qag(&F, sqrt(calE - Psimin) / 1.0e4, sqrt(calE - Psimin), 1.0e-12, 1.0e-12, Nintegration, 6, w, &result, &error);
                    innerintegrandvalues[i] = result;
                    Evalues[i] = calE;
                }

                gsl_interp *fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
                gsl_interp_init(fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
                gsl_interp_accel *fofEacc = gsl_interp_accel_alloc();

                /** @note Write theoretical density profile file for these params (e.g., data/density_profile_NiX_NsY.dat). */
                char base_filename[256];
                sprintf(base_filename, "data/density_profile_Ni%d_Ns%d.dat", Nintegration, Nspline);
                get_suffixed_filename(base_filename, 1, fname, sizeof(fname));
                fp = fopen(fname, "wb"); // Binary mode for fprintf_bin
                for (i = 0; i < num_points; i++)
                {
                    double rr = radius[i];
                    double rho_r = HALO_MASS / normalization * (1.0 / cube(1.0 + sqr(rr / RC)));
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
                    double rho_left = HALO_MASS / normalization * (1.0 / cube(1.0 + sqr(radius[i - 1] / RC)));
                    double rho_right = HALO_MASS / normalization * (1.0 / cube(1.0 + sqr(radius[i + 1] / RC)));
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
                sprintf(base_filename_fofe, "data/f_of_E_Ni%d_Ns%d.dat", Nintegration, Nspline);
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
                    double fE = deriv / (sqrt(5.0) * PI * PI);
                    if (E == 0.0)
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
                            fEval = dval / (sqrt(5.0) * PI * PI) * vtest * vtest * r_fixed * r_fixed;
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
                free(Psivalues);
                free(nPsivalues);
                free(innerintegrandvalues);
                free(Evalues);
                gsl_integration_workspace_free(w);
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
    double result, error, r;
    double alpha = 1.0;
    double calE;
    gsl_integration_workspace *w = gsl_integration_workspace_alloc(1000);
    int i;

    gsl_function F;
    F.function = &massintegrand;
    F.params = &alpha;

    double rmax = 10.0 * RC;
    gsl_integration_qag(&F, 0.0, rmax, 0, 1.0e-12, 1000, 5, w, &result, &error);
    normalization = result;

    int num_points = 10000;
    double *mass = (double *)malloc(num_points * sizeof(double));
    double *radius = (double *)malloc(num_points * sizeof(double));

    for (int i = 0; i < num_points; i++)
    {
        double r = (double)i * rmax / (num_points);
        gsl_integration_qag(&F, 0.0, r, 0, 1.0e-12, 1000, 5, w, &result, &error);
        mass[i] = result * HALO_MASS / normalization;
        radius[i] = r;
    }

    gsl_interp_accel *enclosedmass = gsl_interp_accel_alloc();
    gsl_spline *splinemass = gsl_spline_alloc(gsl_interp_cspline, num_points);
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
    char fname[256];
    FILE *fp;
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
    double *Psivalues = (double *)malloc(num_points * sizeof(double));
    double *nPsivalues = (double *)malloc(num_points * sizeof(double));
    for (i = 0; i < num_points; i++)
    {
        double r = (double)i * rmax / ((double)num_points);
        double r1 = fmax(r, RC / 1000000.0);
        F.function = &Psiintegrand;
        gsl_integration_qagiu(&F, r1, 0, 1e-12, 1000, w, &result, &error);
        double first_term = G_CONST * gsl_spline_eval(splinemass, r1, enclosedmass) / r1;
        double second_term = G_CONST * result * HALO_MASS / normalization;
        Psivalues[i] = (first_term + second_term);
        nPsivalues[i] = -Psivalues[i];
    }

    gsl_interp_accel *Psiinterp = gsl_interp_accel_alloc();
    gsl_spline *splinePsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
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

    /** @brief Create primary inverse spline r(Psi) `splinerofPsi`. */
    gsl_interp_accel *rofPsiinterp = gsl_interp_accel_alloc();
    gsl_spline *splinerofPsi = gsl_spline_alloc(gsl_interp_cspline, num_points);
    gsl_spline_init(splinerofPsi, nPsivalues, radius, num_points);

    /** @brief Calculate primary inner integral I(E) and create primary `fofEinterp`. */
    double *innerintegrandvalues = (double *)malloc((num_points + 1) * sizeof(double));
    double *Evalues = (double *)malloc((num_points + 1) * sizeof(double));
    double Psimin = Psivalues[num_points - 1];
    double Psimax = Psivalues[0];
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

    gsl_interp *fofEinterp = gsl_interp_alloc(gsl_interp_cspline, num_points + 1);
    gsl_interp_init(fofEinterp, Evalues, innerintegrandvalues, num_points + 1);
    gsl_interp_accel *fofEacc = gsl_interp_accel_alloc();

    /**
     * @brief Allocate memory for the main particle data array `particles`.
     * @details Allocates a 2D array `particles[5][npts_initial]`.
     *          Component indices: 0=radius, 1=velocity magnitude (initially), 2=ang. mom.,
     *          3=ID (initial index 0..npts_initial-1), 4=orientation(mu).
     *          Uses `npts_initial` which accounts for potential tidal stripping oversampling.
     * @warning Exits via `CLEAN_EXIT(1)` if allocation fails.
     */
    double **particles = (double **)malloc(5 * sizeof(double *));
    if (particles == NULL)
    {
        printf("Memory allocation failed for particle array pointer\n");
        CLEAN_EXIT(1);
    }
    for (i = 0; i < 5; i++)
    {
        particles[i] = (double *)malloc(npts_initial * sizeof(double));
        if (particles[i] == NULL)
        {
            printf("Memory allocation failed for particle attribute array %d\n", i);
            for(int k=0; k<i; ++k) free(particles[k]);
            free(particles);
            CLEAN_EXIT(1);
        }
    }

    /** @brief Initialize particle arrays with default values (0.0, ID=index). */
    for (i = 0; i < npts_initial; i++)
    {
        particles[0][i] = 0.0;
        particles[1][i] = 0.0;
        particles[2][i] = 0.0;
        particles[3][i] = (double)i; // Initial ID is the index
        particles[4][i] = 0.0;
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
    
    // Seed the generator using time and process ID for uniqueness
    unsigned long int seed = (unsigned long int)time(NULL) + (unsigned long int)getpid();
    gsl_rng_set(g_rng, seed);                      // Set the RNG seed

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
        if (tidal_fraction > 0.0) printf("Initial particle count before stripping: %d\n", npts_initial);
        printf("Generating initial conditions...\n\n");

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
                double currentvalue = vel * vel * gsl_interp_eval_deriv(fofEinterp, Evalues, innerintegrandvalues, Psir - (0.5) * vel * vel, fofEacc);
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
            double mass_sample = mass_frac_sample * HALO_MASS;
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
                double target_func_val = vel * vel * gsl_interp_eval_deriv(fofEinterp, Evalues, innerintegrandvalues, Psir - (0.5) * vel * vel, fofEacc);
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
        /** @note Only show stripping message if --ftidal was used. */
        if (tidal_fraction > 0.0) printf("Tidal stripping: sorting and retaining inner %.1f%% of particles...\n", (1.0 - tidal_fraction) * 100.0);

        /** @note Sort all `npts_initial` particles by radius using basic quadsort. */
        sort_particles_with_alg(particles, npts_initial, "quadsort"); // Sorts by particles[0]

        /** @note Allocate new smaller arrays (`final_particles`) for the `npts` particles to keep. */
        double **final_particles = (double **)malloc(5 * sizeof(double *));
        if (final_particles == NULL)
        {
            printf("Memory allocation failed for final_particles\n");
            CLEAN_EXIT(1);
        }

        /** @brief Copy innermost `npts` particles to final arrays and replace `particles` pointers. */
        for (int i = 0; i < 5; i++) // Loop over components
        {
            final_particles[i] = (double *)malloc(npts * sizeof(double));
            if (final_particles[i] == NULL)
            {
                printf("Memory allocation failed for final_particles[%d]\n", i);
                CLEAN_EXIT(1);
            }
            /** @note Copy only the first `npts` elements (innermost after sort). */
            memcpy(final_particles[i], particles[i], npts * sizeof(double));
            
            /** @note Free original oversized array and update `particles[i]` pointer. */
            free(particles[i]);                // Free the original oversized array
            particles[i] = final_particles[i]; // particles[i] now points to the smaller array
        }
        free(final_particles); // Free the temporary ** structure, not the data arrays

        /** @note Only show completion message if --ftidal was used. */
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
    double tdyn = 1.0 / sqrt((VEL_CONV_SQ * G_CONST) * HALO_MASS / cube(RC));
    double totaltime = (double)tfinal_factor * tdyn; ///< Total simulation time (Myr)
    double dt = totaltime / ((double)Ntimes);        ///< Individual timestep size (Myr)
    printf("Dynamical time tdyn = %.4f Myr\n", tdyn);
    printf("Total simulation time = %.4f Myr (%.1f tdyn)\n", totaltime, (double)tfinal_factor);
    printf("Timestep dt = %.6f Myr\n", dt);

    /** @brief Initialize simulation time tracking and progress reporting. */
    double time = 0.0;                   ///< Current simulation time (Myr)
    double start_time = omp_get_wtime(); ///< Wall-clock start time for timing
    /** @brief Setup progress reporting steps (array `print_steps` holding step numbers for 0%, 5%, ..., 100%). */
    int print_steps[21];
    for (int k = 0; k <= 20; k++) print_steps[k] = (int)floor(k * 0.05 * Ntimes); // Calculate steps for progress output

    /** @brief Flag to determine if simulation phase can be skipped. */
    int skip_simulation = 0; 

    /** @note Print initial progress (0%) if not restarting simulation loop. */
    if (!g_doRestart || !skip_simulation) // Print if not restarting OR if restart requires simulation
        printf("0%% complete, timestep %d/%d, time=%.4f Myr, elapsed=%.2f s\n", print_steps[0], Ntimes, time, omp_get_wtime() - start_time);

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
            fprintf(stderr, "Error: Invalid remapped_id %d encountered at index %d\n", final_rank_id, i);
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

    /** @brief Calculate mass per particle based on *initial* particle count before stripping. Used for M(rank). */
    double deltaM = HALO_MASS / (double)npts_initial;

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
            printf("Error: cannot create all_particle_data output file %s\n", apd_filename);
            CLEAN_EXIT(1);
        }
        fclose(fapd); // Close immediately, file is now ready for appending.
        printf("Initialized empty file for all particle data: %s\n", apd_filename);
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
     * @details Iterates `Ntimes + dtwrite` times (index `j`). The loop body contains
     *          the core integration logic, selecting the method based on `method_select`.
     *          The entire loop is skipped if `skip_simulation` is true (restart mode).
     * @note The loop includes `dtwrite` extra steps beyond `Ntimes`, potentially for
     *       ensuring trajectory arrays (`ext_Ntimes`) are filled for post-loop access.
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[0][i] += drdt * dt;
                    particles[1][i] += dvdt * dt;
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] += dvdt * dt;
                    particles[0][i] += particles[1][i] * (dt / 2.0);
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
                    double dvdt = force + effective_angular_force(r, ell);

                    particles[1][i] = vrad + 0.5 * dvdt * dt;
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
                        double M_enc = ((double)i / (double)npts) * HALO_MASS;
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
                            M_enc = 0.1 * (1.0 / (double)npts) * HALO_MASS;
                        }
                        else
                        {
                            M_enc = ((double)i / (double)npts) * HALO_MASS;
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
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
                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
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

                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
                    double dvdt = force + effective_angular_force(r, ell);
                    particles[1][i] = vrad + 0.5 * c3 * dt * dvdt;
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
                    double force = gravitational_force(r, i, npts, G_CONST, HALO_MASS);
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
                    double force = gravitational_force(r_mid, i, npts, G_CONST, HALO_MASS);
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
                    double force = gravitational_force(r_mid, i, npts, G_CONST, HALO_MASS);
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
                    double force = gravitational_force(r_end, i, npts, G_CONST, HALO_MASS);
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
                // Static variables for AB method history
                static int ab8_bootstrap_done = 0; ///< Flag: 0=needs bootstrap, 1=bootstrap complete.
                static double **f_ab8_r = NULL;    ///< History of dr/dt derivatives [step][orig_id].
                static double **f_ab8_v = NULL;    ///< History of dv/dt derivatives [step][orig_id].
                static double h_ab8;               ///< Sub-step size used during bootstrap.

                // Example AB8 coefficients (currently trivial for debugging).
                static int ab8_num[8] = {23, -16, 5, 0, 0, 0, 0, 0};
                static int ab8_den = 12;

                // We'll do 7 Euler sub-steps for bootstrap:
                static int mini_steps = 7;

                // Allocate AB8 history arrays once.
                if (f_ab8_r == NULL)
                {
                    f_ab8_r = (double **)malloc(8 * sizeof(double *));
                    f_ab8_v = (double **)malloc(8 * sizeof(double *));
                    for (int hh = 0; hh < 8; hh++)
                    {
                        f_ab8_r[hh] = (double *)malloc(npts * sizeof(double));
                        f_ab8_v[hh] = (double *)malloc(npts * sizeof(double));
                    }
                    h_ab8 = dt; // Sub-step size equals dt.
                }

                // If bootstrap not done yet, do it ONCE to fill the 5-step derivative history.
                if (!ab8_bootstrap_done)
                {

                    // Perform sub-steps 0..(mini_steps), i.e. 0..7 inclusive => 5 total.
                    // Only update the state for sub-steps 0..6. After sub-step=7, no further update needed.
#pragma omp single
                    {
                        for (int sub_step = 0; sub_step <= mini_steps; sub_step++)
                        {
                            sort_particles(particles, npts);
#pragma omp parallel for default(shared) schedule(static)
                            for (int idx = 0; idx < npts; idx++)
                            {
                                int orig_id = (int)particles[3][idx];
                                inverse_map[orig_id] = idx;
                            }
                            // Compute derivatives => store in f_ab8_r[sub_step], f_ab8_v[sub_step].
#pragma omp parallel for default(shared) schedule(static)
                            for (int i_eval = 0; i_eval < npts; i_eval++)
                            {
                                int orig_id = (int)particles[3][i_eval];
                                double rr = particles[0][i_eval];
                                double vrad = particles[1][i_eval];
                                double ell = particles[2][i_eval];

                                double drdt = vrad;
                                // Use the gravitational_force and effective_angular_force functions.
                                double force = gravitational_force(rr, i_eval, npts, G_CONST, HALO_MASS);
                                double dvdt = force + effective_angular_force(rr, ell);

                                f_ab8_r[sub_step][orig_id] = drdt;
                                f_ab8_v[sub_step][orig_id] = dvdt;
                            }

                            // If sub_step < 7, do an Euler update in-place:
                            // R += dt*drdt, v += dt*dvdt.
                            if (sub_step < mini_steps)
                            {
#pragma omp parallel for default(shared) schedule(static)
                                for (int i_up = 0; i_up < npts; i_up++)
                                {
                                    int orig_id = (int)particles[3][i_up];
                                    // Read the derivative from f_ab8_r[sub_step], f_ab8_v[sub_step].
                                    double drdt = f_ab8_r[sub_step][orig_id];
                                    double dvdt = f_ab8_v[sub_step][orig_id];

                                    particles[0][i_up] += h_ab8 * drdt;
                                    particles[1][i_up] += h_ab8 * dvdt;
                                }
                            }
                        }
                    }

#pragma omp single
                    {
                        // Mark bootstrap done.
                        ab8_bootstrap_done = 1;
                    }
                }
                else
                {
                    /**
                     * Normal AB8 step each iteration
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
                        // AB8 formula (currently trivial for debugging).
                        int orig_id = (int)particles[3][i];
                        double rr = particles[0][i];
                        double vrad = particles[1][i];

                        double sum_r = 0.0;
                        double sum_v = 0.0;
                        for (int kk = 0; kk < 5; kk++)
                        {
                            sum_r += ab8_num[kk] * f_ab8_r[7 - kk][orig_id];
                            sum_v += ab8_num[kk] * f_ab8_v[7 - kk][orig_id];
                        }

                        double r_next = rr + (dt / (double)ab8_den) * sum_r;
                        double v_next = vrad + (dt / (double)ab8_den) * sum_v;

                        particles[0][i] = r_next;
                        particles[1][i] = v_next;
                    }

                    // We re-sort & compute new derivatives to shift the AB8 history.
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

                    // Recompute the derivatives for the new time => goes into f_ab8_r[7], f_ab8_v[7].
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
                        double force = gravitational_force(rr, i_dbg, npts, G_CONST, HALO_MASS);
                        double dvdt = force + effective_angular_force(rr, ell);

                        f_new_r[0][orig_id] = drdt;
                        f_new_v[0][orig_id] = dvdt;
                    }

#pragma omp single
                    {
                        // SHIFT HISTORY by 1.
                        for (int hh = 0; hh < 7; hh++)
                        {
                            for (int i_s = 0; i_s < npts; i_s++)
                            {
                                f_ab8_r[hh][i_s] = f_ab8_r[hh + 1][i_s];
                                f_ab8_v[hh][i_s] = f_ab8_v[hh + 1][i_s];
                            }
                        }
                        // Put the new derivative in slot #7.
                        for (int i_s = 0; i_s < npts; i_s++)
                        {
                            f_ab8_r[7][i_s] = f_new_r[0][i_s];
                            f_ab8_v[7][i_s] = f_new_v[0][i_s];
                        }

                        free(f_new_r[0]);
                        free(f_new_v[0]);
                        free(f_new_r);
                        free(f_new_v);

                        current_step = j + 1;
                        time += dt;
                    }
                } // End of the "else" block for normal AB8.
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

    // Write theoretical density profile (calculated earlier)
    char suffixed_filename[256];
    get_suffixed_filename("data/density_profile.dat", 1, suffixed_filename, sizeof(suffixed_filename));
    fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
    for (i = 0; i < num_points; i++)
    {
        double rr = radius[i];
        double rho_r = HALO_MASS / normalization * (1.0 / cube(1.0 + sqr(rr / RC)));
        fprintf_bin(fp, "%f %f\n", rr, rho_r);
    }
    fclose(fp);

    // Write theoretical dPsi/dr profile
    get_suffixed_filename("data/dpsi_dr.dat", 1, suffixed_filename, sizeof(suffixed_filename));
    fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
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

    // Write theoretical drho/dPsi profile
    get_suffixed_filename("data/drho_dpsi.dat", 1, suffixed_filename, sizeof(suffixed_filename));
    fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
    for (i = 1; i < num_points - 1; i++)
    {
        double rr = radius[i];
        double rho_left = HALO_MASS / normalization * (1.0 / cube(1.0 + sqr(radius[i - 1] / RC)));
        double rho_right = HALO_MASS / normalization * (1.0 / cube(1.0 + sqr(radius[i + 1] / RC)));
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

    // Write theoretical f(E) profile
    get_suffixed_filename("data/f_of_E.dat", 1, suffixed_filename, sizeof(suffixed_filename));
    fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
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
        double fE = deriv / (sqrt(5.0) * PI * PI);
        if (E == 0.0)
            fE = 0.0;
        fprintf_bin(fp, "%f %f\n", E, fE);
    }
    fclose(fp);

    // Write distribution function at a fixed radius if simulation was run
    if (!skip_file_writes)
    {
        get_suffixed_filename("data/df_fixed_radius.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        {
            double r_fixed = 200.0;
            double Psi_rf = evaluatespline(splinePsi, Psiinterp, r_fixed);
            Psi_rf *= VEL_CONV_SQ;
            double Psimin_test = VEL_CONV_SQ * Psimin;
            double Psimax_test = VEL_CONV_SQ * Psimax;

            int vsteps = 10000;
            int reduce_vsteps = 300; // Cutoff early.
            for (int vv = 0; vv <= vsteps - reduce_vsteps; vv++)
            {
                double vtest = (double)vv * (sqrt(2.0 * (Psi_rf - Psimin_test)) / (vsteps));
                double Etest = Psi_rf - 0.5 * vtest * vtest;
                Etest = Etest / VEL_CONV_SQ;
                double fEval = 0.0;
                if (Etest >= Psimin && Etest <= Psimax)
                {
                    double derivative;
                    int status = gsl_interp_eval_deriv_e(fofEinterp, Evalues, innerintegrandvalues, Etest, fofEacc, &derivative);
                    if (status == GSL_SUCCESS)
                    {
                        fEval = derivative / (sqrt(5.0) * PI * PI) * vtest * vtest * r_fixed * r_fixed;
                    }
                    else
                    {
                        fEval = 0.0;
                        fprintf(stderr, "[DEBUG] gsl_interp_eval_deriv_e failed with status %d for Etest=%.6f\n", status, Etest);
                    }
                }
                else
                {
                    fEval = 0.0;
                    fprintf(stderr, "[DEBUG] Etest=%.6f out of range [%.6f, %.6f]\n", Etest, Psimin_test, Psimax_test);
                }
                fprintf_bin(fp, "%f %f\n", vtest, fEval);
            }
        }
        fclose(fp);
    } // End skip_file_writes block.

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

    // Don't need to compute max_r_all anymore. Simply set to 400.0 kpc.
    double max_r_all = 250.0;

    double rbin_width = max_r_all / 200.0;
    double vbin_width = 320.0 / 200.0;

    double bin_width = max_r_all / 200.0;

    int hist_initial[200][200];
    memset(hist_initial, 0, sizeof(hist_initial));
    int hist_final[200][200];
    memset(hist_final, 0, sizeof(hist_final));

    for (i = 0; i < npts; i++)
    {
        int rbini = (int)(r_initial[i] / rbin_width);
        int vbini = (int)(fabs(v_initial[i]) * (1.0 / kmsec_to_kpcmyr) / vbin_width);
        if (rbini < 200 && vbini < 200 && rbini >= 0 && vbini >= 0)
            hist_initial[rbini][vbini]++;

        rbini = (int)(r_final[i] / rbin_width);
        vbini = (int)(fabs(v_final[i]) * (1.0 / kmsec_to_kpcmyr) / vbin_width);
        if (rbini < 200 && vbini < 200 && rbini >= 0 && vbini >= 0)
            hist_final[rbini][vbini]++;
    }

    // Write initial 2D histogram if simulation was run
    if (!skip_file_writes)
    {
        char suffixed_filename[256];
        get_suffixed_filename("data/2d_hist_initial.dat", 1, suffixed_filename, sizeof(suffixed_filename));
        fp = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
        for (int rr = 0; rr < 200; rr++)
        {
            for (int vv = 0; vv < 200; vv++)
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
        for (int rr = 0; rr < 200; rr++)
        {
            for (int vv = 0; vv < 200; vv++)
            {
                fprintf_bin(fp, "%f %f %d\n", (rr + 0.5) * rbin_width, (vv + 0.5) * vbin_width, hist_final[rr][vv]);
            }
            fprintf_bin(fp, "\n");
        }
        fclose(fp);
    } // End skip_file_writes block.

    int hist_i[200];
    memset(hist_i, 0, sizeof(hist_i));
    int hist_f[200];
    memset(hist_f, 0, sizeof(hist_f));

    for (i = 0; i < npts; i++)
    {
        int b = (int)(r_initial[i] / bin_width);
        if (b < 200 && b >= 0)
            hist_i[b]++;
        b = (int)(r_final[i] / bin_width);
        if (b < 200 && b >= 0)
            hist_f[b]++;
    }

    // Write combined 1D radius histogram if simulation was run
    if (!skip_file_writes)
    {
        char filename[256];
        get_suffixed_filename("data/combined_histogram.dat", 1, filename, sizeof(filename));
        fp = fopen(filename, "wb"); // Binary mode for fprintf_bin
        for (i = 0; i < 200; i++)
        {
            double bin_center = (i + 0.5) * bin_width;
            fprintf_bin(fp, "%f %d %d\n", bin_center, hist_i[i], hist_f[i]);
        }
        fclose(fp);
    } // End skip_file_writes block.

    // Write trajectories for low-ID particles if simulation was run
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

    // Write trajectory for particle ID 0 if simulation was run
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
        // Write trajectories for selected lowest-L particles if simulation was run
        if (!skip_file_writes)
        {
            char suffixed_filename[256];
            get_suffixed_filename("data/lowest_l_trajectories.dat", 1, suffixed_filename, sizeof(suffixed_filename));
            FILE *fp_lowest = fopen(suffixed_filename, "wb"); // Binary mode for fprintf_bin
            if (!fp_lowest)
            {
                printf("Error: cannot open data/lowest_l_trajectories.dat\n");
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
                printf("All Rank files already exist. Skipping further processing.\n");
                goto cleanup_partdata_outsnaps; // Jump past the processing loop
            }
            else if (start_index < 0) // Indicates error or no files found
            {
                start_index = 0; // Default to starting from the beginning
            }
            // Otherwise, start_index is the index of the first snapshot *to be* processed
            printf("Restart: Starting from snapshot index %d (after finding last processed snapshot)\n\n", start_index);
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

            // Sort the PartData array by radius using a critical section for thread-safety
            // if the sort function itself isn't thread-safe (qsort generally isn't reentrant).
            // Assuming sort_by_rad uses qsort.
#pragma omp critical(sort_partarr)
            {
                sort_by_rad(partarr, npts);
            }

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
            /**
             * =========================================================================
             * END DENSITY CALCULATION
             * =========================================================================
             */

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
                sprintf(base_filename, "data/Psi_methodA_t%05d.dat", snap);
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

                /**
                 * =========================================================================
                 * ENERGY CALCULATION FOR PARTICLE DISTRIBUTIONS
                 * =========================================================================
                 *
                 * Calculate total energy for each particle using the formula:
                 * E = Ψ - L²/(2r²) - v²/2
                 *
                 * Components:
                 * - Ψ: Gravitational potential (from spline interpolation)
                 * - L: Angular momentum
                 * - r: Radial position
                 * - v: Radial velocity
                 *
                 * Computed separately for both unsorted and sorted distributions
                 */
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
                    /**
                     * =========================================================================
                     * DEBUG ENERGY COMPUTATION - DYNAMIC ANALYSIS
                     * =========================================================================
                     *
                     * Computes dynamic energy for tracked particle before rewriting files.
                     * Uses PsiA_unsorted to calculate energy for the specific debug ID.
                     * This information is used to validate energy conservation during simulation.
                     */

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

                        /**
                         * =========================================================================
                         * END DEBUG ENERGY COMPUTATION
                         * =========================================================================
                         */
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
                    sprintf(base_filename, "data/Rank_Mass_Rad_VRad_unsorted_t%05d.dat", snap);
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
                    sprintf(base_filename, "data/Rank_Mass_Rad_VRad_sorted_t%05d.dat", snap);
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

            /**
             * =========================================================================
             * CLEANUP SECTION - PARTICLE DATA PROCESSING COMPLETE
             * =========================================================================
             *
             * Release all memory allocated during particle data processing phase.
             * Memory is freed in the reverse order of allocation to prevent memory leaks.
             * This includes thread-local arrays and shared memory structures.
             */
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

        printf("All snapshot processing completed.\n");
        log_message("INFO", "All snapshot processing completed. Cleaning up.");

    cleanup_partdata_outsnaps:
        // Free the memory for particle data arrays
        free(Rank_partdata_outsnaps);
        free(R_partdata_outsnaps);
        free(Vrad_partdata_outsnaps);
    }
    {
        printf("Using file suffix: %s\n", g_file_suffix);

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
    free(Psivalues);
    free(nPsivalues);
    free(innerintegrandvalues);
    free(Evalues);
    gsl_integration_workspace_free(w);
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
    gsl_interp_free(fofEinterp);
    gsl_interp_accel_free(fofEacc);

    free_local_snap_arrays();
    cleanup_all_particle_data();

#ifdef _OPENMP
    // Clean up FFTW threads only if they were initialized.
    fftw_cleanup_threads();
#endif

    return 0;
} // End main function.

/**
 * Calculates the mass integrand for a given radius using a density profile.
 *
 * @param r Radius value
 * @param params Unused parameter (with compiler attribute to prevent warnings)
 * @return Mass integrand value at the specified radius
 */
double massintegrand(double r, void *params __attribute__((unused)))
{
    double startingprofile = 1.0 / cube((1.0 + sqr(r / RC)));
    return r * r * startingprofile;
}

/**
 * Calculates the derivative of density with respect to radius.
 *
 * @param r Radius value
 * @return Derivative of density with respect to radius at the specified point
 */
double drhodr(double r)
{
    return -6.0 * r / (RC * RC) / pow(1.0 + sqr(r / RC), 4.0);
}

/**
 * Calculates the gravitational potential integrand for a given radius.
 *
 * @param rp Radius parameter
 * @param params Parameters containing alpha value
 * @return Potential integrand value at the specified radius
 */
double Psiintegrand(double rp, void *params)
{
    double massintegrand(double r, void *params);
    double alpha = *(double *)params;
    if (rp <= 0.0)
    {
        printf("rp out of range\n");
        CLEAN_EXIT(1);
    }
    return massintegrand(rp, &alpha) / rp;
}


/**
 * Safely evaluates a GSL spline at a given value with robust bounds checking.
 *
 * Handles edge cases including NULL pointers and out-of-range values by clamping
 * to the valid range with a small safety margin to prevent errors.
 *
 * @param spline Pointer to the GSL spline object
 * @param acc Pointer to the GSL interpolation accelerator
 * @param value Value at which to evaluate the spline
 * @return Interpolated value from the spline, or 0.0 if inputs are invalid
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
    double drhodpsi = -(HALO_MASS / normalization) * drhodr(r) / (G_CONST * evaluatespline(splinemass, massarray, r) / (r * r));
    return 2.0 * drhodpsi;
}

/**
 * =========================================================================
 * PARALLEL SORTING ALGORITHM CONSTANTS AND FUNCTIONS
 * =========================================================================
 *
 * @brief Constants and functions controlling the parallel sorting algorithm behavior.
 * @details The sorting implementation uses a parallel chunk-based approach with overlapping
 *          regions between chunks to ensure correct ordering at chunk boundaries.
 */
static const int num_sort_sections = 24; ///< Number of parallel chunks for sorting.
static const int overlap_divisor = 8;    ///< Overlap factor: Overlap = section_size / overlap_divisor.
static const int min_seam_overlap = 50;  ///< Minimum overlap region size between chunks.

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
void insertion_parallel_sort(double **columns, int n)
{
    if (n < num_sort_sections)
    {
        // Fallback to sequential sort for small N
        insertion_sort(columns, n);
        return;
    }

    // Determine chunk boundaries, distributing remainder
    int base_chunk_size = n / num_sort_sections;
    int remainder = n % num_sort_sections;
    int *startIdx = (int *)malloc(num_sort_sections * sizeof(int));
    int *endIdx = (int *)malloc(num_sort_sections * sizeof(int));
    if (!startIdx || !endIdx) { /* Handle error */ CLEAN_EXIT(1); }

    int offset = 0;
    for (int c = 0; c < num_sort_sections; c++)
    {
        int size_c = base_chunk_size + (c < remainder ? 1 : 0);
        startIdx[c] = offset;
        endIdx[c] = offset + size_c - 1;
        offset += size_c;
    }

    // Sort each chunk in parallel using insertion sort
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < num_sort_sections; c++)
    {
        insertion_sort_sub(columns, startIdx[c], endIdx[c]);
    }

    // Compute overlap size for seam fixing
    int minChunkSize = n;
    for (int c = 0; c < num_sort_sections; c++)
    {
        int csize = endIdx[c] - startIdx[c] + 1;
        if (csize < minChunkSize) minChunkSize = csize;
    }
    int overlapSize = minChunkSize / overlap_divisor;
    if (overlapSize < min_seam_overlap) overlapSize = min_seam_overlap;
    if (overlapSize <= 0) overlapSize = 1;

    // Merge/fix the seams in parallel
    int nSeams = num_sort_sections - 1;
#pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < nSeams; s++)
    {
        int c_left = s;
        int c_right = s + 1;

        // Determine boundaries of the combined overlap region
        int leftStart = endIdx[c_left] - overlapSize + 1;
        if (leftStart < startIdx[c_left]) leftStart = startIdx[c_left];

        int rightEnd = startIdx[c_right] + overlapSize - 1;
        if (rightEnd > endIdx[c_right]) rightEnd = endIdx[c_right];

        // Sort the combined overlap region [leftStart..rightEnd]
        if (leftStart <= rightEnd)
        {
            insertion_sort_sub(columns, leftStart, rightEnd);
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
void quadsort_parallel_sort(double **columns, int n)
{
    if (n < num_sort_sections)
    {
        // Fallback to sequential sort for small N
        quadsort(columns, n, sizeof(double *), compare_particles);
        return;
    }

    // Determine chunk boundaries
    int base_chunk_size = n / num_sort_sections;
    int remainder = n % num_sort_sections;
    int *startIdx = (int *)malloc(num_sort_sections * sizeof(int));
    int *endIdx = (int *)malloc(num_sort_sections * sizeof(int));
    if (!startIdx || !endIdx) { /* Handle error */ CLEAN_EXIT(1); }

    int offset = 0;
    for (int c = 0; c < num_sort_sections; c++)
    {
        int size_c = base_chunk_size + (c < remainder ? 1 : 0);
        startIdx[c] = offset;
        endIdx[c] = offset + size_c - 1;
        offset += size_c;
    }

    // Sort each chunk in parallel using quadsort
#pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < num_sort_sections; c++)
    {
        quadsort(&columns[startIdx[c]], endIdx[c] - startIdx[c] + 1, sizeof(double *), compare_particles);
    }

    // Compute overlap size for seam fixing
    int minChunkSize = n;
    for (int c = 0; c < num_sort_sections; c++)
    {
        int csize = endIdx[c] - startIdx[c] + 1;
        if (csize < minChunkSize) minChunkSize = csize;
    }
    int overlapSize = minChunkSize / overlap_divisor;
    if (overlapSize < min_seam_overlap) overlapSize = min_seam_overlap;
    if (overlapSize <= 0) overlapSize = 1;

    // Merge/fix the seams in parallel using quadsort
    int nSeams = num_sort_sections - 1;
#pragma omp parallel for schedule(dynamic)
    for (int s = 0; s < nSeams; s++)
    {
        int c_left = s;
        int c_right = s + 1;

        // Determine boundaries of the combined overlap region
        int leftStart = endIdx[c_left] - overlapSize + 1;
        if (leftStart < startIdx[c_left]) leftStart = startIdx[c_left];

        int rightEnd = startIdx[c_right] + overlapSize - 1;
        if (rightEnd > endIdx[c_right]) rightEnd = endIdx[c_right];

        // Sort the combined overlap region
        if (leftStart <= rightEnd)
        {
            quadsort(&columns[leftStart], rightEnd - leftStart + 1, sizeof(double *), compare_particles);
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
            // Optionally print details of mismatch for deeper debugging
            // fprintf(stderr, "Mismatch at index %d: Algo=%f, Qsort=%f\n", i, columns[i][0], tempCopy[i][0]);
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
void sort_particles_with_alg(double **particles, int npts, const char *sortAlg)
{
    // Transpose data: particles[component][particle] -> columns[particle][component]
    double **columns = (double **)malloc(npts * sizeof(double *));
    if (!columns) { /* Handle error */ CLEAN_EXIT(1); }
#pragma omp parallel for
    for (int i = 0; i < npts; i++)
    {
        columns[i] = (double *)malloc(5 * sizeof(double));
        if (!columns[i]) { /* Handle error */ CLEAN_EXIT(1); }
        for (int j = 0; j < 5; j++)
        {
            columns[i][j] = particles[j][i];
        }
    }

    // Select and apply the sorting algorithm
    const char *method = (sortAlg ? sortAlg : "insertion_parallel"); // Default if NULL

    if (strcmp(method, "quadsort") == 0)
    {
        quadsort_wrapper(columns, npts);
    }
    else if (strcmp(method, "quadsort_parallel") == 0)
    {
        quadsort_parallel_sort(columns, npts);
    }
    else if (strcmp(method, "insertion") == 0) // Added case for sequential insertion
    {
         insertion_sort(columns, npts);
    }
    else // Default to parallel insertion sort
    {
        insertion_parallel_sort(columns, npts);
    }

    // Transpose data back: columns[particle][component] -> particles[component][particle]
#pragma omp parallel for
    for (int i = 0; i < npts; i++)
    {
        for (int j = 0; j < 5; j++)
        {
            particles[j][i] = columns[i][j];
        }
        free(columns[i]); // Free inner arrays
    }
    free(columns);
}

/**
 * Convenience wrapper function for sorting particles with the default algorithm.
 *
 * Calls sort_particles_with_alg using the default sorting algorithm specified in g_defaultSortAlg.
 *
 * @param particles 2D array of particle data to be sorted [component][particle]
 * @param npts Number of particles to sort
 */
void sort_particles(double **particles, int npts)
{
    sort_particles_with_alg(particles, npts, g_defaultSortAlg);
}

/**
 * Comparison function for sorting PartData structures by radius value.
 *
 * Provides a robust comparison that handles NULL pointers and NaN values,
 * ensuring stable sorting behavior even with potentially problematic data.
 *
 * @param a Pointer to the first PartData structure (as void pointer)
 * @param b Pointer to the second PartData structure (as void pointer)
 * @return -1 if a<b, 1 if a>b, 0 if equal
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
 * Sorts an array of PartData structures by their radius value.
 *
 * Includes safety checks for NULL array and invalid counts before
 * proceeding with the sort operation.
 *
 * @param array Array of PartData structures to sort
 * @param npts Number of elements in the array
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

/**
 * =========================================================================
 * SPLINE DATA SORTING UTILITIES
 * =========================================================================
 *
 * @brief Utility functions and structures for sorting spline data arrays.
 * @details Provides mechanisms to sort arrays used for GSL spline creation (like radius `r`
 *          and potential `Psi`) while maintaining the correct correspondence between
 *          paired values after sorting based on one of the arrays (typically radius).
 */

/**
 * @brief Structure for pairing radius and potential values during sorting.
 * @details Used as a temporary structure within `sort_rr_psi_arrays` to maintain
 *          the correspondence between radius and potential values when sorting by radius.
 */
struct RrPsiPair
{
    double rr;  ///< Radius value.
    double psi; ///< Corresponding potential value.
};

/**
 * @brief Comparison function for sorting RrPsiPair structures by radius (`rr`).
 *
 * @param a Pointer to the first RrPsiPair structure (as void pointer).
 * @param b Pointer to the second RrPsiPair structure (as void pointer).
 * @return -1 if a->rr < b->rr, 1 if a->rr > b->rr, 0 if equal.
 *
 * @see RrPsiPair
 * @see sort_rr_psi_arrays
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
 * Sorts radius and potential arrays simultaneously to maintain correspondence.
 *
 * Creates temporary paired structures to ensure that the relationship between
 * radius values and their corresponding potential values is preserved during sorting.
 * This is crucial for correct spline interpolation.
 *
 * @param rrA_spline Array of radius values to be sorted
 * @param psiAarr_spline Array of potential values corresponding to the radius values
 * @param npts Number of points in the arrays (arrays are of size npts+1)
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
