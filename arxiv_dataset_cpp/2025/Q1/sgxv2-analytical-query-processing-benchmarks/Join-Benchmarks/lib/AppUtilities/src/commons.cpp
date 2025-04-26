#include "commons.h"

#include <string.h>
#include <stdlib.h>
#include <getopt.h>
#include "data-types.h"

#include "Logger.hpp"

void parse_args(int argc, char **argv, args_t *params, const struct algorithm_t algorithms[]) {
    int c;
    int i;
    int found;
    bool def_dataset = false;
    bool def_table = false;
    static int sort_r;
    static int sort_s;
    static int mitigation;
    char *ptr;
    char *eptr;
    uint64_t ret;


    while (true) {
        static struct option long_options[] =
        {
            {"sort-r", no_argument, &sort_r, 1},
            {"sort-s", no_argument, &sort_s, 1},
            {"mitigation", no_argument, &mitigation, 1},

            {"r-path", required_argument, 0, 't'},
            {"s-path", required_argument, 0, 'u'}
        };

        int option_index = 0;

        c = getopt_long(argc, argv, "a:c:d:e:l:n:mr:s:t:u:x:y:z:hv",
                        long_options, &option_index);

        if (c == -1) {
            break;
        }
        switch (c) {
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                logger(DBG, "option %s", long_options[option_index].name);
                if (optarg)
                    logger(DBG, "with arg %s", optarg);
                break;
            case 'a':
                if (algorithms == nullptr) {
                    logger(DBG, "Not checking the validity of algorithm when parsing - "
                           "will check in the enclave.");
                } else {
                    i = 0;
                    found = 0;
                    while (i < INT32_MAX) {
                        if (strcmp(optarg, algorithms[i].name) == 0) {
                            params->algorithm = &algorithms[i];
                            found = 1;
                            break;
                        }
                        i++;
                    }
                    if (found == 0) {
                        logger(ERROR, "Join algorithm not found: %s", optarg);
                        exit(EXIT_SUCCESS);
                    }
                }
                strcpy(params->algorithm_name, optarg);
                break;

            case 'c':
                params->alloc_core = atoi(optarg);
                break;

            case 'd':
                if (def_table) {
                    logger(ERROR, "Select a predefined dataset OR specify tables sizes");
                    exit(EXIT_FAILURE);
                }
                def_dataset = 1;
                if (strcmp(optarg, "cache-fit") == 0) {
                    params->r_size = (uint64_t) (10 * 1024 * 1024) / sizeof(tuple_t);
                    params->s_size = (uint64_t) (40 * 1024 * 1024) / sizeof(tuple_t);
                } else if (strcmp(optarg, "cache-exceed") == 0) {
                    params->r_size = (uint64_t) (100 * 1024 * 1024) / sizeof(tuple_t);
                    params->s_size = (uint64_t) (400 * 1024 * 1024) / sizeof(tuple_t);
                } else if (strcmp(optarg, "L") == 0) {
                    params->r_size = 50000000;
                    params->s_size = 200000000;
                } else {
                    logger(ERROR, "Unrecognized dataset: %s", optarg);
                    exit(EXIT_FAILURE);
                }
                break;

            case 'h':
                logger(DBG, "Print help");
                exit(EXIT_SUCCESS);
                break;

            case 'l':
                params->selectivity = atoi(optarg);
                if (params->selectivity < 0 || params->selectivity > 100) {
                    logger(ERROR, "Selectivity should be between 0 and 100");
                    exit(EXIT_FAILURE);
                }
                break;
            case 'm':
                params->materialize = 1;
                break;
            case 'n':
                params->nthreads = atoi(optarg);
                break;
            case 'r':
                if (def_dataset) {
                    logger(ERROR, "Select a predefined dataset OR specify tables sizes");
                    exit(EXIT_FAILURE);
                }
            //                params->r_size = atoi(optarg);
                params->r_size = strtoul(optarg, &eptr, 10);
                def_table = 1;
                break;

            case 's':
                if (def_dataset) {
                    logger(ERROR, "Select a predefined dataset OR specify tables sizes");
                    exit(EXIT_FAILURE);
                }
            //                params->s_size = atoi(optarg);
                params->s_size = strtoul(optarg, &eptr, 10);
                def_table = 1;
                break;

            case 't':
                params->r_from_path = 1;
                strcpy(params->r_path, optarg);
                break;

            case 'u':
                params->s_from_path = 1;
                strcpy(params->s_path, optarg);
                break;

            // size of R in MBs
            case 'x':
                if (def_dataset) {
                    logger(ERROR, "Select a predefined dataset OR specify tables sizes");
                    exit(EXIT_FAILURE);
                }
                ret = (uint64_t) strtoul(optarg, &ptr, 10);
                params->r_size = (uint64_t) ((ret * 1024 * 1024) / sizeof(tuple_t));
                def_table = 1;
                break;

            // size of S in MBs
            case 'y':
                if (def_dataset) {
                    logger(ERROR, "Select a predefined dataset OR specify tables sizes");
                    exit(EXIT_FAILURE);
                }
                ret = (uint64_t) strtoul(optarg, &ptr, 10);
                params->s_size = (uint64_t) ((ret * 1024 * 1024) / sizeof(tuple_t));
                def_table = 1;
                break;

            case 'z':
                params->skew = atof(optarg);
                break;

            default:
                break;
        }
    }

    params->sort_r = sort_r;
    params->sort_s = sort_s;
    params->mitigation = mitigation;

    /* Print remaining command line arguments */
    if (optind < argc) {
        logger(INFO, "remaining arguments: ");
        while (optind < argc) {
            logger(INFO, "%s", argv[optind++]);
        }
    }
}

void print_relation(relation_t *rel, uint32_t num, uint32_t offset) {
    logger(DBG, "****************** Relation sample ******************");
    for (uint64_t i = offset; i < rel->num_tuples && i < num + offset; i++) {
        logger(DBG, "%lu -> %lu", rel->tuples[i].key, rel->tuples[i].payload);
    }
    logger(DBG, "******************************************************");
}
