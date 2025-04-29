/*
 * main.cpp
 * Purpose : program interface, allowing to set options for the execution of the algorithm
 */

#include <iostream>
#include <getopt.h>
#include <unistd.h>

#include "DecisionDiagram.h"
#include "DDColors.h"

/*
 * print_help : prints the usage and parameters of the program
 * program_header : prints some runtime conditions of the program, like the argument string and and when it was called
 */

void print_help(){
    std::cout<<"Usage: ddcolors.exe [options] graph"<<std::endl;
    std::cout<<"Options:"<<std::endl;
    std::cout<<"-h|--help               :Prints this help message."<<std::endl;
    std::cout<<"-m|--method arg         :Which algorithm is to be run to get the chromatic number: i for iterative refinement (default),"<<std::endl;//b for basic as hidden option
    std::cout<<"                         e for exact compilation, h for heuristic (only an upper bound) and f to compute the fractional chromatic number."<<std::endl;
    std::cout<<"-o|--ordering arg       :Option of which vertex ordering is to be used. d for dsatur ordering (default)."<<std::endl;
    std::cout<<"                         l for lexicographic, m for max connected degree, w for minimum width ordering."<<std::endl;
    std::cout<<"-r|--relaxation arg     :IP solver relaxation: l for LP first and then IP (default), i for IP only"<<std::endl;
    std::cout<<"                         and m(X) for solving IP at every Xth iteration instead of the normal LP (default X=100)."<<std::endl;
    std::cout<<"-l|--longest_paths arg  :The number of iterations the longest path preprocessing should be applied, standard is 100."<<std::endl;
    std::cout<<"-p|--preprocessing      :Enables the use of peeling and removing dominated vertices of the graph in a preprocessing step."<<std::endl;
    std::cout<<"-a|--arc_redirection    :Redirects arcs to the most similar node in the next level when constructing the next relaxed decision diagram."<<std::endl;
    std::cout<<"-c|--conflicts arg      :Detect and resolve conflicts in a flow solution after chosen method. m for multiple conflicts (default),"<<std::endl;
    std::cout<<"                         s for single conflict, f(X) for conflicts with associated flow larger than 1e^-X (default X=5, choose 1<=X<=7)."<<std::endl;
    std::cout<<"-d|--decomposition arg  :Specifies the heuristic used in the path decomposition of the flow. o is for preferring 1-arcs (default),"<<std::endl;
    std::cout<<"                         a for paths avoiding conflicts and z for preferring 0-arcs."<<std::endl;
    std::cout<<"-f|--formulation arg    :Allows slight changes to the way the IP/LP is formulated. n is for normal (default), e is for setting first 1-arc to 1, o to add bound on objective value,"<<std::endl;
    std::cout<<"                         b for adding the variable bounds as constraints, c for using obtained coloring bound as upper bound on vars and r to let 0-arcs always be continuous"<<std::endl;
    std::cout<<"-k|--clique             :Looks for a clique whose vertices are fixed as the first ones in the ordering and in the upper bound heuristic."<<std::endl;
    std::cout<<"-z|--randomness         :Enables random tie breaks in the coloring heuristic."<<std::endl;
    std::cout<<"-v|--verbosity arg      :Prints information every xth iteration of the refinement procedure, not only when bounds change (x must be a postive integer)."<<std::endl;
    std::cout<<"-s|--size_limit arg     :Limits the size of the decision diagram to x million nodes in exact compilation (x must be a postive integer)."<<std::endl;
    std::cout<<"-e|--emphasis arg       :Changes the emphasis of CPLEX for the MIP on the exact decision diagram."<<std::endl;
    std::cout<<"-t|--threads arg        :Changes the number of allowed threads for CPLEX to use. If larger than 1, solving will be opportunistic."<<std::endl;
}

#define MAX_PNAME_LEN 128
static const char* color_verion_string = "some version string";
void program_header(int ac, char **av) {
    int   rval     = 0;
    time_t starttime;
    char my_hostname[MAX_PNAME_LEN];
    pid_t my_pid   = getpid();
    int   i;

    rval = gethostname (my_hostname, MAX_PNAME_LEN - 1);
    if(rval){
        throw std::runtime_error("gethostname failed");
    }

    printf("##############################################################\n");

    printf("Running  :");
    for (i = 0; i < ac; ++i) {printf(" %s", av[i]);}
    printf ("\n");

    printf("Machine  : %s, pid: %lld\n",
           my_hostname, (long long) my_pid);

    (void) time(&starttime);

    printf("Date     : %s", ctime(&starttime));

//    printf("Changeset: %s\n", color_verion_string);
    fflush (stdout);
    printf("##############################################################\n");
}

int main(int argc, char *argv[]) {

    if(argc == 1){//no arguments were given
        print_help();
        return 0;
    }

    int opt;
    int option_index = 0;
    static struct option long_options[] = {
            {"help",            no_argument,       nullptr, 'h'},
            {"arc_redirection", no_argument,       nullptr, 'a'},
            {"preprocessing",   optional_argument, nullptr, 'p'},
            {"method",          required_argument, nullptr, 'm'},
            {"conflicts",       required_argument, nullptr, 'c'},
            {"ordering",        required_argument, nullptr, 'o'},
            {"relaxation",      required_argument, nullptr, 'r'},
            {"longest_paths",   required_argument, nullptr, 'l'},
            {"decomposition",   required_argument, nullptr, 'd'},
            {"safe_bound",      no_argument,       nullptr, 's'},
            {"formulation",     required_argument, nullptr, 'f'},
            {"clique",          optional_argument, nullptr, 'k'},
            {"randomness",      no_argument,       nullptr, 'z'},
            {"verbosity",       required_argument, nullptr, 'v'},
            {"size_limit",      required_argument, nullptr, 's'},
            {"emphasis",       required_argument, nullptr, 'e'},
            {"threads",      required_argument, nullptr, 't'},
    };
    Options dd_settings{};
    std::string short_opt = "hap::k::zc:o:r:l:m:d:f:v:s:";
    short_opt += "e:t:";
    while((opt = getopt_long(argc, argv, short_opt.c_str(), long_options, &option_index)) != -1){
        switch(opt){

            default:
            case '?':                                                       //getopt itself will return an error message
                return -1;
            case 'h':
                print_help();
                return -1;

            case 'm':
                if(*optarg == 'i'){
                    dd_settings.algorithm = Options::HeuristicRefinement;
                } else if(*optarg == 'e'){
                    dd_settings.algorithm = Options::ExactCompilation;
                } else if(*optarg == 'h'){
                    dd_settings.algorithm = Options::HeuristicOnly;
                } else if(*optarg == 'f'){
                    dd_settings.algorithm = Options::ExactFractionalNumber;
                } else{
                    std::cout << "The method to compute the chromatic number was not correctly specified." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 'a':
                dd_settings.redirect_arcs = RedirectArcs::MostSimilarNode;
                break;
            case 'c':
                if(*optarg == 's'){
                    dd_settings.find_conflicts = SingleConflict;
                } else if(*optarg == 'm'){
                    dd_settings.find_conflicts = MultipleConflicts;
                } else if(*optarg == 'f'){
                    dd_settings.find_conflicts = LargestFlowConflict;
                    std::string oa(optarg);
                    if( oa != "f" ){
                        oa.erase(oa.begin());
                        dd_settings.largest_conflicts_limit = std::stoi(oa, nullptr, 10);
                        if(not (1 <= dd_settings.largest_conflicts_limit and dd_settings.largest_conflicts_limit <= 7)){
                            std::cout << "The minimum flow for selecting the conflicts is not between 0.1 and 1e^-7." << std::endl;
                            std::cout << "Program failed." << std::endl;
                            return -1;
                        }
                    }
                } else{
                    std::cout << "The conflict detection method was not correctly specified." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 'o':
                if(*optarg == 'l'){
                    dd_settings.vertex_ordering = Graph::OrderType::Lexicographic;
                } else if(*optarg == 'm'){
                    dd_settings.vertex_ordering = Graph::OrderType::MaxConnectedDegree;
                } else if(*optarg == 'd'){
                    dd_settings.vertex_ordering = Graph::OrderType::Dsatur;
                } else if(*optarg == 'o'){
                    dd_settings.vertex_ordering = Graph::OrderType::DsaturOriginal;
                } else if(*optarg == 'w'){
                    dd_settings.vertex_ordering = Graph::OrderType::MinWidth;
                } else{
                    std::cout << "The ordering type was not correctly specified." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;

            case 'r':
                if(*optarg == 'i'){
                    dd_settings.relaxation = Options::IP_Only;
                } else if(*optarg == 'l'){
                    dd_settings.relaxation = Options::LP_First;
                } else if(*optarg == 'm'){
                    dd_settings.relaxation = Options::Mixed;
                    std::string oa(optarg);
                    if( oa != "m" ){
                        oa.erase(oa.begin());
                        dd_settings.ip_frequency = std::stoi(oa, nullptr, 10);
                        if(dd_settings.ip_frequency <= 0){
                            std::cout << "The integer program frequency must be a positive number." << std::endl;
                            std::cout << "Program failed." << std::endl;
                            return -1;
                        }
                    }
                }else{
                    std::cout << "The IP relaxation method was not correctly specified." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 'l':
                dd_settings.num_longest_path_iterations = std::stoi(optarg, nullptr, 10);
                if(dd_settings.num_longest_path_iterations < 0){
                    std::cout << "The number of longest path refinement iterations must not be a negative number." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 'p':
                dd_settings.preprocess_graph = true;
                if(optarg != nullptr and *optarg != '\0'){
                    dd_settings.preprocessing_hint = std::stoi(optarg, nullptr, 10);
                }
                break;
            case 'd':
                if(*optarg == 'o'){
                    dd_settings.path_decomposition = PreferOneArcs;
                } else if(*optarg == 'a'){
                    dd_settings.path_decomposition = AvoidConflicts;
                } else if(*optarg == 'z'){
                    dd_settings.path_decomposition = PreferZeroArcs;
                } else{
                    std::cout << "The decomposition type was not correctly specified." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 'f':
                if(*optarg == 'n'){
                    dd_settings.formulation = Normal;
                } else if(*optarg == 'c'){
                    dd_settings.formulation = VarColorBound;
                }else if(*optarg == 'r'){
                    dd_settings.formulation = OneArcsContinuous;
                }else{
                    std::cout << "The IP/LP formulation type was not correctly specified." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 'k':
                dd_settings.use_clique_in_ordering = true;
                if(optarg != nullptr and *optarg != '\0'){
                    dd_settings.clique_num_branches = std::stoi(optarg, nullptr, 10);
                }
                break;
            case 'z':
                dd_settings.ordering_random_tiebreaks = true;
                break;
            case 'v':
                dd_settings.verbosity_frequency = std::stoi(optarg, nullptr, 10);
                if(dd_settings.verbosity_frequency <= 0){
                    std::cout << "The program output frequency must be a positive number." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 's':
                dd_settings.size_limit = std::stoi(optarg, nullptr, 10);
                if(dd_settings.size_limit <= 0){
                    std::cout << "The size limit of nodes in million must be a positive number." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }else if(5 < dd_settings.size_limit){
                    std::cout << "Allowing " << dd_settings.size_limit <<
                        " million nodes in the exact decision diagram, be sure that is what you want." << std::endl;
                }
                break;
            case 'e':
                dd_settings.MIP_emphasis = std::stoi(optarg, nullptr, 10);
                if(dd_settings.MIP_emphasis < 0 or 4 < dd_settings.MIP_emphasis){
                    std::cout << "CPLEX solving emphasis must be between 0 and 4." << std::endl;
                    std::cout << "Program failed." << std::endl;
                    return -1;
                }
                break;
            case 't':
                dd_settings.num_cores = std::stoi(optarg, nullptr, 10);
                break;
        }
    }

    //optind is the index in argv after going through all the options, now the arguments are given
    if(optind >= argc){
        std::cout << "No input file was given." << std::endl;
        return 0;
    }
    char const *filename = argv[optind];

    try{
        program_header(argc, argv);
        std::cout << "Begin ddcolors:" << std::endl;

        DDColors ddcolors(filename, dd_settings);

        int chromatic_number = ddcolors.run();

        if(dd_settings.algorithm == Options::HeuristicOnly){
            std::cout << "Heuristic bound of input graph is " << chromatic_number << std::endl;
        }else if(dd_settings.algorithm == Options::ExactFractionalNumber){
            std::cout << "Safe lower bound for fractional chromatic number of input graph is " << std::setprecision(20)
            << ddcolors.get_fractional_chromatic_number() << " thus " << chromatic_number << " is a lower bound";
            if(chromatic_number == ddcolors.get_upper_bound()){
                std::cout << ", this is equal to the upper bound of " << ddcolors.get_upper_bound()
                << " and therefore enough to determine the chromatic number." << std::endl;
            } else{
                std::cout << ", this is not equal to the upper bound of " << ddcolors.get_upper_bound()
                << " and therefore not enough to determine the chromatic number." << std::endl;
            }
        } else
            std::cout << "Chromatic number of input graph is " << chromatic_number << std::endl;
    }
    catch(const std::runtime_error &e){
        std::cout << e.what() << std::endl;
        std::cout << "Program failed." << std::endl;
        return -1;
    }
    catch(...){
        std::cout << "Program failed for an unknown reason." << std::endl;
        return -1;
    }
    std::cout << "Finished." << std::endl;
    return 0;
}
