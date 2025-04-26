#include <set>
#include <queue>
#include <boost/dynamic_bitset.hpp>

#include "Continuous.h"

using namespace boost;
using namespace std;
using namespace flowstar;

const int BUFSIZE = 8096;
char buf[BUFSIZE];

// System definition
int xcnt, ucnt;
double safeDist;
int d;
vector<string> xname;
vector<string> uname;
vector<Expression_AST<Real>> xexpr;
vector<Expression_AST<Real>> uexpr;
int m, k;
double period, stepSize;

// Flowstar definition
const int order = 6;
const double eps = 1e-10;
Computational_Setting setting;
Deterministic_Continuous_Dynamics dynamics({});

// Grids and graph
vector<vector<Interval>> grids;
vector<vector<vector<int>>> oneStepGraph;  // [start][meet] 0: not meet, 1: meet
vector<vector<int>> revKStepGraph;
dynamic_bitset<> Xs;
vector<int> safeInitialGrids;  // final answer

void parseModel(char* modelPath) {
    printf("[Info] Parsing model.\n");
    FILE *file = fopen(modelPath, "r");
    fscanf(file, "%d%d%lf%d", &xcnt, &ucnt, &safeDist, &d);
    for (int i = 0; i < xcnt; i++) {
        fscanf(file, "%s", buf);
        xname.push_back(buf);
        stateVars.declareVar(buf);
    }
    for (int i = 0; i < ucnt; i++) {
        fscanf(file, "%s", buf);
        uname.push_back(buf);
        stateVars.declareVar(buf);
    }
    fgets(buf, BUFSIZE, file);
    for (int i = 0; i < xcnt; i++) {
        fgets(buf, BUFSIZE, file);
        xexpr.push_back(Expression_AST<Real>(buf));
    }
    for (int i = 0; i < ucnt; i++) {
        fgets(buf, BUFSIZE, file);
        uexpr.push_back(Expression_AST<Real>(buf));
    }
    fscanf(file, "%lf%lf", &period, &stepSize);
    fscanf(file, "%d%d", &m, &k);
    fclose(file);
}

void buildFlowstar() {
    printf("[Info] Building the setting related to FLOW*.\n");
    setting.setFixedStepsize(stepSize, order);  // stepsize and order for reachability analysis
    setting.setTime(period);  // time horizon for a single control step
    setting.setCutoffThreshold(eps);  // cutoff threshold
    setting.setQueueSize(1000);  // queue size for the symbolic remainder
    setting.printOn();  // print out the steps

    // remainder estimation
    Interval I(-0.01, 0.01);
    vector<Interval> remainder_estimation(stateVars.size(), I);
    setting.setRemainderEstimation(remainder_estimation);

    setting.printOff();
    setting.prepare();

    vector<Expression_AST<Real>> ode = xexpr;
    for (int i = 0; i < ucnt; i++) {
        ode.push_back(Expression_AST<Real>("0"));
    }
    dynamics = Deterministic_Continuous_Dynamics(ode);
}

vector<Interval> curInt;
void buildGrids(int curDim=0) {
    if (curDim == 0) {
        printf("[Info] Building grids.\n");    
    }
    if (curDim == xcnt) {
        grids.push_back(curInt);
        return;
    }
    double blockSize = safeDist * 2 / d;
    for (int i = 0; i < d; i++) {
        double start = -safeDist + i * blockSize;
        double end = -safeDist + (i + 1) * blockSize;
        curInt.push_back(Interval(start, end));
        buildGrids(curDim + 1);
        curInt.pop_back();
    }
}

void getIntersectGridsId(int curDim, int curId, vector<Interval> &region, vector<int> &gridsId) {
    if (curDim == xcnt) {
        gridsId.push_back(curId);
        return;
    }
    double blockSize = safeDist * 2 / d;
    for (int i = 0; i < d; i++) {
        double start = -safeDist + i * blockSize;
        double end = -safeDist + (i + 1) * blockSize;
        if (Interval(start, end).intersect(region[curDim]).width() < eps) {
            continue;
        }
        getIntersectGridsId(curDim + 1, curId * d + i, region, gridsId);
    }
}

void buildOneStepGraph() {
    printf("[Info] Building one-step graph.\n");
    int process = 0;
    int edgeCnt = 0;
    oneStepGraph.resize(grids.size());
    // #pragma omp parallel for reduction(+:edgeCnt) num_threads(4)
    for (int start = 0; start < grids.size(); start++) {
        oneStepGraph[start].resize(2);
        for (int meet = 0; meet < 2; meet++) {
            // #pragma omp critical
            {
                process += 1;
                printf("\r       Process: %.2f%%", 100.0 * process / (grids.size() * 2));
                fflush(stdout);    
            }
            

            // The initial set is same as the current grid
            vector<Interval> initialState = grids[start];
            for (int i = 0; i < ucnt; i++) {
                initialState.push_back(Interval(0));
            }
            Flowpipe initial_set(initialState);
            Result_of_Reachability result;

            // Calculate the input if it meets the deadline.
            if (meet) {
                for (int i = 0; i < ucnt; i++) {
                    TaylorModel<Real> tm_u;
                    uexpr[i].evaluate(tm_u, initial_set.tmvPre.tms, order, initial_set.domain, setting.tm_setting.cutoff_threshold, setting.g_setting);
                    initial_set.tmvPre.tms[xcnt + i] = tm_u;    
                }
            }

            // Move forward one step
            vector<Constraint> unsafeSet;
            vector<Interval> reachableState;
            dynamics.reach(result, setting, initial_set, unsafeSet);
            result.fp_end_of_time.intEval(reachableState, order, setting.tm_setting.cutoff_threshold);
            
            // Check safety and build edge
            bool safe = true;
            for (int i = 0; i < xcnt; i++) {
                double segLen = reachableState[i].width();
                double inLen = reachableState[i].intersect(Interval(-safeDist, safeDist)).width();
                if (abs(segLen - inLen) > eps) {
                    safe = false;
                }
            }
            if (safe) {
                getIntersectGridsId(0, 0, reachableState, oneStepGraph[start][meet]);
                edgeCnt += oneStepGraph[start][meet].size();
            }
        }
    }
    printf("\r       Process: 100.00%%\n");
    printf("[Success] Number of edges: %d\n", edgeCnt);
}

void buildKStepGraph() {
    printf("[Info] Building K-step graph.\n");
    int n = grids.size();
    vector<vector<dynamic_bitset<>>> dp[2];  // grid, miss cnt
    vector<vector<dynamic_bitset<>>> *now = &dp[0];
    vector<vector<dynamic_bitset<>>> *prev = &dp[1];

    // initialization
    for (int i = 0; i < 2; i++) {
        dp[i].resize(n);
        for (int j = 0; j < n; j++) {
            dp[i][j].resize(m + 1);
            for (int k = 0; k <= m; k++) {
                dp[i][j][k].resize(n);
            }
        }
    }
    for (int id = 0; id < n; id++) {
        for (int miss = 0; miss <= m; miss++) {
            (*now)[id][miss].reset();
            (*now)[id][miss].set(id);  // can only reach itself in 0 step
        }
    }

    // transition
    for (int step = k - 1; step >= 0; step--) {
        swap(now, prev);
        for (int id = 0; id < n; id++) {
            // clean reachable
            for (int miss = 0; miss < m; miss++) {
                (*now)[id][miss].reset();
            }

            // try not meet when miss cnt < m
            set<int> unsafeSet;
            for (int miss = 0; miss < m; miss++) {
                if (oneStepGraph[id][0].size() == 0) {
                    unsafeSet.insert(miss);
                }
                for (int reachId: oneStepGraph[id][0]) {
                    (*now)[id][miss] |= (*prev)[reachId][miss + 1];
                    if (!(*prev)[reachId][miss + 1].any()) {
                        unsafeSet.insert(miss);
                    }
                }
            }

            // meet
            for (int miss = 0; miss <= m; miss++) {
                if (oneStepGraph[id][1].size() == 0) {
                    unsafeSet.insert(miss);
                }
                for (int reachId: oneStepGraph[id][1]) {
                    (*now)[id][miss] |= (*prev)[reachId][miss];
                    if (!(*prev)[reachId][miss].any()) {
                        unsafeSet.insert(miss);
                    }
                }
            }
            
            for (int miss: unsafeSet) {
                (*now)[id][miss].reset();
            }
        }
    }

    // final
    int start = 0, end = 0, edge = 0;
    dynamic_bitset<> reach(n);
    revKStepGraph.resize(n);
    Xs.resize(n);
    for (int id = 0; id < n; id++) {
        if ((*now)[id][0].any()) {
            Xs.set(id);
            start++;
            edge += (*now)[id][0].count();
            reach |= (*now)[id][0]; 
            for (int nextId = 0; nextId < n; nextId++) {
                if ((*now)[id][0].test(nextId)) {
                    revKStepGraph[nextId].push_back(id);
                }
            }
        }
    }
    end = reach.count();
    printf("[Success] Start Region Size: %d\n", start);
    printf("          End Region: %d\n", end);
    printf("          Number of Edges: %d\n", edge);
}

void findLargestClosedSubgraph() {
    printf("[Info] Finding the largest closed subgraph.\n");
    int n = grids.size();
    dynamic_bitset<> safe(n);
    dynamic_bitset<> visit(n);
    safe.set();  // all grids are in X0 at the begining
    queue<int> que;
    for (int id = 0; id < n; id++) {
        if (!Xs.test(id)) {
            que.push(id);
            visit.set(id);
        }
    }
    while (!que.empty()) {
        int id = que.front();
        que.pop();
        safe.set(id, 0);
        for (int nextId: revKStepGraph[id]) {
            if (visit.test(nextId)) continue;
            visit.set(nextId);
            que.push(nextId);
        }
    }
    for (int id = 0; id < n; id++) {
        if (safe.test(id)) {
            safeInitialGrids.push_back(id);
        }
    }
    printf("[Success] Safe Initial Region Size: %d\n", safe.count());
}

int main(int argc, char** argv) {
    parseModel(argv[1]);
    buildFlowstar();
    buildGrids();
    buildOneStepGraph();
    buildKStepGraph();
    findLargestClosedSubgraph();
}
