#include <set>
#include <queue>
#include <boost/dynamic_bitset.hpp>

#include "Continuous.h"
#include "gnuplot-iostream.h"

using namespace boost;
using namespace std;
using namespace flowstar;

const int BUFSIZE = 8096;
char buf[BUFSIZE];

// System definition
int xcnt, ucnt;
int d;
vector<string> xname;
vector<string> uname;
vector<Expression_AST<Real>> xexpr;
vector<Expression_AST<Real>> uexpr;
vector<Interval> safeStateInterval;
vector<Interval> initialStateInterval;
int m, k;
double period, stepSize;

// Flowstar definition
int order = 6;
double eps = 1e-10;
int queueSize = 1000;
Interval I(-0.01, 0.01);  // remainder estimation
Computational_Setting setting;
Deterministic_Continuous_Dynamics dynamics({});

// Grids and graph
vector<vector<Interval>> grids;
vector<vector<vector<int>>> oneStepGraph;  // [start][meet] 0: not meet, 1: meet
vector<vector<int>> revKStepGraph;
dynamic_bitset<> Ts, Tk, Ti;

void parseModel(char* modelPath) {
    printf("[Info] Parsing model.\n");
    FILE *file = fopen(modelPath, "r");
    fscanf(file, "%d%d%d", &xcnt, &ucnt, &d);
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
    for (int i = 0; i < xcnt; i++) {
        double start, end;
        fscanf(file, "%lf%lf", &start, &end);
        safeStateInterval.push_back({start, end});
    }
    for (int i = 0; i < xcnt; i++) {
        double start, end;
        fscanf(file, "%lf%lf", &start, &end);
        initialStateInterval.push_back({start, end});
    }
    fclose(file);
}

void buildFlowstar() {
    printf("[Info] Building FLOW* configuration.\n");
    FILE *file = fopen("config.txt", "r");
    fscanf(file, "%d", &order);
    fscanf(file, "%lf", &eps);
    fscanf(file, "%d", &queueSize);
    double start, end;
    fscanf(file, "%lf%lf", &start, &end);
    I = Interval(start, end);
    setting.setFixedStepsize(stepSize, order);  // stepsize and order for reachability analysis
    setting.setTime(period);  // time horizon for a single control step
    setting.setCutoffThreshold(eps);  // cutoff threshold
    setting.setQueueSize(queueSize);  // queue size for the symbolic remainder
    setting.printOn();  // print out the steps
    vector<Interval> remainder_estimation(stateVars.size(), I);
    setting.setRemainderEstimation(remainder_estimation);

    setting.printOff();
    setting.prepare();

    vector<Expression_AST<Real>> ode = xexpr;
    for (int i = 0; i < ucnt; i++) {
        ode.push_back(Expression_AST<Real>("0"));
    }
    dynamics = Deterministic_Continuous_Dynamics(ode);
    fclose(file);
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
    double blockSize = (safeStateInterval[curDim].sup() - safeStateInterval[curDim].inf()) / d;
    for (int i = 0; i < d; i++) {
        double start = safeStateInterval[curDim].inf() + i * blockSize;
        double end = safeStateInterval[curDim].inf() + (i + 1) * blockSize;
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
    double blockSize = (safeStateInterval[curDim].sup() - safeStateInterval[curDim].inf()) / d;
    for (int i = 0; i < d; i++) {
        double start = safeStateInterval[curDim].inf() + i * blockSize;
        double end = safeStateInterval[curDim].inf() + (i + 1) * blockSize;
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
                double inLen = reachableState[i].intersect(safeStateInterval[i]).width();
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
            for (int miss = 0; miss <= m; miss++) {
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
    int edge = 0;
    revKStepGraph.resize(n);
    Ts.resize(n);
    Tk.resize(n);
    for (int id = 0; id < n; id++) {
        if ((*now)[id][0].any()) {
            Ts.set(id);
            edge += (*now)[id][0].count();
            Tk |= (*now)[id][0]; 
            for (int nextId = 0; nextId < n; nextId++) {
                if ((*now)[id][0].test(nextId)) {
                    revKStepGraph[nextId].push_back(id);
                }
            }
        }
    }
    printf("[Success] Start Region Size: %d\n", Ts.count());
    printf("          End Region: %d\n", Tk.count());
    printf("          Number of Edges: %d\n", edge);
}

void findLargestClosedSubgraph() {
    printf("[Info] Finding the largest closed subgraph.\n");
    int n = grids.size();
    dynamic_bitset<> visit(n);
    Ti.resize(n);
    Ti.set();  // all grids are in Ti at the begining
    queue<int> que;
    for (int id = 0; id < n; id++) {
        if (!Ts.test(id)) {
            que.push(id);
            visit.set(id);
        }
    }
    while (!que.empty()) {
        int id = que.front();
        que.pop();
        Ti.set(id, 0);
        for (int nextId: revKStepGraph[id]) {
            if (visit.test(nextId)) continue;
            visit.set(nextId);
            que.push(nextId);
        }
    }
    printf("[Success] Safe Initial Region Size: %d\n", Ti.count());
}

void checkSafety() {
    printf("[Info] Calculating area.\n");
    double area = 1, gridArea = 0;
    for (int d = 0; d < xcnt; d++) {
        Interval dim = initialStateInterval[d];
        area *= dim.width();
    }
    for (int i = 0; i < grids.size(); i++) {
        if (!Ti.test(i)) continue;
        auto &grid = grids[i];
        double nowArea = 1;
        for (int d = 0; d < xcnt; d++) {
            Interval dim = initialStateInterval[d].intersect(grid[d]);
            if (dim.width() < eps) {
                nowArea = 0;
                break;
            }
            nowArea *= dim.width();
        }
        gridArea += nowArea;
    }
    printf("       Initial state region: %f\n", area);
    printf("       Grids Intersection:   %f\n", gridArea);
    if (abs(area - gridArea) / area < 1e-6) {
        printf("       Result: safe\n");
    } else {
        printf("       Result: unsafe\n");
    }
}

void plotGrids() {
    if (grids[0].size() == 1) {
        printf("[Warning] No result image for 1 dimension.\n");
        double l = 1e100, r = 1e-100;
        for (int i = 0; i < grids.size(); i++) {
            if (!Ti.test(i)) continue;
            Interval dim = grids[i][0];
            l = min(l, dim.inf());
            r = max(r, dim.sup());
        }
        if (l < r) {
            printf("          Safe initial region: from %f to %f.\n", l, r);    
        }
        return;
    }
    Gnuplot gp;
    gp << "set terminal svg size 480, 480\n";
    gp << "set output 'output.svg'\n";
    gp << "set xrange [ " << safeStateInterval[0].inf() << " : " << safeStateInterval[0].sup() << " ]\n";
    gp << "set yrange [ " << safeStateInterval[1].inf() << " : " << safeStateInterval[1].sup() << " ]\n";
    vector<int> rowId;
    Interval colInt;
    int prevColId = 0;
    for (int i = 0; i < grids.size(); i++) {
        if (!Ti.test(i)) continue;
        if (rowId.size() && (i / d > prevColId ||  i != rowId.back() + 1)) {
            sprintf(buf, "set object rect from %f,%f to %f,%f fc 'green' fillstyle solid 1.0 noborder\n",
                colInt.inf(), grids[rowId.front()][1].inf(), colInt.sup(), grids[rowId.back()][1].sup());
            gp << buf;
            rowId.clear();
        }
        prevColId = i / d;
        colInt = grids[i][0];
        rowId.push_back(i);
    }
    if (rowId.size()) {
        sprintf(buf, "set object rect from %f,%f to %f,%f fc 'green' fillstyle solid 1.0 noborder\n",
            colInt.inf(), grids[rowId.front()][1].inf(), colInt.sup(), grids[rowId.back()][1].sup());
        gp << buf;
        rowId.clear();
    }
    prevColId = 0;
    for (int i = 0; i < grids.size(); i++) {
        if (!Ts.test(i)) continue;
        if (rowId.size() && (i / d > prevColId ||  i != rowId.back() + 1)) {
            sprintf(buf, "set object rect from %f,%f to %f,%f fc lt 2 fillstyle pattern 4 noborder\n",
                colInt.inf(), grids[rowId.front()][1].inf(), colInt.sup(), grids[rowId.back()][1].sup());
            gp << buf;
            rowId.clear();
        }
        prevColId = i / d;
        colInt = grids[i][0];
        rowId.push_back(i);
    }
    if (rowId.size()) {
        sprintf(buf, "set object rect from %f,%f to %f,%f fc lt 2 fillstyle pattern 4 noborder\n",
            colInt.inf(), grids[rowId.front()][1].inf(), colInt.sup(), grids[rowId.back()][1].sup());
        gp << buf;
        rowId.clear();
    }
    // initial state set
    {
        Interval dim0 = initialStateInterval[0];
        Interval dim1 = initialStateInterval[1];
        sprintf(buf, "set object rect from %f,%f to %f,%f lw 3 fs empty border fc 'blue'\n",
            dim0.inf(), dim1.inf(), dim0.sup(), dim1.sup());
        gp << buf;
    }
    gp << "set nokey\n";
    // gp << "set key outside\n";
    // gp << "set key right top\n";
    gp << "plot NaN title 'Safe state region' with boxes fc 'black', ";
    gp << "NaN title 'Safe initial state region' with boxes fc 'green' fillstyle solid 1.0 noborder, ";
    gp << "NaN title 'Local safety region' with boxes fc lt 2 fillstyle pattern 4, ";
    gp << "NaN title 'Initial state region' with boxes lw 3 fs empty border fc 'blue'\n";
}

int main(int argc, char** argv) {
    parseModel(argv[1]);
    buildFlowstar();
    buildGrids();
    buildOneStepGraph();
    buildKStepGraph();
    findLargestClosedSubgraph();
    checkSafety();
    plotGrids();
}
