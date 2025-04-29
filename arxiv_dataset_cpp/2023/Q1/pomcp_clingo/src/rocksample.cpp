#include "rocksample.h"
#include <limits>
#include <sstream>

#include <unordered_map>

#include "utils.h"

using namespace std;
using namespace UTILS;

ROCKSAMPLE::ROCKSAMPLE(int size, int rocks, std::string asp_file)
:   Grid(size, size),
    Size(size),
    NumRocks(rocks),
    SmartMoveProb(0.95),
    UncertaintyCount(0),
    has_fixed_belief(false)
{
    NumActions = NumRocks + 5;
    NumObservations = 3;
    RewardRange = 20;
    Discount = 0.95;

    if (size == 4 && rocks == 1)
        Init_4_1();
    else if (size == 7 && rocks == 4)
        Init_7_4();
    else if (size == 7 && rocks == 8)
        Init_7_8();
    else if (size == 11 && rocks == 11)
        Init_11_11();
    else if (size == 15 && rocks == 15)
        Init_15_15();
    else if (size == 12 && rocks == 4)
        Init_12_4();
    else if (size == 24 && rocks == 4)
        Init_24_4();
    else if (size == 24 && rocks == 8)
        Init_24_8();
    else
        InitGeneral();

    // Load asp program with rules
    if (asp_file != ""){
        clingo_control.load(asp_file.c_str());

    
        std::ostringstream dist_const;
        dist_const << "#const max_range_dist=" << Clingo::Number(size) << ".";
        clingo_control.add("base", {}, dist_const.str().c_str());  
        std::ostringstream rock_const;
        rock_const << "#const num_rocks=" << Clingo::Number(rocks) << ".";
        clingo_control.add("base", {}, rock_const.str().c_str());  
        clingo_control.ground({{"base", {}}});
    }

}

ROCKSAMPLE::ROCKSAMPLE(int size, int rocks, int x, int y,
           const std::vector<double> &belief)
:   Grid(size, size),
    Size(size),
    NumRocks(rocks),
    SmartMoveProb(0.95),
    UncertaintyCount(0) ,
    has_fixed_belief(true),
    fixed_belief(belief)
{
    NumActions = NumRocks + 5;
    NumObservations = 3;
    RewardRange = 20;
    Discount = 0.95;

    if (size == 4 && rocks == 1)
        Init_4_1();
    else if (size == 7 && rocks == 4)
        Init_7_4();
    else if (size == 7 && rocks == 8)
        Init_7_8();
    else if (size == 11 && rocks == 11)
        Init_11_11();
    else if (size == 15 && rocks == 15)
        Init_15_15();
    else if (size == 12 && rocks == 4)
        Init_12_4();
    else if (size == 24 && rocks == 4)
        Init_24_4();
    else if (size == 24 && rocks == 8)
        Init_24_8();
    else
        InitGeneral();

    StartPos = COORD(x, y);

    // Load asp program with rules
    clingo_control.load("/home/daniele/Documents/pomcp_clingo_orig/asp_test_rocksample_neg_ex_2.lp");
    clingo_control.ground({{"base", {}}});
}

ROCKSAMPLE::ROCKSAMPLE(int size, int rocks, std::string shield_file, bool use_shield)
:   Grid(size, size),
    Size(size),
    NumRocks(rocks),
    SmartMoveProb(0.95),
    UncertaintyCount(0),
    has_fixed_belief(false)
{
    NumActions = NumRocks + 5;
    NumObservations = 3;
    RewardRange = 20;
    Discount = 0.95;

    if (size == 4 && rocks == 1)
        Init_4_1();
    else if (size == 7 && rocks == 4)
        Init_7_4();
    else if (size == 7 && rocks == 8)
        Init_7_8();
    else if (size == 11 && rocks == 11)
        Init_11_11();
    else if (size == 15 && rocks == 15)
        Init_15_15();
    else if (size == 12 && rocks == 4)
        Init_12_4();
    else if (size == 24 && rocks == 4)
        Init_24_4();
    else if (size == 24 && rocks == 8)
        Init_24_8();
    else
        InitGeneral();

    ifstream iff(shield_file);
    iff >> sample_shield_tr;
    if (iff.eof())
        return;

    double threshold;
    iff >> threshold;
    sampling_points.set_threshold(threshold);

    std::string line;
    std::getline(iff, line); // discard threshold line
    std::array<double, 1> p;
    std::vector<std::array<double, 1>> points;

    std::getline(iff, line);
    std::stringstream ssopen(line);
    while (ssopen >> p[0])
        points.emplace_back(p);
    sampling_points.set_points(points);

    // Load asp program with rules
    clingo_control.load("/home/daniele/Documents/pomcp_clingo_orig/asp_test_rocksample_neg_ex_2.lp");
    clingo_control.ground({{"base", {}}});
}

void ROCKSAMPLE::InitGeneral()
{
    HalfEfficiencyDistance = 20; // 20 IS STANDARD
    //HalfEfficiencyDistance = 2; // 2 IS NOISY
    StartPos = COORD(0, Size / 2);
    //RandomSeed(0);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        COORD pos;
        do {
            pos = COORD(random_state() % Size, random_state() % Size);
        } while (Grid(pos) >= 0);
        Grid(pos) = i;
        RockPos.push_back(pos);
    }
}

void RANDOM_ROCKSAMPLE::update(const ROCKSSETUP &rs) {
    assert(rs.num_rocks() == NumRocks);
    HalfEfficiencyDistance = 20; // 20 IS STANDARD
    StartPos = COORD(0, rs.grid_size()/2);
    Grid.SetAllValues(-1);
    RockPos.clear();
    for (int i = 0; i < NumRocks; ++i) {
        Grid(rs.get_rock(i)) = i;
        RockPos.push_back(rs.get_rock(i));
    }
}

void ROCKSAMPLE::Init_4_1()
{
    // Equivalent to RockSample_7_8.pomdpx
    cout << "Using special layout for rocksample(7, 8)" << endl;

    COORD rocks[] =
    {
        COORD(3, 2),
    };

    HalfEfficiencyDistance = 20;
    //HalfEfficiencyDistance = 2; // 2 IS NOISY
    StartPos = COORD(0, 0);
    Grid.SetAllValues(-1);

    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_7_4()
{
    // Equivalent to RockSample_7_8.pomdpx
    cout << "Using special layout for rocksample(7, 8)" << endl;

    COORD rocks[] =
    {
        COORD(1, 1),
        COORD(6, 2),
        COORD(5, 4),
        COORD(4, 6),
    };

    HalfEfficiencyDistance = 5;
    StartPos = COORD(0, 2);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_12_4()
{
    // this special case has rocks only on the left part of the map
    cout << "Using special layout for rocksample(12, 4)" << endl;

    COORD rocks[4];
    for (int i = 0; i < 4; i++) {
        rocks[i] = COORD(random_state() % 6, random_state() % 12);
        for (int j = 0; j < i; j++) {
            if (rocks[i] == rocks[j]) {
                --i;
                break;
            }
        }
    }

    HalfEfficiencyDistance = 5;
    // random starting point in the first half
    StartPos = COORD(0, random_state() % 12);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_24_4()
{
    // this special case has rocks only on the left part of the map
    cout << "Using special layout for rocksample(12, 4)" << endl;

    COORD rocks[4];
    for (int i = 0; i < 4; i++) {
        rocks[i] = COORD(random_state() % 24, random_state() % 24);
        for (int j = 0; j < i; j++) {
            if (rocks[i] == rocks[j]) {
                --i;
                break;
            }
        }
    }

    HalfEfficiencyDistance = 5;
    // random starting point in the first half
    StartPos = COORD(0, random_state() % 24);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_24_8()
{
    // this special case has rocks only on the left part of the map
    cout << "Using special layout for rocksample(12, 4)" << endl;

    COORD rocks[8];
    for (int i = 0; i < 8; i++) {
        rocks[i] = COORD(random_state() % 24, random_state() % 24);
        for (int j = 0; j < i; j++) {
            if (rocks[i] == rocks[j]) {
                --i;
                break;
            }
        }
    }

    HalfEfficiencyDistance = 5;
    // random starting point in the first half
    StartPos = COORD(0, random_state() % 24);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_7_8()
{
    // Equivalent to RockSample_7_8.pomdpx
    cout << "Using special layout for rocksample(7, 8)" << endl;

    COORD rocks[] =
    {
        COORD(2, 0),
        COORD(0, 1),
        COORD(3, 1),
        COORD(6, 3),
        COORD(2, 4),
        COORD(3, 4),
        COORD(5, 5),
        COORD(1, 6)
    };

    HalfEfficiencyDistance = 20;
    StartPos = COORD(0, 3);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_11_11()
{
    // Equivalent to RockSample_11_11.pomdp(x)
    cout << "Using special layout for rocksample(11, 11)" << endl;

    COORD rocks[] =
    {
        COORD(0, 3),
        COORD(0, 7),
        COORD(1, 8),
        COORD(2, 4),
        COORD(3, 3),
        COORD(3, 8),
        COORD(4, 3),
        COORD(5, 8),
        COORD(6, 1),
        COORD(9, 3),
        COORD(9, 9)
    };

    HalfEfficiencyDistance = 20;
    //HalfEfficiencyDistance = 2;
    StartPos = COORD(0, 5);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i)
    {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

void ROCKSAMPLE::Init_15_15()
{
    cout << "Using special layout for rocksample(15, 15)" << endl;

    COORD rocks[] =
    {
        COORD(0, 4),
        COORD(0, 11),
        COORD(1, 8),
        COORD(2, 4),
        COORD(2, 12),
        COORD(3, 9),
        COORD(4, 3),
        COORD(5, 14),
        COORD(6, 0),
        COORD(6, 8),
        COORD(9, 3),
        COORD(10, 2),
        COORD(11, 7),
        COORD(12, 12),
        COORD(14, 9)
    };

    HalfEfficiencyDistance = 20;
    //HalfEfficiencyDistance = 2;
    StartPos = COORD(0, 7);
    Grid.SetAllValues(-1);
    for (int i = 0; i < NumRocks; ++i) {
        Grid(rocks[i]) = i;
        RockPos.push_back(rocks[i]);
    }
}

STATE* ROCKSAMPLE::Copy(const STATE& state) const
{
    const ROCKSAMPLE_STATE& rockstate = safe_cast<const ROCKSAMPLE_STATE&>(state);
    ROCKSAMPLE_STATE* newstate = MemoryPool.Allocate();
    *newstate = rockstate;
    return newstate;
}

void ROCKSAMPLE::Validate(const STATE& state) const
{
    const ROCKSAMPLE_STATE& rockstate = safe_cast<const ROCKSAMPLE_STATE&>(state);
    assert(Grid.Inside(rockstate.AgentPos));
}

STATE* ROCKSAMPLE::CreateStartState() const
{
    ROCKSAMPLE_STATE* rockstate = MemoryPool.Allocate();
    rockstate->AgentPos = StartPos;
    rockstate->Rocks.clear();

    if (has_fixed_belief) {
        for (int i = 0; i < NumRocks; i++)
        {
            ROCKSAMPLE_STATE::ENTRY entry;
            entry.Collected = false;
            entry.Valuable = unif_dist(random_state) <= fixed_belief[i];
            entry.Count = 0;
            entry.Measured = 0;
            entry.ProbValuable = 0.5;
            entry.LikelihoodValuable = 1.0;
            entry.LikelihoodWorthless = 1.0;
            rockstate->Rocks.push_back(entry);
        }
        rockstate->Target = SelectTarget(*rockstate);
    }
    else {
        for (int i = 0; i < NumRocks; i++)
        {
            ROCKSAMPLE_STATE::ENTRY entry;
            entry.Collected = false;
            // if (i<2) entry.Valuable = true;
            // else entry.Valuable = false;
            entry.Valuable = unif_dist(random_state) <= 0.5;
            entry.Count = 0;
            entry.Measured = 0;
            entry.ProbValuable = 0.5;
            entry.LikelihoodValuable = 1.0;
            entry.LikelihoodWorthless = 1.0;
            rockstate->Rocks.push_back(entry);
        }

        rockstate->Target = SelectTarget(*rockstate);
    }
    return rockstate;
}

void ROCKSAMPLE::FreeState(STATE* state) const
{
    ROCKSAMPLE_STATE* rockstate = safe_cast<ROCKSAMPLE_STATE*>(state);
    MemoryPool.Free(rockstate);
}

// return true if the state is final
bool ROCKSAMPLE::Step(STATE& state, int action,
    observation_t& observation, double& reward) const
{
    ROCKSAMPLE_STATE& rockstate = safe_cast<ROCKSAMPLE_STATE&>(state);
    reward = 0;
    observation = E_NONE;

    if (action < E_SAMPLE) // move
    {
        switch (action)
        {
            case COORD::E_EAST:
                if (rockstate.AgentPos.X + 1 < Size)
                {
                    rockstate.AgentPos.X++;
                    break;
                }
                else
                {
                    reward = +10;
                    return true;
                }

            case COORD::E_NORTH:
                if (rockstate.AgentPos.Y + 1 < Size)
                    rockstate.AgentPos.Y++;
                else
                    reward = -100;
                break;

            case COORD::E_SOUTH:
                if (rockstate.AgentPos.Y - 1 >= 0)
                    rockstate.AgentPos.Y--;
                else
                    reward = -100;
                break;

            case COORD::E_WEST:
                if (rockstate.AgentPos.X - 1 >= 0)
                    rockstate.AgentPos.X--;
                else
                    reward = -100;
                break;
        }
    }

    if (action == E_SAMPLE) // sample
    {
        int rock = Grid(rockstate.AgentPos);
        if (rock >= 0 && !rockstate.Rocks[rock].Collected)
        {
            rockstate.Rocks[rock].Collected = true;
            if (rockstate.Rocks[rock].Valuable)
                reward = +10;
            else
                reward = -10;
        }
        else
        {
            reward = -100;
        }
    }

    if (action > E_SAMPLE) // check
    {
        int rock = action - E_SAMPLE - 1;
        assert(rock < NumRocks);
        observation = GetObservation(rockstate, rock);
        rockstate.Rocks[rock].Measured++;

        double distance = COORD::EuclideanDistance(rockstate.AgentPos, RockPos[rock]);
        double efficiency = (1 + pow(2, -distance / HalfEfficiencyDistance)) * 0.5;

        if (observation == E_GOOD)
        {
            rockstate.Rocks[rock].Count++;
            rockstate.Rocks[rock].LikelihoodValuable *= efficiency;
            rockstate.Rocks[rock].LikelihoodWorthless *= 1.0 - efficiency;

        }
        else
        {
            rockstate.Rocks[rock].Count--;
            rockstate.Rocks[rock].LikelihoodWorthless *= efficiency;
            rockstate.Rocks[rock].LikelihoodValuable *= 1.0 - efficiency;
        }
        double denom = (0.5 * rockstate.Rocks[rock].LikelihoodValuable) +
            (0.5 * rockstate.Rocks[rock].LikelihoodWorthless);
        rockstate.Rocks[rock].ProbValuable = (0.5 * rockstate.Rocks[rock].LikelihoodValuable) / denom;
    }

    if (rockstate.Target < 0 || rockstate.AgentPos == RockPos[rockstate.Target])
        rockstate.Target = SelectTarget(rockstate);

    //assert(reward != -100);
    return false;
}

/*
 * explore unlikely scenarios
 */
bool ROCKSAMPLE::Step(const VNODE *const mcts_root, STATE &state, int action, 
        observation_t& observation, double& reward, double min, double max) const
{
    // consider only the check actions
    if (action <= E_SAMPLE)
        return Step(state, action, observation, reward);

    const auto &qnode = mcts_root->Child(action);

    for (int i = 0; i < NumObservations; i++) {
        const VNODE *vnode = qnode.Child(i);

        if (!vnode || vnode->Beliefs().Empty())
            continue;

        const auto &meta = dynamic_cast<const ROCKSAMPLE_METAINFO &>(
            vnode->Beliefs().get_metainfo());
        int rock = action - E_SAMPLE - 1;
        double prob = meta.get_prob_valuable(rock);
        if (prob > min && prob < max) {
            std::cout << "ROCK " << rock << " PROB " << prob << " FORCE OBSERVATION " << i << std::endl;
            reward = 0.0;
            observation = i;
            return false;
        }
    }

    // default behavior
    return Step(state, action, observation, reward);
}

bool ROCKSAMPLE::LocalMove(STATE& state, const HISTORY& history,
    int stepObs, const STATUS& status) const
{
    ROCKSAMPLE_STATE& rockstate = safe_cast<ROCKSAMPLE_STATE&>(state);
    int rock = random_state() % NumRocks;
    rockstate.Rocks[rock].Valuable = !rockstate.Rocks[rock].Valuable;

    if (history.Back().Action > E_SAMPLE) // check rock
    {
        rock = history.Back().Action - E_SAMPLE - 1;
        int realObs = history.Back().Observation;

        // Condition new state on real observation
        int newObs = GetObservation(rockstate, rock);
        if (newObs != realObs)
            return false;

        // Update counts to be consistent with real observation
        if (realObs == E_GOOD && stepObs == E_BAD)
            rockstate.Rocks[rock].Count += 2;
        if (realObs == E_BAD && stepObs == E_GOOD)
            rockstate.Rocks[rock].Count -= 2;
    }
    return true;
}

void ROCKSAMPLE::GenerateLegal(const STATE& state, const HISTORY& history,
    vector<int>& legal, const STATUS& status) const
{

    const ROCKSAMPLE_STATE& rockstate =
        safe_cast<const ROCKSAMPLE_STATE&>(state);

    if (rockstate.AgentPos.Y + 1 < Size)
        legal.push_back(COORD::E_NORTH);

    legal.push_back(COORD::E_EAST);

    if (rockstate.AgentPos.Y - 1 >= 0)
        legal.push_back(COORD::E_SOUTH);

    if (rockstate.AgentPos.X - 1 >= 0)
        legal.push_back(COORD::E_WEST);

    int rock = Grid(rockstate.AgentPos);
    if (rock >= 0 && !rockstate.Rocks[rock].Collected)
        legal.push_back(E_SAMPLE);

    for (rock = 0; rock < NumRocks; ++rock){
        if (!rockstate.Rocks[rock].Collected)
            legal.push_back(rock + 1 + E_SAMPLE);
    }
}

void ROCKSAMPLE::GenerateFromRules(const STATE& state, const BELIEF_STATE &belief,
    vector<int>& actions, const STATUS& status) const
{
    const ROCKSAMPLE_STATE& rs = safe_cast<const ROCKSAMPLE_STATE&>(state);

    Clingo::SymbolVector externals;

    // OBSERVABLE
    int min_rock = -1, min_dist = 100000;
    int max_rock = -1, max_dist = 0;
    int num_sampled = 0;
    for (int i = 0; i < RockPos.size(); i++) {
        int dist = COORD::ManhattanDistance(rs.AgentPos, RockPos[i]);
        externals.push_back(Clingo::Function(
            "dist",
            {Clingo::Number(i), Clingo::Number(dist)}));
        externals.push_back(Clingo::Function(
            "delta_x",
            {Clingo::Number(i), Clingo::Number(RockPos[i].X - rs.AgentPos.X)}));

        externals.push_back(Clingo::Function(
            "delta_y",
            {Clingo::Number(i), Clingo::Number(RockPos[i].Y - rs.AgentPos.Y)}));

        if (rs.Rocks[i].Collected) {
            externals.push_back(Clingo::Function(
                        "sampled", {Clingo::Number(i)}));
            num_sampled++;
        }
        else {
            if (dist < min_dist) {
                min_dist = dist;
                min_rock = i;
            }
            if (dist > max_dist) {
                max_dist = dist;
                max_rock = i;
            }
        }
    }

    if (min_rock != -1) {
        externals.push_back(
            Clingo::Function("min_dist", {Clingo::Number(min_rock)}));
    }
    if (max_rock != -1) {
        externals.push_back(
            Clingo::Function("max_dist", {Clingo::Number(max_rock)}));
    }

    int num_sampled_percentage = (num_sampled/NumRocks) * 100;
    num_sampled_percentage = num_sampled_percentage - (num_sampled_percentage % 25);
    externals.push_back(
        Clingo::Function("num_sampled", {Clingo::Number(num_sampled_percentage)}));

    int best_rock = -1;
    double best_guess = -1.0;
    int worst_rock = -1;
    double worst_guess = 2.0;
    for (int i = 0; i < RockPos.size(); i++) {
        /*
        const auto& meta =
            dynamic_cast<const ROCKSAMPLE_METAINFO&>(belief.get_metainfo());

        // Belief on root, likelyhood otherwise
        double real_val = belief.Empty()
                              ? meta.get_prob_valuable(i)
                              : rs.Rocks[i].LikelihoodValuable;
        */
        double real_val = rs.Rocks[i].ProbValuable;
        // prob arrotondata
        int val = real_val * 100;
        val = val - (val % 10);
        externals.push_back(Clingo::Function(
            "guess", {Clingo::Number(i), Clingo::Number(val)}));

        if (real_val > best_guess) {
            best_guess = real_val;
            best_rock = i;
        }
        if (real_val < worst_guess) {
            worst_guess = real_val;
            worst_rock = i;
        }
    }


    externals.push_back(Clingo::Function("best_guess", {Clingo::Number(best_rock)}));
    externals.push_back(Clingo::Function("worst_guess", {Clingo::Number(worst_rock)}));

    // assign externals
    for (auto s : externals) {
            clingo_control.assign_external(s, Clingo::TruthValue::True);
    }

#if 0
    cout << "--------------------" << endl;
    cout << "total " << clingo_control.symbolic_atoms().length() << endl;
    for (auto &s : clingo_control.symbolic_atoms()) {
        cout << s.symbol().to_string()
            << " ext:" << s.is_external()
            << " pos:" << s.symbol().is_positive()
            << " neg:" << s.symbol().is_negative()
            << endl;

    }
    cout << "--------------------" << endl;
#endif

    bool empty = true;
    for (auto &m : clingo_control.solve()) {
        for (auto &atom : m.symbols()) {
            // print
            empty = false;
            //cout << atom << " ";

            if (strcmp(atom.name(), "north") == 0)
                actions.push_back(COORD::E_NORTH);

            else if (strcmp(atom.name(), "east") == 0 ||
                     strcmp(atom.name(), "exit") == 0)
                actions.push_back(COORD::E_EAST);

            else if (strcmp(atom.name(), "south") == 0)
                actions.push_back(COORD::E_SOUTH);
            else if (strcmp(atom.name(), "west") == 0)
                actions.push_back(COORD::E_WEST);
            else if (strcmp(atom.name(), "sample") == 0)
                actions.push_back(E_SAMPLE);
            else if (strcmp(atom.name(), "check") == 0) {
                int n = atom.arguments()[0].number();
                actions.push_back(E_SAMPLE + n + 1);
            }
        }
    }

    /*
    if (!empty)
        cout << endl;
    */

    // release externals
    for (auto s: externals) {
            //clingo_control.release_external(s);
            clingo_control.assign_external(s, Clingo::TruthValue::False);
    }
}



void ROCKSAMPLE::GenerateFromRulesHardcoded15(const STATE& state, const BELIEF_STATE &belief,
    vector<int>& actions, const STATUS& status) const
{
    // cout << "USING HARDCODED RULES" << endl;
    chrono::steady_clock::time_point start = chrono::steady_clock::now();
    const ROCKSAMPLE_STATE& rs = safe_cast<const ROCKSAMPLE_STATE&>(state);

    std::vector<Target> targets;
    int min_rock = -1, min_dist = 100000;
    int max_rock = -1, max_dist = 0;
    int num_sampled = 0;
    double real_val;
    int val, dist;
    for (int i=0; i<NumRocks; ++i){
        real_val = rs.Rocks[i].ProbValuable;
        val = real_val * 100;
        val = val - (val % 10);
        dist = COORD::ManhattanDistance(rs.AgentPos, RockPos[i]);
        
        if (rs.Rocks[i].Collected){
            num_sampled++;
        }
        //target
        else {
            if (dist<=0 && val<=80) actions.push_back(E_SAMPLE + i + 1); //check valid
            if ((val>=70 && val<=80) || dist<=1){
                Target target;
                target.val = val;
                target.dist = dist;
                target.rock = i;
                target.X = RockPos[i].X;
                target.Y = RockPos[i].Y;
                targets.push_back(target);
            }
        }
    }
    sort(targets.begin(), targets.end(), compare_targets);
    for (auto i=0; i<targets.size(); ++i)
    {
        Target best_target = targets[i];
        if (best_target.val>=90 and best_target.dist<=0) actions.push_back(E_SAMPLE); //sample valid
        if (best_target.val<=50) actions.push_back(E_SAMPLE + best_target.rock + 1); //check valid
        if (best_target.X - rs.AgentPos.X >= 1) actions.push_back(COORD::E_EAST); //east valid
        if (best_target.X - rs.AgentPos.X <= -1) actions.push_back(COORD::E_WEST); //west valid
        if (best_target.Y - rs.AgentPos.Y >= 1) actions.push_back(COORD::E_NORTH); //north valid
        if (best_target.Y - rs.AgentPos.Y <= -1) actions.push_back(COORD::E_SOUTH); //south valid

        if (!(i+1<targets.size() && targets[i+1].dist == targets[i].dist && targets[i+1].val == targets[i].val)) break; 
    }

    //exit (COMMENT FOR AAMAS RESULTS)
    double num_rocks_double = NumRocks;
    int num_sampled_percentage = (num_sampled/num_rocks_double) * 100;
    num_sampled_percentage = num_sampled_percentage - (num_sampled_percentage % 25);
    if (num_sampled_percentage >= 25){
        for (int i=0; i<NumRocks; ++i){
            real_val = rs.Rocks[i].ProbValuable;
            val = real_val * 100;
            val = val - (val % 10);
            dist = COORD::ManhattanDistance(rs.AgentPos, RockPos[i]);
            
            if (!rs.Rocks[i].Collected){
                if ((val<=40) || (dist<=8 && dist >=5)) actions.push_back(COORD::E_EAST);
                break;
            }
        }
    }


}

void ROCKSAMPLE::GeneratePreferred(const STATE& state, const HISTORY& history,
    vector<int>& actions, const STATUS& status) const
{
    static const bool UseBlindPolicy = false;

    if (UseBlindPolicy) {
        actions.push_back(COORD::E_EAST);
        return;
    }

    const auto& rockstate = safe_cast<const ROCKSAMPLE_STATE&>(state);

    // Sample rocks with more +ve than -ve observations
    int rock = Grid(rockstate.AgentPos);
    if (rock >= 0 && !rockstate.Rocks[rock].Collected) {
        int total = 0;
        for (int t = 0; t < history.Size(); ++t) {
            if (history[t].Action == rock + 1 + E_SAMPLE)
            {
                if (history[t].Observation == E_GOOD)
                    total++;
                if (history[t].Observation == E_BAD)
                    total--;
            }
        }
        if (total > 0) {
            actions.push_back(E_SAMPLE);
            return;
        }
    }

    // processes the rocks
    bool all_bad = true;
    bool north_interesting = false;
    bool south_interesting = false;
    bool west_interesting  = false;
    bool east_interesting  = false;

    for (int rock = 0; rock < NumRocks; ++rock) {
        const ROCKSAMPLE_STATE::ENTRY& entry = rockstate.Rocks[rock];
        if (!entry.Collected) {
            int total = 0;
            for (int t = 0; t < history.Size(); ++t) {
                if (history[t].Action == rock + 1 + E_SAMPLE) {
                    if (history[t].Observation == E_GOOD)
                        total++;
                    if (history[t].Observation == E_BAD)
                        total--;
                }
            }

            if (total >= 0) {
                all_bad = false;

                if (RockPos[rock].Y > rockstate.AgentPos.Y)
                    north_interesting = true;
                if (RockPos[rock].Y < rockstate.AgentPos.Y)
                    south_interesting = true;
                if (RockPos[rock].X < rockstate.AgentPos.X)
                    west_interesting = true;
                if (RockPos[rock].X > rockstate.AgentPos.X)
                    east_interesting = true;
            }
        }
    }

    // if all remaining rocks seem bad, then head east
    if (all_bad) {
        actions.push_back(COORD::E_EAST);
        return;
    }

    // generate a random legal move, with the exceptions that:
    //   a) there is no point measuring a rock that is already collected
    //   b) there is no point measuring a rock too often
    //   c) there is no point measuring a rock which is clearly bad or good
    //   d) we never sample a rock (since we need to be sure)
    //   e) we never move in a direction that doesn't take us closer to
    //      either the edge of the map or an interesting rock
    if (rockstate.AgentPos.Y + 1 < Size && north_interesting)
        actions.push_back(COORD::E_NORTH);

    if (east_interesting)
        actions.push_back(COORD::E_EAST);

    if (rockstate.AgentPos.Y - 1 >= 0 && south_interesting)
        actions.push_back(COORD::E_SOUTH);

    if (rockstate.AgentPos.X - 1 >= 0 && west_interesting)
        actions.push_back(COORD::E_WEST);

    for (rock = 0; rock < NumRocks; ++rock) {
        if (!rockstate.Rocks[rock].Collected &&
            rockstate.Rocks[rock].ProbValuable != 0.0 &&
            rockstate.Rocks[rock].ProbValuable != 1.0 &&
            rockstate.Rocks[rock].Measured < 5 &&
            std::abs(rockstate.Rocks[rock].Count) < 2
            ) {
            actions.push_back(rock + 1 + E_SAMPLE);
        }
    }
}

int ROCKSAMPLE::GetObservation(const ROCKSAMPLE_STATE& rockstate, int rock) const
{
    double distance = COORD::EuclideanDistance(rockstate.AgentPos, RockPos[rock]);
    double efficiency = (1.0 + pow(2, -distance/HalfEfficiencyDistance)) * 0.5;

    if (unif_dist(random_state) < efficiency)
        return rockstate.Rocks[rock].Valuable ? E_GOOD : E_BAD;
    else
        return rockstate.Rocks[rock].Valuable ? E_BAD : E_GOOD;
}

int ROCKSAMPLE::SelectTarget(const ROCKSAMPLE_STATE& rockstate) const
{
    int bestDist = Size * 2;
    int bestRock = -1;
    for (int rock = 0; rock < NumRocks; ++rock)
    {
        if (!rockstate.Rocks[rock].Collected
            && rockstate.Rocks[rock].Count >= UncertaintyCount)
        {
            int dist = COORD::ManhattanDistance(rockstate.AgentPos, RockPos[rock]);
            if (dist < bestDist)
                bestDist = dist;
        }
    }
    return bestRock;
}

void ROCKSAMPLE::DisplayBeliefs(const BELIEF_STATE& beliefState,
    std::ostream& ostr) const
{
}

void ROCKSAMPLE::DisplayState(const STATE& state, std::ostream& ostr) const
{
    const ROCKSAMPLE_STATE& rockstate = safe_cast<const ROCKSAMPLE_STATE&>(state);
    ostr << endl;
    for (int x = 0; x < Size + 2; x++)
        ostr << "# ";
    ostr << endl;
    for (int y = Size - 1; y >= 0; y--)
    {
        ostr << "# ";
        for (int x = 0; x < Size; x++)
        {
            COORD pos(x, y);
            int rock = Grid(pos);
            const ROCKSAMPLE_STATE::ENTRY& entry = rockstate.Rocks[rock];
            if (rockstate.AgentPos == COORD(x, y))
                ostr << "* ";
            else if (rock >= 0 && !entry.Collected)
                ostr << rock << (entry.Valuable ? "$" : "X");
            else
                ostr << ". ";
        }
        ostr << "#" << endl;
    }
    for (int x = 0; x < Size + 2; x++)
        ostr << "# ";
    ostr << endl;
}

void ROCKSAMPLE::DisplayObservation(const STATE& state, observation_t observation, std::ostream& ostr) const
{
    switch (observation)
    {
    case E_NONE:
        break;
    case E_GOOD:
        ostr << "Observed good" << endl;
        break;
    case E_BAD:
        ostr << "Observed bad" << endl;
        break;
    }
}

void ROCKSAMPLE::DisplayAction(int action, std::ostream& ostr) const
{
    if (action < E_SAMPLE)
        ostr << COORD::CompassString[action] << endl;
    if (action == E_SAMPLE)
        ostr << "Sample" << endl;
    if (action > E_SAMPLE)
        ostr << "Check " << action - E_SAMPLE << endl;
}

void ROCKSAMPLE::log_problem_info() const {
    XES::logger().add_attributes({
            {"problem", "rocksample"},
            {"RewardRange", RewardRange},
            {"HalfEfficiencyDistance", HalfEfficiencyDistance},
            {"Size", Size},
            {"NumRocks", NumRocks}
        });

    XES::logger().start_list("rocks");
    for (int i = 0; i < RockPos.size(); i++) {
        XES::logger().start_list("rock" + std::to_string(i));
        XES::logger().add_attributes({
                {"number", i},
                {"coord x", RockPos[i].X},
                {"coord y", RockPos[i].Y},
                });
        XES::logger().end_list();
    }
    XES::logger().end_list(); // end rocks
}

void ROCKSAMPLE::log_beliefs(const BELIEF_STATE& beliefState) const {
    std::unordered_map<int, int> dist;
    int id;
    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* state = beliefState.GetSample(i);
        const auto& rockstate = safe_cast<const ROCKSAMPLE_STATE&>(*state);
        
        // Compute state id
        id=0;   
        for (int j = 0; j<rockstate.Rocks.size(); j++) {
            id += rockstate.Rocks[j].Valuable ? pow2(j) : 0;
        }
        // If it is not in dist then add it and initialize to 1
        if (dist.find(id) == dist.end())
            dist[id]= 1;
        else
            dist[id]++;
    }

    const STATE* state = beliefState.GetSample(0);
    const auto& rs = safe_cast<const ROCKSAMPLE_STATE&>(*state);
    XES::logger().add_attribute({"coord x", rs.AgentPos.X });
    XES::logger().add_attribute({"coord y", rs.AgentPos.Y });
    XES::logger().start_list("belief");
    for (const auto &[k, v] : dist)
        XES::logger().add_attribute({std::to_string(k), v});
    XES::logger().end_list();
}

void ROCKSAMPLE::log_state(const STATE& state) const {
    const ROCKSAMPLE_STATE& rs = safe_cast<const ROCKSAMPLE_STATE&>(state);
    XES::logger().start_list("state");
    XES::logger().add_attribute({"coord x", rs.AgentPos.X });
    XES::logger().add_attribute({"coord y", rs.AgentPos.Y });
    XES::logger().start_list("rocks");
    int i = 0;
    for (const auto &r : rs.Rocks) {
        XES::logger().start_list("rock");
        XES::logger().add_attributes({
                {"number", i},
                {"valuable", r.Valuable},
                {"collected", r.Collected}
                });
        XES::logger().end_list();
        ++i;
    }
    XES::logger().end_list(); // end rocks
    XES::logger().end_list(); // end state
}

void ROCKSAMPLE::log_action(int action) const {
    switch (action) {
        case COORD::E_EAST:
            XES::logger().add_attribute({"action", "east"});
            return;

        case COORD::E_NORTH:
            XES::logger().add_attribute({"action", "north"});
            return;

        case COORD::E_SOUTH:
            XES::logger().add_attribute({"action", "south"});
            return;

        case COORD::E_WEST:
            XES::logger().add_attribute({"action", "west"});
            return;
        case E_SAMPLE:
            XES::logger().add_attribute({"action", "sample"});
            return;
        default:
            int rock = action - E_SAMPLE - 1;
            XES::logger().add_attribute({"action", "check " + std::to_string(rock)});
            return;
    }
}

void ROCKSAMPLE::log_observation(const STATE& state, observation_t observation) const {
    switch (observation) {
        case E_NONE:
            XES::logger().add_attribute({"observation", "none"});
            return;
        case E_GOOD:
            XES::logger().add_attribute({"observation", "good"});
            return;
        case E_BAD:
            XES::logger().add_attribute({"observation", "bad"});
            return;
    }
}

void ROCKSAMPLE::log_reward(double reward) const {
    XES::logger().add_attribute({"reward", reward});
}

void ROCKSAMPLE::set_belief_metainfo(VNODE *v, const SIMULATOR &) const {
    v->Beliefs().set_metainfo(ROCKSAMPLE_METAINFO(NumRocks), *this);
}

void ROCKSAMPLE::pre_shield(const BELIEF_STATE &belief, std::vector<int> &legal_actions) const {
    const STATE* state = belief.GetSample(0);
    const auto& rs = safe_cast<const ROCKSAMPLE_STATE&>(*state);
    const auto& meta =
        dynamic_cast<const ROCKSAMPLE_METAINFO&>(belief.get_metainfo());

    // moving is always legal
    if (rs.AgentPos.Y - 1 >= 0)
        legal_actions.push_back(COORD::E_SOUTH);
    if (rs.AgentPos.Y + 1 < Size)
        legal_actions.push_back(COORD::E_NORTH);
    if (rs.AgentPos.X - 1 >= 0)
        legal_actions.push_back(COORD::E_WEST);
    legal_actions.push_back(COORD::E_EAST);
               
    // checks are always legal
    for (int rock = 0; rock < NumRocks; ++rock) {
        int action = E_SAMPLE + 1 + rock;
        legal_actions.push_back(action);
    }

    // only sample if safe
    int rock = Grid(rs.AgentPos);
    if (rock >= 0 && !rs.Rocks[rock].Collected) {
        double p = meta.get_prob_valuable(rock);
        if (p >= sample_shield_tr)
            legal_actions.push_back(E_SAMPLE);
        else if (sampling_points.is_in_threshold({p}))
            legal_actions.push_back(E_SAMPLE);
    }
}

Classification ROCKSAMPLE::check_rule(const BELIEF_META_INFO &m, int a, double t) const {
    const ROCKSAMPLE_METAINFO &meta = safe_cast<const ROCKSAMPLE_METAINFO &>(m);

    int rock = -1;
    COORD coord(meta.x(), meta.y());
    for (int i = 0; i < NumRocks; ++i)
    {
        if (coord == RockPos[i]) {
            rock = i;
            break;
        }
    }

    if (a == E_SAMPLE) {
        if ( rock >= 0 && !meta.collected(rock) && meta.get_prob_valuable(rock) >= t)
            return TRUE_POSITIVE; // the rule correctly describes the action
        else
            return FALSE_NEGATIVE; // the rule should have described this action
    }
    else {
        if (rock == -1 || meta.collected(rock) || meta.get_prob_valuable(rock) < t)
            return TRUE_NEGATIVE; // the rule correctly avoid sampling
        else
            return FALSE_POSITIVE; // the rule falsely describe this action
    }
}

void ROCKUPDATER::update() {
    setup.randomize();
    assert(setup.is_valid());
    if (real)
        real->update(setup);
    if (sim)
        sim->update(setup);
}

void RANDOM_ROCKSAMPLE::log_run_info() const {
    XES::logger().start_list("rocks");
    for (int i = 0; i < RockPos.size(); i++) {
        XES::logger().start_list("rock" + std::to_string(i));
        XES::logger().add_attributes({
                {"number", i},
                {"coord x", RockPos[i].X},
                {"coord y", RockPos[i].Y},
                });
        XES::logger().end_list();
    }
    XES::logger().end_list(); // end rocks
}

void RANDOM_ROCKSAMPLE::log_problem_info() const {
    XES::logger().add_attributes({
            {"problem", "rocksample"},
            {"RewardRange", RewardRange},
            {"HalfEfficiencyDistance", HalfEfficiencyDistance},
            {"Size", Size},
            {"NumRocks", NumRocks}
        });
}

void RANDOM_ROCKSAMPLE::pre_run() {
    if (updater) {
        updater->update();
    }
}
