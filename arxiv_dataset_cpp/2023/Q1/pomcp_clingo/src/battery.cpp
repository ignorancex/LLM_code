#include "battery.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <z3.h>
#include <z3++.h>

using namespace std;
using namespace UTILS;

BATTERY::BATTERY(int l, std::vector<int> s)
    : length(l), stops(s), unif_dist(0.0, 1.0) {
    NumActions = 3;
    NumObservations = BATTERY_MAX + 2;
    RewardRange = 60;
    Discount = 0.99;

    nofuel_penalty = -100;
    check_penalty = -1;
    step_penalty = -1;
    recharge_penalty = -10;
    wrong_charge_penalty = -100;
    end_reward = 100;

    prob_correct_obs = 0.95;
}

STATE* BATTERY::Copy(const STATE& state) const // Makes a copy of the state state
{
    const BATTERY_STATE& bmState = safe_cast<const BATTERY_STATE&>(state);
    BATTERY_STATE* newstate = MemoryPool.Allocate();
    *newstate = bmState;
    return newstate;
}

void BATTERY::Validate(const STATE& state) const
{
    const BATTERY_STATE& s = safe_cast<const BATTERY_STATE&>(state);
    assert(s.battery_level <= BATTERY_MAX);
    assert(s.battery_level >= 0);
    assert(s.pos < length);
}

// Used by standard planner
STATE* BATTERY::CreateStartState() const
{
    BATTERY_STATE* bmState = MemoryPool.Allocate();
    bmState->pos = 0;
    bmState->battery_level = BATTERY_MAX;

    return bmState;
}

void BATTERY::FreeState(STATE* state) const
{
    BATTERY_STATE* bmState = safe_cast<BATTERY_STATE*>(state);
    MemoryPool.Free(bmState);
}


bool BATTERY::Step(STATE& state, int action,
        observation_t& observation, double& reward) const
{
    BATTERY_STATE& s = safe_cast<BATTERY_STATE&>(state);

    if (action == ACTION_RECHARGE) {
        observation = OBS_NONE;
        if (std::find(stops.begin(), stops.end(), s.pos) != stops.end()) {
            s.battery_level = BATTERY_MAX;
            reward = recharge_penalty;
        }
        else
            reward = wrong_charge_penalty;
        return false;
    }

    if (action == ACTION_CHECK) {
        reward = check_penalty;

        bool correct = unif_dist(random_state) < prob_correct_obs;
        if (correct)
            observation = s.battery_level;
        else if (s.battery_level == BATTERY_MAX)
            observation = s.battery_level - 1;
        else if (s.battery_level == 0)
            observation = 1;
        else
            observation = unif_dist(random_state) < 0.5 ? s.battery_level - 1 :
                s.battery_level + 1;

        return false;
    }

    // ACTION_ADVANCE

    observation = OBS_NONE;

    if (s.battery_level == 0) {
        reward = nofuel_penalty;
        return true;
    }

    s.pos += 1;
    s.battery_level =
        unif_dist(random_state) < 0.8 ? s.battery_level - 1 : s.battery_level;

    if (s.pos == length) {
        reward = end_reward;
        observation = 1;
        return true;
    }
    else {
        reward = step_penalty;
        return false;
    }
}

bool BATTERY::LocalMove(STATE& state, const HISTORY& history,
        int stepObs, const STATUS& status) const
{
    return false;
}

// Puts in legal a set of legal actions that can be taken from state
void BATTERY::GenerateLegal(const STATE& state, const HISTORY& history,
        vector<int>& legal, const STATUS& status) const
{
    const BATTERY_STATE& s = safe_cast<const BATTERY_STATE&>(state);
    legal.push_back(0);
    legal.push_back(1);
    if (std::find(stops.begin(), stops.end(), s.pos) != stops.end())
        legal.push_back(2);
}

void BATTERY::GeneratePreferred(const STATE& state, const HISTORY& history,
        vector<int>& actions, const STATUS& status) const
{
    const BATTERY_STATE& s = safe_cast<const BATTERY_STATE&>(state);
    actions.push_back(0);
    actions.push_back(1);
    if (std::find(stops.begin(), stops.end(), s.pos) != stops.end())
        actions.push_back(2);
}

// Display methods -------------------------
void BATTERY::DisplayBeliefs(const BELIEF_STATE& beliefState,    // Displays all states in beliefState
        std::ostream& ostr) const
{
    cout << "BATTERY::DisplayBeliefs start" << endl;
    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* s = beliefState.GetSample(i);
        DisplayState(*s,cout);
    }
    cout << "BATTERY::DisplayBeliefs end" << endl;
}

void BATTERY::DisplayBeliefIds(const BELIEF_STATE& beliefState,    // Displays all states in beliefState
        std::ostream& ostr) const
{
    cout << "BATTERY::DisplayBeliefIds: [";
    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* s = beliefState.GetSample(i);
        DisplayStateId(*s, cout);cout <<"; ";
    }
    cout << "BATTERY::DisplayBeliefs end" << endl;
}

void BATTERY::DisplayBeliefDistribution(const BELIEF_STATE& beliefState,
        std::ostream& ostr) const
{
    // TODO
}

void BATTERY::DisplayState(const STATE& state, std::ostream& ostr) const
{
    const BATTERY_STATE& bmState = safe_cast<const BATTERY_STATE&>(state);
    ostr << endl;

    // Display segment difficulties
    ostr << "## STATE ############" << endl;
    ostr << "Position: i:" << bmState.pos << endl;
    ostr << "Battery: " << bmState.pos << endl;
}

void BATTERY::DisplayObservation(const STATE& state, observation_t observation, std::ostream& ostr) const // Prints the observation
{
    if (observation == 0)
        ostr << "Observed None" << endl;
}

void BATTERY::DisplayAction(int action, std::ostream& ostr) const
{
    if (action == 0)  
        ostr << "Action: move (0)" << endl;
    if (action == 1) 
        ostr << "Action: recharge (1)" << endl;
}


void BATTERY::DisplayStateId(const STATE& state, std::ostream& ostr) const
{
    // TODO
}

void BATTERY::log_problem_info() const {
    XES::logger().add_attributes({
            {"problem", "battery"},
            {"RewardRange", RewardRange},
            {"max battery", BATTERY_MAX},
            {"path length", length}
        });
}

void BATTERY::log_beliefs(const BELIEF_STATE& beliefState) const {
    std::vector<int> dist(BATTERY_MAX+1, 0);
    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* state = beliefState.GetSample(i);
        const BATTERY_STATE& bmState = safe_cast<const BATTERY_STATE&>(*state);
        dist[bmState.battery_level]++;
    }

    const BATTERY_STATE& bmState =
        safe_cast<const BATTERY_STATE&>(*beliefState.GetSample(0));

    XES::logger().add_attribute({"pos", bmState.pos });
    XES::logger().start_list("belief");
    for (int i = 0; i < dist.size(); i++) {
        if (dist[i] > 0) 
            XES::logger().add_attribute({std::to_string(i), dist[i]});
    }
    XES::logger().end_list();
}

void BATTERY::log_state(const STATE& state) const {
    const auto& s = safe_cast<const BATTERY_STATE&>(state);
    XES::logger().start_list("state");
    XES::logger().add_attribute({"position", s.pos});
    XES::logger().add_attribute({"battery", s.battery_level});
    XES::logger().end_list();

    XES::logger().end_list();
}

void BATTERY::log_action(int action) const {
    XES::logger().add_attribute({"action", action});
}

void BATTERY::log_observation(const STATE& state, observation_t observation) const {
    XES::logger().add_attribute({"observation", observation});
}

void BATTERY::log_reward(double reward) const {
    XES::logger().add_attribute({"reward", reward});
}

void BATTERY::set_belief_metainfo(VNODE *v, const SIMULATOR &) const {
    v->Beliefs().set_metainfo(BATTERY_METAINFO(), *this);
}

void BATTERY::pre_shield(const BELIEF_STATE &belief, std::vector<int> &legal_actions) const {
    static const double prob_success[] = {0.0,
                                          0.00032000000000000013,
                                          0.006720000000000002,
                                          0.05792000000000001,
                                          0.26272000000000006,
                                          0.67232,
                                          1.0,
                                          1.0,
                                          1.0,
                                          1.0,
                                          1.0};

    const auto& s = safe_cast<const BATTERY_STATE&>(*belief.GetSample(0));
    const auto& meta =
        dynamic_cast<const BATTERY_METAINFO&>(belief.get_metainfo());

    if (std::find(stops.begin(), stops.end(), s.pos) != stops.end()) {
        double prob = 0.0;
        for (int i =0; i <= BATTERY_MAX; i++) {
            prob += prob_success[i] * meta.get_battery_prob(i);
        }
        if (prob < 0.95)
            legal_actions.push_back(2);
        else
            legal_actions.push_back(1);
    }
    else {
        legal_actions.push_back(0);
        legal_actions.push_back(1);
    }
}

