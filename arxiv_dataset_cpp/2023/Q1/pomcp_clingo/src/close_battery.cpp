#include "close_battery.h"
#include "utils.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <clingo.hh>

using namespace std;
using namespace UTILS;

CLOSE_BATTERY::CLOSE_BATTERY(int l, std::vector<int> s, std::string asp_file)
    : length(l), stops(s), unif_dist(0.0, 1.0) {
    NumActions = 3;
    NumObservations = CLOSE_BATTERY_MAX + 1 + 3;
    RewardRange = 50;
    Discount = 1.00;

    nofuel_penalty = -100.0;
    check_penalty = -0.1;
    step_penalty = -0.1;
    recharge_penalty = -3.0;
    wrong_charge_penalty = -100.0;
    end_reward = 100.0;

    prob_correct_obs = 0.90;
    //prob_open_station = 0.80;
    prob_open_station = 1.0;

    if (asp_file != "") {
        clingo_control.load(asp_file.c_str());
        clingo_control.ground({{"base", {}}});
    }
}

STATE* CLOSE_BATTERY::Copy(const STATE& state) const // Makes a copy of the state state
{
    const CLOSE_BATTERY_STATE& bmState = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    CLOSE_BATTERY_STATE* newstate = MemoryPool.Allocate();
    *newstate = bmState;
    return newstate;
}

void CLOSE_BATTERY::Validate(const STATE& state) const
{
    const CLOSE_BATTERY_STATE& s = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    assert(s.battery_level <= CLOSE_BATTERY_MAX);
    assert(s.battery_level >= 0);
    assert(s.pos < length);
}

STATE* CLOSE_BATTERY::CreateStartState() const
{
    CLOSE_BATTERY_STATE* bmState = MemoryPool.Allocate();
    bmState->pos = 0;
    bmState->battery_level = CLOSE_BATTERY_MAX;
    bmState->open_stations = {};
    for (int i = 0; i < stops.size(); i++)
        bmState->open_stations.push_back(
                unif_dist(random_state) < prob_open_station);

    return bmState;
}

void CLOSE_BATTERY::FreeState(STATE* state) const
{
    CLOSE_BATTERY_STATE* bmState = safe_cast<CLOSE_BATTERY_STATE*>(state);
    MemoryPool.Free(bmState);
}

int CLOSE_BATTERY::get_station_from_pos(const CLOSE_BATTERY_STATE& state, int pos) const {
    for (int i = 0; i < stops.size(); i++)
        if (pos == stops[i])
            return i;
    return -1;
}

int CLOSE_BATTERY::get_next_station_pos(const CLOSE_BATTERY_STATE& state, int pos) const {
    for (int i = 0; i < stops.size(); i++)
        if (pos < stops[i])
            return stops[i];
    return -1;
}

bool CLOSE_BATTERY::Step(STATE& state, int action,
        observation_t& observation, double& reward) const
{
    CLOSE_BATTERY_STATE& s = safe_cast<CLOSE_BATTERY_STATE&>(state);

    if (action == CB_ACTION_RECHARGE) {
        observation = CLOSE_BATTERY_OBS_NONE;
        int station = get_station_from_pos(s, s.pos);
        if (station != -1 && s.open_stations[station]) {
            s.battery_level = CLOSE_BATTERY_MAX;
            reward = recharge_penalty;
        }
        else
            reward = wrong_charge_penalty;
        return false;
    }

    if (action == CB_ACTION_CHECK) {
        reward = check_penalty;

        bool correct = unif_dist(random_state) < prob_correct_obs;
        if (correct)
            observation = s.battery_level;
        else if (s.battery_level == CLOSE_BATTERY_MAX)
            observation = s.battery_level - 1;
        else if (s.battery_level == 0)
            observation = 1;
        else
            observation = unif_dist(random_state) < 0.5 ? s.battery_level - 1 :
                s.battery_level + 1;

        return false;
    }

    // ACTION_ADVANCE

    observation = CLOSE_BATTERY_OBS_NONE;

    if (s.battery_level == 0) {
        reward = nofuel_penalty;
        return true;
    }

    s.pos += 1;
    s.battery_level =
        unif_dist(random_state) < 0.8 ? s.battery_level - 1 : s.battery_level;

    int station = get_station_from_pos(s, s.pos);
    if (station == -1)
        observation = CLOSE_BATTERY_OBS_NONE;
    else if (s.open_stations[station])
        observation = CLOSE_BATTERY_OBS_OPEN;
    else 
        observation = CLOSE_BATTERY_OBS_CLOSE;

    if (s.pos == length) {
        reward = end_reward;
        observation = CLOSE_BATTERY_OBS_NONE;
        return true;
    }
    else {
        reward = step_penalty;
        return false;
    }
}

bool CLOSE_BATTERY::LocalMove(STATE& state, const HISTORY& history,
        int stepObs, const STATUS& status) const
{
    return false;
}

// Puts in legal a set of legal actions that can be taken from state
void CLOSE_BATTERY::GenerateLegal(const STATE& state, const HISTORY& history,
        vector<int>& legal, const STATUS& status) const
{
    const CLOSE_BATTERY_STATE& s = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    legal.push_back(CB_ACTION_CHECK);
    legal.push_back(CB_ACTION_ADVANCE);
    if (std::find(stops.begin(), stops.end(), s.pos) != stops.end())
        legal.push_back(CB_ACTION_RECHARGE);
}

void CLOSE_BATTERY::GeneratePreferred(const STATE& state, const HISTORY& history,
        vector<int>& actions, const STATUS& status) const
{
    const CLOSE_BATTERY_STATE& s = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    actions.push_back(0);
    actions.push_back(1);
    if (std::find(stops.begin(), stops.end(), s.pos) != stops.end())
        actions.push_back(2);
}

void CLOSE_BATTERY::GenerateFromRules(const STATE& state,
        const BELIEF_STATE &belief, std::vector<int>& actions,
        const STATUS& status) const
{
    const CLOSE_BATTERY_STATE& s = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    Clingo::SymbolVector externals;

    if (!belief.has_metainfo())
        return;

    const auto& meta =
        dynamic_cast<const CLOSE_BATTERY_METAINFO&>(belief.get_metainfo());

    if (meta.get_total() == 0) {
        return;
    }

    std::vector<int> cumulative_battery(CLOSE_BATTERY_MAX+1, 0);

    for (int i = CLOSE_BATTERY_MAX; i >= 0; --i) {
        cumulative_battery[i] = int(meta.get_battery_prob(i) * 100.0) - int(meta.get_battery_prob(i) * 100.0) % 10;
    }
    for (int i = CLOSE_BATTERY_MAX-1; i >= 0; --i)
        cumulative_battery[i] += cumulative_battery[i+1];

    for (int i = 0; i <= CLOSE_BATTERY_MAX; ++i) {
        externals.push_back(
            Clingo::Function("guess", {Clingo::Number(i),
                                       Clingo::Number(cumulative_battery[i])}));
    }

    int station = get_station_from_pos(s, s.pos);
    if (station != -1) {
        externals.push_back(
            Clingo::Function("at_station", {}));
        double real_prob = meta.get_station_prob(station);
        int prob = int(real_prob * 100.0) - (int(real_prob * 100.0) % 10);
        externals.push_back(
            Clingo::Function("open", {Clingo::Number(prob)}));
        //std::cout << "OPEN " << real_prob << " " << prob << " " << *externals.rbegin() << std::endl;
    }

    int next = get_next_station_pos(s, s.pos);
    if (next != -1) {
        double real_prob = meta.get_station_prob(next);
        int prob = int(real_prob * 100.0) - (int(real_prob * 100.0) % 10);
        externals.push_back(
            Clingo::Function("open_next", {Clingo::Number(prob)}));
        //std::cout << "NEXT " << real_prob << " " << prob << " " << *externals.rbegin() << std::endl;
    }

    if (next != -1) {
        // dist from next
        externals.push_back(
            Clingo::Function("dist_station_next", {Clingo::Number(next - s.pos)}));
    }
    else {
        // next "station" is the end, use length
        externals.push_back(
            Clingo::Function("dist_station_next", {Clingo::Number(length - s.pos)}));
    }

    // assign externals
    for (auto s : externals) {
        //std::cout << s << std::endl;
        clingo_control.assign_external(s, Clingo::TruthValue::True);

    }

    for (auto &m : clingo_control.solve()) {
        for (auto &atom : m.symbols()) {
            if (strcmp(atom.name(), "recharge") == 0)
                actions.push_back(CB_ACTION_RECHARGE);
            else if (strcmp(atom.name(), "check") == 0)
                actions.push_back(CB_ACTION_CHECK);
            else if (strcmp(atom.name(), "advance") == 0)
                actions.push_back(CB_ACTION_ADVANCE);
        }
    }

    // release externals
    for (auto s: externals) {
            clingo_control.assign_external(s, Clingo::TruthValue::False);
    }

    /*
    if (!actions.empty()) {
        for (auto a : actions)
            std::cout << a << " ";
        std::cout << std::endl;
    }
    */
}

void CLOSE_BATTERY::pre_run() {
    if (updater) {
        updater->update();
    }
}

// Display methods -------------------------
void CLOSE_BATTERY::DisplayBeliefs(const BELIEF_STATE& beliefState,    // Displays all states in beliefState
        std::ostream& ostr) const
{
    cout << "CLOSE_BATTERY::DisplayBeliefs start" << endl;
    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* s = beliefState.GetSample(i);
        DisplayState(*s,cout);
    }
    cout << "CLOSE_BATTERY::DisplayBeliefs end" << endl;
}

void CLOSE_BATTERY::DisplayBeliefIds(const BELIEF_STATE& beliefState,    // Displays all states in beliefState
        std::ostream& ostr) const
{
    cout << "CLOSE_BATTERY::DisplayBeliefIds: [";
    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* s = beliefState.GetSample(i);
        DisplayStateId(*s, cout);cout <<"; ";
    }
    cout << "CLOSE_BATTERY::DisplayBeliefs end" << endl;
}

void CLOSE_BATTERY::DisplayBeliefDistribution(const BELIEF_STATE& beliefState,
        std::ostream& ostr) const
{
    // TODO
}

void CLOSE_BATTERY::DisplayState(const STATE& state, std::ostream& ostr) const
{
    const CLOSE_BATTERY_STATE& bmState = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    ostr << endl;

    // Display segment difficulties
    ostr << "## STATE ############" << endl;
    ostr << "Position: i:" << bmState.pos << endl;
    ostr << "Battery: " << bmState.pos << endl;
}

void CLOSE_BATTERY::DisplayObservation(const STATE& state, observation_t observation, std::ostream& ostr) const // Prints the observation
{
    if (observation == 0)
        ostr << "Observed None" << endl;
}

void CLOSE_BATTERY::DisplayAction(int action, std::ostream& ostr) const
{
    if (action == 0)  
        ostr << "Action: move (0)" << endl;
    if (action == 1) 
        ostr << "Action: recharge (1)" << endl;
}


void CLOSE_BATTERY::DisplayStateId(const STATE& state, std::ostream& ostr) const
{
    // TODO
}

void CLOSE_BATTERY::log_problem_info() const {
    XES::logger().add_attributes({
            {"problem", "closed battery"},
            {"RewardRange", RewardRange},
            {"max battery", CLOSE_BATTERY_MAX},
            {"path length", length}
        });
}

void CLOSE_BATTERY::log_run_info() const {
    XES::logger().start_list("stations");
    for (int i = 0; i < stops.size(); i++) {
        XES::logger().start_list("station " + std::to_string(i));
        XES::logger().add_attributes({
                {"number", i},
                {"position", stops[i]},
                });
        XES::logger().end_list();
    }
    XES::logger().end_list(); // end stations
}

void CLOSE_BATTERY::log_beliefs(const BELIEF_STATE& beliefState) const {
    std::vector<int> battery_dist(CLOSE_BATTERY_MAX+1, 0);
    std::vector<int> open_distr(stops.size(), 0);
    std::vector<int> closed_distr(stops.size(), 0);

    for (int i = 0; i < beliefState.GetNumSamples(); i++){
        const STATE* state = beliefState.GetSample(i);
        const CLOSE_BATTERY_STATE& bmState = safe_cast<const CLOSE_BATTERY_STATE&>(*state);
        battery_dist[bmState.battery_level]++;
        for (int j = 0; j < stops.size(); j++) {
            if (bmState.open_stations[j])
                open_distr[j]++;
            else
                closed_distr[j]++;
        }
    }

    const CLOSE_BATTERY_STATE& bmState =
        safe_cast<const CLOSE_BATTERY_STATE&>(*beliefState.GetSample(0));

    XES::logger().add_attribute({"pos", bmState.pos });
    XES::logger().start_list("belief");

    XES::logger().start_list("battery level");
    for (int i = 0; i < battery_dist.size(); i++) {
        if (battery_dist[i] > 0) 
            XES::logger().add_attribute({std::to_string(i), battery_dist[i]});
    }
    XES::logger().end_list(); // end battery

    XES::logger().start_list("open stations");
    for (int i = 0; i < stops.size(); i++) {
        if (open_distr[i] > 0 || closed_distr[i] > 0) {
            XES::logger().add_attribute({
                    "station " + std::to_string(i) + " open", 
                    open_distr[i]});
            XES::logger().add_attribute({
                    "station " + std::to_string(i) + " closed", 
                    closed_distr[i]});
        }
    }
    XES::logger().end_list(); // end battery

    XES::logger().end_list(); // end belief
}

void CLOSE_BATTERY::log_state(const STATE& state) const {
    const auto& s = safe_cast<const CLOSE_BATTERY_STATE&>(state);
    XES::logger().start_list("state");
    XES::logger().add_attribute({"position", s.pos});
    XES::logger().add_attribute({"battery", s.battery_level});
    XES::logger().end_list();

    XES::logger().end_list();
}

void CLOSE_BATTERY::log_action(int action) const {
    switch (action) {
        case CB_ACTION_CHECK:
            XES::logger().add_attribute({"action", "check"});
            return;
        case CB_ACTION_ADVANCE:
            XES::logger().add_attribute({"action", "advance"});
            return;
        case CB_ACTION_RECHARGE:
            XES::logger().add_attribute({"action", "recharge"});
            return;
    }
}

void CLOSE_BATTERY::log_observation(const STATE& state, observation_t observation) const {
    if ( observation < CLOSE_BATTERY_MAX) {
        XES::logger().add_attribute({"observation",
                "Battery level " + std::to_string(observation)});
    }
    else {
        switch (observation) {
            case CLOSE_BATTERY_OBS_OPEN :
                XES::logger().add_attribute({"observation", "Open"});
                return;
            case CLOSE_BATTERY_OBS_CLOSE :
                XES::logger().add_attribute({"observation", "Close"});
                return;
            case CLOSE_BATTERY_OBS_NONE :
                XES::logger().add_attribute({"observation", "None"});
                return;
        }
    }
}

void CLOSE_BATTERY::log_reward(double reward) const {
    XES::logger().add_attribute({"reward", reward});
}

void CLOSE_BATTERY::set_belief_metainfo(VNODE *v, const SIMULATOR &) const {
    v->Beliefs().set_metainfo(CLOSE_BATTERY_METAINFO(), *this);
}

void CLOSE_BATTERY::pre_shield(const BELIEF_STATE &belief, std::vector<int> &legal_actions) const {
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

    const auto& s = safe_cast<const CLOSE_BATTERY_STATE&>(*belief.GetSample(0));
    const auto& meta =
        dynamic_cast<const CLOSE_BATTERY_METAINFO&>(belief.get_metainfo());

    if (std::find(stops.begin(), stops.end(), s.pos) != stops.end()) {
        double prob = 0.0;
        for (int i =0; i <= CLOSE_BATTERY_MAX; i++) {
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

int CLOSE_BATTERY::get_tree_count(int action) const {
    // tree count (hard coded)
    int base_count = 10;

    double check_w    = 1.0 - (7997.0 / 19009.0);
    double advance_w  = 1.0 - (8734.0 / 19009.0);
    double recharge_w = 1.0 - (1894.0 / 19009.0);

    if (CB_ACTION_CHECK   )
        return check_w * base_count;
    if (CB_ACTION_ADVANCE)
        return advance_w * base_count;
    if (CB_ACTION_RECHARGE)
        return recharge_w * base_count;
}

