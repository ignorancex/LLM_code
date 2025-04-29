#include "simulator.h"

using namespace std;
using namespace UTILS;

SIMULATOR::KNOWLEDGE::KNOWLEDGE()
:   TreeLevel(LEGAL),
    RolloutLevel(LEGAL),
    SmartTreeCount(10),
    dynamic_tree_count(false),
    SmartTreeValue(1.0)
{
}

SIMULATOR::STATUS::STATUS()
:   Phase(TREE),
    Particles(CONSISTENT)
{
}

SIMULATOR::SIMULATOR() 
:   Discount(1.0),
    NumActions(0),
    NumObservations(0),
    RewardRange(1.0),
    clingo_logger([](Clingo::WarningCode, char const *message) { cout << message << endl;}),
    clingo_control({{}, clingo_logger, 20})
{
}

SIMULATOR::SIMULATOR(int numActions, observation_t numObservations, double discount)
:   NumActions(numActions),
    NumObservations(numObservations),
    Discount(discount),
    clingo_logger([](Clingo::WarningCode, char const *message) { cout << message << endl;}),
    clingo_control({{}, clingo_logger, 20})
{ 
    assert(discount > 0 && discount <= 1);
}

SIMULATOR::~SIMULATOR() 
{ 
}

void SIMULATOR::Validate(const STATE& state) const 
{ 
}

bool SIMULATOR::LocalMove(STATE& state, const HISTORY& history,
    observation_t stepObs, const STATUS& status) const
{
    return true;
}

void SIMULATOR::GenerateLegal(const STATE& state, const HISTORY& history, 
    std::vector<int>& actions, const STATUS& status) const
{
    for (int a = 0; a < NumActions; ++a)
        actions.push_back(a);
}

void SIMULATOR::GeneratePreferred(const STATE& state, const HISTORY& history, 
    std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRules(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

void SIMULATOR::GenerateFromRulesHardcoded15(const STATE& state, const BELIEF_STATE &belief, 
        std::vector<int>& actions, const STATUS& status) const
{
}

int SIMULATOR::SelectRandom(const STATE &state, const HISTORY &history,
                            const BELIEF_STATE &belief,
                            const STATUS &status) const {
    static vector<int> actions;

    if (Knowledge.RolloutLevel >= KNOWLEDGE::RULES)
    {
        actions.clear();
        if (!Knowledge.UseHardcoded)
            GenerateFromRules(state, belief, actions, status);
        else
            GenerateFromRulesHardcoded15(state, belief, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }

    if (Knowledge.RolloutLevel >= KNOWLEDGE::SMART)
    {
        actions.clear();
        GeneratePreferred(state, history, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }
        
    if (Knowledge.RolloutLevel >= KNOWLEDGE::LEGAL)
    {
        actions.clear();
        GenerateLegal(state, history, actions, status);
        if (!actions.empty())
            return actions[Random(actions.size())];
    }

    return Random(NumActions);
}

void SIMULATOR::Prior(const STATE* state, const HISTORY& history,
    VNODE* vnode, const STATUS& status) const
{
    static vector<int> actions;
    
    if (Knowledge.TreeLevel == KNOWLEDGE::PURE || state == nullptr)
    {
        vnode->SetChildren(0, 0);
        return;
    }
    else
    {
        if (Knowledge.TreeLevel <= KNOWLEDGE::SMART) {
            vnode->SetChildren(+LargeInteger, -Infinity);

            if (Knowledge.TreeLevel >= KNOWLEDGE::LEGAL) {
                actions.clear();
                GenerateLegal(*state, history, actions, status);

                for (vector<int>::const_iterator i_action = actions.begin();
                     i_action != actions.end(); ++i_action) {
                    int a = *i_action;
                    QNODE &qnode = vnode->Child(a);
                    qnode.Value.Set(0, 0);
                    qnode.AMAF.Set(0, 0);
                }
            }

            if (Knowledge.TreeLevel >= KNOWLEDGE::SMART) {
                actions.clear();
                GeneratePreferred(*state, history, actions, status);

                for (vector<int>::const_iterator i_action = actions.begin();
                     i_action != actions.end(); ++i_action) {
                    int a = *i_action;
                    QNODE &qnode = vnode->Child(a);
                    qnode.Value.Set(Knowledge.SmartTreeCount,
                                    Knowledge.SmartTreeValue);
                    qnode.AMAF.Set(Knowledge.SmartTreeCount,
                                   Knowledge.SmartTreeValue);
                }
            }

            /*
            std::cout << "prior" << std::endl;
            for (int action = 0; action < NumActions; action++) {
                QNODE& qnode = vnode->Child(action);
                std::cout << qnode.Value.GetValue() << " " << qnode.Value.GetCount() << std::endl;
            }
            std::cout << std::endl;
            */
        }

        else if (Knowledge.TreeLevel >= KNOWLEDGE::RULES)
        {
            vnode->SetChildren(0, 0);

            actions.clear();
            if (!Knowledge.UseHardcoded)
                GenerateFromRules(*state, vnode->Beliefs(), actions, status);
            else
                GenerateFromRulesHardcoded15(*state, vnode->Beliefs(), actions, status);

            for (auto a : actions) {
                QNODE& qnode = vnode->Child(a);
                if (!Knowledge.dynamic_tree_count) {
                    qnode.Value.Set(Knowledge.SmartTreeCount, Knowledge.SmartTreeValue);
                    qnode.AMAF.Set(Knowledge.SmartTreeCount, Knowledge.SmartTreeValue);
                }
                else {
                    qnode.Value.Set(Knowledge.SmartTreeCount, get_tree_count(a));
                    qnode.AMAF.Set(Knowledge.SmartTreeCount, get_tree_count(a));
                }
            }

            actions.clear();
            GenerateLegal(*state, history, actions, status);

            for (int a = 0; a < NumActions; a++) {
                if (find(actions.begin(), actions.end(), a) == actions.end()) {
                    QNODE &qnode = vnode->Child(a);
                    qnode.Value.Set(+LargeInteger, -Infinity);
                    qnode.AMAF.Set(+LargeInteger, -Infinity);
                }
            }
        }
    }
}

bool SIMULATOR::HasAlpha() const
{
    return false;
}

void SIMULATOR::AlphaValue(const QNODE& qnode, double& q, int& n) const
{
}

void SIMULATOR::UpdateAlpha(QNODE& qnode, const STATE& state) const
{
}

void SIMULATOR::DisplayBeliefs(const BELIEF_STATE& beliefState, 
    ostream& ostr) const
{
}

void SIMULATOR::DisplayState(const STATE& state, ostream& ostr) const 
{
}

void SIMULATOR::DisplayAction(int action, ostream& ostr) const 
{
    ostr << "Action " << action << endl;
}

void SIMULATOR::DisplayObservation(const STATE& state, observation_t observation, ostream& ostr) const
{
    ostr << "Observation " << observation << endl;
}

void SIMULATOR::DisplayReward(double reward, std::ostream& ostr) const
{
    ostr << "Reward " << reward << endl;
}

double SIMULATOR::GetHorizon(double accuracy, int undiscountedHorizon) const 
{ 
    if (Discount == 1)
        return undiscountedHorizon;
    return log(accuracy) / log(Discount);
}
