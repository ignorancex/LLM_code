#include <motion_analysis/motion_analysis.h>
#include <iostream>
#include <boost/algorithm/string.hpp>
#include <regex>
#include <clocale>

PREDICATE(motion_is_linear, 1)
{
    // The first argument is a trajectory --> cast into string and parse
    std::string first_arg((char*)PL_A1);
    Trajectory traj = Trajectory::fromProlog(first_arg);

    // A motion is linear if it is both linear in translation and rotation
    return motion_is_linear_trans(traj) && motion_is_linear_rot(traj);
}

bool motion_is_linear_trans(const Trajectory& traj)
{
    // The translation component of a Trajectory is linear if the normalized difference vectors are almost the same
    // Skip near-zero difference vectors
    Eigen::Vector3d firstDiffNorm = (traj.datapoints_[1].trans_ - traj.datapoints_[0].trans_).normalized();
    for (size_t i = 0; i < traj.datapoints_.size() - 1; ++i)
    {
        Datapoint dp1 = traj.datapoints_[i];
        Datapoint dp2 = traj.datapoints_[i+1];
        Eigen::Vector3d diffNorm = (dp2.trans_ - dp1.trans_).normalized();
        if (diffNorm.isApprox(Eigen::Vector3d::Zero()))
        {
            continue;
        }
        if (!diffNorm.isApprox(firstDiffNorm, TRANS_LINEAR_EPS))
        {
            return false;
        }
    }
    return true;
}

bool motion_is_linear_rot(const Trajectory& traj)
{
    // The rotation component of a Trajectory is linear if the normalized difference quaternions are almost the same

    // Difference quaternions: https://stackoverflow.com/a/22167097
    Eigen::Quaterniond firstDiffNorm = (traj.datapoints_[1].rot_ * traj.datapoints_[0].rot_.inverse()).normalized();
    for (size_t i = 0; i < traj.datapoints_.size() - 1; ++i)
    {
        Datapoint dp1 = traj.datapoints_[i];
        Datapoint dp2 = traj.datapoints_[i+1];
        Eigen::Quaterniond diffNorm = (dp2.rot_ * dp1.rot_.inverse()).normalized();
        if (!diffNorm.isApprox(firstDiffNorm, ROT_LINEAR_EPS))
        {
            return false;
        }
    }
    return true;
}

Trajectory Trajectory::fromProlog(const std::string& trajString)
{
    // 1: Split datapoints apart
    std::vector<std::string> datapointStrings;
    std::regex re("[0-9].*?\\]\\]");
    std::smatch sm;
    std::string str(trajString);
    while(std::regex_search(str, sm, re))
    {
        datapointStrings.push_back(sm.str());
        str = sm.suffix();
    }
    // 2: Parse individual datapoints
    std::vector<Datapoint> datapoints;
    for (std::string dpString : datapointStrings)
    {
        datapoints.push_back(Datapoint::fromProlog(dpString));
    }
    return Trajectory(datapoints);
}

Datapoint Datapoint::fromProlog(const std::string& datapointString)
{
    std::vector<std::string> tokens;

    // Set locale, because std::stod is locale dependent
    const std::string oldLocale=std::setlocale(LC_NUMERIC,nullptr);
    std::setlocale(LC_NUMERIC,"C");

    // Extract timestamp
    auto split_at = datapointString.find("-[");
    double timestamp = std::stod(datapointString.substr(0, split_at));
    std::string rest = datapointString.substr(split_at + 2, datapointString.size());

    // Split reference CS, trans and rot
    std::regex re("(.+),(\\[.+\\]),(\\[.+\\])");
    std::smatch sm;
    std::regex_search(rest, sm, re);
    std::string referenceFrame = sm.str(1);
    std::string transStr = sm.str(2).substr(1, sm.str(2).size() - 2);
    std::string rotStr = sm.str(3).substr(1, sm.str(3).size() - 3);

    // Translation
    boost::split(tokens, transStr, boost::is_any_of(","));
    Eigen::Vector3d trans(std::stod(tokens[0]), std::stod(tokens[1]), std::stod(tokens[2]));
    // Rotation
    boost::split(tokens, rotStr, boost::is_any_of(","));
    Eigen::Quaterniond quat(std::stod(tokens[3]), std::stod(tokens[0]), std::stod(tokens[1]), std::stod(tokens[2]));

    std::setlocale(LC_NUMERIC,oldLocale.c_str());
    return Datapoint(timestamp, referenceFrame, trans, quat);
}