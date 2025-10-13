#include <turboreg/rigid_transform.hpp>
#include <torch/torch.h>
#include <turboreg/rigid_transform.hpp>

using namespace turboreg;

RigidTransform::RigidTransform()
    : R(torch::eye(3, torch::kFloat)), t(torch::zeros({3}, torch::kFloat)) {}

RigidTransform::RigidTransform(const torch::Tensor &trans)
{
    R = trans.slice(0, 0, 3).slice(1, 0, 3);
    t = trans.slice(0, 0, 3).slice(1, 3, 4).squeeze();
}

RigidTransform::RigidTransform(const torch::Tensor &rotation, const torch::Tensor &translation)
    : R(rotation.clone()), t(translation.clone()) {}

void RigidTransform::copy(const torch::Tensor &R_in, const torch::Tensor &t_in)
{
    R = R_in.clone();
    t = t_in.clone();
}

void RigidTransform::copy(const RigidTransform &other)
{
    this->copy(other.getRotation(), other.getTranslation());
}

torch::Tensor RigidTransform::getTransformation() const
{
    torch::Tensor trans = torch::eye(4, torch::kFloat);
    trans.slice(0, 0, 3).slice(1, 0, 3).copy_(R);
    trans.slice(0, 0, 3).slice(1, 3, 4).copy_(t.unsqueeze(1));
    return trans;
}

torch::Tensor RigidTransform::getRotation() const
{
    return R;
}

torch::Tensor RigidTransform::getTranslation() const
{
    return t;
}

torch::Tensor RigidTransform::transformPoints(const torch::Tensor &pts) const
{
    return torch::matmul(R, pts) + t.unsqueeze(1);
}

int RigidTransform::countInliers(const torch::Tensor &kptsSrc, const torch::Tensor &kptsDst, float metricThresh) const
{
    torch::Tensor kptsSrcRotated = transformPoints(kptsSrc);
    torch::Tensor diff = kptsSrcRotated - kptsDst;
    torch::Tensor norms = diff.norm(2, 0);
    torch::Tensor inlierMask = norms < metricThresh;
    return inlierMask.sum().item<int>();
}
