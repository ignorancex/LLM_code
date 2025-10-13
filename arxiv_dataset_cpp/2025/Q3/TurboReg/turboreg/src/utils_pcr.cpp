#include <turboreg/utils_pcr.hpp>
#include <turboreg/utils_debug.hpp>
#include <cmath>
#include <turboreg/utils_pcr.hpp>

double turboreg::calculateRotationError(const torch::Tensor &est, const torch::Tensor &gt)
{
    float tr = (est.transpose(0, 1).matmul(gt)).trace().item<float>();
    return acos(std::min(std::max((tr - 1.0f) / 2.0f, -1.0f), 1.0f)) * 180.0f / M_PI;
}

double turboreg::calculateTranslationError(const torch::Tensor &est, const torch::Tensor &gt)
{
    torch::Tensor t = est - gt;
    return t.norm().item<float>() * 100.0f; // cm
}

bool turboreg::evaluationEst(const torch::Tensor &est,
                   const torch::Tensor &gt,
                   double reThresh,
                   double teThresh,
                   double &RE,
                   double &TE)
{
    torch::Tensor rotationEst = est.narrow(0, 0, 3).narrow(1, 0, 3);
    torch::Tensor rotationGt = gt.narrow(0, 0, 3).narrow(1, 0, 3);
    torch::Tensor translationEst = est.narrow(0, 0, 3).narrow(1, 3, 1);
    torch::Tensor translationGt = gt.narrow(0, 0, 3).narrow(1, 3, 1);

    RE = calculateRotationError(rotationEst, rotationGt);
    TE = calculateTranslationError(translationEst, translationGt);

    if (0 <= RE && RE <= reThresh && 0 <= TE && TE <= teThresh)
    {
        return true; 
    }
    return false; 
}

