#include "CQUADIntegrator.hpp"

#include <cstddef>
#include <functional>

#include <gsl/gsl_integration.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
// public

CQUADIntegrator::CQUADIntegrator(const std::function<double(double)> &integrand, const std::size_t workSpaceSize)
	: Integrand(integrand),
	  WorkSpaceSize(workSpaceSize),
	  Workspace(gsl_integration_cquad_workspace_alloc(WorkSpaceSize)),
	  GSLFunction({&gsl_integrand, this})
{
}

void CQUADIntegrator::integrate(const double lowerIntegrationLimit, const double upperIntegrationLimit,
								const double absoluteErrorBound, const double relativeErrorBound,
								double &result, double &error, std::size_t &evaluationNumber) const
{
	gsl_integration_cquad(&GSLFunction, lowerIntegrationLimit, upperIntegrationLimit, absoluteErrorBound, relativeErrorBound, Workspace, &result, &error, &evaluationNumber);
}

void CQUADIntegrator::integrate(const double lowerIntegrationLimit, const double upperIntegrationLimit,
								const double absoluteErrorBound, const double relativeErrorBound,
								double &result, double &error) const
{
	gsl_integration_cquad(&GSLFunction, lowerIntegrationLimit, upperIntegrationLimit, absoluteErrorBound, relativeErrorBound, Workspace, &result, &error, NULL);
}

void CQUADIntegrator::integrate(const double lowerIntegrationLimit, const double upperIntegrationLimit,
								const double absoluteErrorBound, const double relativeErrorBound,
								double &result) const
{
	gsl_integration_cquad(&GSLFunction, lowerIntegrationLimit, upperIntegrationLimit, absoluteErrorBound, relativeErrorBound, Workspace, &result, NULL, NULL);
}

CQUADIntegrator::~CQUADIntegrator()
{
	gsl_integration_cquad_workspace_free(Workspace);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// private

double CQUADIntegrator::gsl_integrand(const double variable, void *parameters)
{
	CQUADIntegrator *This = static_cast<CQUADIntegrator *>(parameters);

	return This->Integrand(variable);
}