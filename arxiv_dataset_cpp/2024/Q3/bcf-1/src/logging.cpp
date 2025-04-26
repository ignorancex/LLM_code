#include <RcppArmadillo.h>
#include "logging.h"

using namespace Rcpp;

Logger::Logger(){
    level = 0;
    depth = 0;
}

void Logger::setLevel(int levelIn){
    level = levelIn;
}
void Logger::startContext(){
    depth +=1;
}
void Logger::stopContext(){
    depth +=-1;
}

void Logger::getVectorHead(Rcpp::NumericVector x, char s[100]){
    std::sprintf(s,"%f, %f, %f, %f, %f, %f, %f, %f, %f, %f ... ", x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
}

void Logger::getVectorHead(std::vector<double> x, char s[100]){
    std::sprintf(s,"%f, %f, %f, %f, %f, %f, %f, %f, %f, %f... ", x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
}

void Logger::getVectorHead(double* x, char s[100]){
    std::sprintf(s,"%f, %f, %f, %f, %f, %f, %f, %f, %f, %f... ", x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8],x[9]);
}

void log_iter(std::string context, int current, int final, double sigma, double mscale, double bscale0, double bscale1, Logger& logger) {
    char logBuff[100];
    logger.log("==============================================");
    sprintf(logBuff, "MCMC iteration: %d of %d ", current, final);
    logger.log(logBuff + context);
    sprintf(logBuff, "sigma %f, mscale %f, bscale0 %f, bscale1 %f",sigma, mscale, bscale0, bscale1);
    logger.log(logBuff);
    logger.log("==============================================");
}

void log_fit(std::vector<double>& y, double* allfit, double* allfit_con, double* allfit_mod, Logger& logger, bool verbose) {
    char logBuff[100];
    if (verbose) {
        logger.getVectorHead(y, logBuff);
        Rcout << "           y: " <<  logBuff << "\n";

        logger.getVectorHead(allfit, logBuff);
        Rcout << "Current Fit : " <<  logBuff << "\n";

        logger.getVectorHead(allfit_con, logBuff);
        Rcout << "allfit_con  : " <<  logBuff << "\n";

        logger.getVectorHead(allfit_mod, logBuff);
        Rcout << "allfit_mod  : " <<  logBuff << "\n";
    }
}

void log_status(NumericVector& z_, std::vector<double>& y, double* allfit, double* ri, double mscale, double bscale0, double bscale1, Logger& logger) {
    char logBuff[100];

    logger.log("Printing current status of fit");

    logger.getVectorHead(z_, logBuff);
    Rcout << "\n          z : " <<  logBuff << "\n";

    logger.getVectorHead(y, logBuff);
    Rcout << "          y : " <<  logBuff << "\n";

    logger.getVectorHead(allfit, logBuff);
    Rcout << "Fit - Tree  : " <<  logBuff << "\n";

    logger.getVectorHead(ri, logBuff);
    Rcout << "     resid  : " <<  logBuff << "\n\n";

    Rcout <<" MScale: " << mscale << "\n";

    Rcout <<" bscale0 : " << bscale0 << "\n";

    Rcout <<" bscale1 : " << bscale1 << "\n\n";
}

void Logger::log(std::string text){
    if (level > 0){
        for(int didx=0;didx<depth;didx++){
            Rcout << "--";
        }
        if(depth>0){
            Rcout << " ";
        }

        Rcout << text << "\n";
    }
}
