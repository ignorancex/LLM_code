
#include "Tools.h"
#include "Methods.h"

#include "Ciphertext.h"
#include "NTL/ZZX.h"
#include "Scheme.h"
#include "TestScheme.h"
#include "SecretKey.h"
#include "TimeUtils.h"
#include <cmath>

#include "SerializationUtils.h"  // error: ‘SerializationUtils’ has not been declared

#include "Tools.h"
#include <EvaluatorUtils.h>
#include <NTL/BasicThreadPool.h>
#include <NTL/RR.h>
#include <NTL/ZZ.h>

#include <iomanip>
#include <set>
#include <map>
#include <typeinfo>
#include <vector>
#include <string>
#include <algorithm>    // std::shuffle
#include <vector>        // std::vector
#include <random>       // std::default_random_engine
#include <chrono>       // std::chrono::system_clock
#include <string>

#include <unistd.h>



int CNNtraining(double **traindata, int *trainlabel, long trainfactorDim,
        long trainsampleDim, double **testdata, int *testlabel,
        long testfactorDim, long testsampleDim) {

    int labelnum = Tools::classnum(trainlabel, trainsampleDim);
    if (labelnum != Tools::classnum(testlabel, testsampleDim)) {
        cout
                << " Tools::classnum(trainlabel, trainsampleDim) != Tools::classnum(testlabel, testsampleDim) "
                << endl;
        exit(-1);
    }

    cout << "# of class label : " << endl;
    cout << labelnum << endl;

    int **trainlabel1hot, **testlabel1hot;
    Tools::oneHotEncoding(trainlabel, trainsampleDim, trainlabel1hot);
    Tools::oneHotEncoding(testlabel, testsampleDim, testlabel1hot);
    Tools::printData(trainlabel1hot, labelnum, trainsampleDim);

    cout << "X := " << endl;
    cout << "trainfactorDim = " << trainfactorDim << endl;
    Tools::printData(traindata, trainfactorDim, 2);
    double **zInvB = Tools::zInvBFromTraindata(traindata, trainfactorDim,
            trainsampleDim);
    cout << "after invert " << endl;
    Tools::printData(zInvB, trainfactorDim, 2);

    long fdimBits = (long) ceil(log2(trainfactorDim)); //ceil(x) : Rounds x upward, returning the smallest integral value that is not less than x.
    long cdimBits = (long) ceil(log2(labelnum)); //log2(x) : Returns the binary (base-2) logarithm of x.

    cout << "fdimBits = " << (fdimBits) << endl;
    long fdimNums = (1 << fdimBits);
    cout << "2^fdimBits = fdimNums = " << fdimNums << endl;

    cout << "cdimBits = " << (cdimBits) << endl;
    long cdimNums = (1 << cdimBits);
    cout << "2^cdimBits = cdimNums = " << cdimNums << endl;

    long logN = 16;
    long logQ = 990; // 991.300840336 > logQ  to ensure 128-bit security level. Security Parameter λ

    long slots = 1 << (logN - 1);
    long sBits = (long) ceil(log2(slots));

    long batch = long(slots / (1 << fdimBits)); // batch is the Number of samples encrypted in a single CipherText.
    long bBits = (long) ceil(log2(batch));

    long rnum = (long) ceil((double) trainsampleDim / batch); // To Divide the whole Data into Several CiphertTexts

    cout << endl << endl;
    cout << "trainfactorDim = " << trainfactorDim << endl << "trainsampleDim = "
            << trainsampleDim << endl;
    cout << "batch = " << batch << ", slots = " << slots << ", rnum = " << rnum
            << endl;
    cout << "logQ = " << logQ << ", logN = " << logN << ", cdimBits = "
            << cdimBits << ", fdimBits = " << fdimBits << endl;

    long wBits = 40;                             // Δ (delta)
    long pBits = 20;

    TimeUtils timeutils;
    timeutils.start("Scheme generating...");
    Ring ring;
    SecretKey secretKey(ring);
    Scheme scheme(secretKey, ring);
    //scheme.addLeftRotKeys(secretKey);
    //scheme.addRightRotKeys(secretKey);

    scheme.addLeftRotKeys(secretKey);
    std::set<int> leftset, rightset;
    std::set<int>::iterator it;
    int pos[] = { +0, +1, +2, 3, +4, 5, 6, 7, +8, 9, 10, +16, +32, +64, +128, +256, +512, +1024,
            +1536, +2048, +2560, +3072, +3584, +4096, +4608, +5120, +5632,
            +6144, +6656, +7168, +7680, +8192, +8704, +9216, +9728, +10240,
            +10752, +11264, +11776, +12288, +12800, +13312, +13824, +14336,
            +14848, +15360, +15872, +16384, +16896, +17408, +17920, +18432,
            +18944, +19456, +19968, +20480, +20992, +21504, +22016, +22528,
            +23040, +23552, +24064, +24576, +25088, +25600, +26112, +26624,
            +27136, +27648, +28160, +28672, +29184, +29696, +30208, +30720,
            +31232, +31744, +32256  };
    for (long i = 0; i < 79; ++i) {
        //leftset.insert(pos[i] );
        scheme.addLeftRotKey(secretKey, pos[i]);
    }
    scheme.addRightRotKeys(secretKey);
    int ppos[] = {10, 20, 40, 80, 160, 320 };
    for(long i=0; i < 6; ++i) {
      //rightset.insert(ppos[i] );
      scheme.addRightRotKey(secretKey, ppos[i] );
    }
    timeutils.stop("Scheme generation");

    timeutils.start("BootKey generating...");   
//    long logp = 35;
//    long logq = logp + 10; //< suppose the input ciphertext of bootstrapping has logq = logp + 10
//    long logSlots = 9 * 4; //fdimBits//< larger logn will make bootstrapping tech much slower
//    long logT = 4; //< this means that we use Taylor approximation in [-1/T,1/T] with double angle fomula
//    long logI = 4;
//    timeutils.start("Bootstrap Key generating");
//    //scheme.addBootKey(secretKey, logn, logq+logI);    
//    scheme.addBootKey(secretKey, logSlots, logq + 4);
//    timeutils.stop("Bootstrap Key generated");
    timeutils.stop("BootKey generation");

    ////////////////////////////////////////////////// Data owner ////////////////////////////////////////////////// 
    Ciphertext *encTrainData = new Ciphertext[rnum];
    Ciphertext *encTrainLab1 = new Ciphertext[rnum]; // Train Label One-Hot Encoding
    Ciphertext encW8ghtData; // = new Ciphertext[labelnum];   other method to achieve this goal might not work
    Ciphertext encVelocData;  // = new Ciphertext[labelnum];   //velocity
    Ciphertext encZInvBData;

    Ciphertext encZerosData;  // encZerosData = Enc(0)

    timeutils.start("encTrainData generating...");   
    // encTrainData
    for (long i = 0; i < rnum - 1; ++i) {
        complex<double> *pzData = new complex<double> [slots](); //();

        for (long j = 0; j < batch; ++j) {
            for (long l = 0; l < trainfactorDim; ++l) {
                pzData[fdimNums * j + l].real(traindata[batch * i + j][l]);
            }
        }
        scheme.encrypt(encTrainData[i], pzData, slots, wBits, logQ);
        SerializationUtils::writeCiphertext(encTrainData[i], "encTrainData["+ std::to_string(i) +"].txt");

        delete[] pzData;
    }
    complex<double> *pzData = new complex<double> [slots](); //();

    for (long j = 0; j < trainsampleDim - batch * (rnum - 1); ++j) {
        for (long l = 0; l < trainfactorDim; ++l) {
            pzData[fdimNums * j + l].real(traindata[batch * (rnum - 1) + j][l]);
        }
    }
    scheme.encrypt(encTrainData[rnum - 1], pzData, slots, wBits, logQ);
    timeutils.stop("encTrainData generation");
    SerializationUtils::writeCiphertext(encTrainData[rnum - 1], "encTrainData["+ std::to_string(rnum - 1) +"].txt");

    delete[] pzData;

    timeutils.start("encTrainLab1 encrypting...");  
    // encTrainLab1   :   encrypt trainlabel1hot
    for (long i = 0; i < rnum - 1; ++i) {
        complex<double> *pzData = new complex<double> [slots](); //();

        for (long j = 0; j < batch; ++j) {
            for (long l = 0; l < labelnum; ++l) {
                pzData[fdimNums * j + l].real(trainlabel1hot[batch * i + j][l]);
            }
        }
        scheme.encrypt(encTrainLab1[i], pzData, slots, wBits, logQ);
        SerializationUtils::writeCiphertext(encTrainLab1[i], "encTrainLab1["+ std::to_string(i) +"].txt");

        delete[] pzData;
    }
    complex<double> *pzDatae = new complex<double> [slots](); //();

    for (long j = 0; j < trainsampleDim - batch * (rnum - 1); ++j) {
        for (long l = 0; l < labelnum; ++l) {
            pzDatae[fdimNums * j + l].real(
                    trainlabel1hot[batch * (rnum - 1) + j][l]);
        }
    }
    scheme.encrypt(encTrainLab1[rnum - 1], pzDatae, slots, wBits, logQ);
    timeutils.stop("encTrainLab1 generation");
    SerializationUtils::writeCiphertext(encTrainLab1[rnum - 1], "encTrainLab1["+ std::to_string(rnum - 1) +"].txt");

    delete[] pzDatae;


    // encW8ghtData
    complex<double> *pzDatad = new complex<double> [slots](); //();

    for (long j = 0; j < labelnum; ++j) {   // Ensure: labelnum <= batch
        for (long l = 0; l < trainfactorDim; ++l) {
            pzDatad[fdimNums * j + l].real( 0.0128 );
        }
    }
    scheme.encrypt(encW8ghtData, pzDatad, slots, wBits, logQ);

    delete[] pzDatad;

    // encVelocData
    encVelocData.copy(encW8ghtData);
    encVelocData.n = slots;

    timeutils.start("encZInvBData encrypting...");  
    // encZInvBData   :   trainfactorDim, trainsampleDim
    complex<double> *pzDataf = new complex<double> [slots](); //();

    for (long j = 0; j < labelnum; ++j) {
        for (long l = 0; l < trainfactorDim; ++l) {
            pzDataf[fdimNums * j + l].real(zInvB[j][l]);
        }
    }
    scheme.encrypt(encZInvBData, pzDataf, slots, wBits, logQ);
    timeutils.stop("encZInvBData generation");
    SerializationUtils::writeCiphertext(encZInvBData, "encZInvBData.txt");

    delete[] pzDataf;

    complex<double> *dcw = scheme.decrypt(secretKey, encZInvBData);
    Tools::printData(dcw, fdimNums, 12);


    cout << "CurrentRSS (MB): " << ( Tools::getCurrentRSS() /1024.0/1024.0 ) << endl;
    cout << "PeakRSS    (MB): " << ( Tools::getPeakRSS() /1024.0/1024.0 )    << endl;
    ////////////////////////////////////////////////// Data owner //////////////////////////////////////////////////

    //////////////////////////////////////////////// Model provider ////////////////////////////////////////////////

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ///////////////////////////////////////////////// Cloud server /////////////////////////////////////////////////

    // encZerosData
    complex<double> *pzDatag = new complex<double> [slots](); //();
    scheme.encrypt(encZerosData, pzDatag, slots, wBits, logQ);
    delete[] pzDatag;



    double alpha0, alpha1, eta, gamma;
    double enccor, encauc, truecor, trueauc;

    alpha0 = 0.01;
    alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;

    for (long iter = 0; iter < 3; ++iter) {

        cout << endl << " !!! START " << iter + 1 << " ITERATION !!! " << endl;
        cout << "--------------------------------------------------------------------------------" << endl << endl;

        cout << "CurrentRSS (MB): " << ( Tools::getCurrentRSS() /1024.0/1024.0 ) << endl;
        cout << "PeakRSS    (MB): " << ( Tools::getPeakRSS() /1024.0/1024.0 )    << endl;

        timeutils.start( string("The ") + to_string(iter + 1) + string("-th iteration begins ...") );  

        eta = (1 - alpha0) / alpha1;
        //double gamma = 10. / (iter + 1) / trainsampleDim;
        double gamma = 1.0 / (iter + 1) / trainsampleDim;

        Ciphertext *encProbData = new Ciphertext[rnum];


        Ciphertext *rotatedct = new Ciphertext[batch];
        NTL_EXEC_RANGE(batch, first, last);
        for (long r=first; r < last; ++r) {
            scheme.leftRotateFast(rotatedct[r], encVelocData, r * fdimNums);
        }
        NTL_EXEC_RANGE_END;


        int lg = int( ceil( log(trainfactorDim) / log(2) ) ); 
        Ciphertext* tempct = new Ciphertext[ batch ];


        cout << "Enter Into Outside Loop" << endl;
        //z = np.dot(X, velocity.T)
        for (int cti = 0; cti < rnum; ++cti) {

            // sub-matrix[X] * velocity.T
            // encTrainData[cti] * encVelocData
            encProbData[cti].copy(encZerosData);
            encProbData[cti].n = slots;
 

            cout << "Enter Into Inside Loop" << endl << endl;

            cout << "calculating the " << cti << "-th ciphertext encrypting sub-matrix  *   velcity.T" << endl << endl;

            // dealing with each rotatedct[rowidx]
            //NTL_EXEC_RANGE(batch, first, last);
            long first = 0, last = batch;
            for (long rowidx=first; rowidx < last; ++rowidx) {

                Ciphertext rotatedctidx;
                rotatedctidx.copy( rotatedct[rowidx] );
                rotatedctidx.n = slots;

                if(rotatedctidx.logq > encTrainData[cti].logq)
                    scheme.modDownToAndEqual(rotatedctidx, encTrainData[cti].logq);
                if(rotatedctidx.logq < encTrainData[cti].logq)
                    scheme.modDownToAndEqual(encTrainData[cti], rotatedctidx.logq);
                assert(encTrainData[cti].logq == rotatedctidx.logq);

                scheme.multAndEqual(rotatedctidx, encTrainData[cti]);


                for (long i=0; i < lg; ++i) {
                    //Ciphertext tempct;
                    scheme.leftRotateFast(tempct[rowidx], rotatedctidx, int( pow(2, i) ) );
                    if(rotatedctidx.logp > tempct[rowidx].logp) 
                        scheme.reScaleByAndEqual(rotatedctidx,  rotatedctidx.logp-tempct[rowidx].logp);
                    if(rotatedctidx.logp < tempct[rowidx].logp) 
                        scheme.reScaleByAndEqual(tempct[rowidx], tempct[rowidx].logp-rotatedctidx.logp);
                    if(rotatedctidx.logq > tempct[rowidx].logq) 
                        scheme.modDownToAndEqual(rotatedctidx,  tempct[rowidx].logq);
                    if(rotatedctidx.logq < tempct[rowidx].logq) 
                        scheme.modDownToAndEqual(tempct[rowidx], rotatedctidx.logq);

                    scheme.addAndEqual(rotatedctidx, tempct[rowidx]);
                }


                double* filtermatrix = new double[slots]();
                for (long idx=0; idx < batch; ++idx)
                    filtermatrix[idx * fdimNums] = 1;
                Ciphertext filterct;
                scheme.encrypt(filterct, filtermatrix, slots, wBits, rotatedctidx.logq);

                if(rotatedctidx.logq > filterct.logq) 
                    scheme.modDownToAndEqual(rotatedctidx,  filterct.logq);
                if(rotatedctidx.logq < filterct.logq) 
                    scheme.modDownToAndEqual(filterct, rotatedctidx.logq);
                scheme.multAndEqual(rotatedctidx, filterct);



                for (long i=0; i < lg; ++i) {
                    //Ciphertext tempct;
                    scheme.rightRotateFast(tempct[rowidx], rotatedctidx, int( pow(2, i) ) );
                    if(rotatedctidx.logp > tempct[rowidx].logp) 
                        scheme.reScaleByAndEqual(rotatedctidx,  rotatedctidx.logp-tempct[rowidx].logp);
                    if(rotatedctidx.logp < tempct[rowidx].logp) 
                        scheme.reScaleByAndEqual(tempct[rowidx], tempct[rowidx].logp-rotatedctidx.logp);
                    if(rotatedctidx.logq > tempct[rowidx].logq) 
                        scheme.modDownToAndEqual(rotatedctidx,  tempct[rowidx].logq);
                    if(rotatedctidx.logq < tempct[rowidx].logq) 
                        scheme.modDownToAndEqual(tempct[rowidx], rotatedctidx.logq);

                    scheme.addAndEqual(rotatedctidx, tempct[rowidx]);
                }



                filtermatrix = new double[slots]();
                for (long idx=0; idx < batch; ++idx)
                    if ((idx + rowidx) % batch < labelnum)
                        filtermatrix[idx * fdimNums + (idx + rowidx) % batch] = 1;

                for (long idx = 0; idx < batch; ++idx) {
                    for (long col = labelnum; col < fdimNums; ++col)
                        filtermatrix[idx * fdimNums + col] = 0;
                }

                scheme.encrypt(filterct, filtermatrix, slots, wBits, rotatedctidx.logq);
                delete[] filtermatrix;

                if(rotatedctidx.logq > filterct.logq) 
                    scheme.modDownToAndEqual(rotatedctidx,  filterct.logq);
                if(rotatedctidx.logq < filterct.logq) 
                    scheme.modDownToAndEqual(filterct, rotatedctidx.logq);
                scheme.multAndEqual(rotatedctidx, filterct);
                filterct.kill();   


                if(encProbData[cti].logp > rotatedctidx.logp) 
                    scheme.reScaleByAndEqual(encProbData[cti],  encProbData[cti].logp-rotatedctidx.logp);
                if(encProbData[cti].logp < rotatedctidx.logp) 
                    scheme.reScaleByAndEqual(rotatedctidx, rotatedctidx.logp-encProbData[cti].logp);
                if(encProbData[cti].logq > rotatedctidx.logq) 
                    scheme.modDownToAndEqual(encProbData[cti],  rotatedctidx.logq);
                if(encProbData[cti].logq < rotatedctidx.logq) 
                    scheme.modDownToAndEqual(rotatedctidx, encProbData[cti].logq);  

            //}
            //NTL_EXEC_RANGE_END;
// Bugs Might Stem From This Block !
            //for (long rowidx=0; rowidx < batch; ++rowidx) {
                scheme.addAndEqual(encProbData[cti], rotatedctidx);
                rotatedctidx.kill();
            }
            //delete[] rotatedct;

            cout << "Xsub * Velcity" << endl;
            dcw = scheme.decrypt(secretKey, encProbData[cti]);
            Tools::printData(dcw, fdimNums, 1);

            cout << "calculating the " << cti << "-th ciphertext :: Sigmoid()" << endl << endl;
            ////# Sigmoid(x) ~ poly3 = 0.5 + 0.10679534503216294.*x + -0.00038503259805075.*x.^3; (lambda = 128)
            ////# Sigmoid(x) ~ poly3 = 0.5 + 0.10679534503216294.*x * (1 − 0.003605331.*x.^2); (lambda = 128)
            ////P = 1.0 / ( 1.0 + np.exp(-z) ) 
            //P = hlambda(z)
            //z == encProbData[cti]   >>  -hlambda(encProbData[cti])
            Ciphertext ctx, ctx2;
            ctx.copy(encProbData[cti]); ctx.n = slots;

            scheme.square(ctx2, ctx);
            scheme.reScaleByAndEqual(ctx2,  ctx.logp);  

            scheme.multByConstAndEqual(ctx,   -0.106795345, ctx.logp);

            scheme.multByConstAndEqual(ctx2,  -0.003605331, ctx2.logp);
            scheme.addConstAndEqual(ctx2, 1.0, ctx2.logp);

            if(ctx.logq > ctx2.logq) 
                scheme.modDownToAndEqual(ctx,  ctx2.logq);
            if(ctx.logq < ctx2.logq) 
                scheme.modDownToAndEqual(ctx2, ctx.logq);
            assert(ctx.logq == ctx2.logq);            
            scheme.multAndEqual(ctx, ctx2);
            scheme.reScaleByAndEqual(ctx,  ctx2.logp);

            //Problems Stem from Here !!! 
            scheme.addConstAndEqual(ctx, -0.5, ctx.logp);

            encProbData[cti].copy(ctx);
            encProbData[cti].n = slots;

            ctx2.kill();  ctx.kill();

            cout << "-P" << endl;
            dcw = scheme.decrypt(secretKey, encProbData[cti]);
            Tools::printData(dcw, fdimNums, 1);


            //gradient = (Y - P).T.dot(X)
            // (Y - P) = -P + Y   >>   encProbData[cti]
            if(encProbData[cti].logp > encTrainLab1[cti].logp)
                scheme.reScaleByAndEqual(encProbData[cti],  encProbData[cti].logp-encTrainLab1[cti].logp);
            if(encProbData[cti].logp < encTrainLab1[cti].logp) 
                scheme.reScaleByAndEqual(encTrainLab1[cti], encTrainLab1[cti].logp-encProbData[cti].logp);
            if(encProbData[cti].logq > encTrainLab1[cti].logq) 
                scheme.modDownToAndEqual(encProbData[cti],  encTrainLab1[cti].logq);
            if(encProbData[cti].logq < encTrainLab1[cti].logq) 
                scheme.modDownToAndEqual(encTrainLab1[cti], encProbData[cti].logq); 
            assert(encProbData[cti].logq == encTrainLab1[cti].logq);
            assert(encProbData[cti].logp == encTrainLab1[cti].logp);
            scheme.addAndEqual(encProbData[cti], encTrainLab1[cti]);

            cout << "encTrainLab1[cti].logp " << encTrainLab1[cti].logp << endl;
            cout << "encTrainLab1[cti].logq " << encTrainLab1[cti].logq << endl;


            cout << "encProbData[cti].logp " << encProbData[cti].logp << endl;
            cout << "encProbData[cti].logq " << encProbData[cti].logq << endl;

            cout << "Y - P" << endl << endl;
            dcw = scheme.decrypt(secretKey, encProbData[cti]);
            Tools::printData(dcw, fdimNums, 1);


            cout << "calculating the " << cti << "-th ciphertext :: Part of Gradient" << endl << endl;

            //gradient = (Y - P).T.dot(X)
            double* filtermatrix = new double[slots]();
            for (long idx = 0; idx < batch; ++idx) {
                for (long col = 0; col < labelnum; ++col)
                    filtermatrix[idx * fdimNums + col] = 1;
            }
            Ciphertext filterct;
            scheme.encrypt(filterct, filtermatrix, slots, wBits, encProbData[cti].logq);
            delete[] filtermatrix;

            if(encProbData[cti].logq > filterct.logq) 
                scheme.modDownToAndEqual(encProbData[cti],  filterct.logq);
            if(encProbData[cti].logq < filterct.logq) 
                scheme.modDownToAndEqual(filterct, encProbData[cti].logq);
            scheme.multAndEqual(encProbData[cti], filterct);

            if(encProbData[cti].logp > filterct.logp) 
                scheme.reScaleByAndEqual(encProbData[cti],  encProbData[cti].logp-filterct.logp);
            if(encProbData[cti].logp < filterct.logp) 
                scheme.reScaleByAndEqual(filterct, filterct.logp-encProbData[cti].logp);
            if(encProbData[cti].logq > filterct.logq) 
                scheme.modDownToAndEqual(encProbData[cti],  filterct.logq);
            if(encProbData[cti].logq < filterct.logq) 
                scheme.modDownToAndEqual(filterct, encProbData[cti].logq); 

            filterct.kill();

            cout << "Y - P" << endl << endl;
            dcw = scheme.decrypt(secretKey, encProbData[cti]);
            Tools::printData(dcw, fdimNums, 1);
            cout << "encProbData[cti].logp " << encProbData[cti].logp << endl;
            cout << "encProbData[cti].logq " << encProbData[cti].logq << endl;
            

            // // * labelnum +[1]) is to leave [1] column for an incomplete column shifting
            // if(fdimNums - labelnum > (ceil(trainfactorDim / labelnum) +1) * labelnum +1) {
            //     int lglabel = int( ceil( log(trainfactorDim) / log(2) ) ); 
            //     int lglabel = int( floor( log(512 / 10.0) / log(2) ) ); 
            //     cout << endl << "lglabel: " << lglabel <<  "\t" << 10 * int( pow(2, lglabel) ) << endl;

            //     int* rolist = new int[1+lglabel]();
            //     for (int i = 0; i < 1+lglabel; ++i) {
            //         cout << 10 * int( pow(2, i) ) << endl;
            //         rolist[i] = 10 * int( pow(2, i) );
            //     }

            //     int featureNum = 501;
            //     cout << "A" << (ceil(featureNum / 10.0) +1) * 10 << endl;
            //     auto newcolnum =   (ceil(featureNum / 10.0) +1) * 10;
            //     for(int i = lglabel; i>=0; --i) {
            //         while(newcolnum>=rolist[i]) {
            //             cout << rolist[i] << "\t";
            //             newcolnum -= rolist[i];
            //         }

            //     }
            // }

            //trainfactorDim = 441; labelnum = 10; 160 = 10 * 2^4; 320 = 10 * 2^5; 
            assert(160 + 320 > trainfactorDim + 2 * labelnum);

            Ciphertext tempct, tempct160;
            for (long i=0; i < 5; ++i) {

                if(i == 4) {
                    tempct160.copy(encProbData[cti]);
                    tempct160.n = slots;
                }
 
                scheme.rightRotateFast(tempct, encProbData[cti], labelnum * int( pow(2, i) ) );
                if(encProbData[cti].logp > tempct.logp) 
                    scheme.reScaleByAndEqual(encProbData[cti],  encProbData[cti].logp-tempct.logp);
                if(encProbData[cti].logp < tempct.logp) 
                    scheme.reScaleByAndEqual(tempct, tempct.logp-encProbData[cti].logp);
                if(encProbData[cti].logq > tempct.logq) 
                    scheme.modDownToAndEqual(encProbData[cti],  tempct.logq);
                if(encProbData[cti].logq < tempct.logq) 
                    scheme.modDownToAndEqual(tempct, encProbData[cti].logq);

                scheme.addAndEqual(encProbData[cti], tempct);
            }

            scheme.rightRotateFast(tempct, tempct160, 320 );
            if(encProbData[cti].logp > tempct.logp) 
                scheme.reScaleByAndEqual(encProbData[cti],  encProbData[cti].logp-tempct.logp);
            if(encProbData[cti].logp < tempct.logp) 
                scheme.reScaleByAndEqual(tempct, tempct.logp-encProbData[cti].logp);
            if(encProbData[cti].logq > tempct.logq) 
                scheme.modDownToAndEqual(encProbData[cti],  tempct.logq);
            if(encProbData[cti].logq < tempct.logq) 
                scheme.modDownToAndEqual(tempct, encProbData[cti].logq);
            scheme.addAndEqual(encProbData[cti], tempct);

            tempct160.kill(); tempct.kill();

            cout << "Padding[Y - P]" << endl << endl;
            dcw = scheme.decrypt(secretKey, encProbData[cti]);
            Tools::printData(dcw, fdimNums, 1);
            cout << "encProbData[cti].logp " << encProbData[cti].logp << endl;
            cout << "encProbData[cti].logq " << encProbData[cti].logq << endl; 


        }

        cout << "gradient = (Y - P).T.dot(X)" << endl;
        Ciphertext encGradtData;
        encGradtData.copy(encZerosData);
        encGradtData.n = slots;

        Ciphertext* encTemptData = new Ciphertext[rnum];
        for (int rotidx = 0; rotidx < labelnum; ++rotidx) {
            Ciphertext encAccumData;
            encAccumData.copy(encZerosData);
            encAccumData.n = slots;
            for (int cti = 0; cti < rnum; ++cti) {
                // encProbData rotation
                // !!addLeftRotationKey
                scheme.leftRotateFast(encTemptData[cti], encProbData[cti], rotidx );


                if(encTemptData[cti].logq > encTrainData[cti].logq) 
                    scheme.modDownToAndEqual(encTemptData[cti],  encTrainData[cti].logq);
                if(encTemptData[cti].logq < encTrainData[cti].logq) 
                    scheme.modDownToAndEqual(encTrainData[cti], encTemptData[cti].logq);
                scheme.multAndEqual(encTemptData[cti], encTrainData[cti]);
                scheme.reScaleByAndEqual(encTemptData[cti],  encTrainData[cti].logp);


                if(encTemptData[cti].logp > encAccumData.logp) 
                    scheme.reScaleByAndEqual(encTemptData[cti],  encTemptData[cti].logp-encAccumData.logp);
                if(encTemptData[cti].logp < encAccumData.logp) 
                    scheme.reScaleByAndEqual(encAccumData, encAccumData.logp-encTemptData[cti].logp);
                if(encTemptData[cti].logq > encAccumData.logq) 
                    scheme.modDownToAndEqual(encTemptData[cti],  encAccumData.logq);
                if(encTemptData[cti].logq < encAccumData.logq) 
                    scheme.modDownToAndEqual(encAccumData, encTemptData[cti].logq);
                scheme.addAndEqual(encAccumData, encTemptData[cti]);
            }


            Ciphertext encResutData;
            encResutData.copy(encZerosData);
            encResutData.n = slots;
            for (int cti = 0; cti < batch; ++cti) {

                Ciphertext encTemppData;
                scheme.leftRotateFast(encTemppData, encAccumData, fdimNums * cti );

                if(encResutData.logp > encTemppData.logp) 
                    scheme.reScaleByAndEqual(encResutData,  encResutData.logp-encTemppData.logp);
                if(encResutData.logp < encTemppData.logp) 
                    scheme.reScaleByAndEqual(encTemppData, encTemppData.logp-encResutData.logp);
                if(encResutData.logq > encTemppData.logq) 
                    scheme.modDownToAndEqual(encResutData,  encTemppData.logq);
                if(encResutData.logq < encTemppData.logq) 
                    scheme.modDownToAndEqual(encTemppData, encResutData.logq);

                scheme.addAndEqual(encResutData, encTemppData);

                encTemppData.kill();

            }

    

            double* filtermatrix = new double[slots]();
            for (long idx = 0; idx < labelnum; ++idx) {
                for (long col = 0; col < fdimNums; ++col)
                    if( (col+labelnum-idx   +     rotidx)%labelnum == 0 )
                    filtermatrix[idx * fdimNums + col] = 1;
            }

            for (long idx = 0; idx < labelnum; ++idx) {
                for (long col = trainfactorDim; col < fdimNums; ++col)
                    filtermatrix[idx * fdimNums + col] = 0;
            }

            Ciphertext filterct;
            scheme.encrypt(filterct, filtermatrix, slots, wBits, encResutData.logq);
            delete[] filtermatrix;


            if(encResutData.logq > filterct.logq) 
                scheme.modDownToAndEqual(encResutData,  filterct.logq);
            if(encResutData.logq < filterct.logq) 
                scheme.modDownToAndEqual(filterct, encResutData.logq);
            scheme.multAndEqual(encResutData, filterct);
            scheme.reScaleByAndEqual(encResutData,  filterct.logp);

            filterct.kill();


            cout << endl << "part of gradient" << endl;
            dcw = scheme.decrypt(secretKey, encResutData);
            Tools::printData(dcw, fdimNums, 2);
            cout << "encResutData.logp " << encResutData.logp << endl;
            cout << "encResutData.logq " << encResutData.logq << endl; 




            if(encResutData.logp > encGradtData.logp) 
                scheme.reScaleByAndEqual(encResutData,  encResutData.logp-encGradtData.logp);
            if(encResutData.logp < encGradtData.logp) 
                scheme.reScaleByAndEqual(encGradtData, encGradtData.logp-encResutData.logp);
            if(encResutData.logq > encGradtData.logq) 
                scheme.modDownToAndEqual(encResutData,  encGradtData.logq);
            if(encResutData.logq < encGradtData.logq) 
                scheme.modDownToAndEqual(encGradtData,  encResutData.logq);

            scheme.addAndEqual(encGradtData, encResutData);


            encResutData.kill();
            encAccumData.kill();
        }

        for (long i = 0; i < rnum; ++i)
            encTemptData[i].kill();
        delete[] encTemptData;
        

        cout << endl << "Gradient:" << endl;
        cout << "encGradtData.logp " << encGradtData.logp << endl;
        cout << "encGradtData.logq " << encGradtData.logq << endl; 
        dcw = scheme.decrypt(secretKey, encGradtData);
        Tools::printData(dcw, fdimNums, 2);


        //MG = np.multiply(invBrow, gradient) 
        if(encGradtData.logq > encZInvBData.logq) 
            scheme.modDownToAndEqual(encGradtData,  encZInvBData.logq);
        if(encGradtData.logq < encZInvBData.logq) 
            scheme.modDownToAndEqual(encZInvBData, encGradtData.logq);
        scheme.multAndEqual(encGradtData, encZInvBData);
        scheme.reScaleByAndEqual(encGradtData, encZInvBData.logp);

        cout << endl << "Quadratic Gradient:" << endl;
        cout << "encGradtData.logp " << encGradtData.logp << endl;
        cout << "encGradtData.logq " << encGradtData.logq << endl; 
        dcw = scheme.decrypt(secretKey, encGradtData);
        Tools::printData(dcw, fdimNums, 2);





        Ciphertext encQGradData;
        scheme.multByConst(encQGradData, encGradtData, 1. + gamma, pBits);

        if(encW8ghtData.logp > encQGradData.logp) 
            scheme.reScaleByAndEqual(encW8ghtData,  encW8ghtData.logp-encQGradData.logp);
        if(encW8ghtData.logp < encQGradData.logp) 
            scheme.reScaleByAndEqual(encQGradData,  encQGradData.logp-encW8ghtData.logp);
        if(encW8ghtData.logq > encQGradData.logq) 
            scheme.modDownToAndEqual(encW8ghtData,  encQGradData.logq);
        if(encW8ghtData.logq < encQGradData.logq) 
            scheme.modDownToAndEqual(encQGradData,  encW8ghtData.logq);

        Ciphertext ctmpw;
        scheme.add(ctmpw, encW8ghtData, encQGradData);                // encGrad[i] has already self-multiplied with gamma
                                                                        // ctmpw = encVData[i] - encGrad[i]
        scheme.multByConst(encW8ghtData, ctmpw, 1. - eta, pBits);        // encVData[i] = ( 1. - eta ) * ctmpw
        

        scheme.multByConstAndEqual(encVelocData, eta, pBits);            // encWData[i] = eta * encWData[i]
        

        if (encW8ghtData.logq > encVelocData.logq) 
            scheme.modDownToAndEqual(encW8ghtData, encVelocData.logq);
        if (encW8ghtData.logq < encVelocData.logq) 
            scheme.modDownToAndEqual(encVelocData, encW8ghtData.logq);
        assert(encW8ghtData.logp == encVelocData.logp);

        scheme.addAndEqual(encW8ghtData, encVelocData);                   // encVData[i] = encVData[i] + encWData[i]
                                                         // encVData[i] = ( 1. - eta ) * ctmpw + eta * encWData[i]

        scheme.reScaleByAndEqual(encW8ghtData, pBits);
        encVelocData.copy(ctmpw);

        ctmpw.kill();
        encQGradData.kill();

        encGradtData.kill();


        alpha0 = alpha1;
        alpha1 = (1. + sqrt(1. + 4.0 * alpha0 * alpha0)) / 2.0;
        cout << " !!! STOP " << iter + 1 << " ITERATION !!! " << endl << endl << endl;



        for (long i = 0; i < batch; ++i)
            tempct[i].kill();
        delete[] tempct;

        for (long i = 0; i < batch; ++i)
            rotatedct[i].kill();
        delete[] rotatedct;


        for (long i = 0; i < rnum; ++i)
            encProbData[i].kill();
        delete[] encProbData;


        timeutils.stop( string("The ") + to_string(iter + 1) + string("-th iteration ends ...") ); 


        dcw = scheme.decrypt(secretKey, encW8ghtData);
        cout << endl << "Weight: " << endl;
        Tools::printData(dcw, fdimNums, 22);
        cout << "encW8ghtData.logp " << encW8ghtData.logp << endl;
        cout << "encW8ghtData.logq " << encW8ghtData.logq << endl;

        /////////////////////////////////////////////////////////////////////////////
        ////////////////////////  TEST MODEL on TESTING DATA  ///////////////////////
        /////////////////////////////////////////////////////////////////////////////
        double *wData = new double[slots];
        for ( long idx = 0; idx < slots; ++idx) 
                wData[idx] = dcw[idx].real();
        cout << "TEST MODEL on TESTDATASET ACC: " <<
        Tools::calculateACC(testdata, testlabel, testfactorDim, testsampleDim, wData, fdimNums, labelnum) << endl;
        cout << "TEST MODEL on TESTDATASET SLE: " <<
        Tools::calculateSLE(testdata, testlabel1hot, testfactorDim, testsampleDim, wData, fdimNums, labelnum) << endl;
        delete[] wData;


        /////////////////////////////////////////////////////////////////////////////
        //        BOOTSTRAPPING                                                    //
        //             Step 1. Obtain various encW8ghtData[i] from encW8ghtData    //
        //             Step 2. Bootstrap encW8ghtData[i]                           //
        //             Step 3. Combine various encW8ghtData[i] into encW8ghtData   //
        /////////////////////////////////////////////////////////////////////////////
        if ( false ) { //|| encW8ghtData.logq <= 300  && iter < numIter-1 || encW8ghtData.logq < wBits && iter == numIter-1) {


            cout << " +-------------------- encW8ghtData --------------------+ " << endl;
            cout << " encW8ghtData.logp = " << encW8ghtData.logp << ", encW8ghtData.logq = " << encW8ghtData.logq << endl;
            complex<double>* dcii = scheme.decrypt(secretKey, encW8ghtData);
            for (long j = 0; j < 12; ++j) {
                for (long l = 0; l < 12; ++l) {
                    cout << setiosflags(ios::fixed) << setprecision(10) << dcii[fdimNums * j + l].real() << "\t";
                }
                cout << endl << setiosflags(ios::fixed) << setprecision(7);
            }
            cout << " +-------------------- encW8ghtData --------------------+ " << endl ;

            timeutils.start("Use Bootrap To Recrypt Ciphertext");
            cout << endl << " ----------------------- Use Bootrap To Recrypt Ciphertext ----------------------- " << endl;
            //MyBootStrap(scheme, secretKey, encW8ghtData, labelnum, slots, batch, fdimNums, logp, logq, logQ, logT, logI);
            cout << endl << " ----------------------- Use Bootrap To Recrypt Ciphertext ----------------------- " << endl;
            timeutils.stop("Use Bootrap To Recrypt Ciphertext");


            cout << " x-------------------- encW8ghtData --------------------x " << endl;
            cout << " encW8ghtData.logp = " << encW8ghtData.logp << ", encW8ghtData.logq = " << encW8ghtData.logq << endl;
            dcii = scheme.decrypt(secretKey, encW8ghtData);
            for (long j = 0; j < 12; ++j) {
                for (long l = 0; l < 12; ++l) {
                    cout << setiosflags(ios::fixed) << setprecision(10) << dcii[fdimNums * j + l].real() << "\t";
                }
                cout << endl << setiosflags(ios::fixed) << setprecision(7);
            }
            cout << " x-------------------- encW8ghtData --------------------x " << endl ;

            delete[] dcii;

        }
        /////////////////////////////////////////////////////////////////////////////
        //        BOOTSTRAPPING                                                    //
        //             Over and Out                                                //
        /////////////////////////////////////////////////////////////////////////////




    }

    cout << "CurrentRSS (MB): " << ( Tools::getCurrentRSS() /1024.0/1024.0 ) << endl;
    cout << "PeakRSS    (MB): " << ( Tools::getPeakRSS() /1024.0/1024.0 )    << endl;

    return 0;
}



void MyBootStrap(Scheme &scheme, SecretKey &secretKey,
        Ciphertext &encWData, long labelnum, long slots, long batch,
        long fdimNums, long logp, long logq, long logQ, long logT, long logI) {

    /*****************************************************************************************\
     *                      Get various encVData[i] from encVData                            *
     *             a new fresh ciphertext ecnVData[i] with bigger modulo is ready            *
    \*****************************************************************************************/
    Ciphertext *encRowsData = new Ciphertext[labelnum];
    long pBits = 20;

    cout << "Encrypting Special Ciphertext; " << endl;
    NTL_EXEC_RANGE(labelnum, first, last);
    for (long r = 0; r < labelnum; ++r) {
        complex<double> *pcData = new complex<double> [slots]();
        for (long l = 0; l < fdimNums; ++l) 
            pcData[fdimNums * r + l] = 1;
        
        scheme.encrypt(encRowsData[r], pcData, slots, pBits, encWData.logq);
        delete[] pcData;
    }
    NTL_EXEC_RANGE_END;
    
    cout << "Special Ciphertext @ encVData; " << endl;
    NTL_EXEC_RANGE(labelnum, first, last);
    for (long r = first; r < last; ++r) {
        scheme.multAndEqual(encRowsData[r], encWData); 
        //scheme.reScaleByAndEqual(encRowsData[r], );   // delay&save this operations to bootstrapping
    }
    NTL_EXEC_RANGE_END;
    

    Ciphertext zeros;
    complex<double> *pdData = new complex<double> [slots]();
    scheme.encrypt(zeros, pdData, slots, encRowsData[0].logp, encRowsData[0].logq);
    delete[] pdData;
    NTL_EXEC_RANGE(labelnum, first, last);
    for (long r = first; r < last; ++r) {
        Ciphertext rot,res;
        res.copy(zeros);
        res.n = slots;
        for (long l = 0; l < batch; ++l) {
            scheme.leftRotateFast(rot, encRowsData[r], l * fdimNums);
            scheme.addAndEqual(res, rot);
        }
        rot.kill();
        encRowsData[r].copy(res);
        encRowsData[r].n = slots;
        res.kill();

        

        if (encRowsData[r].logp > logp) 
            scheme.reScaleByAndEqual(encRowsData[r], encRowsData[r].logp - logp);
        if (encRowsData[r].logq > logq)  
            scheme.modDownToAndEqual(encRowsData[r], logq);
    }
    NTL_EXEC_RANGE_END


 

    /*****************************************************************************************\
     *                                BOOTSTRAPPING BEGINNING                                *
     *               Now, encRowsData[r] contains all the information you want !             *
    \*****************************************************************************************/

    cout << " ------------------------ BOOTSTRAPPING BEGINNING ------------------------ " << endl;
    cout << "encRowsData[0] logp before: " << encRowsData[0].logp << endl;
    cout << "encRowsData[0] logq before: " << encRowsData[0].logq << endl;
    //NTL_EXEC_RANGE(labelnum, first, last);
    long first = 0, last = labelnum;
    for (long r = first; r < last; ++r) {
        assert(encRowsData[r].logp == logp);
        assert(encRowsData[r].logq == logq);
        cout << "bootstrapping the " << r << "-th encRowsData" << endl;
        encRowsData[r].n = fdimNums;
        scheme.bootstrapAndEqual(encRowsData[r], logq, logQ, logT, logI);
        encRowsData[r].n = slots;
    }
    //NTL_EXEC_RANGE_END;    
    cout << "encRowsData[0] logp before: " << encRowsData[0].logp << endl;
    cout << "encRowsData[0] logq before: " << encRowsData[0].logq << endl;

    cout << " ------------------------ BOOTSTRAPPING FINISHED ------------------------ " << endl;


    /*****************************************************************************************\
     *                     Combine various encVData[i] into encVData[0]                      *
     *                so just bootstrapping one ciphertext ecnVData[0] is OK                 *
    \*****************************************************************************************/


    Ciphertext* encTemptData = new Ciphertext[labelnum];
    cout << "Encrypting Special Ciphertext; " << endl;
    NTL_EXEC_RANGE(labelnum, first, last);
    for (long r = 0; r < labelnum; ++r) {
        complex<double> *pcData = new complex<double> [slots]();
        for (long l = 0; l < fdimNums; ++l) 
            pcData[fdimNums * r + l] = 1;
        
        scheme.encrypt(encTemptData[r], pcData, slots, pBits, encRowsData[r].logq);
        delete[] pcData;
    }
    NTL_EXEC_RANGE_END;
    

    cout << "Special Ciphertext @ encVData; " << endl;
    NTL_EXEC_RANGE(labelnum, first, last);
    for (long r = first; r < last; ++r) {
        scheme.multAndEqual(encTemptData[r], encRowsData[r]); 
        //scheme.reScaleByAndEqual(encTemptData[r], encRowsData[r].logp);   // delay&save this operations to bootstrapping
    }
    NTL_EXEC_RANGE_END;
    

    for (long r = 1; r < labelnum; ++r) {
        scheme.addAndEqual(encTemptData[0], encTemptData[r]);
    }
    scheme.reScaleByAndEqual(encTemptData[0], encRowsData[0].logp);

    // The Ciphertext after bootstrapping would lose some logp due to the above operation !
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    encWData.copy(encTemptData[0]);
    encWData.n = slots;



    zeros.kill();
}
