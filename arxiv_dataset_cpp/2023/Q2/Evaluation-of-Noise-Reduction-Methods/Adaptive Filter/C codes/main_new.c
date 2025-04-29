#include <stdio.h>
#include <math.h>

// read input signals
[in_noisy_1,Fs] = audioread('Unp_Front_SB_90_180.wav');
[in_noisy_2,Fs] = audioread('Unp_Rear_SB_90_180.wav');
[in_clean,Fs2] = audioread('Clean3.wav');

int N = sizeof(in_noisy_1)/sizeof(float);
int n_frame_coherance = 640;
int n_frame_adaptive = 320;

//Initialise parameters for coherence function
float coherence_final[N] = {0}; // not sure

//Initialise parameters for VAD
float value = N/n_frame_adaptive;
int numFrames = ceil(value);
float outputSignal[N] = {0};  // not sure

//Initialize VAD flag arrays
float iVad[numFrames] = {0};
float fVad[numFrames] = {0}

// Initialize Energy Variables
double energyMin = 0;
double energyMax = 0;
double deltaEmin = 1;
double deltaEmax = 1;
double lambda = 1;
double threshold = 0;
double initialEnergyMin = 0;

// Initialize RMSE, periodicity, and energy ratio arrays
float RMSEArray[numFrames] = {0};
float periodicityArray[numFrames] = {0};
float ratioArray[numFrames] = {0};
float thresholdArray[numFrames] = {0};

// Initialise parameters for NLMS adaptive filter 
int M = 70;  //70 best
float xx[M] = {0};
float w1[M] = {0};

float y[numFrames] = {0}; //y = []; not sure
float e[numFrames] = {0}; //e = []; not sure



//initialising parameters for frequency shaper
double H_th_val = 45/20;                // H_th = db2mag(45);   db2mag == y = 10.^(ydb/20);  db2mag(45)= 45/20 
float H_th = pow(10, H_th_val);         //Threshold of hearing

double P_th_val = 70/20;   
float P_th = pow(10, P_th_val);         // Threshold of pain

float Lower_limit = 2000;               // Lower limit of the hearing loss range
float Upper_limit = 5000;               // Upper limit of the hearing loss range
float a=1000;                           // Smoothing area
float fre_shap_final[numFrames] = {0};  // fre_shap_final = []; not sure


// Loop starts here
int first = 1;
int last = n_frame_coherance;
int frameCounter = 1;

int i = 0;
int j = 0;

while (numFrames > frameCounter)
{
    float noisy1[] = in_noisy_1[first:last];
    float noisy2[] = in_noisy_2[first:last];

    // %Coherance function
    coherance_out=Coherence(noisy1,noisy2,Fs); // call coherence function
    if (last ==N){
        coherance_out=coherance_out[1+(n_frame_adaptive/4):end];
    }
    else
    {
        coherance_out=coherance_out[1+(n_frame_adaptive/4):5*n_frame_adaptive/4];
    }

    //coherence_final = [coherence_final coherance_out'];  not sure
    float len = sizeof(coherance_out)/sizeof(float);
    for (i; i<len; i++)
    {
        coherence_final[i] = coherance_out[i];
    }

    // %Voice activity detector
    [out,iVad,fVad,RMSEArray,periodicityArray,ratioArray,thresholdArray,energyMin,energyMax,deltaEmin,deltaEmax,lambda,threshold,initialEnergyMin]=VAD(coherance_out,Fs,frameCounter,n_frame_adaptive,iVad,fVad,RMSEArray,periodicityArray,ratioArray,thresholdArray,energyMin,energyMax,deltaEmin,deltaEmax,lambda,threshold,initialEnergyMin);

    //outputSignal=[outputSignal out']; not sure
    float len1 = sizeof(out)/sizeof(float);
    for (i; i<le1n; i++)
    {
        outputSignal[i] = out[i];
    }

    //NLMS Adaptive filter
    [e1, y1, w1, xx] = NLMS(coherance_out, out, w1, xx, M); // call NLMS

    y=[y y1']; // two arry concat
    e=[e e1'];

    //Frequency shaping
    fre_shaper = Freqshaper(y1,Fs,H_th,P_th,Lower_limit,Upper_limit,a); // call Freqshaper
    fre_shap_final = [fre_shap_final fre_shaper'']; // two arry concat

    frameCounter = frameCounter + 1;
    first = first+n_frame_adaptive;

    if (last + n_frame_adaptive > N)
    {
        last = N;
    }
    else
    {
        last = last + n_frame_adaptive;
    }   

}

// final signals

float vad[] = outputSignal;
float nlms[] = y; // nlms = y'
float e[]=e; // e=e'
float coherence_final[] = coherence_final; //coherence_final=coherence_final';
float fre_shap_final[] = fre_shap_final; //fre_shap_final = fre_shap_final';

// normalization before getting SNRs
nlms = ProcessAudio(in_noisy_1,nlms);
noisy1 = ProcessAudio(in_noisy_1,in_noisy_1);
noisy2 = ProcessAudio(in_noisy_1,in_noisy_2);
coherence_final=ProcessAudio(in_noisy_1,coherence_final);
clean = ProcessAudio(in_noisy_1,in_clean);
vad = ProcessAudio(in_noisy_1,vad);
fre_shap_final = ProcessAudio(in_noisy_1,fre_shap_final);

//calculate SNR
void norm1(float[] signal1,float[] signal2)
{
    int len = sizeof(signal1)/sizeof(float);
    float sum = 0;
    
    for (int i=0; i<len; i++)
    {
        float val = signal1[i]-signal2[i]
        sum += sqrt(val*val);
    }
    return sum;
}

void norm2(float[] signal1)
{
    int len = sizeof(signal1)/sizeof(float);
    float sum = 0;
    
    for (int i=0; i<len; i++)
    {
        sum += sqrt(signal1[i]*signal1[i]);
    }
    return sum;
}

float SNR_noisy1 = 20*log10(norm1(in_clean)/norm2(in_noisy_1-in_clean));
float SNR_noisy2 = 20*log10(norm1(in_clean)/norm2(in_noisy_2-in_clean));
float SNR_coherence = 20*log10(norm1(in_clean)/norm2(coherence_final-in_clean(1:length(coherence_final))));
float SNR_nlms = 20*log10(norm1(in_clean)/norm2(nlms-in_clean(1:length(nlms))));




