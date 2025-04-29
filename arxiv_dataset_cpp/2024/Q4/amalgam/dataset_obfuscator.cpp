/*
This is the dataset obfuscator written in C++ using the libtorch library.

Author: Sifat Ut Taki
*/

#include <torch/torch.h>
#include <torch/script.h>
#include <chrono>
#include <cstdlib>
#include <pthread.h>
#include <string> 

using namespace std::chrono;

using namespace std;

#define NUM_THREADS 20

torch::Tensor trainset;
torch::Tensor testset;
torch::Tensor aug_indices;

torch::Tensor aug_trainset;
torch::Tensor aug_testset;

int num_channel;
int img_dim;

int aug_dim;
int aug_tensor_dim;

int aug_tensor_size;
int aug_size;

int test_setSize;
int train_setSize;


void *AugData(void *threadId) {
    long tid = (long) threadId;

    int train_startIdx = tid * train_setSize;
    int test_startIdx = tid * test_setSize;
    int train_endIdx = train_startIdx + train_setSize;
    int test_endIdx = test_startIdx + test_setSize;

    // Trainset augmentation
    for (int data_i = train_startIdx; data_i < train_endIdx; data_i++) {

        torch::Tensor aug_tensors = torch::zeros({num_channel, aug_tensor_size});
        int aug_image_shape[2] = {aug_tensor_dim, aug_tensor_dim};
        torch::Tensor aug_img = torch::zeros({num_channel, aug_tensor_dim, aug_tensor_dim});

        torch::Tensor img = trainset[data_i];

        for (int c = 0; c < num_channel; c++) {
            torch::Tensor img_flat = torch::flatten(img[c]);
            int aug_index = 0;
            int j = 0;

            int aug_index_val = aug_indices[c][aug_index].item<int>();

            for (int i = 0; i < aug_tensor_size; i++) {
                if (i == aug_index_val) {
                    aug_tensors[c][i] = 0;      // replace with a preferred randomizer
                    aug_index++;
                    aug_index_val = aug_indices[c][aug_index].item<int>();
                } else {
                    aug_tensors[c][i] = img_flat[j];
                    j++;
                }
            }
            aug_img[c] = aug_tensors[c].view({aug_tensor_dim, aug_tensor_dim});
        }

        aug_trainset[data_i] = aug_img;
    }

    // Testset augmentation
    for (int data_i = test_startIdx; data_i < test_endIdx; data_i++) {

        // cout << "train index: " << data_i << endl;

        torch::Tensor aug_tensors = torch::zeros({num_channel, aug_tensor_size});
        int aug_image_shape[2] = {aug_tensor_dim, aug_tensor_dim};
        torch::Tensor aug_img = torch::zeros({num_channel, aug_tensor_dim, aug_tensor_dim});

        torch::Tensor img = testset[data_i];

        for (int c = 0; c < num_channel; c++) {
            torch::Tensor img_flat = torch::flatten(img[c]);
            int aug_index = 0;
            int j = 0;

            int aug_index_val = aug_indices[c][aug_index].item<int>();

            for (int i = 0; i < aug_tensor_size; i++) {
                if (i == aug_index_val) {
                    aug_tensors[c][i] = 0;      // replace with a preferred randomizer
                    aug_index++;
                    aug_index_val = aug_indices[c][aug_index].item<int>();
                } else {
                    aug_tensors[c][i] = img_flat[j];
                    j++;
                }
            }
            aug_img[c] = aug_tensors[c].view({aug_tensor_dim, aug_tensor_dim});
        }

        aug_testset[data_i] = aug_img;
    }
}

int main() {

    torch::jit::script::Module container = torch::jit::load("path/to/dataset");
    trainset = container.attr("train_set").toTensor();
    testset = container.attr("test_set").toTensor();
    torch::Tensor aug_indices_all = container.attr("aug_indices").toTensor();

    int train_samples = trainset.sizes()[0];
    int test_samples = testset.sizes()[0];

    num_channel = testset.sizes()[1];
    img_dim = testset.sizes()[2];

    test_setSize = (int) test_samples / 20;
    train_setSize = (int) train_samples / 20;

    float aug_levels[4] = {0.25, 0.5, 0.75, 1.0};

    for(int level = 0; level < 4; level++) {

        float aug_level = aug_levels[level];

        aug_dim = (int) img_dim * aug_level;
        aug_tensor_dim = img_dim + aug_dim;
        aug_tensor_size = aug_tensor_dim * aug_tensor_dim;
        aug_size = aug_tensor_size - (img_dim * img_dim);
        aug_indices = aug_indices_all[level];

        aug_trainset = torch::zeros({train_samples, num_channel, aug_tensor_dim, aug_tensor_dim});
        aug_testset = torch::zeros({test_samples, num_channel, aug_tensor_dim, aug_tensor_dim});

        pthread_t threads[NUM_THREADS];
        // struct ThreadArgs td[NUM_THREADS];
        pthread_attr_t attr;
        void *status;
        int rc;
        int i;

        // Initialize and set thread joinable
        pthread_attr_init(&attr);
        pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

        auto start_time = high_resolution_clock::now();

        for(i = 0; i < NUM_THREADS; i++) {
            rc = pthread_create(&threads[i], NULL, AugData, (void *)i);
            
            if (rc) {
                cout << "Error:unable to create thread," << rc << endl;
                exit(-1);
            }
        }

        // free attribute and wait for the other threads
        pthread_attr_destroy(&attr);
        for( i = 0; i < NUM_THREADS; i++ ) {
            rc = pthread_join(threads[i], &status);
            if (rc) {
                cout << "Error:unable to join," << rc << endl;
                exit(-1);
            }
        }

        auto stop_time = high_resolution_clock::now();
        auto duration = duration_cast<seconds>(stop_time - start_time);

        cout << "Aug time: " << duration.count() << endl;

        string train_name = "amalgam_trainset" + to_string(aug_level) + ".pt";
        string test_name = "amalgam_testset" + to_string(aug_level) + ".pt";
        
        torch::save(aug_trainset, train_name);
        torch::save(aug_testset, test_name);
    }
    pthread_exit(NULL);
}
