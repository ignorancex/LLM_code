#include <global/net_headers.hpp>
#include <utils/aux.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

bool read_imagenet_classes(const std::string &fileName, std::vector<std::string> &classes)
{

    // Open the File
    std::ifstream in(fileName.c_str());

    // Check if object is valid
    if (!in)
    {
        std::cout << "x> Not able to open Imagenet classes file at: " << fileName << std::endl;
        return false;
    }

    std::string str;
    // Read the next line from File untill it reaches the end.
    while (std::getline(in, str))
    {
        // Line contains string of length > 0 then save it in vector
        if (str.size() > 0)
            classes.push_back(str);
    }

    if (classes.size() != __LINEAR_ROW_SIZE__)
    {
        std::cout << "x> There shoud be " << __LINEAR_ROW_SIZE__ << " classes defined in file at: " << fileName << ", but it was: " << classes.size() << std::endl;
        return false;
    }

    //Close The File
    in.close();
    return true;
}

void concat(dType_8u *image_0, dType_8u *output_0, int const &outChan, int const &out_size)
{
    // int thread_work_end[__GLOBAL_WORKER_NUMBER__];
    // for (int global_id = 0; global_id < __GLOBAL_WORKER_NUMBER__; global_id++)
    // {
    //     thread_work_end[global_id] = (((global_id + 1) * outChan / __GLOBAL_WORKER_NUMBER__) * out_size * out_size);
    // }

    // int per_worker = 0;
    // dType_8u *ptr = output_0;
    for (int idx = 0; idx < outChan * out_size * out_size; idx++)
    {
        image_0[idx] = output_0[idx];
    }
}

void print_qnt_results(dType_8u *image, int out_ch, int out_size)
{
    int ch = 0;
    for (int i = 0; i < out_ch * out_size * out_size; i++)
    {
        if ((i % (out_size * out_size) == 0))
        {
            PRINT("\nOUT_CHAN=", ch)
            ch++;
        }
        float tmp;
        tmp = (int)image[i];
        std::cout << tmp << " ";
    }
    std::cout << std::endl; //Empty stdout buffer.
}

void saveNPYFile(dType_8u *out, const std::string &filename, dType_Reg out_ch, dType_Reg im_Size)
{
    std::vector<dType_8u> output_norm_vect(out_ch * im_Size * im_Size);

    for (int i = 0; i < im_Size; i++)
    {
        for (int j = 0; j < im_Size; j++)
        {
            for (int ch = 0; ch < out_ch; ch++)
            {
                output_norm_vect[ch * (im_Size * im_Size) + (i * im_Size) + j] = out[(i * im_Size * out_ch) + (j * out_ch) + ch];
            }
        }
    }
    const long unsigned int dim[3] = {(unsigned long)out_ch, (unsigned long)im_Size, (unsigned long)im_Size};
    npy::SaveArrayAsNumpy(filename, false, 3, dim, output_norm_vect);
}
