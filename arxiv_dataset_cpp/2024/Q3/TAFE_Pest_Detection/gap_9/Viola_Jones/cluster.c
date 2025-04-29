 /*Copyright (C) 2023-2024 University of Bologna, Italy.
 *  Created on: Apr 21, 2022
 *      Author: olmo baldoni
 *      email:  olmo.baldoni@studio.unibo.it
 *  Modified by 
 *          Carlo Montanari calo.montanari4@studio.unibo.it
 *          Luca Bompani luca.bompani5@unibo.it
 */


#include "cluster.h"
#include <math.h>



PI_L1 MyIntImage l1_intimage;
PI_L1 MySize l1_minSize;
PI_L1 MySize l1_maxSize;

// Cluster entry point
void cluster_entry(void *arg)
{
    ClusterParams *params = (ClusterParams*) arg;
    MySize sz;
    float l1_scaleFactor;
    //loading cascade parameters
    load_cl_l1_cascade(params->cascade);
    params->cascade = &l1_cascade;
    
    //setting up pointer to imege data and parameters for execution
    params->intimage = &l1_intimage;
    l1_minSize = params->minSize;
    l1_maxSize = params->maxSize;
    l1_scaleFactor = params->scaleFactor;
    

    float factor;

    int start;
    int j = 0;  
    setImage(params->image->width, params->image->height, params->nneighbor);
    
    
    
    for (factor = 1;factor<2 ;factor *= params->scaleFactor) {
        
        MyPoint my_final_offset;
        MyPoint *final_offset=&my_final_offset;   

        sz.width = params->image->width/factor;
        sz.height = params->image->height/factor;
    
        if (sz.width < params->cascade->orig_window_size.width || sz.height < params->cascade->orig_window_size.height) 
		{
            break;
        }
        int overlapp = 20*factor;
        
        setImage(sz.width, sz.height, params->nneighbor);

        nearestNeighbor(params->image, params->nneighbor);
        
        
        int k;	//starting point subimage
	    int j = 0;	//100pxl counter
	    int new_line;
        

        //TODO remove case switch it works right now but it is hardcoded.
        switch (overlapp)
        {
        case 20:
            for(int n_tile = 0; n_tile < 12; n_tile++){
               

                Copy_tile_image(params->current_tile, params->nneighbor, n_tile, overlapp, final_offset);
                detectInsects(params->current_tile, params->intimage, params->intimagesq, params->cascade, factor, final_offset);
                
            }

            break;
        case 24:
            for(int n_tile = 0; n_tile < 8; n_tile++){
                
                Copy_tile_image(params->current_tile, params->nneighbor, n_tile, overlapp, final_offset);
                detectInsects(params->current_tile, params->intimage, params->intimagesq, params->cascade, factor, final_offset);
            }
            break;
        case 28:
            for(int n_tile = 0; n_tile < 6; n_tile++){
                Copy_tile_image(params->current_tile, params->nneighbor, n_tile, overlapp, final_offset);
                detectInsects(params->current_tile, params->intimage, params->intimagesq, params->cascade, factor, final_offset);
            }
            break;
        case 34:
            for(int n_tile = 0; n_tile < 6; n_tile++){
                Copy_tile_image(params->current_tile, params->nneighbor, n_tile, overlapp, final_offset);
                detectInsects(params->current_tile, params->intimage, params->intimagesq, params->cascade, factor, final_offset);
            }
            break;
        case 41:
            for(int n_tile = 0; n_tile < 2; n_tile++){
                Copy_tile_image(params->current_tile, params->nneighbor, n_tile, overlapp, final_offset);
                detectInsects(params->current_tile, params->intimage, params->intimagesq, params->cascade, factor, final_offset);
            }
            break;
        default:
                final_offset->x = 0;
                final_offset->y = 0;
                detectInsects(params->current_tile, params->intimage, params->intimagesq, params->cascade, factor, final_offset);
            break;
        }
    }

}




// Cluster Function da parallelizzare
void PCS_cluster(void *arg) {

    int core_id = pi_core_id();


    // Casing of void to parameters for execution
    
    PCS_Params *params = (PCS_Params*) arg;
    
    // Cascade parameters setup
    MyCascade *_cascade = params->cascade;
    int y2 = params->y2;
    int x2 = params->x2;
    float _factor = params->factor;
    
   

    for(int z = core_id; z < x2*y2; z+=NUM_CORES){
        

        int y = z / x2;
        int x = z % x2;

        MyPoint p = {x,y};
        int result = runCascadeClassifier(_cascade, p);
        
        if (result > 0) {
        
        MyRect r = {round(params->final_offset->x*_factor + x*_factor), round(params->final_offset->y*_factor + y*_factor), round(_cascade->orig_window_size.width*_factor), round(_cascade->orig_window_size.width*_factor)};	
        push(r);
        }



}

   
 
}




