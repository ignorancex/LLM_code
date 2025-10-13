
		#include <math.h>
		#include <stdio.h>
		// sub-functions
		__host__ __device__ double dot(double p[3],double q[3])
		{return(p[0]*q[0]+p[1]*q[1]+p[2]*q[2]);}

		__host__ __device__ double norm2(double p[3])
		{return(sqrt(dot(p,p)));}

		__host__ __device__ double Sqrt(double x)
		{return(sqrt(x));}

		__host__ __device__ int check_intersection(double p[3],double q[3],double q_1[3],double d1, double d2)
		{
		//intersection_state equals 1 if p-centered voxel intersects 
		//the knot arc (q ; q_1)
		int intersection_state;  
		intersection_state=0;
		double xi;
		double kp_kp1[3];  // Point between the arc (q ; q_1) 

		for (int s=0;s<=50;s++){
			xi=-1+s*2.0/50;
			kp_kp1[0]=(1-xi)*0.5*q[0]+(1+xi)*0.5*q_1[0]-p[0];
			kp_kp1[1]=(1-xi)*0.5*q[1]+(1+xi)*0.5*q_1[1]-p[1];
			kp_kp1[2]=(1-xi)*0.5*q[2]+(1+xi)*0.5*q_1[2]-p[2];
			double d;
			d=d1+s*fabs(d2-d1)/50.0;
			if (norm2(kp_kp1)<d){
				intersection_state=1;
			}
			//else{
				//intersection_state=0;
				//int dum=0;
			//}	
		}
		return(intersection_state);}
		// END of sub-functions
		
		__global__ void scalar_field(double field[@dim_grid], double knot[@nkp][3],
						double infx,
						double infy,
						double infz,
						double res,
						double reach[@nkp],/*[@dim_grid],*/
						int is_link,
						int batch_number)
		{
		int n_knot_points=@nkp;
		/*
		int dim_grid_x=@dim_grid_x;
		int dim_grid_y=@dim_grid_y;
		int dim_grid_z=@dim_grid_z;
		*/
		int this_thread=threadIdx.x+blockDim.x*blockIdx.x+@dim_grid*batch_number; //batch_number*blockDim.x;
		int k=(this_thread)/(@nx*@ny);// z-index
		int i=(this_thread-k*(@nx*@ny))%@nx; 	// x-index
		int j=(this_thread-k*(@nx*@ny))/@ny;	// y-index
				//threadIdx.z+blockDim.z*blockIdx.z;  	
		//if(i>=51 || j>=51 || k>=51){printf("not in grid: (%d,%d,%d),batch:%d, threadinfo:%d,%d,\n",i,j,k,batch_number,threadIdx.x,blockIdx.x);}
				//printf("node coords GPU: %d,%d,%d - block dim %d\n",i,j,k, blockDim.x);
		//if ((threadIdx.x<1024) ){///////////
			if(i>=0 || i<(@nx)  || j>=0 || j<(@ny)  || k>=0 || k<(@nz) ){
				for(int par=0;par<n_knot_points ; par++){
					double A[3];
					double B[3];
					double C[3];
					A[0]=infx+(i)*res;
					A[1]=infy+(j)*res;
					A[2]=infz+(k)*res;
	
					B[0]=knot[par][0];
					B[1]=knot[par][1];
					B[2]=knot[par][2];
	
					int sig;
					if (par<(n_knot_points-1)){
						sig=par+1;
					}
					else{
						if (is_link==0){
							sig=0;
						}
						else{
							sig=par;
						}
					}
					C[0]=knot[sig][0];
					C[1]=knot[sig][1];
					C[2]=knot[sig][2];
	
					int intersec;
					//intersec=check_intersection(A,B,C,res*Sqrt(3),res*Sqrt(3));
					intersec=check_intersection(A,B,C,reach[par],reach[sig]);
					
					if (intersec==1) {
						if (par==(n_knot_points-1) && (is_link==1)){
						}
						else{
							field[threadIdx.x+blockDim.x*blockIdx.x]=-1;
						}
		
					}
					else{
					}
					//else{
					//}
				}
				//if(i==0 || i==(@dim_grid -1) || j==0 || j==(@dim_grid -1) || k==0 || k==(@dim_grid -1)){field[i][j][k]=1;}
				
			}//
		//if (i==0 && j==0){field[i][j][k]=-1;}else{field[i][j][k]=1;}
		//}//field[threadIdx.x+blockDim.x*blockIdx.x]=-1; // if thread<1024
			__syncthreads();
		}
	