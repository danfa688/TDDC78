#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "ppmio.h"
#include "blurfilter.h"
#include "gaussw.h"
#include <mpi.h>
#include <math.h>
#include <stdlib.h>


void calculate_local_problem_size(const int x,const int y,const int r,const int np, const int me, int lproblem[][2]);

void calculate_local_allocation_size(int ldata[][2], int lproblem[][2],int const ysize ,const int me, const int np, const int radius);

int main (int argc, char *argv[]) {
   int radius;
   int xsize, ysize, colmax;

#define MAX_RAD 3000

    double w[MAX_RAD];
    int np, me, buff[3];
    pixel* local;
    pixel* src;
    struct timespec stime, etime;
    MPI_Init( &argc, &argv );
    MPI_Comm com = MPI_COMM_WORLD;
    MPI_Comm_size( com, &np );
    MPI_Comm_rank( com, &me );
	MPI_Status status;
    int lproblem[np][2];
    int ldata[np][2];
    int i;
    
    if(me == 0){
		src = malloc(MAX_PIXELS*sizeof(*src));
    }

    
    //Create mpi structure
    pixel item;

	MPI_Datatype pixel_mpi;
	int block_lengths [] = {1 , 1, 1};
	MPI_Datatype block_types [] = {MPI_CHAR, MPI_CHAR, MPI_CHAR};
	MPI_Aint start, displ[3];

	MPI_Address(&item, &start);
	MPI_Address(&item.r, &displ[0]);
	MPI_Address(&item.g, &displ[1]);
	MPI_Address(&item.b, &displ[2]);

	displ[0] -= start;
	displ[1] -= start;
	displ[2] -= start;
	MPI_Type_struct(3, block_lengths, displ, block_types, &pixel_mpi);

	MPI_Type_commit( &pixel_mpi);
	//STOP Create mpi structure

    /* Take care of the arguments */

    if (me == 0) { // read image at process 0:

		if (argc != 4) {
			fprintf(stderr, "Usage: %s radius infile outfile\n", argv[0]);
			exit(1);
		}
		radius = atoi(argv[1]);
		if((radius > MAX_RAD) || (radius < 1)) {
			fprintf(stderr, "Radius (%d) must be greater than zero and less than %d\n", radius, MAX_RAD);
			exit(1);
		}

		// read file
		if(read_ppm (argv[2], &xsize, &ysize, &colmax, (char *) src) != 0)
		    exit(1);

		if (colmax > 255) {
			fprintf(stderr, "Too large maximum color-component value\n");
			exit(1);
		}

		printf("Has read the image, generating coefficients\n");
		
		//read problem size into buf
		buff[0]=xsize;
		buff[1]=ysize;
		buff[2]=radius;

    }

    /* filter */
    get_gauss_weights(radius, w);  //Only needs to be calculated once

	if (me == 0) {
		printf("Calling filter %d\n", me);

		clock_gettime(CLOCK_REALTIME, &stime);
	}
	
	// Single-Broadcast of size from P0 to P1...P(np-1):
	MPI_Bcast( buff, 3, MPI_INT, 0, com );
	// Extract problem size from buff; allocate space:
	
	if(me != 0){
		xsize=buff[0];
		ysize=buff[1];
		radius=buff[2];
	}
	
	//Calculates lproblem, size and lines
	calculate_local_problem_size(xsize,ysize,radius,np ,me , lproblem);
	//Calculates ldata from lproblem, ldata contains size and lines of the problem and radius
	calculate_local_allocation_size(ldata, lproblem, ysize, me, np, radius);
	
	//Allocate local memory
	pixel* local_src;
    local_src = malloc(ldata[me][1]*xsize*sizeof(*local_src));
	
	//Send data to other processes
	if(me == 0){
		for(i=1; i<np;i++){
			MPI_Send( &(src[ldata[i][0]*xsize]), ldata[i][1]*xsize, pixel_mpi, i, 10, com );
		}
		memcpy(local_src, src, ldata[me][1]*xsize*3);	//Copy from src to local src in process 0
	}else{
		MPI_Recv(local_src, ldata[me][1]*xsize, pixel_mpi, 0, 10, com, &status);
	}
	
    //blurfilter(xsize, ysize, src, radius, w);
    
	//Get data from other processes
	//MPI_Gather


	if(me == 0)
	{
    	clock_gettime(CLOCK_REALTIME, &etime);
	}
	if(me == 0){
    	printf("Filtering took: %g secs\n", (etime.tv_sec  - stime.tv_sec) + 1e-9*(etime.tv_nsec  - stime.tv_nsec)) ;
	}

    // write result
    if(me == 0){
    	printf("Writing output file\n");
    }
    
    if(me ==2){
    	if(write_ppm (argv[3], xsize, ldata[me][1], (char *)local_src) != 0){
      		exit(1);
      	}
    }
    
	if(me == 0){
		printf("lproblem:\n");
		for(i=0; i<np; i++){
			printf("[%d][%d]\n",lproblem[i][0],lproblem[i][1]);
		}
		printf("ldata:\n");
		for(i=0; i<np; i++){
			printf("[%d][%d]\n",ldata[i][0],ldata[i][1]);
		}
	}
	MPI_Finalize();
}


void calculate_local_problem_size(const int x,const int y,const int r,const int np, const int me, int lproblem[][2]){
	int linesize, rest;
	int lysize[np];
	linesize = floor(y/np);
	rest = y%np;
	int i;
	int k;
	for (i=0; i<np; i++){
		if(i<rest){
			lysize[i]=linesize+1;
		}else{
			lysize[i]=linesize;
		}
	}
	int tmp;
	for (i=0; i<np; i++){
		tmp=0;
		for (k=0; k<i; k++){
			tmp += lysize[k];
		}
		
		lproblem[i][0]=tmp;
		lproblem[i][1]=lysize[i];
	}
}

void calculate_local_allocation_size(int ldata[][2], int lproblem[][2],int const ysize ,const int me, const int np, const int radius){
	int i;
	for(i=0; i<np; i++){
		if(lproblem[i][0] > radius){
			ldata[i][0]=lproblem[i][0]-radius;
		}else{
			ldata[i][0]=0;
		}
		
		if(lproblem[i][0]+lproblem[i][1]-1 + radius < ysize){
			ldata[i][1]=lproblem[i][0]-ldata[i][0]+lproblem[i][1] + radius;
		}else{
			ldata[i][1]=ysize-ldata[i][0];
		}
	}
}






















