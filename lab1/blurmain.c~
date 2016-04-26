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


void calculate_local_problem_size(const int x,const int y,const int np, int lproblem[][2]);

void calculate_local_allocation_size(int ldata[][2], int lproblem[][2],int const ysize, const int np, const int radius);

int write_txt (const char* fname, const int radius, const double time, const int np);

int main (int argc, char *argv[]) {
   int radius;
   int xsize, ysize, colmax;

#define MAX_RAD 3000

    double w[MAX_RAD];
    int np, me, counter, imagecounter, buff[3];
    pixel* local_src;
    pixel* src;
	double starttime, endtime;
    MPI_Init( &argc, &argv );
    MPI_Comm com = MPI_COMM_WORLD;
    MPI_Comm_size( com, &np );
    MPI_Comm_rank( com, &me );
	MPI_Status status;
    int lproblem[np][2];	//contains startline[0] and number-of-lines[1] for each process 
    int ldata[np][2];		//as above including radius, i.e. all data needed to calculate the problem
    int i;
    char* imagename;
	char* filename ; 
    //Create mpi structure
    pixel item;

	MPI_Datatype pixel_mpi;
	int block_lengths [] = {1 , 1, 1};
	MPI_Datatype block_types [] = {MPI_UNSIGNED_CHAR,MPI_UNSIGNED_CHAR,MPI_UNSIGNED_CHAR};
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
    for(imagecounter = 0; imagecounter<4; imagecounter++)
    {
    	switch(imagecounter)
    	{
    		case 0: 
    		imagename = "im1.ppm";
			filename = "im1blur.txt";
			break;
			case 1: 
    		imagename = "im2.ppm";
			filename = "im2blur.txt";
			break;
			case 2: 
    		imagename = "im3.ppm";
			filename = "im3blur.txt";
			break;
			case 3: 
    		imagename = "im4.ppm";
			filename = "im4blur.txt";
			break;
    	}
		for(counter = 0; counter < 10; counter ++)
		{
			radius= pow(2,counter);
			if (me == 0) { // read image at process 0:
				src = malloc(MAX_PIXELS*sizeof(*src));
				if((radius > MAX_RAD) || (radius < 1)) {
					fprintf(stderr, "Radius (%d) must be greater than zero and less than %d\n", radius, MAX_RAD);
					exit(1);
				}

				// read file
				if(read_ppm (imagename, &xsize, &ysize, &colmax, (char *) src) != 0)
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
		
				printf("Calling filter\n");
			}
	
			starttime = MPI_Wtime();
			// Single-Broadcast of size from P0 to P1...P(np-1):
			MPI_Bcast( buff, 3, MPI_INT, 0, com );
			// Extract problem size from buff; allocate space:
	
			if(me != 0){
				xsize=buff[0];
				ysize=buff[1];
				radius=buff[2];
			}
			get_gauss_weights(radius, w);  //Theoretically only needs to be calculated once
							//and then sent to other processes
			//Calculates lproblem, size and lines
			calculate_local_problem_size(xsize,ysize,np ,me , lproblem);
			//Calculates ldata from lproblem, ldata contains size and lines of the problem and radius
			calculate_local_allocation_size(ldata, lproblem, ysize, me, np, radius);
	
			//Allocate local memory
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
	
			int ydiff = lproblem[me][0] - ldata[me][0];
			blurfilter(xsize, ysize, local_src, radius, w, ydiff, lproblem[me][1], ldata[me][1]);
		
			//Get data from other processes
			MPI_Gather(&local_src[ydiff*xsize], lproblem[me][1]*xsize, pixel_mpi, src, lproblem[me][1]*xsize, pixel_mpi, 0, com);
	
			endtime = MPI_Wtime(); 
	
			if(me == 0)
			{
				printf("Filtering took: %f secs\n", (endtime-starttime)) ;

				printf("Filename: %s, radius: %d \n", filename, radius);
		
				if(write_txt (filename, radius, (endtime-starttime), np ) != 0){
			  		exit(1);
			  	}
			}
		}
	}
	MPI_Finalize();
}


void calculate_local_problem_size(const int x,const int y,const int np, int lproblem[][2]){
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

void calculate_local_allocation_size(int ldata[][2], int lproblem[][2],int const ysize , const int np, const int radius){
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

int write_txt (const char* fname, const int radius, const double time, const int np) {

  FILE * fp;
  int errno = 0;

  if (fname == NULL) fname = "\0";
  fp = fopen (fname, "a");
  if (fp == NULL) {
    fprintf (stderr, "write_txt failed to open %s: %s\n", fname,strerror (errno));
    return 1;
  }
  
  fprintf(fp, "%d %d %f \n", np, radius, time);
  if (fclose (fp) == EOF) {
    perror ("Close failed");
    return 3;
  }
  return 0;
}






















