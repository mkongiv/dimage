#ifndef _DIMAGE_RT_H_
#define _DIMAGE_RT_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <sys/time.h>
#include <omp.h>

char logfile[32];
static int msg_id = 0;

#define mydebug(rr,msg) {if (rr == 0) printf ("MSG%d : %s\n", msg_id++, msg); }

#define DIMAGE_SF 1

#define DIMAGE_MAX_GRID_DIMS 10
  
#ifdef DIMAGE_LOG
#define log_msg(msg) { FILE * ff  = fopen (logfile, "a"); fprintf (ff,"MSG%d : %s\n", msg_id++, msg); fclose (ff);}
#define log_cond(msg) { FILE * ff  = fopen (logfile, "a");  fprintf (ff,"MSG%d : %s\n", msg_id++, msg);  fclose (ff);}
#define log_num(msg,num) { FILE * ff  = fopen (logfile, "a"); fprintf (ff,"MSG%d: %s: %d\n", msg_id++, msg, num);  fclose (ff);}
#define log_commvec(msg,cv,nd) { FILE * ff  = fopen (logfile, "a"); int yy; fprintf (ff,"MSG%d %s: ", msg_id++,msg); for (yy = 0; yy < nd; yy++) fprintf (ff,"%d ",cv[yy]); fprintf (ff,"\n\n");  fclose (ff);}
#else
#define log_msg(msg) {}
#define log_cond(msg) {}
#define log_num(msg,num) {}
#define log_commvec(msg,cv,nd) {}
#endif

#define dimage_min(aa,bb) ((aa) <= (bb) ? (aa) : (bb))
#define dimage_max(aa,bb) ((aa) >= (bb) ? (aa) : (bb))
#define next_tile(tt,DIMAGE_TS) (((tt)+1)*(DIMAGE_TS))
#define dimage_div(xx,TT) ((xx)/(TT))
#define dimage_acc(ref,NT2,DIMAGE_TS2,ti1,ti2,pi1,pi2) ref[(ti1) * (NT2) + (ti2)][(pi1) * (DIMAGE_TS2) + (pi2)]

#define aceil(n,d) (dimage_ceil(n,d))
#define dimage_ceil(n,d)  (int)(ceil(((double)(n))/((double)(d))))
#define dimage_floor(n,d) (int)(floor(((double)(n))/((double)(d))))


#define dimage_is_comm_subset(s1,s2) ((s1 == s2) ? 1 : ((s2 == -1) ? 1 : 0))

#define dimage_proc_coord_1D(ar,thedim) dimage_proc_dim(ar,1,thedim,DIMAGE_GRID_DIMS[0],1)
#define dimage_proc_coord_2D(ar,thedim) dimage_proc_dim(ar,2,thedim,DIMAGE_GRID_DIMS[1],1)
#define dimage_proc_coord_3D(ar,thedim) dimage_proc_dim(ar,3,thedim,DIMAGE_GRID_DIMS[1],DIMAGE_GRID_DIMS[2])

// Macros used to access an array in a linearized form. Global coordinates must be assembled previously.
#define DIMAGE_ACC_LIN1D(i0,n0) ((i0))
#define DIMAGE_ACC_LIN2D(i0,i1,n0,n1) ((i0) * (n1) + (i1))
#define DIMAGE_ACC_LIN3D(i0,i1,i2,n0,n1,n2) ((i0) * (n1) * (n2) + (i1) * (n2) + (i2))

// Macros used for accessing an array using separate tile and point coordinates.

#define DIMAGE_ACC_TILE1D(t0,i0,ts0,nt0) ((t0) * (ts0) + (i0))
#define DIMAGE_ACC_TILE2D(t0,t1,i0,i1,ts0,ts1,nt0,nt1) (((t0) * (nt1) + (t1)) * (ts0) * (ts1) + (i0) * (ts1) + i1)
#define DIMAGE_ACC_TILE3D(t0,t1,t2,i0,i1,i2,ts0,ts1,ts2,nt0,nt1,nt2) (DIMAGE_ACC_LIN3D(t0,t1,t2,nt0,nt1,nt2) * (ts0) * (ts1) * (ts2) + DIMAGE_ACC_LIN3D(i0,i1,i2,ts0,ts1,ts2))

#define DIMAGE_PTR_TILE1D(t0,ts0,nt0) ((t0) * (ts0))
#define DIMAGE_PTR_TILE2D(t0,t1,ts0,ts1,nt0,nt1) (((t0) * (nt1) + (t1)) * (ts0) * (ts1))
#define DIMAGE_PTR_TILE3D(t0,t1,t2,ts0,ts1,ts2,nt0,nt1,nt2) (DIMAGE_ACC_LIN3D(t0,t1,t2,nt0,nt1,nt2) * (ts0) * (ts1) * (ts2))

#define DIMAGE_TILE_COMP(i,N,p) ((i)/((N)/(p)))
#define DIMAGE_POINT_COMP(i,N,p) ((i)%((N)/(p)))
#define DIMAGE_ACC_SLICE2D(t0, t1, i0, i1, s0, s1, np) (DIMAGE_ACC_LIN2D(DIMAGE_TILE_COMP((i1),s1,np),DIMAGE_TILE_COMP((i0),s0,np),np,np) * ((s0/np)*(s1/np)) + DIMAGE_POINT_COMP((i1),s1,np) * (s0/np) + DIMAGE_POINT_COMP((i0),s0,np))
#define DIMAGE_ACC_SLICE3D(t0, t1, t2, i0, i1, i2, n0, n1, n2, np) (DIMAGE_ACC_LIN3D(t0,t1,t2,nt0,nt1,nt2) * (tvol) + DIMAGE_ACC_LIN3D(i0,i1,i2,s0,s1,s2))

#define DIMAGE_INIT_DIAG_2D(row,col,s1,s2) ((row) == (col) ? 2.0 : 0.0)
#define DIMAGE_INIT_DIAG_3D(xx,yy,zz,s1,s2,s3) ((xx) == (yy) && (yy) == (zz) ? 2.0 : 0.0)

void
write_to_file_tile1D (const char * name, int ** ranks, double * mat, int s0, int nt0);
             
void         
write_to_file_tile2D (const char * name, int ** ranks, double * mat, int s0, int s1, int nt0, int nt1);
             
void
write_to_file_tile3D (const char * name, int ** ranks, double * mat, int s0, int s1, int s2, int nt0, int nt1, int nt2);

int 
dimage_proc_dim (int dimage_rank, int n_proc_dim, int pdim, int extent1, int extent2);

void
dimage_2d_free (double ** dimage_array, int pbs, int tile_size);

double *
dimage_tile2d_alloc (int ts1, int ts2);

double *
dimage_alloc_buffer (int numitems);

double *
dimage_1d_tile_alloc (int ts0, int nt0);

double *
dimage_2d_tile_alloc (int ts1, int ts2, int nt1, int nt2);

double *
dimage_3d_tile_alloc (int ts0, int ts1, int ts2, int nt0, int nt1, int nt2);

double ** 
dimage_2d_alloc (int pbs1, int pbs2, int ts1, int ts2);


void init_log_file_1D (int pd0);

void init_log_file_1D_with_rank (int pd0, int dimage_rank);

void init_log_file_2D (int pd0, int pd1);

void init_log_file_2D_with_rank (int pd0, int pd1, int dimage_rank);

void init_log_file_3D (int pd0, int pd1, int pd2);

void init_log_file_3D_with_rank (int pd0, int pd1, int pd2, int dimage_rank);

void
generate_datafile_1D (const char * filename, unsigned long long n1, double given);

void
generate_datafile_2D (const char * filename, unsigned long long n1, unsigned long long n2, double given);

void
generate_datafile_3D (const char * filename, unsigned long long n1, unsigned long long n2, unsigned long long n3, double given);


void
dump_slice (const char * name, int pd0, int pd1, double * mat, int nrows, int ncols);

void dump_tile (double * mout, int pd0, int pd1, int kk);


void
read_from_file_tile1D (const char * filename, double * mat, int n0, int s0, int t0, int * block_count);

void
read_from_file_tile2D (const char * filename, double * mat, int n0, int n1, int s0, int s1, int t0, int t1, int * );

void
read_from_file_tile3D (const char * filename, double * mat, int n0, int n1, int n2, int s0, int s1, int s2, int t0, int t1, int t2, int *);

void
compute_comm_vector (int dimage_rank, 
  int * used, // iteration space dimensions appearing in access function. Example: S[i,j,k]->A[i,k] = [0,-1,1,-2] 
  int * imap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * amap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * commvec,// communication vectors; 0 means no-comm along that dimension
  int tranpose); 

void
compute_generator_comm_vector (int dimage_rank, 
  int * used, // iteration space dimensions appearing in access function. Example: S[i,j,k]->A[i,k] = [0,-1,1,-2] 
  int * imap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * amap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * commvec,// communication vectors; 0 means no-comm along that dimension
  int tranpose); 

int 
compute_communicator_color (int dimage_rank, int n_proc_dim, int * cv);


double rtclock();


double *
read_matrix_from_file (const char * filename, unsigned long long nitems);

void
write_to_file_matrix_1D (const char * filename, double * mat, unsigned long long nitems);

void
write_to_file_matrix_2D 
(const char * filename, double * mat, unsigned long long n1, unsigned long long n2);

void
write_to_file_matrix_3D (const char * filename, double * mat, unsigned long long n1, unsigned long long n2, unsigned long long n3);

void
check_array_tile2D (const char * filename, double * mat_dist, double * mat_ref, int n0, int n1, int s0, int s1, int t0, int t1, int * block_count);

void
check_array_tile3D (const char * filename, double * mat_dist, double * mat_ref, 
  int n0, int n1, int n2, int s0, int s1, int s2, int t0, int t1, int t2, int * block_count);


void 
dimage_collect_tile_coordinates_1D (double * mat, int * tile_map, int ss1, int maxnt1, int nt1);

void 
dimage_collect_tile_coordinates_2D (double * mat, int * tile_map, int ss1, int ss2, int maxnt1, int maxnt2, int nt1, int nt2);

void 
dimage_collect_tile_coordinates_3D (double * mat, int * tile_map, int ss1, int ss2, int ss3, int maxnt1, int maxnt2, int maxnt3, int nt1, int nt2, int nt3);

int * 
dimage_alloc_tile_map_1D (int np0);

int * 
dimage_alloc_tile_map_2D (int np0, int np1);

int * 
dimage_alloc_tile_map_3D (int np0, int np1, int np2);

double * 
dimage_fetch_tile_ptr_1D (double * mat, int * tile_map, int ss0, int t0, int nt0);

double * 
dimage_fetch_tile_ptr_2D (double * mat, int * tile_map, int ss0, int ss1, int t0, int t1, int nt0, int nt1);

double * 
dimage_fetch_tile_ptr_3D (double * mat, int * tile_map, int ss0, int ss1, int ss2, int t0, int t1, int t2, int nt0, int nt1, int nt2);

void
dimage_store_tile_map_1D (const char * prefix_name, int ** ranks, int * map, int nt0);

void
dimage_store_tile_map_2D (const char * prefix_name, int ** ranks, int * map, int nt0, int nt1);

void
dimage_store_tile_map_3D (const char * prefix_name, int ** ranks, int * map, int nt0, int nt1, int nt2);

#define DIMAGE_SET_TILE_COORDINATE_1D(arr, t0) {(arr-DIMAGE_TILE_HEADER_SIZE)[0] = t0;}
#define DIMAGE_SET_TILE_COORDINATE_2D(arr, t0, t1) {(arr-DIMAGE_TILE_HEADER_SIZE)[0] = t0; (arr-DIMAGE_TILE_HEADER_SIZE)[1] = t1;}
#define DIMAGE_SET_TILE_COORDINATE_3D(arr, t0, t1, t2) {(arr-DIMAGE_TILE_HEADER_SIZE)[0] = t0; (arr-DIMAGE_TILE_HEADER_SIZE)[1] = t1; (arr-DIMAGE_TILE_HEADER_SIZE)[2] = t2;}


int
dimage_is_lcsr (int dimage_rank, 
  int ** dimage_ranks,
  int * used,
  int * imap,
  int * amap,
  int * commvec);

// Wrappers for external operators used in testing and benchmarking.

void 
dimage_gemm (double * mat_out, double * mat_in, double * mat_ker, int ni, int nj, int nk, double alpha, double beta);

void 
dimage_ttm (double * mat_out, double * mat_in, double * mat_ker, int ni, int nj, int nk, int nl, double alpha, double beta);

void 
dimage_ttv (double * mat_out, double * mat_in, double * mat_ker, int ni, int nj, int nk, double alpha, double beta);

void 
dimage_mttkrp (double * mat_out, double * mat_in, double * mat_ker1, double * mat_ker2, int ni, int nj, int nk, int nl, double alpha, double beta);

void 
dimage_mttkrp2 (double * mat_out, double * mat_in, double * mat_ker1, double * mat_ker2, int ni, int nj, int nk, int nl, double alpha, double beta);

void 
dimage_mttkrp3 (double * mat_out, double * mat_in, double * mat_ker1, double * mat_ker2, int ni, int nj, int nk, int nl, double alpha, double beta);

void 
dimage_trans (double * mat_out, double * mat_in, int ni, int nj, double alpha, double beta);


#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

void print_coords (int nd, int * cc);
int dimage_compute_max_colors (int nd, int * dims);
void dimage_rank_to_coords (int nd, int rank, int * dims, int * cc);
int dimage_compute_color_from_comm_vec (int nd, int * dims, int * cc, int * cv);

#endif // _DIMAGE_RT_H_
