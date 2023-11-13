#include "dimage-rt.h"
#include <omp.h>

#ifdef DIMAGE_MKL
#include <mkl.h>
#endif

extern const int DIMAGE_GRID_DIMS[];

void
compute_generator_comm_vector (int dimage_rank, 
  int * used, // iteration space dimensions appearing in access function. Example: S[i,j,k]->A[i,k] = [0,-1,1,-2] 
  int * imap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * amap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * commvec,// communication vectors; 0 means no-comm along that dimension
  int transpose) 
{
  // Example: A[1,-1,0,-2], PROCS={2,3,5}, comm_size = 6, color = 2x3x5 / 6 = 5
  int comm_size = 1;
  int comm_world_size = 1;
  int ii;
  int np = 0;
  for (ii = 0, np = 0; DIMAGE_GRID_DIMS[ii] > 0; ii++, np++)
  {
    comm_world_size *= DIMAGE_GRID_DIMS[ii];
  }
  // By default, assume that we will *NOT* have communication along any 
  // directions of the processor space.
  log_num ("np" , np);
  for (ii = 0; ii < np; ii++)
  {
    commvec[ii] = 1;
  }
  int all_reduce = 0;
  int any_mu = 0;
  int any_pi = 0;
  int both_unmapped = 0;
  int last_mu = -1;
  int matched = 0;
  int is_red = 0;
  int red_dim[np];
  int n_red_dim = 0;
  for (ii = 0; ii < np; ii++)
  {
    red_dim[ii] = 0;
  }
  int n_used = 0;
  int mapped_pi = 0;
  int mapped_mu = 0;
  int mapped_red = 0;
  int matched_dim = 0;
  int n_iter_dim = 0;
  for (ii = 0; used && used[ii] >= -1; ii++)
  {
    int adim = used[ii];
    n_used += (adim >= 0 ? 1 : 0);
    n_iter_dim ++;
    int dim_pi = (adim >= 0 ? amap[adim] : -1);
    mapped_pi += (dim_pi >= 0 ? 1 : 0);
    int dim_mu = imap[ii];
    mapped_mu += (dim_mu >= 0 ? 1 : 0);
    mapped_red += (dim_mu >= 0 && used[ii] == -1 ? 1 : 0);
    matched_dim += (dim_mu == dim_pi && dim_mu >= 0 ? 1 : 0);
  }
  log_num ("Num.dim : ", n_iter_dim);
  log_num ("Num.used: ", n_used);
  log_num ("Mapped mu: ", mapped_mu);
  log_num ("Mapped pi: ", mapped_pi);
  log_num ("Mapped red: ", mapped_red);
  log_num ("Matched dim: ", matched_dim);
  if (mapped_red > 0 && mapped_pi > 0)
  {
    // Reduction case along a subset of processor grid dimensions.
    // Processor grid sliced because of the mu/pi matching.
    // No communication along the matching dimension.
    for (ii = 0; ii < np; ii++)
      commvec[ii] = 1;
    log_num ("Case 1A (red + match) ", 1000);
    for (ii = 0; matched_dim > 0 && used && used[ii] >= -1; ii++)
    {
      int adim = used[ii];    
      int dim_pi = (adim >= 0 ? amap[adim] : -1);
      int dim_mu = imap[ii];
      if (dim_pi == dim_mu && dim_mu >= 0)
        commvec[dim_pi] = 0;
    }
  }
  if (mapped_red > 0 && mapped_pi == 0)
  {
    // Reduction case along a subset of processor grid dimensions.
    // Array fully replicated, so no matching will be possible.
    // Assume communication along the reduction dimension only.
    for (ii = 0; ii < np; ii++)
      commvec[ii] = 0;
    log_num ("Case 1B (red + no-match) ", 1500);
    for (ii = 0; used && used[ii] >= -1; ii++)
    {
      int adim = used[ii];    
      int dim_mu = imap[ii];
      int dim_pi = (adim >= 0 ? amap[adim] : -1);
      if (adim == -1 && dim_mu >= 0)
        commvec[dim_mu] = 1;
      if (adim >= 0 && dim_pi == -1 && dim_mu >= 0)
        commvec[dim_mu] = 1;
    }
  }
  if (mapped_red == 0 && matched_dim > 0 && matched_dim <= n_used)
  {
    for (ii = 0; ii < np; ii++)
      commvec[ii] = 1;
    log_num ("Case 2 (only match) ", 2000);
    for (ii = 0; matched_dim > 0 && used && used[ii] >= -1; ii++)
    {
      int adim = used[ii];    
      int dim_pi = (adim >= 0 ? amap[adim] : -1);
      int dim_mu = imap[ii];
      if (dim_pi == dim_mu && dim_mu >= 0)
        commvec[dim_pi] = 0;
    }
  }
  if (mapped_red == 0 && matched_dim == 0)
  {
    log_num ("Case 3 (no-match, generator-case) ", 4000);
    for (ii = 0; ii < np; ii++)
    {
      commvec[ii] = 0;
    }
    for (ii = 0; used && used[ii] >= -1; ii++)
    {
      int adim = used[ii];    
      int dim_pi = (adim >= 0 ? amap[adim] : -1);
      int dim_mu = imap[ii];
      if (dim_mu >= 0 && dim_pi == -1)
        commvec[dim_mu] = 1;
    }
  }
  for (ii = 0; ii < np; ii++)
  {
    if (commvec[ii] > 0)
      comm_size *= DIMAGE_GRID_DIMS[ii];
  }
  log_num ("Communicator size: ", comm_size);
  return;
  #define ATA(a,e) log_num ("weird",(e)); assert ((e) >= 0 && (e) < np); a[e]
  for (ii = 0; used && used[ii] >= -1; ii++)
  {
    if (used[ii] >= 0)
    {
      int adim = used[ii];
      int dim_mu = imap[ii];
      int dim_pi = (adim >= 0 ? amap[adim] : -1);

      if (dim_mu >= 0)
      {
        any_mu = 1;
      }
      
      if (dim_pi >= 0)
        any_pi = 1;

      if (dim_mu == dim_pi && dim_mu >= 0)
      {
        log_num ("dim_mu",dim_mu);
        ATA(commvec,dim_mu) = 0;
        matched ++;
      }

      if (dim_mu >= 0 && dim_pi == -1) // && DIMAGE_GRID_DIMS[dim_mu] > 1)
      {
        both_unmapped = 1; 
        last_mu = dim_mu;
      }

      if (dim_mu == -1 && dim_pi == -1)
      {
        both_unmapped = 1;
      }
      
      log_num ("ii",ii);
      log_num ("used",used[ii]);
      log_num ("mu",dim_mu);
      log_num ("pi",dim_pi);
    }
    // Some iterator is not used in the AF, but it's mapped to the grid.
    // This means it is the reduction dimension.
    if (used[ii] == -1 && imap[ii] >= 0)
    {
      int dim_mu = imap[ii];
      assert (dim_mu < np);
      if (dim_mu >= 0 && DIMAGE_GRID_DIMS[dim_mu] > 1) {
        log_num ("red-ii",ii);
        log_num ("not used",used[ii]);
        log_num ("mu",dim_mu);
        red_dim[n_red_dim++] = dim_mu;
        any_mu = 1;
        is_red = 1;
      }
    }
  }
  if (any_mu && both_unmapped && matched < np && !is_red && np > 1)
  {
    log_num ("DEBUG ", 1);
    for (ii = 0; ii < np; ii++)
      commvec[ii] = 1;
    for (ii = 0; used && used[ii] >= -1; ii++)
    {
      if (used[ii] >= 0)
      {
        log_num ("red-ii (used)",ii);
        int adim = used[ii];
        int dim_mu = imap[ii];
        int dim_pi = (adim >= 0 ? amap[adim] : -1);
        if (dim_mu == dim_pi && dim_mu >= 0)
        {
          ATA(commvec,dim_mu) = 0; 
          commvec[dim_mu] = 0;
          break;
        }
        if (dim_mu == -1 && dim_pi == -1 && last_mu >= 0)
        {
          log_num ("last_mu ", last_mu);
          int kk;
          for (kk = 0; kk < np; kk++)
            if (kk != last_mu)
              commvec[kk] = 0;
          break;
        }
      }
    }
  }
  if ((!any_mu && !any_pi && !is_red)) 
  {
    log_num ("DEBUG ", 2);
    for (ii = 0; ii < np; ii++)
      commvec[ii] = 0;
  }
  if (is_red && any_mu)
  {
    // Handle all-reduce cases.
    int deac = 0;
    log_num ("DEBUG ", 3);
    // First case: Try to find if any data-space dimension is mapped.
    // If so, then we don't have communication along a mapped dimension.
    for (ii = 0; ii < np; ii++)
      commvec[ii] = 1;
    for (ii = 0; used && used[ii] >= -1; ii++)
      if (used[ii] >= 0)
    {
      int pi_map = amap[used[ii]];
      if (pi_map >= 0)
      {
        commvec[pi_map] = 0;
        deac++;
      }
    }
    // Second case: We didn't find any mapped data-space dimension.
    // Hence, we will have communication along the iteration-space dimension
    // that is a reduction dimension in the computation.
    // We de-activate all the dimensions of the communication vector
    // and activate the ones on which we find a mapped reduction dimension.
    if (!deac)
    {
      for (ii = 0; ii < np; ii++)
        commvec[ii] = 0;
      for (ii = 0; used && used[ii] >= -1; ii++)
        if (used[ii] < 0)
      {
        int mu_map = imap[ii];
        if (mu_map >= 0)
        {
          commvec[mu_map] = 1;
        }
      }
    }
  }
  // Compute the communicator size, just for reference.
  for (ii = 0; ii < np; ii++)
  {
    log_num ("DEBUG ", 4);
    if (commvec[ii] > 0)
      comm_size *= DIMAGE_GRID_DIMS[ii];
  }
  log_num ("Comm size", comm_size);
}

void
compute_comm_vector (int dimage_rank, 
  int * used, // iteration space dimensions appearing in access function. Example: S[i,j,k]->A[i,k] = [0,-1,1,-2] 
  int * imap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * amap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * commvec,// communication vectors; 0 means no-comm along that dimension
  int transpose) 
{
  // Example: A[1,-1,0,-2], PROCS={2,3,5}, comm_size = 6, color = 2x3x5 / 6 = 5
  int comm_size = 1;
  int comm_world_size = 1;
  int ii;
  int np = 0;
  for (ii = 0, np = 0; DIMAGE_GRID_DIMS[ii] > 0; ii++, np++)
  {
    comm_world_size *= DIMAGE_GRID_DIMS[ii];
  }
  // By default, assume that we will have communication along all directions of the
  // processor space.
  for (ii = 0; ii < np; ii++)
  {
    commvec[ii] = 1;
  }
  for (ii = 0; used && used[ii] >= -1; ii++)
    if (used[ii] >= 0)
  {
    int adim = used[ii];
    int p4is = imap[ii];
    int p4ds = amap[adim];
    if ((p4is >= 0 && p4ds >= 0 && p4is == p4ds) || (p4is >= 0 && p4ds == -1))
    {
      commvec[p4is] = 0;
    }
    if ((p4is == p4ds && p4ds == -1)) 
    {
      // Nothing to do. Data replicated, work serialized.
    }
  }
  comm_size = 1;
  for (ii = 0; ii < np; ii++)
  {
    log_num ("DEBUG ", 3);
    if (commvec[ii] > 0)
      comm_size *= DIMAGE_GRID_DIMS[ii];
  }
  log_num ("Comm size", comm_size);
}

#define dimage_pow(b,e) ((e) == 2 ? (b)*(b) : (e) == 1 ?  (b) : (e) == 3 ? (b)*(b)*(b) : assert (0))

void print_coords (int nd, int * cc)
{
  printf ("<");
  int ii;
  for (ii = 0; ii < nd - 1; ii++)
    printf ("%d, ", cc[ii]);
  printf ("%d>\n", cc[nd-1]);
}


int compute_max_colors (int nd, int * dims)
{
  if (nd == 2)
  {
    int G0 = dims[0];
    int G1 = dims[1];
    return 1 + G0 + G1 + 1;
  }
  if (nd == 3)
  {
    int G0 = dims[0];
    int G1 = dims[1];
    int G2 = dims[2];
    return 1 + G0 * G1 + G0 * G2 + G1 * G2 + G0 + G1 + G2 + 1;
  }
  if (nd == 1)
  {
    return 1 + 1;
  }
  return -1;
}

void rank_to_coords_3D (int rank, int * dims, int * cc)
{
  int total = dims[0] * dims[1] * dims[2];
  int n01 = dims[0] * dims[1];
  int n02 = dims[0] * dims[2];
  int n12 = dims[1] * dims[2];
  cc[0] = rank / n12;
  cc[1] = (rank / dims[2]) % dims[1];
  cc[2] = rank % dims[2];
}

void rank_to_coords_2D (int rank, int * dims, int * cc)
{
  int total = dims[0] * dims[1];
  cc[0] = rank / dims[1];
  cc[1] = rank % dims[1];
}

void rank_to_coords_1D (int rank, int * dims, int * cc)
{
  cc[0] = rank;
}

void dimage_rank_to_coords (int nd, int rank, int * dims, int * cc)
{
  if (nd == 1)
  {
    rank_to_coords_1D (rank, dims, cc);
    return;
  }
  if (nd == 2)
  {
    rank_to_coords_2D (rank, dims, cc);
    return;
  }
  if (nd == 3)
  {
    rank_to_coords_3D (rank, dims, cc);
    return;
  }
  assert (0 && "Invalid grid dimension for function rank_to_coords");
}


/*
 * @dims: dimensions of the grid.
 * @cc: coordinates of a process in the grid.
 * @cv: 3D communication vector. Each component can only be 0 or 1.
 */
int compute_color_from_comm_vec_3D (int * dims, int * cc, int * cv)
{
  int ret = 0;

  int G0 = dims[0];
  int G1 = dims[1];
  int G2 = dims[2];

  int C0 = cc[0];
  int C1 = cc[1];
  int C2 = cc[2];

  int V0 = cv[0];
  int V1 = cv[1];
  int V2 = cv[2];

  int n01 = G0 * G1;
  int n02 = G0 * G2;
  int n12 = G1 * G2;

  int offset;

  offset = 1;

  // Communication vector along direction <1,0,0>.
  // All processes [*,c1,c2] map to the same communicator.
  // Hence we need to produce G1 * G2 different colors. 
  // After "using" G1 * G2 colors, we advance the offset.
  if (V0 == 1 && V1 == 0 && V2 == 0)
  {
    ret = offset + C1 * G2 + C2;
  }
  offset += n12;

  // Communication vector along direction <0,1,0>.
  // All processes [c0,*,c2] map to the same communicator.
  // Hence we need to produce G0 * G2 different colors. 
  // After "using" G0 * G2 colors, we advance the offset.
  if (V0 == 0 && V1 == 1 && V2 == 0)
  {
    ret = offset + C0 * G2 + C2;
  }
  offset += n02;

  // Communication vector along direction <0,0,1>.
  // All processes [c0,c1,*] map to the same communicator.
  // Hence we need to produce G0 * G1 different colors. 
  // After "using" G0 * G1 colors, we advance the offset.
  if (V0 == 0 && V1 == 0 && V2 == 1)
  {
    ret = offset + C0 * G1 + C1;
  }
  offset += n01;

  // Communication vector along direction <0,1,1>.
  // All processes [c0,*,*] map to the same communicator.
  // Hence we need to produce G0 different colors. 
  // After "using" G0 colors, we advance the offset.
  if (V0 == 0 && V1 == 1 && V2 == 1)
  {
    ret = offset + C0;
  }
  offset += G0;

  // Communication vector along direction <1,1,0>.
  // All processes [*,*,c2] map to the same communicator.
  // Hence we need to produce G2 different colors. 
  // After "using" G2 colors, we advance the offset.
  if (V0 == 1 && V1 == 1 && V2 == 0)
  {
    ret = offset + C2;
  }
  offset += G2;

  // Communication vector along direction <1,0,1>.
  // All processes [*,c1,*] map to the same communicator.
  // Hence we need to produce G1 different colors. 
  // After "using" G1 colors, we advance the offset.
  if (V0 == 1 && V1 == 0 && V2 == 1)
  {
    ret = offset + C1;
  }
  offset += G1;

  // Communication vector along direction <1,1,1>.
  // All processes [*,*,*] map to the same communicator.
  // Hence we need to produce 1 different colors. 
  // After "using" 1 colors, we advance the offset.
  if (V0 == 1 && V1 == 1 && V2 == 1)
  {
    ret = offset;
  }
  offset += 1;

  // Will return 0 if no condition was met.
  return ret;
}


/*
 * @dims: dimensions of the grid.
 * @cc: coordinates of a process in the grid.
 * @cv: 2D communication vector. Each component can only be 0 or 1.
 */
int compute_color_from_comm_vec_2D (int * dims, int * cc, int * cv)
{
  int ret = 0; // Reserve color 0 for no-communication.

  int G0 = dims[0];
  int G1 = dims[1];

  int C0 = cc[0];
  int C1 = cc[1];

  int V0 = cv[0];
  int V1 = cv[1];

  int n01 = G0 * G1;

  int offset;

  offset = 1;

  // Communication vector along direction <1,0>.
  // All processes [*,c1] map to the same communicator.
  // Hence we need to produce G1 different colors. 
  // After "using" G1 colors, we advance the offset.
  if (V0 == 1 && V1 == 0)
  {
    ret = offset + C1;
  }
  offset += G1;

  // Communication vector along direction <0,1>.
  // All processes [c0,*] map to the same communicator.
  // Hence we need to produce G0 different colors. 
  // After "using" G0 colors, we advance the offset.
  if (V0 == 0 && V1 == 1)
  {
    ret = offset + C0;
  }
  offset += G0;

  // Communication vector along direction <1,1>.
  // All processes [*,*] map to the same communicator.
  // Hence we need to produce 1 different colors. 
  // After "using" 1 colors, we advance the offset.
  if (V0 == 1 && V1 == 1)
  {
    ret = offset;
  }
  offset += 1;

  // Will return 0 if no condition was met.
  return ret;
}


/*
 * @dims: dimensions of the grid.
 * @cc: coordinates of a process in the grid.
 * @cv: 1D communication vector. Each component can only be 0 or 1.
 */
int compute_color_from_comm_vec_1D (int * dims, int * cc, int * cv)
{
  int ret = 0;

  int G0 = dims[0];

  int C0 = cc[0];

  int V0 = cv[0];

  int offset;

  offset = 1;

  // Communication vector along direction <1>.
  // All processes [*] map to the same communicator.
  // Hence we need to produce 1 different colors. 
  // After "using" 1 colors, we advance the offset.
  if (V0 == 1)
  {
    ret = offset;
  }
  offset += 1;

  // Will return 0 if no condition was met.
  return ret;
}

int dimage_compute_color_from_comm_vec (int nd, int * dims, int * cc, int * cv)
{
  if (nd == 2)
    return compute_color_from_comm_vec_2D (dims, cc, cv);
  if (nd == 3)
    return compute_color_from_comm_vec_3D (dims, cc, cv);
  if (nd == 1)
    return compute_color_from_comm_vec_1D (dims, cc, cv);
  assert (0 && "Invalid grid dimension for function compute_color_from_comm_vec ");
}

int 
dimage_proc_dim (int dimage_rank, int n_proc_dim, int pdim, int extent1, int extent2)
{
  if (n_proc_dim == 2)
  {
    if (pdim == 0)
      return dimage_rank / extent1;
    else
      return dimage_rank % extent1;
  }
  if (n_proc_dim == 3)
  {
    int plane_extent = (extent1 * extent2);
    if (pdim == 0)
      return dimage_rank / plane_extent;
    if (pdim == 1)
      return (dimage_rank % plane_extent) / extent2;
    if (pdim == 2)
      return dimage_rank % extent2;
  }
  if (n_proc_dim == 1)
  {
    return dimage_rank % extent1;
  }
  assert (0 && "Unsupported processor configuration.");
  return -1;
}


// lcgr = Locally-computed slice-replicated: 
// Only one rank computes on a replicate slice. Need broadcast instead of reduction.
int
dimage_is_lcsr (int dimage_rank, 
  int ** dimage_ranks,
  int * used, // iteration space dimensions appearing in access function. Example: S[i,j,k]->A[i,k] = [0,-1,1,-2] 
  int * imap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * amap, // iteration space processor map. Example S[i,j,k]->P[*,*] = [-1,-1,-1]
  int * commvec)// communication vectors; 0 means no-comm along that dimension
{
  assert (0 && "Do not use. Not fully implemented.");
  return 0;
  int dd;
  for (dd = 0; amap[dd] != -2; dd++)
    if (amap[dd] == -1)
  {
    int ss;
    for (ss = 0; imap[ss] != -2; ss++)
    {
      int adim = imap[ss];
      if (adim >= 0 && adim == dd)
      {
        int sdim = imap[adim];
        if (sdim >= 0 && *dimage_ranks[0] != *dimage_ranks[2])
          return dimage_rank;
      }
    }
  }
  return -1;
}


double *
dimage_tile2d_alloc (int ts1, int ts2)
{
  int ne = ts1 * ts2;
  double * ret = (double*)(malloc (sizeof(double) * ne));
  unsigned long long i0;
  unsigned long long i1;
  for (i0 = 0; i0 < ts1; i0++)
    for (i1 = 0; i1 < ts2; i1++)
      ret[i0 * ts2 + i1] = 0.0;
  return ret;
}


double *
dimage_alloc_buffer (int numitems)
{
  double * ret = (double*)(malloc (sizeof(double) * (numitems)));
  unsigned long long i0;
  for (i0 = 0; i0 < numitems; i0++)
    ret[i0] = 0.0;
  return ret;
}

double *
dimage_1d_tile_alloc (int ts0, int nt0)
{
  double * ret;
  unsigned long long stride = ts0 + DIMAGE_TILE_HEADER_SIZE;
  ret = (double*)(malloc (sizeof(double) * stride * nt0));
  double * ptr = ret;
  unsigned long long i0;
  int t0;
  for (t0 = 0; t0 < nt0; t0++)
  {
    int kk;
    for (kk = 0; kk < DIMAGE_TILE_HEADER_SIZE; kk++)
      ptr[kk] = -1.0;
    for (i0 = 0; i0 < ts0; i0++)
      ptr[DIMAGE_TILE_HEADER_SIZE + i0] = 0.0;
    ptr += stride;
  }
  return ret;
}


double *
dimage_2d_tile_alloc (int ts1, int ts2, int nt0, int nt1)
{
  double * ret;
  unsigned long long stride = ts1 * ts2 + DIMAGE_TILE_HEADER_SIZE;
  ret = (double*)(malloc (sizeof(double) * (stride) * nt0 * nt1));
  double * ptr = ret;
  unsigned long long i0;
  unsigned long long i1;
  int offset = 0;
  int kk;
  int t0, t1;
  for (t0 = 0; t0 < nt0; t0++)
    for (t1 = 0; t1 < nt1; t1++)
  {
    for (kk = 0; kk < DIMAGE_TILE_HEADER_SIZE; kk++)
      ptr[kk] = -1.0;
    for (i0 = 0; i0 < ts1; i0++)
      for (i1 = 0; i1 < ts2; i1++)
        ptr[DIMAGE_TILE_HEADER_SIZE + i0 * ts2 + i1] = 0.0;
    ptr += stride;
  }
  return ret;
}

double *
dimage_3d_tile_alloc (int ts0, int ts1, int ts2, int nt0, int nt1, int nt2)
{
  double * ret;
  unsigned long long stride = ts0 * ts1 * ts2 + DIMAGE_TILE_HEADER_SIZE;
  ret = (double*)(malloc (sizeof(double) * (stride) * nt0 * nt1 * nt2));
  double * ptr = ret;
  unsigned long long i0;
  unsigned long long i1;
  unsigned long long i2;
  int t0;
  int t1;
  int t2;
  for (t0 = 0; t0 < nt0; t0++)
    for (t1 = 0; t1 < nt1; t1++)
      for (t2 = 0; t2 < nt2; t2++)
  {
    int kk;
    for (kk = 0; kk < DIMAGE_TILE_HEADER_SIZE; kk++)
      ptr[kk] = -1.0;
    for (i0 = 0; i0 < ts0; i0++)
      for (i1 = 0; i1 < ts1; i1++)
        for (i2 = 0; i2 < ts2; i2++)
          ptr[DIMAGE_TILE_HEADER_SIZE + i0 * ts1 * ts2 + i1 * ts2 + i2] = 0.0;
    ptr += stride;
  }
  return ret;
}

void
dimage_2d_free (double ** dimage_array, int pbs, int tile_size)
{
  int ii;
  for (ii = 0; ii < dimage_div(pbs, tile_size); ii++)
    free (dimage_array[ii]);
  free (dimage_array);
}

double ** dimage_2d_alloc (int pbs1, int pbs2, int ts1, int ts2)
{
  int i1, i2;
  int nt1 = dimage_div(pbs1, ts1);
  int nt2 = dimage_div(pbs2, ts2);
  int outer_nt1_nt2 = nt1 * nt2;
  int inner_i1_i2 = ts1 * ts2;
  double ** ret;
  ret = (double**)(malloc (sizeof(double*)*(outer_nt1_nt2)));
  for (i1 = 0; i1 < outer_nt1_nt2; i1++)
  {
    ret[i1] = (double*)(malloc (sizeof(double)*(inner_i1_i2)));
  }
  return ret;
}

void init_log_file_1D (int pd0)
{
  #ifdef DIMAGE_LOG
  sprintf (logfile, "phases_p_%d", pd0);
  FILE * ff  = fopen (logfile, "w");
  fclose (ff);
  #endif
}

void init_log_file_1D_with_rank (int pd0, int dimage_rank)
{
  #ifdef DIMAGE_LOG
  sprintf (logfile, "phases_p_%d", pd0);
  FILE * ff  = fopen (logfile, "w");
  fprintf (ff, "Rank: %d: [%d]\n", dimage_rank, pd0);
  fclose (ff);
  #endif
}

void init_log_file_2D (int pd0, int pd1)
{
  #ifdef DIMAGE_LOG
  sprintf (logfile, "phases_p_%d_%d", pd0, pd1);
  FILE * ff  = fopen (logfile, "w");
  fclose (ff);
  #endif
}

void init_log_file_2D_with_rank (int pd0, int pd1, int dimage_rank)
{
  #ifdef DIMAGE_LOG
  sprintf (logfile, "phases_p_%d_%d", pd0, pd1);
  FILE * ff  = fopen (logfile, "w");
  fprintf (ff, "Rank: %d: [%d,%d]\n", dimage_rank, pd0, pd1);
  fclose (ff);
  #endif
}

void init_log_file_3D (int pd0, int pd1, int pd2)
{
  #ifdef DIMAGE_LOG
  sprintf (logfile, "phases_p_%d_%d_%d", pd0, pd1, pd2);
  FILE * ff  = fopen (logfile, "w");
  fclose (ff);
  #endif
}

void init_log_file_3D_with_rank (int pd0, int pd1, int pd2, int dimage_rank)
{
  #ifdef DIMAGE_LOG
  sprintf (logfile, "phases_p_%d_%d_%d", pd0, pd1, pd2);
  FILE * ff  = fopen (logfile, "w");
  fprintf (ff, "Rank: %d: [%d,%d,%d]\n", dimage_rank, pd0, pd1, pd2);
  fclose (ff);
  #endif
}

void
dump_slice (const char * name, int pd0, int pd1, double * mat, int nrows, int ncols)
{
  int i1, i2;
  char filename[64];
  sprintf (filename, "%s_%d_%d", name, pd0, pd1);
  FILE * ff = fopen (filename, "w");
  for (i1 = 0; i1 < nrows; i1++)
  {
    for (i2 = 0; i2 < ncols; i2++)
    {
      fprintf (ff, "%lf ", mat[i1 * ncols + i2]);
    }
    fprintf (ff,"\n");
  }
}

void
write_to_file_tile1D (const char * name, int ** ranks, double * mat, int s0, int nt0)
{
  #ifdef DIMAGE_LOG
  int i0;
  char filename[64];
  for (i0 = 0; ranks[i0]; i0++);
  int grid_dim = i0;
  if (grid_dim == 1)
    sprintf (filename, "%s_%d.data", name, *ranks[0]);
  if (grid_dim == 2)
    sprintf (filename, "%s_%d_%d.data", name, *ranks[0], *ranks[1]);
  if (grid_dim == 3)
    sprintf (filename, "%s_%d_%d_%d.data", name, *ranks[0], *ranks[1], *ranks[2]);
  FILE * ff = fopen (filename, "w");
  int t0;
  int block_stride = s0 + DIMAGE_TILE_HEADER_SIZE;
  int coor;
  double * ptr = mat;
  for (t0 = 0; t0 < nt0; t0++)
  {
    for (coor = 0; coor < 1; coor++)
      fprintf (ff, "%0.lf ", ptr[coor]);
    fprintf (ff, "\n");
    for (i0 = 0; i0 < s0; i0++)
    {
      fprintf (ff, "%lf ", ptr[DIMAGE_TILE_HEADER_SIZE + i0]);
    }
    fprintf (ff,"\n");
    ptr += block_stride;
  }
  fclose (ff);
  #endif
}

void
write_to_file_tile2D (const char * name, int ** ranks, double * mat, int s0, int s1, int nt0, int nt1)
{
  #ifdef DIMAGE_LOG
  int i0, i1;
  char filename[64];
  for (i0 = 0; ranks[i0]; i0++);
  int grid_dim = i0;
  if (grid_dim == 1)
    sprintf (filename, "%s_%d.data", name, *ranks[0]);
  if (grid_dim == 2)
    sprintf (filename, "%s_%d_%d.data", name, *ranks[0], *ranks[1]);
  if (grid_dim == 3)
    sprintf (filename, "%s_%d_%d_%d.data", name, *ranks[0], *ranks[1], *ranks[2]);
  FILE * ff = fopen (filename, "w");
  int offset = DIMAGE_TILE_HEADER_SIZE;
  int t0, t1;
  int coor;
  double * ptr = mat;
  int block_stride = s0 * s1 + DIMAGE_TILE_HEADER_SIZE;
  for (t0 = 0; t0 < nt0; t0++)
    for (t1 = 0; t1 < nt1; t1++)
  {
    for (coor = 0; coor < 2; coor++)
      fprintf (ff, "%0.lf ", ptr[coor]);
    fprintf (ff, "\n");
    for (i0 = 0; i0 < s0; i0++)
    {
      for (i1 = 0; i1 < s1; i1++)
      {
        fprintf (ff, "%lf ", ptr[offset + i0 * s1 + i1]);
      }
      fprintf (ff,"\n");
    }
    ptr += block_stride;
  }
  fclose (ff);
  #endif
}

void
write_to_file_tile3D (const char * name, int ** ranks, double * mat, int s0, int s1, int s2, int nt0, int nt1, int nt2)
{
  #ifdef DIMAGE_LOG
  int i0, i1, i2;
  char filename[64];
  for (i0 = 0; ranks[i0]; i0++);
  int grid_dim = i0;
  if (grid_dim == 1)
    sprintf (filename, "%s_%d.data", name, *ranks[0]);
  if (grid_dim == 2)
    sprintf (filename, "%s_%d_%d.data", name, *ranks[0], *ranks[1]);
  if (grid_dim == 3)
    sprintf (filename, "%s_%d_%d_%d.data", name, *ranks[0], *ranks[1], *ranks[2]);
  FILE * ff = fopen (filename, "w");
  int block_stride = s0 * s1 * s2 + DIMAGE_TILE_HEADER_SIZE;
  int offset = DIMAGE_TILE_HEADER_SIZE;
  double * ptr = mat;
  int t0, t1, t2;
  for (t0 = 0; t0 < nt0; t0++)
    for (t1 = 0; t1 < nt1; t1++)
      for (t2 = 0; t2 < nt2; t2++)
  {
    int coor;
    for (coor = 0; coor < 3; coor++)
      fprintf (ff, "%0.lf ", ptr[coor]);
    fprintf (ff, "\n");
    for (i0 = 0; i0 < s0; i0++)
    {
      for (i1 = 0; i1 < s1; i1++)
      {
        for (i2 = 0; i2 < s2; i2++)
        {
          fprintf (ff, "%lf ", ptr[offset + i0 * (s1 * s2) + i1 * s2 + i2]);
        }
        fprintf (ff,"\n");
      }
      fprintf (ff,"\n");
    }
    ptr += block_stride;
  }
  fclose (ff);
  #endif
}


void
read_from_file_tile1D (const char * filename, double * mat, int n0, int s0, int t0, int * block_count)
{
  int i0;
  FILE * ff = NULL;
  #ifdef INIT_MAT
  ff = fopen (filename, "r");
  #endif
  int lb0 = t0 * s0;
  int ub0 = (t0 + 1) * s0 - 1;
  if (s0 == n0)
  {
    lb0 = 0;
    ub0 = n0 - 1;
  }
  int count = 0;
  assert (n0 % s0 == 0);
  int offset = DIMAGE_TILE_HEADER_SIZE;
  double * tmp = (mat + (*block_count) * (s0 + offset));
  tmp[0] = (double)(t0);
  #ifdef INIT_MAT
  for (i0 = 0; i0 < n0; i0++)
  {
    double val = 1.0;
    int nb;
    if (ff)
      nb = fscanf (ff, "%lf ", &val);
    if (i0 >= lb0 && i0 <= ub0)
      tmp[offset + count++] = val;
  }
  #endif
  (*block_count) += 1;
  if (ff)
    fclose (ff);
}

/*
 * Read a single tile at coordinates t0, t1, ... from a single linearized matrix.
 *
 */
void
read_from_file_tile2D (const char * filename, double * mat, int n0, int n1, int s0, int s1, int t0, int t1, int * block_count)
{
  int i0, i1;
  FILE * ff = NULL;
  #ifdef INIT_MAT
  ff = fopen (filename, "r");
  #endif
  int lb0 = t0 * s0;
  int ub0 = (t0 + 1) * s0 - 1;
  if (s0 == n0)
  {
    lb0 = 0;
    ub0 = n0 - 1;
  }
  int lb1 = t1 * s1;
  int ub1 = (t1 + 1) * s1 - 1;
  if (s1 == n1)
  {
    lb1 = 0;
    ub1 = n1 - 1;
  }
  int count = 0;
  assert (n1 % s1 == 0);
  int offset = DIMAGE_TILE_HEADER_SIZE;
  double * tmp = (mat + (*block_count) * (s0*s1 + offset));
  tmp[0] = (double)(t0);
  tmp[1] = (double)(t1);
  #ifdef INIT_MAT
  for (i0 = 0; i0 < n0; i0++)
  {
    for (i1 = 0; i1 < n1; i1++)
    {
      double val = (i0 + 0.5) / (i1 + 1.1);
      int nb;
      if (ff)
        nb = fscanf (ff, "%lf ", &val);
      if (i0 >= lb0 && i0 <= ub0 && i1 >= lb1 && i1 <= ub1)
        tmp[offset + count++] = val;
    }
  }
  #endif
  (*block_count) += 1;
  if (ff)
    fclose (ff);
}


void
read_from_file_tile3D (const char * filename, double * mat, int n0, int n1, int n2, int s0, int s1, int s2, int t0, int t1, int t2, int *block_count)
{
  int i0, i1, i2;
  FILE * ff = NULL;
  #ifdef INIT_MAT
  ff = fopen (filename, "r");
  #endif 
  int lb0 = t0 * s0;
  int ub0 = (t0 + 1) * s0 - 1;
  if (s0 == n0)
  {
    lb0 = 0;
    ub0 = n0 - 1;
  }
  int lb1 = t1 * s1;
  int ub1 = (t1 + 1) * s1 - 1;
  if (s1 == n1)
  {
    lb1 = 0;
    ub1 = n1 - 1;
  }
  int lb2 = t2 * s2;
  int ub2 = (t2 + 1) * s2 - 1;
  if (s2 == n2)
  {
    lb2 = 0;
    ub2 = n2 - 1;
  }
  int count = 0;
  assert (n0 % s0 == 0);
  assert (n1 % s1 == 0);
  assert (n2 % s2 == 0);
  unsigned long long offset = DIMAGE_TILE_HEADER_SIZE;
  double * tmp = (mat + (*block_count) * (s0*s1*s2 + offset));
  tmp[0] = (double)(t0);
  tmp[1] = (double)(t1);
  tmp[2] = (double)(t2);
  #ifdef INIT_MAT
  for (i0 = 0; i0 < n0; i0++)
  {
    for (i1 = 0; i1 < n1; i1++)
    {
      for (i2 = 0; i2 < n2; i2++)
      {
        double val = (i1 + 0.5 + i2) / (i0 + 1.1);
        int nb;
        if (ff)
          nb = fscanf (ff, "%lf ", &val);
        if (i0 >= lb0 && i0 <= ub0 && i1 >= lb1 && i1 <= ub1 && i2 >= lb2 && i2 <= ub2)
          tmp[offset + count++] = val;
      }
    }
  }
  #endif
  (*block_count) += 1;
  if (ff)
    fclose (ff);
}


// Routine used in dimage-op.c files.
double *
read_matrix_from_file (const char * filename, unsigned long long nitems)
{
  unsigned long long ii;
  FILE * ff = fopen (filename, "r");
  if (!ff)
    return NULL;
  double * ret = dimage_alloc_buffer (nitems);
  if (!ret)
    return NULL;
  for (ii = 0; ii < nitems; ii++)
  {
    int nb = fscanf (ff, "%lf", &ret[ii]);
  }
  fclose (ff);
  return ret;
}

void
write_to_file_matrix_1D (const char * filename, double * mat, unsigned long long nitems)
{
  unsigned long long ii;
  FILE * ff = fopen (filename, "w");
  if (!ff)
    return;
  if (!mat)
  {
    fclose (ff);
    return;
  }
  for (ii = 0; ii < nitems; ii++)
  {
    int nb = fprintf (ff, "%.6lf ", mat[ii]);
  }
  fclose (ff);
}

void
write_to_file_matrix_2D 
(const char * filename, double * mat, unsigned long long n1, unsigned long long n2)
{
  unsigned long long ii;
  unsigned long long jj;
  FILE * ff = fopen (filename, "w");
  if (!ff)
    return;
  if (!mat)
  {
    fclose (ff);
    return;
  }
  #ifdef INIT_MAT
  unsigned long long offset = DIMAGE_TILE_HEADER_SIZE;
  for (ii = 0; ii < n1; ii++)
  {
    for (jj = 0; jj < n2; jj++)
    {
      int nb = fprintf (ff, "%.6lf ", mat[ii * n2 + jj]);
    }
    fprintf (ff, "\n");
  }
  #endif
  fclose (ff);
}

void
write_to_file_matrix_3D (const char * filename, double * mat, unsigned long long n1, unsigned long long n2, unsigned long long n3)
{
  unsigned long long ii;
  unsigned long long jj;
  unsigned long long kk;
  FILE * ff = fopen (filename, "w");
  if (!ff)
    return;
  if (!mat)
  {
    fclose (ff);
    return;
  }
  #ifdef INIT_MAT
  for (ii = 0; ii < n1; ii++)
  {
    for (jj = 0; jj < n2; jj++)
    {
      for (kk = 0; kk < n3; kk++)
      {
        int nb = fprintf (ff, "%.6lf ", mat[ii * n2 * n3 + jj * n3 + kk]);
      }
      fprintf (ff, "\n");
    }
    fprintf (ff, "\n");
  }
  #endif
  fclose (ff);
}


/*
 * Generate a datafile for a 1D array.
 */
void
generate_datafile_1D (const char * filename, unsigned long long n1, double given)
{
  unsigned long long ii;
  FILE * ff = fopen (filename, "w");
  if (!ff)
    return;
  for (ii = 0; ii < n1; ii++)
  {
    #ifndef USE_INIT_DIAGONAL
    double val = 1.0;
    #else
    double val = 1.0;
    #endif
    int nb = fprintf (ff, "%.6lf ", val);
  }
  fclose (ff);
}


/*
 * Generate a datafile for a 2D matrix.
 */
void
generate_datafile_2D (const char * filename, unsigned long long n1, unsigned long long n2, double given)
{
  unsigned long long ii;
  unsigned long long jj;
  FILE * ff = fopen (filename, "w");
  if (!ff)
    return;
  for (ii = 0; ii < n1; ii++)
  {
    for (jj = 0; jj < n2; jj++)
    {
      int ti = ii / 4;
      int tj = jj / 4;
      int block = ti * 2 + tj;
      #ifndef USE_INIT_DIAGONAL
      double val = block * given;
      #else
      double val = (ii == jj ? given : 0.0);
      #endif
      int nb = fprintf (ff, "%.6lf ", val);
    }
    fprintf (ff, "\n");
  }
  fclose (ff);
}

/*
 * Generate a datafile for a 3D matrix.
 */
void
generate_datafile_3D (const char * filename, unsigned long long n1, unsigned long long n2, unsigned long long n3, double given)
{
  unsigned long long ii;
  unsigned long long jj;
  unsigned long long kk;
  FILE * ff = fopen (filename, "w");
  if (!ff)
    return;
  for (ii = 0; ii < n1; ii++)
  {
    for (jj = 0; jj < n2; jj++)
    {
      for (kk = 0; kk < n3; kk++)
      {
        int ti = ii / 4;
        int tj = jj / 4;
        int tk = kk / 4;
        int block = ti * 4 + tj * 2 + tk;
        #ifndef USE_INIT_DIAGONAL
        double val = block * given;
        #else
        double val = (jj == kk ? given : 0.0);
        #endif
        int nb = fprintf (ff, "%.6lf ", val);
      }
      fprintf (ff, "\n");
    }
    fprintf (ff, "\n");
  }
  fclose (ff);
}





#define CHECK_FIXED
void
check_array_tile1D (const char * filename, double * mat_dist, double * mat_ref, int n0, int s0, int t0, int * block_count)
{
  #if defined(INIT_MAT) && defined(CHECK_FIXED)
  unsigned long long i0;
  char fullfilename[64];
  sprintf (fullfilename, "%s_%d_check.mat", filename, t0);
  FILE * ff = fopen (fullfilename, "a");
  unsigned long long count = 0;
  assert (n0 % s0 == 0);
  int offset = DIMAGE_TILE_HEADER_SIZE;
  int stride = s0 + DIMAGE_TILE_HEADER_SIZE;
  int block_stride = stride * (*block_count);
  double * tmp1 = (mat_dist);
  double * tmp2 = (mat_ref);
  double diffs = 0.0;
  for (i0 = 0; i0 < s0; i0++)
  {
    double dd;
    dd = tmp1[count];
    dd -= tmp2[count];
    diffs += (dd >= 0 ? dd : -dd);
    count++;
  }
  assert (ff && "File not open. Aborting ...");
  if (diffs < 0.0000001)
    fprintf (ff,"block (%d): %lf (PASS)\n",t0,diffs);
  else
    fprintf (ff,"block (%d): %lf (FAIL)\n",t0,diffs);
  (*block_count) += 1;
  fclose (ff);
  #endif
}

void
check_array_tile2D (const char * filename, double * mat_dist, double * mat_ref, int n0, int n1, int s0, int s1, int t0, int t1, int * block_count)
{
  #if defined(INIT_MAT) && defined(CHECK_FIXED)
  int i0, i1;
  char fullfilename[64];
  sprintf (fullfilename, "%s_%d_%d_check.mat", filename, t0, t1);
  FILE * ff = fopen (fullfilename, "w");
  unsigned long long count = 0;
  assert (n0 % s0 == 0);
  assert (n1 % s1 == 0);
  int offset = DIMAGE_TILE_HEADER_SIZE;
  int stride = s0 * s1 + DIMAGE_TILE_HEADER_SIZE;
  int block_stride = stride * (*block_count);
  double * tmp1 = (mat_dist );
  double * tmp2 = (mat_ref );
  double diffs = 0.0;
  for (i0 = 0; i0 < s0; i0++)
  {
    for (i1 = 0; i1 < s1; i1++)
    {
      double dd;
      dd = tmp1[count];
      dd -= tmp2[count];
      diffs += (dd >= 0 ? dd : -dd);
      count++;
    }
  }
  assert (ff && "File not open. Aborting ...");
  if (diffs < 0.0000001)
    fprintf (ff,"block (%d,%d): %lf (PASS)\n",t0,t1,diffs);
  else
    fprintf (ff,"block (%d,%d): %lf (FAIL)\n",t0,t1,diffs);
  (*block_count) += 1;
  fclose (ff);
  #endif
}


void
check_array_tile3D (const char * filename, double * mat_dist, double * mat_ref, 
  int n0, int n1, int n2, int s0, int s1, int s2, int t0, int t1, int t2, int * block_count)
{
  #if defined(INIT_MAT) && defined(CHECK_FIXED)
  int i0, i1, i2;
  char fullfilename[64];
  sprintf (fullfilename, "%s_%d_%d_%d_check.mat", filename, t0, t1, t2);
  FILE * ff = fopen (fullfilename, "a");
  unsigned long long count = 0;
  assert (n0 % s0 == 0);
  assert (n1 % s1 == 0);
  assert (n2 % s2 == 0);
  int offset = DIMAGE_TILE_HEADER_SIZE;
  int stride = s0 * s1 * s2 + DIMAGE_TILE_HEADER_SIZE;
  int block_stride = stride * (*block_count);
  double * tmp1 = (mat_dist);
  double * tmp2 = (mat_ref);
  double diffs = 0.0;
  for (i0 = 0; i0 < s0; i0++)
  {
    for (i1 = 0; i1 < s1; i1++)
    {
      for (i2 = 0; i2 < s2; i2++)
      {
        double dd;
        dd = tmp1[count];
        dd -= tmp2[count];
        diffs += (dd >= 0 ? dd : -dd);
        count++;
      }
    }
  }
  assert (ff && "File not open. Aborting ...");
  if (diffs < 0.0000001)
    fprintf (ff,"block (%d,%d,%d): %lf (PASS)\n",t0,t1,t2,diffs);
  else
    fprintf (ff,"block (%d,%d,%d): %lf (FAIL)\n",t0,t1,t2,diffs);
  (*block_count) += 1;
  fclose (ff);
  #endif
}


void 
dimage_collect_tile_coordinates_1D (double * mat, int * tile_map, int ss1, int maxnt1, int nt1)
{
  int ii;
  int block_stride = ss1 + DIMAGE_TILE_HEADER_SIZE;
  int stride = maxnt1;
  double * ptr = mat;
  int loc = 0;
  for (ii = 0; ii < nt1; ii++)
  {
    int t0 = (int)ptr[0];
    tile_map[t0] = loc;
    ptr = (double*)(ptr + block_stride);
    loc++;
  }
}

void 
dimage_collect_tile_coordinates_2D (double * mat, int * tile_map, int ss1, int ss2, int maxnt1, int maxnt2, int nt1, int nt2)
{
  int ii, jj;
  int block_stride = ss1 * ss2 + DIMAGE_TILE_HEADER_SIZE;
  int stride = maxnt2;
  double * ptr = mat;
  int loc = 0;
  for (ii = 0; ii < nt1; ii++)
    for (jj = 0; jj < nt2; jj++)
  {
    int t0 = (int)ptr[0];
    int t1 = (int)ptr[1];
    tile_map[t0 * stride + t1] = loc;
    ptr = (double*)(ptr + block_stride);
    loc++;
  }
}

void 
dimage_collect_tile_coordinates_3D (double * mat, int * tile_map, int ss1, int ss2, int ss3, int maxnt1, int maxnt2, int maxnt3, int nt1, int nt2, int nt3)
{
  int ii, jj, kk;
  int block_stride = ss1 * ss2 * ss3 + DIMAGE_TILE_HEADER_SIZE;
  int stride1 = maxnt2 * maxnt3;
  int stride2 = maxnt3;
  double * ptr = mat;
  int loc = 0;
  for (ii = 0; ii < nt1; ii++)
    for (jj = 0; jj < nt2; jj++)
      for (kk = 0; kk < nt3; kk++)
  {
    int t0 = (int)ptr[0];
    int t1 = (int)ptr[1];
    int t2 = (int)ptr[2];
    tile_map[t0 * stride1 + t1 * stride2 + t2] = loc;
    ptr = (double*)(ptr + block_stride);
    loc++;
  }
}

int * 
dimage_alloc_tile_map_1D (int np0)
{
  int * ret = (int*)(malloc (sizeof(int) * np0));
  int ii;
  int count = 0;
  for (ii = 0; ii < np0; ii++)
  {
    ret[count++] = -1.0;
  }
  return ret;
}

int * 
dimage_alloc_tile_map_2D (int np0, int np1)
{
  int * ret = (int*)(malloc (sizeof(int) * np0 * np1));
  int ii;
  int jj;
  int count = 0;
  for (ii = 0; ii < np0; ii++)
    for (jj = 0; jj < np1; jj++)
  {
    ret[count++] = -1.0;
  }
  return ret;
}

int * 
dimage_alloc_tile_map_3D (int np0, int np1, int np2)
{
  int nelem = sizeof(int) * np0 * np1 * np2;
  int * ret = (int*)(malloc (nelem));
  int i0;
  int i1;
  int i2;
  int count = 0;
  for (i0 = 0; i0 < np0; i0++)
    for (i1 = 0; i1 < np1; i1++)
      for (i2 = 0; i2 < np2; i2++)
  {
    ret[count++] = -1.0;
  }
  return ret;
}

double * 
dimage_fetch_tile_ptr_1D (double * mat, int * tile_map, int ss0, int t0, int nt0)
{
  int stride = ss0 + DIMAGE_TILE_HEADER_SIZE;
  int idx = (int)(tile_map [t0]);
  return (mat + stride * idx + DIMAGE_TILE_HEADER_SIZE);
}

double * 
dimage_fetch_tile_ptr_2D (double * mat, int * tile_map, int ss0, int ss1, int t0, int t1, int nt0, int nt1)
{
  int stride = ss0 * ss1 + DIMAGE_TILE_HEADER_SIZE;
  int idx = (int)(tile_map [t0 * nt1 + t1]);
  return (mat + stride * idx + DIMAGE_TILE_HEADER_SIZE);
}

double * 
dimage_fetch_tile_ptr_3D (double * mat, int * tile_map, int ss0, int ss1, int ss2, int t0, int t1, int t2, int nt0, int nt1, int nt2)
{
  int stride = ss0 * ss1 * ss2 + DIMAGE_TILE_HEADER_SIZE;
  int idx = (int)(tile_map [t0 * nt1 * nt2 + t1 * nt2 + t2]);
  return (mat + stride * idx + DIMAGE_TILE_HEADER_SIZE);
}

void
dimage_store_tile_map_1D (const char * prefix_name, int ** ranks, int * map, int nt0)
{
  char filename[64];
  int i0;
  for (i0 = 0; ranks[i0]; i0++);
  int grid_dim = i0;
  if (grid_dim == 1)
    sprintf (filename, "%s_%d.tmap", prefix_name, *ranks[0]);
  if (grid_dim == 2)
    sprintf (filename, "%s_%d_%d.tmap", prefix_name, *ranks[0], *ranks[1]);
  if (grid_dim == 3)
    sprintf (filename, "%s_%d_%d_%d.tmap", prefix_name, *ranks[0], *ranks[1], *ranks[2]);
  FILE * ff = fopen (filename, "w");
  for (i0 = 0; i0 < nt0; i0++)
  {
    fprintf (ff, "%d: %d\n", i0, map[i0]);
  }
  fclose (ff);
}

void
dimage_store_tile_map_2D (const char * prefix_name, int ** ranks, int * map, int nt0, int nt1)
{
  char filename[64];
  int i0;
  int i1;
  for (i0 = 0; ranks[i0]; i0++);
  int grid_dim = i0;
  if (grid_dim == 1)
    sprintf (filename, "%s_%d.tmap", prefix_name, *ranks[0]);
  if (grid_dim == 2)
    sprintf (filename, "%s_%d_%d.tmap", prefix_name, *ranks[0], *ranks[1]);
  if (grid_dim == 3)
    sprintf (filename, "%s_%d_%d_%d.tmap", prefix_name, *ranks[0], *ranks[1], *ranks[2]);
  FILE * ff = fopen (filename, "w");
  for (i0 = 0; i0 < nt0; i0++)
    for (i1 = 0; i1 < nt1; i1++)
  {
    fprintf (ff, "%d %d: %d\n", i0, i1, map[i0 * nt1 + i1]);
  }
  fclose (ff);
}

void
dimage_store_tile_map_3D (const char * prefix_name, int ** ranks, int * map, int nt0, int nt1, int nt2)
{
  char filename[64];
  int i0;
  int i1;
  int i2;
  for (i0 = 0; ranks[i0]; i0++);
  int grid_dim = i0;
  if (grid_dim == 1)
    sprintf (filename, "%s_%d.tmap", prefix_name, *ranks[0]);
  if (grid_dim == 2)
    sprintf (filename, "%s_%d_%d.tmap", prefix_name, *ranks[0], *ranks[1]);
  if (grid_dim == 3)
    sprintf (filename, "%s_%d_%d_%d.tmap", prefix_name, *ranks[0], *ranks[1], *ranks[2]);
  FILE * ff = fopen (filename, "w");
  for (i0 = 0; i0 < nt0; i0++)
    for (i1 = 0; i1 < nt1; i1++)
      for (i2 = 0; i2 < nt2; i2++)
  {
    fprintf (ff, "%d %d %d: %d\n", i0, i1, i2, map[i0 * nt1 * nt2 + i1 * nt2 + i2]);
  }
  fclose (ff);
}

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


void 
dimage_gemm (double * mat_out, double * mat_in, double * mat_ker, int ni, int nj, int nk, double alpha, double beta)
{
  int ii, jj, kk;
  #ifndef DIMAGE_MKL
  #pragma omp parallel for private(kk,jj)
  for (ii = 0; ii < ni; ii++)
  {
    for (kk = 0; kk < nk; kk++)   
      for (jj = 0; jj < nj; jj++)   
    {
      mat_out[ii * nj + jj] += mat_in[ii * nk + kk] * mat_ker[kk * nj + jj];
    }
  }
  #else
  cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, ni, nj, nk, 1.0, mat_in, nk, mat_ker, nj, 1.0, mat_out, nj);
  #endif
}
