## DiMage: The DIstributed MApping GEnerator.
## Authors: Martin Kong, Raneem Abu Yosef, Atanas Rountev, P. Sadayappan
## Maintainer: Martin Kong.
## Copyright 2023. Ohio State University.

import sys 
import re
import os
import math
import signal
from timeit import default_timer as timer
from fractions import gcd

DIMAGE_MAX_TRIES=1 #3
DIMAGE_DEBUG=False
DO_REF=True
DIMAGE_PY_SCRIPT="dimage.py"
MEM2COMP_RATIO = 80
## Tweak variable below to accommodate possible internal local memory used.
DIMAGE_CAP_FACTOR = 1
# NOTE: Tweak the option below to switch between fixed sized grids and 
# parametric grids.
DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY=False
DIMAGE_EXCLUDE_CROSSDIM_MAP_SOLUTIONS=True
DIMAGE_EXCLUDE_CROSSDIM_MAP_SOLUTIONS=False
DIMAGE_FAST_SOL=False
DIMAGE_OPTION_DO_CHECK=True
DIMAGE_USE_RHO_REPLICATION_FACTOR=False
DIMAGE_OPTION_DEBUG=0   # Higher is more verbose.

PER_DIM = 1
PER_PROC = 2
USE_MODULO = False
DIM_UNMAPPED = -1
DIM_NOT_USED=-2
DIMAGE_INT = 'int'
DIMAGE_DT = 'double'
DIMAGE_COMPUTE_COLOR_FUNC = 'dimage_compute_color_from_comm_vec'
DIMAGE_GRID_DIMS = 'DIMAGE_GRID_DIMS'
DIMAGE_TILE_HEADER_SIZE=8
DIMAGE_TILE_HEADER_MACRO='DIMAGE_TILE_HEADER_SIZE'
DIMAGE_BUFFER_ALLOCATOR='dimage_alloc_buffer'
DIMAGE_TILE_ALLOCATOR_1D='dimage_1d_tile_alloc'
DIMAGE_TILE_ALLOCATOR_2D='dimage_2d_tile_alloc'
DIMAGE_TILE_ALLOCATOR_3D='dimage_3d_tile_alloc'
DIMAGE_TILE_ALLOCATOR_4D='dimage_4d_tile_alloc'
DIMAGE_TILE_MAP_ALLOCATOR='dimage_alloc_tile_map'
DIMAGE_COLLECT_TILE_MAP_FUNC='dimage_collect_tile_coordinates'
DIMAGE_STORE_TILE_MAP_FUNC='dimage_store_tile_map'
DIMAGE_FETCH_TILE_FUNC='dimage_fetch_tile_ptr'
DIMAGE_SET_TILE_COORD_FUNC='DIMAGE_SET_TILE_COORDINATE'
ALLOC_MODE_FULL=0
ALLOC_MODE_SLICE=1
ALLOC_MODE_TILE=2
DIMAGE_CEIL_DEF='#define dimage_ceil(n,d)  ceil((((double)(n))*(DIMAGE_SF))/((double)(d)))'
DIMAGE_CEIL='aceil'
DIMAGE_SF_DEF="#define DIMAGE_SF 1"
DIMAGE_SF='DIMAGE_SF'
BASE_INDENT='  '
COMM_TYPE_LOCAL=0
COMM_TYPE_LOCAL_SLICE=1
COMM_TYPE_GATHER_SLICE=2
COMM_TYPE_ALLRED=3
COMM_TYPE_P2P=4
COLLECTIVE_ALLGATHER='MPI_Allgather'
COLLECTIVE_ALLREDUCE='MPI_Allreduce'
DIMAGE_PROC_COORD_FUNC='dimage_rank_to_coords'
ACC_TYPE_TILE=0
ACC_TYPE_SLICE=1
ACC_TYPE_LIN=2
ACC_TYPE_ERROR=-42
L2_LOOP_GENMODE_FULL=0
L2_LOOP_GENMODE_LB=1
L2_LOOP_GENMODE_UB=2
DIMAGE_KERNEL_FUNCALL='dimage_gemm'
DIMAGE_ACC_LIN='DIMAGE_ACC_LIN'
DIMAGE_ACC_TILE='DIMAGE_ACC_TILE'
DIMAGE_ACC_SLICE='DIMAGE_ACC_SLICE'
DIMAGE_TILE_POINTER='DIMAGE_PTR_TILE'
WRITE_TO_FILE_FUNC='write_to_file'
READ_FROM_FILE_FUNC='read_from_file'
WRITE_MATRIX_TO_FILE='write_to_file_matrix'
ARRAY_CHECK_FUNC='check_array'
DIMAGE_INIT_DIAG='DIMAGE_INIT_DIAG'
REDUCE_OP_ADD='MPI_SUM'
DIMAGE_PROC_RANK='dimage_rank'
DIMAGE_RANK_ARRAY='dimage_ranks'
DIMAGE_PROC_COORDS='dimage_coords'
DIMAGE_CLOCK='rtclock()'
DIMAGE_START_TIMER='timer_start'
COMM_SIZE_VAR='procs_in_comm'
DIMAGE_BLOCK_COUNT='block_count'
DIMAGE_REFBLOCK_COUNT='ref_block_count'
MPI_COMM_SIZE='MPI_Comm_size'
DIMAGE_TIMEOUT=180
DIMAGE_OBJ_COMM_ONLY=1
DIMAGE_OBJ_COMM_COMP=2
DIMAGE_CHECK_NO_CHECK=0
DIMAGE_CHECK_READ_REF_ARRAY=1
DIMAGE_CHECK_CALL_CHECK=2
DEBUG_BLOCK_SIZE_OP_TEN_DIM=False
DEBUG_REF_USED_DIM=False

def timeout_message (signum, frame):
  print ("Solver timed-out ({} seconds)".format (DIMAGE_TIMEOUT))
  raise exception ("[DIMAGE:TIMEOUT]")
#
#signal.signal (signal.SIGALRM, timeout_message)
#signal.alarm (DIMAGE_TIMEOUT)


## Methods scaled with DIMAGE_SF:
# - stmt.get_slice_vol_by_name (ref, PP)
# - ref.reference_get_local_volume (self, stmt)
# - ref.get_dimension_size_as_val (self, stmt, dd, PP)
# - stmt.collect_tile_trip_counts (producers)
# - ref.get_extent_as_str (self, dd)
# - stmt.build_loop_structure (self,

def prod (ll):
  ret = 1
  for xx in ll:
    ret = ret * xx
  return ret

# Return False if the per_node capacity is 0, 0K, 0k, 0M or 0m,
# and True otherwise.
def include_capacity_constraints (per_node_cap):
  pnc = re.sub ('[KkMm]','', per_node_cap)
  if (pnc.find ("0") == 0):
    return False
  return True

def iceil(num,den):
  return int(math.ceil(num/(1.0*den)))

def comm_type_str(ct):
  if (ct == COMM_TYPE_LOCAL):
    return 'LOCAL'
  if (ct == COMM_TYPE_ALLRED):
    return 'ALLRED'
  if (ct == COMM_TYPE_LOCAL_SLICE):
    return 'LOCAL_SLICE'
  if (ct == COMM_TYPE_GATHER_SLICE):
    return 'GATHER_SLICE'
  if (ct == COMM_TYPE_P2P):
    return 'P2P'
  return 'ERROR'

def get_mpi_datatype (dtype):
  if (dtype == 'double'):
    return 'MPI_DOUBLE'
  if (dtype == 'float'):
    return 'MPI_FLOAT'
  if (dtype == 'int'):
    return 'MPI_INT'
  return 'ERROR'

def estimate_per_node_requirement (scol, PP, procs):
  max_cap = 0
  caps = []
  for ss in scol:
    stmt = scol[ss]
    caps.append (stmt.estimate_memory_requirements ())
  max_req = max(caps)
  if (DIMAGE_DT == 'double'):
    max_req *= 8
  if (DIMAGE_DT == 'float'):
    max_req *= 4
  gsize = 1
  unit = 'B'
  if (max_req >= 2**20):
    max_req = float(max_req) / 2**20
    unit = 'MB'
  elif (max_req >= 2**10):
    max_req = float(max_req) / 2**10
    unit = 'KB'
  print   ("Single-node requirement : {} {}".format (max_req, unit ))
  if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
    for pp in procs:
      gsize *= pp
      req_per_node = int(math.ceil (max_req/gsize))
      print ("Requirement by {} nodes : {} {}".format (gsize, req_per_node, unit))
  else:
    n_proc_dim = PP.get_num_dim ()
    max_procs = PP.get_max_procs ()
    proc_per_dim = max_procs ** (1.0 / n_proc_dim)
    for pp in range(n_proc_dim):
      nodes_level = proc_per_dim ** (pp + 1)
      req_per_node = max_req * 1.0 / nodes_level
      print ("Requirement at level {} - {} nodes: {} {}".format (pp + 1, nodes_level, req_per_node, unit))



## Z3 optimization flags and optimization options.
class Comm_Opt_Form:
  def __init__ (self, output_filename, procvec):
    self.decl = []
    self.cstr = []
    self.modelfile = output_filename
    self.pvec = procvec
    self.options = ""
    self.options += "':algebraic_number_evaluator', False, "
    self.options += "':arith_ineq_lhs', False, " 
    #self.options += "':elim_to_real', True, "
    self.options += "':eq2ineq', False, " 
    self.options += "':expand_nested_stores', True, "
    self.options += "':gcd_rounding', False, "
    self.options += "':ignore_patterns_on_ground_qbody', True, "
    self.options += "':flat', False, "
    self.options += "':ite_extra_rules', True, "
    #self.options += "':max_memory', 7516192768, "
    #self.options += "':max_memory', 10737418240, "
    self.options += "':pull_cheap_ite', True, "
    self.options += "':push_ite_arith', True, "
    #self.options += "':push_to_real', False, "
    self.options += "':som', True, "
    self.options += "':som_blowup', 1000, "
    self.options += "':sort_store', True, "
    self.options += "':sort_sums', True, "
    self.options += "':split_concat_eq', True, "
    self.options += "':blast_select_store', True, "
    self.options += "':expand_select_ite', True"
    # testing
    #self.options += ",':local_ctx', True"
    #self.options += ",':cache_all', True"
    #self.options += ",':gcd_rounding', True"
    #self.options += ",':rewrite_patterns', True"
    #self.options += ",':expand_store_eq', True"
    self.options += ",':hoist_mul', True"
    self.options += ",':hoist_ite', True"
    #for dd,pp in enumerate(self.pvec):
    #  cstr = 'p{} == {}'.format (dd, pp)
    #  self.add_cstr (cstr)
    #  print ("Processor constraint : {}".format (cstr))

  def assemble_decl (self):
    ret = ""
    for dd in self.decl:
      if (not ret == ""):
        ret += "\n"
      ret = ret + dd
    return ret

  def assemble_cstr (self):
    ret = ""
    for cc in self.cstr:
      if (not ret == ""):
        ret += ", "
      ret = ret + cc
    return ret

  def print_decl_debug (self):
    variables = self.assemble_decl ()
    print ('Declared variables: {}'.format (variables))

  def print_cstr_debug (self):
    constraints = self.assemble_cstr ()
    print ('Formulation : {}'.format (constraints))

  def add_cstr (self, new_cstr):
    self.cstr.append (new_cstr)

  def add_var (self, new_decl):
    self.decl.append (new_decl)

  def write_chunk (self, ff, chunk, chunk_id):
    cmnt = '## Chunk No. {} \n'.format (chunk_id)
    ff.write (cmnt)
    #ff.write (chunk)
    cmd = 'term = simplify (And ({}), {})\n'.format (chunk, self.options)
    #cmd = 'term = simplify (And ({}))\n'.format (chunk)
    ff.write (cmd)
    cmd = 'opt.add (term)\n'
    ff.write (cmd)
    ff.write ('\n')

  ## Write the COF to a python file script.
  def write_formulation (self, glob_obj_ub, n_fails):
    MAX_CHUNK = 250
    #MAX_CHUNK = 150
    variables = self.assemble_decl ()
    constraints = self.assemble_cstr ()
    ff = open (self.modelfile, 'w')
    ff.write ('from z3 import *\n')
    #ff.write ('opt = Optimize ()\n')
    ## qfnra = Quantifier Free Polynomial Real Arithmetic (e..g x^2 + y^2 < 1)
    ## qfnia = Quantifier Free Non-Linear Integer Arithmetic 
    ## ufnia = What does UF stand for? NIA = Non-Linear Integer Arithmetic 
    #ff.write ("opt = Then('simplify',With('ufnia',':arith.min',True),'qfnia','qfnra').solver ()\n")
    topts = ''
    topts += "':arith.min',True"
    #topts += ","
    #topts += "':arith.solver',2"
    topts += ","
    topts += "':arith.nl.rounds',1048576"
    topts += ","
    topts += "':arith.nl.delay',1000"
    topts += ","
    topts += "':qi.quick_checker',2"
    topts += ","
    topts += "':arith.nl.gr_q',50"
#
    topts += ","
    topts += "':algebraic_number_evaluator', False, "
    topts += "':arith_ineq_lhs', False, " 
    #topts += "':elim_to_real', True, "
    topts += "':eq2ineq', False, " 
    topts += "':expand_nested_stores', True, "
    topts += "':gcd_rounding', False, "
    topts += "':ignore_patterns_on_ground_qbody', True, "
    topts += "':flat', False, "
    topts += "':ite_extra_rules', True, "
    #topts += "':max_memory', 7516192768, "
    #topts += "':max_memory', 10737418240, "
    topts += "':pull_cheap_ite', True, "
    topts += "':push_ite_arith', True, "
    #topts += "':push_to_real', False, "
    topts += "':som', True, "
    topts += "':som_blowup', 1000, "
    topts += "':sort_store', True, "
    topts += "':sort_sums', True, "
    topts += "':split_concat_eq', True, "
    topts += "':blast_select_store', True, "
    topts += "':expand_select_ite', True"
    # testing
    #self.options += ",':local_ctx', True"
    #self.options += ",':cache_all', True"
    #self.options += ",':gcd_rounding', True"
    #self.options += ",':rewrite_patterns', True"
    #self.options += ",':expand_store_eq', True"
    self.options += ",':hoist_mul', True"
    self.options += ",':hoist_ite', True"
    #ff.write ("opt = Then('simplify',With('ufnia',':arith.min',True)).solver ()\n")
    #ff.write ("opt = Then(With('qfnra',arith.min=True)).solver ()\n")
    #ff.write ("opt = Then('simplify','ufnia').solver ()\n")
    ff.write ("opt = Then('simplify',With('ufnia',{})).solver ()\n".format (topts))
    #ff.write ('set_option (rational_to_decimal=True)\n')
    ff.write ('\n')
    ff.write (variables)
    ff.write ('\n')
    ff.write ('## Formulation Objectives\n')
    #ff.write ('K_obj = opt.minimize (K_prog)\n')
    #ff.write ('P_obj = opt.maximize (O_par)\n')
    #ff.write ('G_obj = opt.minimize (G_prog)\n')
    if (glob_obj_ub > 0):
      #nfails = 1
      base_scale = n_fails
      left_scale = 1
      right_scale = 1
      if (DIMAGE_FAST_SOL):
        left_scale = base_scale + 2
        right_scale = base_scale + 1
      iter_g_obj_cstr = '{} * G_prog < {} * {}'.format (left_scale, right_scale, glob_obj_ub)
      ff.write ('opt.add ({})\n'.format (iter_g_obj_cstr))
    ff.write ('\n')
    chunk = ""
    count = 0
    chunk_id = 1
    cache = {}
    for cc in self.cstr:
      if (cc in cache):
        #print ("[INFO] Skipping duplicated constraint: {}".format (cc))
        continue
      cache[cc] = 1
      if (count > 0):
        chunk += ", "
      count += 1
      count += cc.count (',') 
      chunk = chunk + cc
      if (count >= MAX_CHUNK):
        self.write_chunk (ff, chunk, chunk_id)
        count = 0
        chunk = ""
        chunk_id += 1
    # Write last chunk
    self.write_chunk (ff, chunk, chunk_id)   
    # Script epilogue
    #ff.write ('for k, v in opt.statistics ():\n')
    #ff.write ('  print ("K is {} - V is {}".format (k,v))\n')
    #ff.write ('\n')
    #ff.write ('print(opt.check())\n')
    ff.write ('if (opt.check() != unsat):\n')
    ff.write ('  sol = opt.model ()\n')
    ff.write ('  for vv in sol:\n')
    ff.write ('    print(vv, sol[vv])\n')
    ff.write ('else:\n')
    ff.write ('  print("unsat")\n')
    ff.close ()


class Dist:
  def __init__(self, pset, stmts, cgstmts, cfilename):
    self.PP = pset
    self.stmts = stmts
    self.CG = cgstmts
    self.cfilename = cfilename
    self.ff = None
    self.arrays = None
    self.producers = {}
    self.last_writer = {}
  
  def writeln (self, line):
    self.ff.write (line + "\n")

  def empty_line (self):
    self.writeln ('')

  def indent (self):
    self.ff.write ('  ')

  def print_processor_geometry (self):
    sizes = self.PP.get_sizes ()
    for pp in sizes:
      macro_name = self.PP.get_dim_macro_name (pp)
      self.writeln ('#define {} ({})'.format (macro_name, sizes[pp]))

  # Dist.collect_arrays
  def collect_arrays (self):
    ret = {}
    for ss in self.stmts:
      stm = self.stmts[ss]
      ret = stm.collect_arrays (ret)
    return ret

  def collect_last_op_writer (self):
    ret = {}
    for aa in self.arrays:
      ref = self.arrays[aa]
      last = None
      for ss in self.CG:
        if (ss.writes_to (ref)):
          last = ss
      if (last != None):
        ret[aa] = last
    self.last_writer = ret
    for ss in self.CG:
      ss.set_last_writer_map (self.last_writer)
    return ret

    
  def collect_communicators (self):
    comms = {}
    for sid in self.stmts:
      ss = self.stmts[sid]
      comms = ss.collect_communicators (comms)
    return comms


  def declare_communicators (self, comms):
    #self.empty_line ()
    #for cc in comms:
    #  self.writeln ('int {};'.format (cc))
    self.writeln ('// Communicator used in program, one per {array,statement}')
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.declare_communicators (self.ff)

  def declare_used_ispace_dimensions (self):
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.generate_udim_declarations (self.ff)

  def declare_amap_declarations (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      ref.generate_ref_amap_declarations (self.ff)

  def declare_imap_declarations (self):
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.generate_stmt_imap_declarations (self.ff)

  def generate_communicators (self):
    for sid in self.stmts:
      ss = self.stmts[sid]
      ss.generate_communicators (self.ff)
    
  ## Dist.generate_operators ():
  ## Generate baseline operators.
  def generate_operators (self):
    #for sid in self.stmts:
    mrap = {} # mrap is "Most Recent Array Producer"
    for ss in self.CG:
      self.empty_line ()
      new_arr = ss.generate_operator (self.ff, self.PP, self.producers, mrap)
      if (new_arr != None and not new_arr in self.producers):
        self.producers[new_arr] = ss
    self.empty_line ()

  ## Dist.generate_single_node_reference_dag ():
  ## Generate single node reference operators to perform a final check.
  def generate_single_node_reference_dag (self):
    self.writeln ('void reference ()')
    self.writeln ('{')
    mrap = {} # mrap is "Most Recent Array Producer"
    decl = '  int {} = 0;'.format (DIMAGE_BLOCK_COUNT)
    self.writeln (decl)
    for ii in range(10):
      it = '  int t{} = 0;'.format (ii)
      self.writeln (it)
      it = '  int i{};'.format (ii)
      self.writeln (it)
    for ss in self.CG:
      self.empty_line ()
      if (not ss.is_data_generator ()):
        self.writeln ('  {')
      new_arr = ss.generate_single_node_operator (self.ff, self.PP, self.producers, mrap)
      if (new_arr != None and not new_arr in self.producers):
        self.producers[new_arr] = ss
      if (not ss.is_data_generator ()):
        self.writeln ('  }')
    self.empty_line ()
    self.writeln ('  // check call goes here')
    self.empty_line ()
    for ss in self.CG:
      if (ss.is_data_generator ()):
        arr_name = 'sna_' + ss.accs[0].get_name ()
        free_call = '  free ({});'.format (arr_name)
        self.writeln (free_call)
    self.writeln ('}\n')


  def declare_arrays (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_name ()
      line = '{} * {};'.format (DIMAGE_DT, varname)
      self.writeln (line)

  def declare_tile_maps (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_tile_map_name ()
      line = '{} * {};'.format (DIMAGE_INT, varname)
      self.writeln (line)


  def codegen (self, sys_argv, avg_call_solv, iters, num_fails, solset):
    self.arrays = self.collect_arrays ()
    dist.collect_last_op_writer ()
    self.ff = open (self.cfilename, "w")
    self.writeln ('// Script {} invoked with : {}'.format (DIMAGE_PY_SCRIPT, sys_argv))
    self.writeln ('// Average solver call: {}; Iterations: {}; Fails = {}\n'.format (avg_call_solv, iters, num_fails))
    self.writeln ('// K_RATIO of communication-to-computation: {};\n'.format (MEM2COMP_RATIO))
    self.writeln ('// Optimal value found (G_prog): {}\n'.format (solset['G_prog']))
    self.writeln ('#include "dimage-rt.h"')
    self.empty_line ()
    self.writeln ('#ifndef DIMAGE_TILE_HEADER_SIZE')
    self.writeln ('#define DIMAGE_TILE_HEADER_SIZE {}'.format (DIMAGE_TILE_HEADER_SIZE))
    self.writeln ('#endif')
    self.empty_line ()
    self.writeln ('int {};'.format (DIMAGE_PROC_RANK))
    self.writeln ('int {}[DIMAGE_MAX_GRID_DIMS];'.format (DIMAGE_PROC_COORDS))
    self.PP.declare_processor_coordinate_variables (self.ff)
    comms = self.collect_communicators ()
    self.empty_line ()
    self.declare_communicators (comms)
    self.empty_line ()
    self.declare_timers ()
    self.empty_line ()
    self.writeln ("// Processor-space grid")
    self.PP.generate_processor_space_declarations (self.ff)
    self.empty_line ()
    self.writeln ("// Iteration-space to processor-space mappings")
    self.declare_imap_declarations ()
    self.empty_line ()
    self.writeln ("// Data-space to processor-space mappings")
    self.declare_amap_declarations ()
    self.empty_line ()
    self.writeln ("// Iteration-space to data-space mappings")
    self.writeln ("// (similar to poly. access functions).")
    self.declare_used_ispace_dimensions ()
    self.print_processor_geometry ()
    self.empty_line ()
    self.writeln ("// Declare arrays as global variables")
    self.declare_arrays ()
    self.empty_line ()
    self.writeln ("// Declare arrays for tile-maps (dictionaries) as global variables")
    self.declare_tile_maps ()
    self.empty_line ()
    self.generate_operators ()
    if (option_check):
      self.generate_single_node_reference_dag ()
    self.generate_main (option_check)
    self.ff.close ()

  def insert_operator_calls (self, option_check):
    self.indent ()
    self.writeln ('// Computing baseline')
    if (option_check):
      self.indent ()
      self.writeln ('reference ();')
      self.empty_line ()
    self.indent ()
    self.writeln ('// Operator calls')
    for ss in self.CG:
      self.indent ()
      self.writeln ('log_msg ("Calling operator {}");'.format (ss.get_name()))
      self.indent ()
      ss.insert_operator_call (self.ff)

  def deallocate_arrays (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_name ()
      line = 'free ({});'.format (varname)
      self.indent ()
      self.writeln (line)

  def deallocate_tile_maps (self):
    for aa in self.arrays:
      ref = self.arrays[aa]
      varname = ref.get_tile_map_name ()
      line = 'free ({});'.format (varname)
      self.indent ()
      self.writeln (line)

  def get_total_computation_timer (self):
    return 'timer_total_computation'

  def get_total_communication_timer (self):
    return 'timer_total_communication'

  def get_generator_computation_timer (self):
    return 'timer_generator_computation'

  def get_generator_communication_timer (self):
    return 'timer_generator_communication'

  def get_operator_computation_timer (self):
    return 'timer_operator_computation'

  def get_operator_communication_timer (self):
    return 'timer_operator_communication'

  def get_full_operator_timer (self):
    return 'timer_operator_full'

  def declare_timers (self):
    self.writeln ('// Declaring timers ')
    self.writeln ('double timer_start;\n')
    self.writeln ('double {} = 0.0;\n'.format (self.get_total_computation_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_total_communication_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_generator_computation_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_generator_communication_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_operator_computation_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_operator_communication_timer ()))
    self.writeln ('double {} = 0.0;\n'.format (self.get_full_operator_timer ()))
    for ss in self.CG:
      ss.declare_timer (self.ff)
    self.empty_line ()


  def reduce_timer (self, src_timer, dst_timer, use_sum):
    self.indent ()
    red_op = 'MPI_MAX'
    if (use_sum):
      red_op = 'MPI_SUM'
    self.writeln ('double {};'.format (dst_timer))
    self.indent ()
    self.writeln ('MPI_Reduce (&{}, &{}, 1, MPI_DOUBLE, {}, 0, MPI_COMM_WORLD);'.format (src_timer, dst_timer, red_op))

  def accumulate_timers (self):
    self.empty_line ()
    self.indent ()
    self.writeln ('// Aggregating timers ')
    total_comp_timer = self.get_total_computation_timer ()
    total_comm_timer = self.get_total_communication_timer ()
    generator_comp_timer = self.get_generator_computation_timer ()
    generator_comm_timer = self.get_generator_communication_timer ()
    operator_comp_timer = self.get_operator_computation_timer ()
    operator_comm_timer = self.get_operator_communication_timer ()
    operator_full_timer = self.get_full_operator_timer ()
    for ss in self.CG:
      self.indent ()
      line = '{} += {};'.format (total_comp_timer, ss.get_local_computation_timer ())
      self.writeln (line)
      self.indent ()
      line = '{} += {};'.format (total_comm_timer, ss.get_local_communication_timer ())
      self.writeln (line)
      if (ss.is_data_generator ()):
        self.indent ()
        line = '{} += {};'.format (generator_comp_timer, ss.get_local_computation_timer ())
        self.writeln (line)
        self.indent ()
        line = '{} += {};'.format (generator_comm_timer, ss.get_local_communication_timer ())
        self.writeln (line)
      elif (not ss.is_data_sink ()):
        self.indent ()
        line = '{} += {};'.format (operator_comp_timer, ss.get_local_computation_timer ())
        self.writeln (line)
        self.indent ()
        line = '{} += {};'.format (operator_comm_timer, ss.get_local_communication_timer ())
        self.writeln (line)
        self.indent ()
        line = '{} = {} + {};'.format (operator_full_timer, operator_comm_timer, operator_comp_timer)
        self.writeln (line)
    if (option_include_all):
      self.empty_line ()
      self.indent ()
      self.writeln ('printf ("Local computation time (sec): %.6lf\\n", {});'.format (total_comp_timer))
      self.indent ()
      self.writeln ('printf ("Local communication time (sec): %.6lf\\n", {});'.format (total_comm_timer))
      self.empty_line ()
      self.indent ()
      self.writeln ('printf ("Generator-only (local) computation time (sec): %.6lf\\n", {});'.format (generator_comp_timer))
      self.indent ()
      self.writeln ('printf ("Generator-only (local) communication time (sec): %.6lf\\n", {});'.format (generator_comm_timer))
      self.empty_line ()
      self.indent ()
      self.writeln ('printf ("Operator-only (local) computation time (sec): %.6lf\\n", {});'.format (operator_comp_timer))
      self.indent ()
      self.writeln ('printf ("Operator-only (local) communication time (sec): %.6lf\\n", {});'.format (operator_comm_timer))
      self.indent ()
      self.writeln ('printf ("Operator-only (local) total time (sec): %.6lf\\n", {});'.format (operator_full_timer))
    ## Collect max timers.
    timer_var = 'timer_comp_max'
    self.reduce_timer (operator_comp_timer, timer_var, False)
    timer_var = 'timer_comm_max'
    self.reduce_timer (operator_comm_timer, timer_var, False)
    timer_var = 'timer_total_max'
    self.reduce_timer (operator_full_timer, timer_var, False)
    ## Collect sum timers.
    timer_var = 'timer_comp_sum'
    self.reduce_timer (operator_comp_timer, timer_var, True)
    timer_var = 'timer_comm_sum'
    self.reduce_timer (operator_comm_timer, timer_var, True)
    timer_var = 'timer_total_sum'
    self.reduce_timer (operator_full_timer, timer_var, True)
    self.indent ()
    self.writeln ('if (dimage_rank == 0) {')
    self.indent ()   
    timer_var = 'timer_comp_max'
    self.writeln ('  printf ("Max. compute-time: %lf\\n", {});'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_comm_max'
    self.writeln ('  printf ("Max. communication-time: %lf\\n", {});'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_total_max'
    self.writeln ('  printf ("Max. total-time: %lf\\n", {});'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_comp_sum'
    self.writeln ('  printf ("Avg. compute-time: %lf\\n", {}/dimage_cw);'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_comm_sum'
    self.writeln ('  printf ("Avg. communication-time: %lf\\n", {}/dimage_cw);'.format (timer_var))
    self.indent ()   
    timer_var = 'timer_total_sum'
    self.writeln ('  printf ("Avg. total-time: %lf\\n", {}/dimage_cw);'.format (timer_var))
    self.indent ()
    self.writeln ('}')
    

  def generate_main (self, option_check):
    self.writeln ('int main(int argc, char** argv) {')
    self.indent ()
    self.writeln ('MPI_Init(NULL, NULL);')
    self.empty_line ()
    self.indent ()
    self.writeln ('int dimage_cw;')
    self.indent ()
    self.writeln ('MPI_Comm_size (MPI_COMM_WORLD, &dimage_cw);')
    self.indent ()
    self.writeln ('MPI_Comm_rank(MPI_COMM_WORLD, &{});'.format (DIMAGE_PROC_RANK))
    self.PP.init_processor_coordinates (self.ff)
    self.indent ()
    n_proc_dim = self.PP.get_num_dim ()
    proc_coord_list = self.PP.get_processor_coordinate_str_list ()
    self.writeln ('init_log_file_{}D_with_rank ({}, {});'.format (n_proc_dim, proc_coord_list, DIMAGE_PROC_RANK))
    self.indent ()
    self.writeln ('int comm_color;')
    self.indent ()
    self.writeln ('int comm_vec[{}];'.format (self.PP.get_num_dim ()))
    self.empty_line ()
    self.generate_communicators ()
    self.empty_line ()
    self.insert_operator_calls (option_check)
    self.indent ()
    self.empty_line ()
    self.deallocate_arrays ()
    self.empty_line ()
    self.deallocate_tile_maps ()
    self.empty_line ()
    self.indent ()
    self.accumulate_timers ()
    self.empty_line ()
    self.indent ()
    self.writeln ('MPI_Finalize ();')
    self.indent ()
    self.writeln ('return 0;')
    self.writeln ('}');

  ## Generate a Makefile specific to the input *.rel file.
  def gen_makefile (self):
    mf = open('Makefile','w')
    mf.write ('MPICC=mpicc\n')
    mf.write ('MPIRUN=mpirun\n')
    mf.write ('MPIOPTS=-np {} --use-hwthread-cpus --oversubscribe\n'.format (self.PP.get_max_procs ()))
    procs=self.PP.get_max_procs ()
    mf.write ('OSCRUN=srun\n')
    mf.write ('OSCOPTS=--nodes={} --ntasks={} --ntasks-per-node=1 --cpus-per-task=28\n'.format (procs, procs))
    mf.write ('DIMAGERT=dimage-rt.c\n')
    debug_opts = ' -D DIMAGE_LOG -D DIMAGE_DEBUG -D USE_INIT_DIAGONAL -D INIT_MAT'
    defs_bench = ' -D DIMAGE_TILE_HEADER_SIZE={} -D DIMAGE_KERNEL_LOOP  '.format (DIMAGE_TILE_HEADER_SIZE)
    defs_loop = ' -D DIMAGE_TILE_HEADER_SIZE={} -D DIMAGE_KERNEL_LOOP -D INIT_MAT '.format (DIMAGE_TILE_HEADER_SIZE)
    defs_no_loop = ' -D DIMAGE_TILE_HEADER_SIZE={} '.format (DIMAGE_TILE_HEADER_SIZE)
    check_debug_opts = ' -D DIMAGE_LOG -D DIMAGE_DEBUG -D INIT_MAT'
    mf.write ('DEBUGOPTS={}\n'.format (debug_opts))
    mf.write ('CHECK_DEBUG_OPTS={}\n'.format (check_debug_opts))
    mf.write ('SRCS={} code/dimage-rt.c\n'.format (self.cfilename))
    bin_name = re.sub ('\.c','.exe',self.cfilename)
    bin_debug_name = re.sub ('\.c','.debug.exe',self.cfilename)
    bin_debug_loop_name = re.sub ('\.c','.debug-loop.exe',self.cfilename)
    mklflags=' -O3 -qopenmp -fPIC -lmpi -ilp64 -lmpi_ilp64 -mkl=parallel -D DIMAGE_MKL $(MKL_CLUSTER_LIBS) '
    intelflags=' -O3 -qopenmp -fPIC -lmpi -ilp64 -lmpi_ilp64  '
    mf.write ('\n')
    mf.write ('all: dist\n')
    mf.write ('\n')
    mf.write ('dist: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_bench, bin_name))
    mf.write ('\n')
    mf.write ('mkl: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\tmpiicc -I. -I code {} $(SRCS) {} -o {} \n'.format (mklflags, defs_no_loop, bin_name))
    mf.write ('\n')
    mf.write ('intel: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\tmpiicc -I. -I code {} $(SRCS) -o {} \n'.format (intelflags,bin_name))
    mf.write ('\n')
    mf.write ('debug: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) $(DEBUGOPTS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_no_loop, bin_debug_name))
    mf.write ('\n')
    mf.write ('debug-loop: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) $(DEBUGOPTS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_loop, bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('debug-loop-hard: {} code/dimage-rt.c\n'.format (self.cfilename))
    mf.write ('\t$(MPICC) -I. -I code $(SRCS) $(CHECK_DEBUG_OPTS) {} -o {} -fopenmp -O3 -lm\n'.format (defs_loop, bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('bench:\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_name))
    mf.write ('\n')
    mf.write ('run-osc:\n')
    mf.write ('\t$(OSCRUN) $(OSCOPTS) ./{}\n'.format (bin_name))
    mf.write ('\n')
    mf.write ('check-debug: gendata baseline debug\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_debug_name))
    mf.write ('\n')
    mf.write ('check-debug-loop: gendata baseline debug-loop\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('check-hard-debug-loop: gendata-hard baseline-hard debug-loop-hard\n')
    mf.write ('\t$(MPIRUN) $(MPIOPTS) ./{}\n'.format (bin_debug_loop_name))
    mf.write ('\n')
    mf.write ('baseline:\n')
    for ss in self.CG:
      if (ss.is_data_generator () or ss.is_data_sink ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c $(DEBUGOPTS) {} -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('baseline-hard:\n')
    for ss in self.CG:
      if (ss.is_data_generator () or ss.is_data_sink ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c $(CHECK_DEBUG_OPTS) {} -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('gendata:\n')
    for ss in self.CG:
      if (not ss.is_data_generator ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c {} $(DEBUGOPTS) -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('gendata-hard:\n')
    for ss in self.CG:
      if (not ss.is_data_generator ()):
        continue
      bin_file = ss.get_operator_bin_filename ()
      c_file = ss.get_operator_c_filename ()
      mf.write ('\t$(MPICC) -I. -I code code/dimage-rt.c {} $(CHECK_DEBUG_OPTS) -o {} {} -fopenmp -O3 -lm; ./{}\n'.format (c_file, bin_file, defs_loop, bin_file))
    mf.write ('\n')
    mf.write ('clean:\n')
    mf.write ('\trm -f {} *.data phases_* *.mat\n'.format (bin_name))
    mf.close ()

    

class Reference:
  def __init__(self, form, PP, NP):
    self.name = ""
    self.np = NP
    self.PP = PP
    self.ndim = 0
    self.cof = form
    self.dims = {}
    self.sizes = {}
    self.map = {}
    self.data = None
    self.uses_slice = False
    self.last_access = 'ERROR'
    self.is_allgat_out_slice = False
    self.is_allgat_in_slice = False
    self.free_list = ''

  def init_from_file (self, ff):
    line = ff.readline ()
    line = line.strip ()
    parts = line.split (':')
    self.name = parts[0]
    dimlist = parts[1].split (',')
    for dd,dname in enumerate(dimlist):
      self.dims[dd] = dname
      self.map[dd] = DIM_UNMAPPED
      self.ndim += 1
    sizes = parts[2].split(',')
    for dd,dsize in enumerate(sizes):
      self.sizes[dd] = dsize

  ## Retrieve the most recently generated access.
  def get_precollective_buffer_access (self):
    return self.last_access

  ## Reference.set_precollective_buffer_access ():
  ## Store the most recently generated access.
  def set_precollective_buffer_access (self, new_acc):
    self.last_access = new_acc

  def set_is_allgat_out_slice (self, bool_val):
    self.is_allgat_out_slice = bool_val

  def get_is_allgat_out_slice (self):
    return self.is_allgat_out_slice

  def set_is_allgat_in_slice (self, bool_val):
    self.is_allgat_in_slice = bool_val

  def get_is_allgat_in_slice (self):
    return self.is_allgat_in_slice
  

  def get_matrix_filename (self, op_name = ''):
    atop = ''
    if (op_name != ""):
      atop = '_at_{}'.format (op_name)
    fname = '{}{}.mat'.format (self.name, atop)
    return fname

  def estimate_memory_requirements (self):
    vol = 1
    for dsize in self.sizes:
      vol *= int(self.sizes[dsize])
    return vol

  def get_data (self):
    return self.data

  def get_dims (self):
    return self.dims

  def get_dim_name (self, adim):
    if (adim >= len(self.dims)):
      sys.exit (42)
    return self.dims[adim]

  def show_data (self):
    if (self.data == None):
      print ("[{}]: No data found".format (self.name))
      return
    print ("[{}] data:".format (self.name))
    N0 = int(self.sizes[0])
    N1 = int(self.sizes[1])
    for ii in range(N0):
      line = ""
      for jj in range(N1):
        line += "{:.6} ".format (self.data[ ii * N1 + jj])
      print (line)
    print ()
  
  def gen_matrix_data (self):
    fname = self.get_matrix_filename ()
    mat = open (fname, 'w')
    if (self.ndim == 2):
      N0 = int(self.sizes[0])
      N1 = int(self.sizes[1])
      self.data = [0] * (N0 * N1)
      for ii in range(N0):
        for jj in range(N1):
          val = ((ii + (abs(self.map[1]) + 1.0) * N1) * 1.0 + (jj + ord(self.name[0])) + abs(self.map[0]) + 1.0) / (N0 * N1 * 1.0)
          mat.write ('{:.6f} '.format (val))
          index = ii * N1 + jj
          self.data[index] = val
        mat.write ('\n')
    if (self.ndim == 1):
      N0 = int(self.sizes[0])
      self.data = [0] * (N0)
      for ii in range(N0):
        val = (ii * 1.0)  / (N0 * 1.0)
        mat.write ('{} '.format (val))
        index = ii
        self.data[index] = val
    if (self.ndim == 3):
      N0 = int(self.sizes[0])
      N1 = int(self.sizes[1])
      N2 = int(self.sizes[2])
      self.data = [0] * (N0 * N1 * N2)
      for ii in range(N0):
        for jj in range(N1):
          for kk in range(N2):
            val = ((((ii + kk) % N1) * N1) * 1.0 + jj * N2 + ((kk + jj) % N1)) / (ii * kk + jj * N2 + 1.0)
            mat.write ('{} '.format (val))
            index = ii * N1 * N2 + jj * N2 + kk
            self.data[index] = val
          mat.write ('\n')
        mat.write ('\n')
    mat.close ()
  
  ## Show the array name and its sizes by printing it to stdout.
  def show_info(self):
    print ("  Reference: {} (ndim={})".format (self.name, self.ndim))
    for dd in self.dims:
      print ("  --> Dim {}: {} ({})".format (dd, self.dims[dd], self.sizes[dd]))

  ## Return the reference as a string. Used for debugging.
  def get_as_str (self):
    ret = ''
    for dd in self.dims:
      if (ret != ''):
        ret += ','
      ret += str(self.dims[dd])
    ret = '{}[{}]'.format (self.name, ret)
    return ret



  def get_pi_map (self):
    return self.map

  def get_pi_var_by_dim_name (self, dim_name, pdim):
    ret = ''
    for adim in self.dims:
      if (self.dims[adim] == dim_name):
        return self.get_map_varname (adim, pdim)
    return 'ERROR'

  def is_fully_replicated (self):
    for dd in self.dims:
      if (self.map[dd] >= 0):
        return False
    return True
    
  def is_replicated_at_dim (self, adim):
    if (adim >= len(self.map)):
      print ('ERROR @ is_replicated_at_dim ')
      sys.exit (42)
    return self.map[adim] == -1

  def is_pi_map_dim_equal (self, iter_name, mu_pdim):
    for dd in self.dims:
      if (self.dims[dd] == iter_name):
        if (self.map[dd] == mu_pdim and mu_pdim >= 0):
          print ("\t\t Reference {} at dimension {} matched proc.dim {}".format (self.name, dd, mu_pdim))
          return True
        return False
    return False

  def is_mu_map_dim_strict_subset_of_pi (self, iter_name, mu_pdim):
    for dd in self.dims:
      if (self.dims[dd] == iter_name):
        if (self.map[dd] == -1 and mu_pdim >= 0):
          return True
        return False
    return False

  def show_maps (self):
    print ("\tArray {} mappings".format (self.name))
    for dd in self.map:
      print ("\t\t{}[{}]: {}".format (self.name, dd, self.map[dd]))

  # Draw tikz graph for statement using its mapping
  def print_tikz_graph (self, fout, par_x, par_y):
    #print ("\tArray {} mappings".format (self.name))
    for dd in self.map:
      pi = self.map[dd]
      #print ("\t\t{}[{}]: {}".format (self.name, dd, self.map[dd]))
      #\foreach \j in {0,...,4}
      nodename='{}_i{}'.format (self.name, dd)
      dimname=self.dims[dd]
      nodelabel = '{\\large\\textbf{ ' + '{}[i{}]'.format (self.name, dd) + '}}'
      if (pi < 0):
        nodelabel = '{\\large\\textbf{' + '{}[i{}]=*'.format (self.name, dd) + '}}'
      x=par_x
      y=par_y - dd
      command = '\\node[shape=rectangle,draw=red,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
      if (pi >= 0):
        procdim = 'p{}'.format (pi)
        src = '({}.west)'.format (nodename)
        tgt =  '({})'.format (procdim)
        command = '\path [{}] {} edge node[right] {} {};'.format ('->,line width=1mm,red',src,'{}',tgt)
        fout.write (command + '\n')
    return len(self.map)


  def get_ref_dim (self):
    return len(self.dims)

  def pretty_print_map (self, df):
    df.write ('<')
    for dd in self.map:
      if (dd > 0):
        df.write (', ')
      map_dim = self.map[dd]
      if (map_dim >= 0):
        df.write ('{}'.format (map_dim))
      else:
        df.write ('{}=*'.format (map_dim))
    df.write ('>')

  ## Method to add a generic pre-assembled constraint to the COF object
  ## and to the formulation file.
  def add_constraint (self, mf, cstr):
    self.writeln (mf, 'opt.add ({})'.format (cstr))
    self.cof.add_cstr (cstr)

  ## Reference.get_name (): Return the name of the array
  def get_name (self):
    return self.name

  ## Reference.get_tile_map_name (): Return the name of tile-map
  ## variable / dictionary.
  def get_tile_map_name (self, ext_buffer = None):
    if (ext_buffer != None):
      return 'TM_{}'.format (ext_buffer)
    return 'TM_{}'.format (self.name)


  def get_tile_name (self, intermediate = None):
    if (intermediate == None):
      return 'tile_{}'.format (self.name)
    return 'tile_{}'.format (intermediate)

  def get_sna_ref_name (self, intermediate = None):
    if (intermediate == None):
      return 'sna_{}'.format (self.name)
    return 'sna_{}'.format (intermediate)

  def get_sna_reference_filename (self):
    return 'ref_{}'.format (self.name)

  def get_tile_header_size (self, all_tiles):
    proc_geom = self.PP.get_processor_geometry_list_from_map (self.map, all_tiles)
    proc_geom = re.sub (',',' *', proc_geom)
    return '({} * {})'.format (DIMAGE_TILE_HEADER_SIZE, proc_geom)

  ## Reference.get_name_for_check (): Return the name for the reference array
  ## used in calls to check_arrayND().
  def get_name_for_check (self):
    return 'ref_{}'.format (self.name)

  ## Return the number of dimensions of the current array object.
  def get_num_dim (self):
    return len(self.dims)

  def get_iter_name (self, adim):
    return self.dims[adim]

  def get_map_varname (self, idim, pdim):
    varname = 'pi_{}_i{}_p{}'.format (self.name, idim, pdim)
    return varname
    
  def writeln(self, mf, line):
    mf.write (line + "\n")

  def set_lower_bound (self, mf, varname, lb):
    cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_upper_bound (self, mf, varname, ub):
    cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds (self, mf, varname, lb, ub):
    cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    #cmd = 'opt.add ({} >= {}, {} <= {})'.format (varname, lb, varname, ub)
    #self.writeln (mf, cmd)
    self.set_bounds (mf, varname, lb, ub)

  def is_dimension_mapped (self, adim):
    if (self.map[adim] >= 0):
      return True
    return False

  def get_array_communicator_at_statement (self, stmt_name):
    varname = 'dimage_comm_{}_at_{}'.format (self.name, stmt_name)
    return varname

  def get_dimension_communicator_at_statement (self, adim, stmt_name):
    dim_name = 'dimage_comm_{}_dim{}_at_{}'.format (self.name, adim, stmt_name)
    return dim_name

  def collect_communicators_for_statement (self, stmt_name, comms):
    for dd in self.map:
      if (self.is_dimension_mapped (dd)):
        comm_name = self.get_dimension_communicator_at_statement (dd, stmt_name)
        comms[comm_name] = comm_name
    return comms

  def declare_variable (self, mf, varname, decl):
    if (decl == None):
      print ("Exiting")
      sys.exit(42)
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname  
    return decl

  ## Add a \pi (boolean) variable to the COF object.
  def declare_map_vars (self, mf, decl):
    if (decl == None):
      print ("[ERROR] Dictionary is None")
      sys.exit (42)
    NP = self.np
    for dd in self.dims:
      for pp in range(NP):
      #print ("Dim {}: {}".format (dd, self.dims[dd]))
        varname = self.get_map_varname (dd, pp)
        #print ("Varname received: {}".format (varname))
        decl = self.declare_variable (mf, varname, decl)
        self.set_bounds_boolean (mf, varname)
    return decl

  ## Reference.get_sum_pi_var_along_dim ():
  ## Sum pi vars along all the processors p of iter space dimension i
  ## or sum all the pi vars along the same processor-space p dimension.
  def get_sum_pi_var_along_dim (self, idim, pdim):
    varname = ''
    if (idim < 0 and pdim < 0):
      print ("[ERROR] Both in [{}]. Both idim and pdim are -1".format ('get_sum_pi_var_along_dim'))
      sys.exit (42)
    if (idim == -1):
      varname = 'sum_pi_{}_iX_p{}'.format (self.name, pdim)
    else:
      varname = 'sum_pi_{}_i{}_pX'.format (self.name, idim)
    return varname

  ## Sum all the pi variables along an iteration-space dimension or
  ## along a processor-space dimension.
  def set_sum_bound_along_dim (self, mf, mode, dim, ub, decl):
    nn = self.ndim
    if (mode == PER_DIM):
      nn = self.np
    if (mode == PER_DIM):
      self.writeln (mf, '## per dim: np = {}'.format (self.np))
    else:
      self.writeln (mf, '## per proc: np = {}'.format (self.ndim))
    cstr = ""
    # By default, assume we are summing along all the iteration-space 
    # variables for a fixed processor.
    pi_sum_var = self.get_sum_pi_var_along_dim (-1, dim)
    if (mode == PER_DIM):
      pi_sum_var = self.get_sum_pi_var_along_dim (dim, -1)
    decl = self.declare_variable (mf, pi_sum_var, decl)
    for kk in range(nn):
      if (not cstr == ""):
        cstr += " + "
      varname = ""
      if (mode == PER_DIM):
        varname = self.get_map_varname (dim, kk)
      if (mode == PER_PROC):
        varname = self.get_map_varname (kk, dim)
      cstr += varname
    #cstr += " <= {}".format (ub)
    cstr = '{} == {}'.format (pi_sum_var, cstr)
    cmd = "opt.add ({})".format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    cstr = '{} >= 0, {} <= {}'.format (pi_sum_var, pi_sum_var, ub)
    cmd = "opt.add ({})".format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    return decl

  def set_dim_sum_bounds (self, mf, decl):
    for dd in range(self.ndim):
      decl = self.set_sum_bound_along_dim (mf, PER_DIM, dd, 1, decl)
    return decl

  def set_proc_sum_bounds (self, mf, decl):
    #for dd in range(self.ndim):
    for dd in range(self.np):
      decl = self.set_sum_bound_along_dim (mf, PER_PROC, dd, 1, decl)
    return decl
      
  def link_dimensions (self, mf, pp, dd, dim, muvar):
    for dd in range(self.ndim):
      if (self.dims[dd] == dim):
        pivar = self.get_map_varname (dd, pp)
        cstr = '{} >= {}'.format (muvar, pivar)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)

  # Return variable for computing data slices.
  def get_block_size_var (self, dim):
    varname = 'DS_{}_{}'.format (self.name, dim)
    return varname

  def get_rho_varname (self):
    return 'rho_{}'.format (self.name)

  def get_rho_dim_varname (self, adim):
    return 'rho_{}_i{}'.format (self.name, adim)

  # Declare all rho variables for current array.
  def declare_replication_variables (self, mf, decl):
    rho_var = self.get_rho_varname ()
    decl = self.declare_variable (mf, rho_var, decl)
    for dd in self.dims:
      rho_var = self.get_rho_dim_varname (dd)
      decl = self.declare_variable (mf, rho_var, decl)
    return decl

  def bound_replication_variables (self, mf):
    rho_var = self.get_rho_varname ()
    self.set_bounds_boolean (mf, rho_var)
    for dd in self.dims:
      rho_var = self.get_rho_dim_varname (dd)
      self.set_bounds_boolean (mf, rho_var)

  ## Link the rho variable of an array to all of its rho_<array>_dim variables.
  def link_rho_variables (self, mf):
    main_rho_var = self.get_rho_varname ()
    USE_INEQ = False
    if (USE_INEQ):
      for dd in self.dims:
        rho_dim_var = self.get_rho_dim_varname (dd)
        cstr = '{} <= {}'.format (main_rho_var, rho_dim_var)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)
    else:
      cstr = ''
      for dd in self.dims:
        if (cstr != ''):
          cstr += ' * '
        rho_dim_var = self.get_rho_dim_varname (dd)
        cstr += rho_dim_var
      cstr = '{} == {}'.format (main_rho_var, cstr)
      cmd = 'opt.add ({}) ## link_rho_variables'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)

  ## Reference.define_rho_expression ():
  ## The below function is buggy. It allows the rho variable to be 1,
  ## when any of the placement variables is 1.
  def define_rho_expression (self, mf, adim, rho_dim_var):
    cstr = ''
    # MK - Optimization [09/14/2022]:
    # Sum of pi variables is bounded to be between 0 and 1.
    # So we convert the ugly product to 1 - sum pi
    # OLD approach:
    #for pp in range(self.np):
    #  pi_var = self.get_map_varname (adim, pp)
    #  if (cstr != ''):
    #    cstr += " * "
    #  cstr += '(1 - {})'.format (pi_var)
    # New approach:
    #cstr += '1 - ('
    #nt = 0
    #for pp in range(self.np):
    #  pi_var = self.get_map_varname (adim, pp)
    #  if (nt > 0):
    #    cstr += " + "
    #  nt += 1
    #  cstr += '{}'.format (pi_var)
    #cstr += ')'
    sum_pi_var = self.get_sum_pi_var_along_dim (adim, -1)
    #expr = '{} == {}'.format (rho_dim_var, cstr)
    ## NOTE: An array dimension is replicated if it's not mapped.
    ## To truly represent replication we define each rho_Ref_dim variable as
    ## rho_Ref_dim == 1 - sum of same rhos along the same dim across all p-dimensions.
    expr = '{} == 1 - {}'.format (rho_dim_var, sum_pi_var)
    cstr = 'opt.add ({})'.format (expr)
    self.writeln (mf, cstr)
    self.cof.add_cstr (expr)

  ## Reference.define_rho_expression_new ():
  def define_rho_expression_new (self, mf, adim, rho_dim_var):
    cstr = ''
    for pp in range(self.np):
      pi_var = self.get_map_varname (adim, pp)
      cstr = '{} >= (1 - {})'.format (rho_dim_var, pi_var)
      cmd_cstr = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd_cstr)
      self.cof.add_cstr (cstr)

  ## Reference.link_replication_to_placement () : 
  ## Create a constraint where the replication variable, \rho^{A}_{l}, 
  ## upper-bounds the placement variables. When the \rho variable is set
  ## to 1, all of the placement variables, the \pi^{S}_{l,p}, will become 0,
  ## meaning that the array A will not be replicated along any processor-space
  ## dimension p. Further, if an array A is replicated along all space dimensions,
  ## then the array is effectively replicated.
  def link_replication_to_placement (self, mf):
    self.writeln (mf, '## Replication expression of array {}: prod (1-pi^F_[k,p])'.format (self.name))
    for dd in self.dims:
      ## NOTE: In some cases we could prefer un-replicated (distributed arrays). 
      ## Replication often comes with communication cost in the form of all-reduce.
      rho_dim_var = self.get_rho_dim_varname (dd)
      self.define_rho_expression (mf, dd, rho_dim_var)


  ## Return the array dimension size, as an integer, given the associated
  ## iterator name used to access it in the rels input file.
  def get_array_extent_by_dim_name (self, dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == dim_name):
        return int(self.sizes[dd])
    return -1

  def get_portion_expression (self, pbs, proc_var):
    Nportion = ""
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      Nportion = '{}'.format (int(math.ceil (int(pbs) * 1.0 / int(proc_var))))
    else:
      Nportion = '({} / {})'.format (pbs, proc_var)
    return Nportion

  def set_block_function (self, mf, bvar, dim, pbs):
    cstr_sum = ""
    cstr_prod = ""
    # Model Optimization (SEP/2022):
    # The loop below creates the expression: N / (sum Pi x pi_var  + prod (1 - pi))
    # Which is expanded and simplified into:
    # sum ( pi_var x N/Pi ) + N * (1 - sum pi)
    # The above works because sum of pi variables is guaranteed to be upper bounded by 1.
    ## for pp in range(self.np):
    ##   proc_var = self.PP.get_proc_dim_symbol (pp)
    ##   pi_var = self.get_map_varname (dim, pp)
    ##   term1 = '{} * {}'.format (proc_var, pi_var)
    ##   term2 = '(1 - {})'.format (pi_var)
    ##   if (pp > 0):
    ##   cstr_sum += term1
    ##   cstr_prod += term2
    ## cstr = '{} == {} / ({} + {})'.format (bvar, pbs, cstr_sum, cstr_prod) 
    block_sizes = []
    for pp in range(self.np):
      proc_var = self.PP.get_proc_dim_symbol (pp)
      pi_var = self.get_map_varname (dim, pp)
      #Nportion = '{}'.format (int(math.ceil (int(pbs) * 1.0 / int(proc_var))))
      Nportion = self.get_portion_expression (pbs, proc_var)
      if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
        block_sizes.append (int(Nportion))
      else:
        cstr_lb = '{} >= {} * {}'.format (bvar, Nportion, pi_var)
        self.cof.add_cstr (cstr_lb)
        cmd = 'opt.add ({})'.format (cstr_lb)
        self.writeln (mf, cmd)
      if (pp > 0):
        cstr_sum += " + "
        cstr_prod += " + "
      cstr_sum += '{} * {}'.format (Nportion, pi_var)
      cstr_prod += pi_var
    cstr_prod = self.get_sum_pi_var_along_dim (dim, -1)
    cstr = '{} == {} + {} - {} * ({})'.format (bvar, cstr_sum, pbs, pbs, cstr_prod)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    # Set the min block size with a constraint
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      min_block_size = min(block_sizes)
      cstr = '{} >= {}'.format (bvar, min_block_size)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
    # Modulo condition for selecting perfect multiples of the processor dimensions.
    if (USE_MODULO):
      for pp in range(self.np):
        proc_var = 'p{}'.format (pp)
        cstr = '{} % {} == 0'.format (pbs, proc_var)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)

  ## Declare variables with prefix 'DS_'
  def declare_block_variables (self, mf, decl):
    for dd in range(self.ndim):
      varname = self.get_block_size_var (dd)
      decl = self.declare_variable (mf, varname, decl)
      size = self.sizes[dd]
      #print ("Ref {} {} = {}".format (self.name, dd, self.sizes[dd]))
      self.set_upper_bound (mf, varname, size)
      self.set_block_function (mf, varname, dd, size)
    return decl

  def is_dim_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return True
    return False

  ## Reference: return a vector with 0s and 1s representing
  ## whether an iteration space dimension is used to access
  ## a reference.
  def get_vector_used_dims (self, stmt):
    ret=[0] * stmt.get_num_dim ()
    if (DEBUG_REF_USED_DIM):
      print ("stmt = {}, vec01 before = {}".format (stmt.get_name (), ret))
    for dd in range(self.ndim):
      iter_name = self.dims[dd]
      idim = stmt.get_dim_by_name (iter_name)
      ret[idim] = 1
    if (DEBUG_REF_USED_DIM):
      print ("==> vec01 after = {}".format (ret))
    return ret

  def get_dim_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return dd
    return -1

  def get_dim_size_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return int(self.sizes[dd])
    return 0

  ## @Reference:
  ## Return the pi mapping of the dimension corresponding
  ## to parameter iter_dim_name, if it is used.
  ## As the value -1 for a pi means that it is replicated,
  ## DIM_NOT_USED (-2) denotes 'not used'
  def get_pi_by_dim_name_if_used (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return DIM_NOT_USED

  def get_pi_by_name (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return DIM_NOT_USED

  def get_pi_dim_map (self, adim):
    if (adim >= len(self.map)):
      print ("[ERROR] Dimension requested does not exist ({} > {})".format (adim, len(self.map)))
      sys.exit (42)
    return self.map[adim]

  # Return the processor dimension associated to a data
  # space dimension by the name of the iterator used to access it.
  def get_proc_map_by_dim_name (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return -1


  def proc_map_match (self, iter_dim_name, ispace_pdim):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        ds_pdim = self.map[dd]
        if (ds_pdim == ispace_pdim):
          return True
    return False

  def all_pi_mu_match (self, stmt):
    for dd in self.dims:
      iter_name = self.dims[dd]
      stmt_pdim = stmt.get_proc_map_by_dim_name (iter_name)
      array_pdim = self.map[dd]
      if (stmt_pdim != array_pdim):
        return False
    return True


  ## Return the product of all extents of the array.
  def get_fixed_ub (self):
    ret = 1
    if (option_debug >= 2):
      print ("Sizes used by statement {}".format (self.name))
    for ss in self.sizes:
      if (option_debug >= 2):
        print ("{} ".format (self.sizes[ss]))
      ret *= int(self.sizes[ss])
    return ret
      
  ## Return the name of the capacity constraint for a given statement.
  def get_volume_var (self):
    varname = 'req_{}'.format (self.name)
    return varname

  ## Declare volume variables (capacity variables) together with their defining 
  ## expressions. Volumes are computes from the block size associated to the 
  ## array and the dimensions being accessed.
  ## Expressions resulting are of the form: req_{aa} = \prod_{adim} DS_{aa,adim}
  def define_volume_var (self, mf, decl):
    volvar = self.get_volume_var ()
    decl = self.declare_variable (mf, volvar, decl)
    prod_str = ""
    for dd in range(self.ndim):
      if (dd > 0):
        prod_str += " * "
      varname = self.get_block_size_var (dd)
      prod_str += varname
      # Lower bound the
      cstr = '{} >= {}'.format (volvar, varname)
      cmd = 'opt.add ({}) ## Generated by define_volume_var'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    ## Alternate eqs below between == and >=
    cstr = '{} == {}'.format (volvar, prod_str)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, '## Generated by define_volume_var')
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    ub = self.get_fixed_ub ()
    ## Volume var has a fixed upper bound.
    cstr = '{} <= {}'.format (volvar, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    return decl

  # Return variable name of the form: LM_<stmt_name>_<array_name>
  ## In the paper, we refer to the 'LM' variables as '\lambda'.
  ## Variable to determine if access to an array is done in a local fashion.
  def get_match_variable (self, stmt_name):
    varname = 'LM_{}_{}'.format (stmt_name, self.name)
    return varname

  ## Local computation (local match) dimension variable.
  ## A boolean variable that represents if a computation is local.
  ## This happens when the array in question is not mapped to any
  ## processor dimension, or when both the mu and pi variables
  ## of the array and statement match.
  def get_match_dim_variable (self, stmt_name, dim):
    varname = 'LM_{}_{}_i{}'.format (stmt_name, self.name, dim)
    return varname

  def get_mu_variable (self, stmt_name, dd, pp):
    varname = 'mu_{}_i{}_p{}'.format (stmt_name, dd, pp)
    return varname

  def get_phi_variable (self, stmt_name, idim, adim, pdim):
    varname = 'phi_{}_{}_i{}_a{}_p{}'.format (stmt_name, self.name, idim, adim, pdim)
    return varname

  ## Reference.get_replication_comm_factor_variable ():
  def get_replication_comm_factor_variable (self, stmt_name):
    varname = 'Nrep_{}_{}'.format (stmt_name, self.name)
    return varname

  ## Reference.get_replication_comm_factor_variable ():
  def get_replication_comm_factor_dim_variable (self, stmt_name, pdim):
    varname = 'Nrep_{}_{}_p{}'.format (stmt_name, self.name, pdim)
    return varname

  ## Reference.get_replication_out_factor_expr (): Return the product
  ## expression of all Nrep_ variables.
  def get_replication_out_factor_expr (self, stmt_name):
    ret = ''
    for pp in range(self.PP.get_num_dim ()):
      if (pp > 0):
        ret += ' + '
      ret += self.get_replication_comm_factor_dim_variable (stmt_name, pp)
      #ret += '(1 - {})'.format (self.get_sum_pi_var_along_dim (-1, pp))
    return ret

  ## This function creates constraints of the form:
  ## LM_{stmt,ref} = (1 - sum pi_{stmt,ref,dim}) + sum pi_{stmt,ref_dim} x mu_{stmt,ref,dim}
  ## Each LM_{stmt,ref} is a boolean variable.
  ## In the paper we refer to the LM variables as lambda.
  def declare_matching_variables (self, mf, stmt_name, idim, sum_mu_var, dim_name, decl):
    # Must receive the array of mapped dimensions of the current statement.
    for dd in range(self.ndim):
      if (self.dims[dd] == dim_name):
        uvar = self.get_match_dim_variable (stmt_name, idim)
        decl = self.declare_variable (mf, uvar, decl)
        self.set_bounds (mf, uvar, 0, 1)
        if (DIMAGE_USE_RHO_REPLICATION_FACTOR):
          N_rho_dim = self.get_replication_comm_factor_dim_variable (stmt_name, dd)
          if (not N_rho_dim in decl):
            decl = self.declare_variable (mf, N_rho_dim, decl)
            self.set_bounds (mf, N_rho_dim, 0, 1) #, self.PP.get_max_procs ()) ## CHECK that the LB is 1 and not 0
        sum1 = ""
        sum2 = ""
        sum3 = ""
        # Factor to scale outgoing communication when tensor is replicated
        outfactor = "" 
        for pp in range(self.np):
          pivar = self.get_map_varname (dd, pp)
          muvar = self.get_mu_variable (stmt_name, idim, pp)
          proc_vardim_size = self.PP.get_varname (pp)
          #match_term = '({} + {})*({} - {})'.format (muvar, pivar, muvar, pivar)
          match_term = '({} * {})'.format (muvar, pivar)
          #match_term = '({} + {})/2'.format (muvar, pivar)
          # NOTE: We do not need this, after all.
          #local_cstr = '{} <= {}'.format (pivar, muvar)
          #self.add_constraint (mf, local_cstr)
          if (pp > 0):
            sum1 += " + "
            sum2 += " + "
            sum3 += " + "
            outfactor += " + "
          sum1 += pivar
          sum2 += match_term
          sum3 += '({}+{})%2'.format (muvar,pivar)
          outfactor += '{} * ({} - {})'.format (proc_vardim_size, muvar, pivar)
        ## Reminder: idim is the iteration space dimension, but within the
        ## reference it has another dimension position.
        ## Fetch the sum_pi variable for the given data-space dimension.
        sum1 = self.get_sum_pi_var_along_dim (dd, -1)
        #sum1 = sum_mu_var  #self.get_sum_pi_var_along_dim (dd, -1)
        match_expr = '{} == (1 - {} + {} - ({}))'.format (uvar, sum1, sum2, sum3)
        match_expr = '{} == (1 - {} + {})'.format (uvar, sum1, sum2)
        self.add_constraint (mf, match_expr)
        if (DIMAGE_USE_RHO_REPLICATION_FACTOR):
          #replication_expr = '{} == (2 - {} + {})'.format (N_rho_dim, self.get_rho_varname (), outfactor)
          replication_expr = '{} >= (1 - {} + {})'.format (N_rho_dim, self.get_rho_varname (), outfactor)
          self.add_constraint (mf, replication_expr)
    return decl

  ## Reference.set_rho_var_dim_constraints ():
  ## Insert constraints to activate the rho_dim_var depending on whether its 
  ## a data replication scenario or a reduction scenario.
  def set_rho_var_dim_constraints (self, mf, decl, stmt):
    for pp in range(self.PP.get_num_dim ()):
      N_rho_dim = self.get_replication_comm_factor_dim_variable (stmt.get_name (), pp)
      if (not N_rho_dim in decl):
        decl = self.declare_variable (mf, N_rho_dim, decl)
        self.set_bounds (mf, N_rho_dim, 0, 1) 
        cstr = '{} >= 1 - {}'.format (N_rho_dim, self.get_sum_pi_var_along_dim (-1, pp))
        self.add_constraint (mf, cstr)
        red_expr = stmt.get_sum_reduction_mu_expr_along_dim (self, pp)
        if (red_expr != ''):
          cstr = '{} >= ({})'.format (N_rho_dim, red_expr)
          self.add_constraint (mf, cstr)
    return decl


  # One boolean variable per stmt, per array
  def declare_matching_variables_with_phi (self, mf, stmt_name, idim, dim_name, decl):
    # Must receive the array of mapped dimensions of the current statement.
    for dd in range(self.ndim):
      if (self.dims[dd] == dim_name):
        uvar = self.get_match_dim_variable (stmt_name, idim)
        decl = self.declare_variable (mf, uvar, decl)
        self.set_bounds (mf, uvar, 0, 1)
        sum1 = ""
        sum2 = ""
        for pp in range(self.np):
          pivar = self.get_map_varname (dd, pp)
          muvar = self.get_mu_variable (stmt_name, idim, pp)
          phivar = self.get_phi_variable (stmt_name, idim, dd, pp)
          decl = self.declare_variable (mf, phivar, decl)
          self.set_bounds (mf, phivar, 0, 1)
          #match_term = '{} * {}'.format (pivar, muvar)
          self.add_constraint (mf, '{} <= {}'.format (phivar, pivar))
          self.add_constraint (mf, '{} <= {}'.format (phivar, muvar))
          match_term = phivar
          if (pp > 0):
            sum1 += " + "
            sum2 += " + "
          sum1 += pivar
          sum2 += match_term
        match_expr = '{} == (1 - {} + {})'.format (uvar, sum1, sum2)
        self.add_constraint (mf, match_expr)
    return decl

  # Return *READ* communication variable for statement name given.
  def get_stmt_read_ref_comm_var (self, stmt_name):
    varname = 'ReadK_{}_{}'.format (stmt_name, self.name)
    return varname

  # Return *WRITE* communication variable for statement name given.
  def get_stmt_write_ref_comm_var (self, stmt_name):
    varname = 'WriteK_{}_{}'.format (stmt_name, self.name)
    return varname

  # Declare communication varible K_{stmt}_{array}
  def define_stmt_ref_comm_var (self, mf, stmt_name, decl):
    commvar = self.get_stmt_read_ref_comm_var (stmt_name)
    decl = self.declare_variable (mf, commvar, decl)
    return decl

  # Return the name of the variable representing if an array slice is
  # locally mapped.
  def get_local_ref_vol_var (self, stmt_name):
    varname = 'Local_{}_{}'.format (stmt_name, self.name)
    return varname

  def define_stmt_ref_local_vol_var (self, mf, stmt_name, decl):
    localvar = self.get_local_ref_vol_var (stmt_name)
    decl = self.declare_variable (mf, localvar, decl)
    return decl

  def get_local_ref_dim_vol_var (self, stmt_name, idim):
    varname = 'Local_{}_{}_i{}'.format (stmt_name, self.name, idim)
    return varname

  def define_stmt_ref_dim_local_vol_var (self, mf, stmt_name, idim, decl):
    localvar = self.get_local_ref_dim_vol_var (stmt_name, idim)
    decl = self.declare_variable (mf, localvar, decl)
    return decl

  def extract_dims_from_pi_var (self, pi_var):
    parts = pi_var.split ("_")
    idim_str = re.sub ("i","",parts[2])
    idim = int(idim_str)
    pdim_str = re.sub ("p","",parts[3])
    pdim = int(pdim_str)
    return (idim,pdim)

  ## Extract the pi mappings from the solution set and store 
  ## them in the map attribute.
  def extract_mappings_from_solution_set (self, solset):
    for vv in solset:
      if (vv.find ("sum") == 0):
        continue
      piprefix='pi_{}_'.format (self.name)
      #if (vv.find ("pi_") == 0 and vv.find (self.name) > 0):
      if (vv.find (piprefix) == 0):
        if (int(solset[vv]) == 1):
          idim, pdim = self.extract_dims_from_pi_var (vv)
          self.map[idim] = pdim

  def get_udim_varname (self, stmt_name):
    varname = 'DIMAGE_UDIM_{}_{}'.format (stmt_name, self.name)
    return varname

  def get_amap_varname (self):
    varname = 'DIMAGE_AMAP_{}'.format (self.name)
    return varname

  ## Generate a global array declaration finalized with a '-2'
  def generate_ref_amap_declarations (self, mf):
    dimlist = ""
    for dd in self.map:
      dimlist += '{}'.format (self.map[dd])
      dimlist += ', '
    dimlist += '-2'
    varname = self.get_amap_varname ()
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def indent (self, df):
    df.write ('  ')

  def is_map_transposed (self):
    pdim = []
    for dd in self.dims:
      if (self.map[dd] >= 0):
        pdim.append (self.map[dd])
    if (len(pdim) <= 1):
      return False
    if (len(pdim) == 2):
      if (pdim[0] > pdim[1]):
        return True
      return False
    print ("Unhandled case. Will abort.")
    sys.exit (42)
    return False
      

    print ('WTF WHY: len={}'.format (len(self.map)))
    sys.exit (42)
    return False

  ## @Reference
  def generate_communicators_at_statement (self, df, stmt):
    comm_var = self.get_array_communicator_at_statement (stmt.get_name ())
    self.indent (df)
    df.write ('// Compute color for communicator #{} used by array [{}] at statement {}()\n'.format (comm_var, self.name, stmt.get_name ()))
    rank = DIMAGE_PROC_RANK
    udim_var = self.get_udim_varname (stmt.get_name ())
    imap_var = stmt.get_imap_varname ()
    amap_var = self.get_amap_varname ()
    commvec = 'comm_vec'
    comm_variant = ''
    if (not stmt.is_data_sink () and stmt.is_output_array (self)):
      comm_variant = 'generator_'
    is_trans = 0
    if (self.is_map_transposed ()):
      is_trans = 1
    line = 'compute_{}comm_vector ({}, {}, {}, {}, {}, {});\n'.format (comm_variant, rank, udim_var, imap_var, amap_var, commvec, is_trans)
    self.indent (df)
    df.write (line)
    comm_color = 'comm_color'
    nprocdim = self.np
    # Build call to dimage_compute_color_from_comm_vec.
    line = '{} = {} ({}, {}, {}, {});\n'.format (comm_color, DIMAGE_COMPUTE_COLOR_FUNC, nprocdim, DIMAGE_GRID_DIMS, DIMAGE_PROC_COORDS, commvec)
    self.indent (df)
    df.write (line)
    line = 'MPI_Comm_split (MPI_COMM_WORLD, {}, {}, &{});\n'.format (comm_color, rank, comm_var)
    self.indent (df)
    df.write (line)
    self.indent (df)
    df.write ('log_num("Communicator {} color", {});\n'.format (comm_var,comm_color))
    self.indent (df)
    df.write ('log_commvec("Comm.vector {} ", {}, {});\n'.format (commvec,commvec,self.np))

  def declare_communicator_at_statement (self, df, stmt):
    comm_var = self.get_array_communicator_at_statement (stmt.get_name ())
    df.write ('MPI_Comm {};\n'.format (comm_var))

  # Reference.get_num_proc_along_dim_at_current ():
  # Return the number of processors along the data space dimension @dd.
  # If the data dimension is unmapped, return '1' since we assume it's a local
  # computation.
  def get_num_proc_along_dim_at_current (self, dd, PP):
    if (dd >= len(self.map)):
      print ('[ERROR@get_num_proc_along_dim]: Invalid dimension requested.')
      sys.exit (42)
    pdim = self.map[dd]   
    if (pdim < 0):
      return '1'
    return str(PP.get_dim_size (pdim))
    
  # Return the dimension size, possibly tiled, of the current array and 
  # dimension. Will take into account the number of processors and the processor
  # geometry.
  # NOTE: the @dd argument must be an index in the array reference, not the
  # statement.
  def get_dimension_size_as_str (self, stmt, dd, PP, alloc_mode):
    #if (stmt != None):
    #  print ("Error from stmt {} and dimension {}".format (stmt.get_name (), dd))
    num = int(self.sizes[dd])
    pdim = self.map[dd]
    iter_name = self.dims[dd]
    denum = 1
    #denum = self.PP.lcm ()
    tag = 'base'
    if (alloc_mode == ALLOC_MODE_SLICE):
      if (stmt != None and pdim == -1):  # -1 == unmapped
        # Statement could still access in a tiled fashion.
        # Check if statement is mapped at dimension @dd.
        stmt_idim = stmt.get_dim_by_name (iter_name)
        stmt_pdim = stmt.get_proc_dim_map (stmt_idim)
        denum = 1 
        tag = 'asm1'
      elif (stmt != None and pdim >= 0):  # -1 == unmapped
        denum = PP.lcm ()
        tag = 'asm2'
      elif (stmt == None and pdim == -1):
        denum = 1 #PP.lcm () #PP.get_dim_size (pdim)
        tag = 'asm3'
      elif (stmt == None and pdim >= 0):
        denum = PP.get_dim_size (pdim)
        tag = 'asm4'
      else:
        sys.exit (42)
    if (alloc_mode == ALLOC_MODE_TILE):
      if (stmt != None and pdim == -1):
        denum = PP.lcm ()
        tag = 'tag-CF alloc'
      elif (stmt != None and pdim >= 0):
        denum = PP.lcm ()
        stmt_idim = stmt.get_dim_by_name (iter_name)
        stmt_pdim = stmt.get_proc_dim_map (stmt_idim)
        denum = PP.lcm () # / PP.get_dim_size (stmt_pdim)
        tag = 'tag-CF'
      elif (stmt == None and pdim == -1):
        denum = 1
        tag = 'pi-rep'
      elif (stmt == None and pdim >= 0):
        denum = PP.get_dim_size (pdim)
        tag = 'pi-part'
      else:
        sys.exit (42)
    ceilfunc = DIMAGE_CEIL
    div_expr = '{}({},{}) /* {} */'.format (ceilfunc, num, denum, tag)
    #if (denum == 1):
    #  div_expr = '{} /* {} */'.format (num, tag)
    return div_expr

  def get_full_dimension_size_as_str (self, stmt, dd, PP):
    num = int(self.sizes[dd])
    return '{}'.format (num)


  def get_num_proc_along_dim (self, stmt, dd, PP):
    num = int(self.sizes[dd])
    pdim = self.map[dd]
    iter_name = self.dims[dd]
    denum = 1
    if (stmt != None):  # -1 == unmapped
      # Statement could still access in a tiled fashion.
      # Check if statement is mapped at dimension @dd.
      stmt_idim = stmt.get_dim_by_name (iter_name)
      if (stmt_idim >= 0):
        stmt_pdim = stmt.get_proc_dim_map (stmt_idim)
        if (option_debug >= 2):
          print ("\t[INFO@get_num_proc_along_dim]: producer {} array {}, dimension {} : map = {}".format (stmt.get_name (), self.name, iter_name, stmt_pdim))
        if (stmt_pdim >= 0):
          denum = PP.get_dim_size (stmt_pdim)
    if (pdim >= 0):
      denum = max(denum, PP.get_dim_size (pdim))
    return str(denum)

  ## Return an expression representing the data slice of an array dimension.
  ## Original array extent is divided by the number of processors if mapped.
  ## Combinations of mu and pi mappings are considered.
  def get_dimension_size_as_val (self, stmt, dd, PP):
    num = int(self.sizes[dd])
    pdim = self.map[dd]
    iter_name = self.dims[dd]
    denum = 1
    tag = 'dsav1'
    if (stmt != None and pdim == -1):  # -1 == unmapped
      # Statement could still access in a tiled fashion.
      # Check if statement is mapped at dimension @dd.
      stmt_idim = stmt.get_dim_by_name (iter_name)
      if (stmt_idim >= 0):
        stmt_pdim = stmt.get_proc_dim_map (stmt_idim)
        if (stmt_pdim >= 0):
          denum = PP.get_dim_size (stmt_pdim)
          tag = 'dsav2'
        else:
          denum = 1
          tag = 'dsav3'
    if (pdim >= 0):
      denum = PP.get_dim_size (pdim)
      tag = 'dim-size'
    dim_size = '{} (({}), {} /* {} */)'.format (DIMAGE_CEIL, num, denum, tag)
    if (int(denum) == 1):
      dim_size = '{}'.format (num)
    return dim_size

  ## Return the number of tiles (blocks) along a data dimension (@dd).
  def reference_get_num_mapped_tiles_at_dim (self, stmt, dd):
    lcm = self.PP.lcm ()
    pdim = self.map[dd]
    if (stmt == None):
      if (pdim == -1):
        return lcm
      return lcm / self.PP.get_dim_size (pdim)
    iter_name = self.dims[dd]
    stmt_pdim = stmt.get_mu_dim_map_by_name (iter_name)
    if (stmt_pdim == pdim and pdim >= 0): 
      return lcm / self.PP.get_dim_size (pdim)
    if (stmt_pdim == pdim and pdim == -1): 
      return lcm 
    if (stmt_pdim >= 0 and pdim == -1): 
      return lcm / self.PP.get_dim_size (stmt_pdim)
    ## If the computation is unmapped (mu < 0) and 
    ## the data is partitioned (pi >= 0), we fetch
    ## the entire slice.
    if (stmt_pdim < 0 and pdim >= 0):
      return '(' + str(lcm / self.PP.get_dim_size (pdim)) + '/* mu < 0, pi >= 0 */ )'
    return 99999999
    
  ## Reference.get_aggregated_tile_header_space(): Compute the aggregated 
  ## payload associated to all tile headers.
  def get_aggregated_tile_header_space (self, stmt):
    ret = str(DIMAGE_TILE_HEADER_SIZE)
    for dd in self.dims:
      ret += ' * '
      ret += str(self.reference_get_num_mapped_tiles_at_dim (stmt, dd))
    return ret

  def get_dimension_size_as_str_list (self, stmt, PP, alloc_mode):
    ## alloc_mode is one of ALLOC_MODE_FULL, ALLOC_MODE_SLICE and ALLOC_MODE_TILE
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ', '
      if (alloc_mode == ALLOC_MODE_FULL):
        ret += self.get_full_dimension_size_as_str (stmt, dd, PP)
      else:
        ret += self.get_dimension_size_as_str (stmt, dd, PP, alloc_mode)
    return ret

  ## Return the array extent give a dimension.
  def get_extent_as_str (self, dd):
    num = '{}'.format (self.sizes[dd])
    return num

  def get_extent_as_str_by_dim_name (self, iter_name):
    for dd in self.sizes:
      if (self.dims[dd] == iter_name):
        return self.get_extent_as_str (dd)
    return "ERROR"

  ## @Reference: Return a list of array extents separated by commas.
  def get_array_extents_as_str_list (self):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ', '
      ret += self.get_extent_as_str (dd)
    return ret

  ## Return the array extent give a dimension.
  ## This method is exclusively used by get_mapped_array_extents_as_str_list.
  def get_mapped_extent_as_str (self, dd, use_full, ext_mu):
    if (dd >= len(self.dims)):
      print ("ERROR: Requested array dimension {}, but array only has {} dimensions.".format (dd, len(self.dims)))
      sys.exit (42)
    pi_dim = self.map[dd]
    ret = '{}'.format (self.sizes[dd])
    if (pi_dim >= 0 and not use_full):
      denum = self.PP.get_dim_size (pi_dim)
      if (int(denum) > 1):
        ret = '{}({},{})'.format (DIMAGE_CEIL,ret,denum)
    elif (ext_mu >= 0 and not use_full):
      denum = self.PP.get_dim_size (ext_mu)
      if (int(denum) > 1):
        ret = '{}({},{})'.format (DIMAGE_CEIL,ret,denum)
    return ret

  ## @Reference: Return a list of mapped array extents separated by commas.
  ## Returned extents are the original extents divided by the number
  ## of processors along their mapped dimension.
  ## Argument ext_mu determines if we are accessing a local
  ## buffer and not a 'frozen' layout. If ext_mu == -1, then we ignore it.
  def get_mapped_array_extents_as_str_list (self, stmt = None, use_full = True, is_write = False):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ', '
      ext_mu = DIM_UNMAPPED
      if (stmt != None and is_write):
        iter_name = self.dims[dd]
        ext_mu = stmt.get_mu_dim_map_by_name (iter_name)
      ret += self.get_mapped_extent_as_str (dd, use_full, ext_mu)
    return ret

  def get_array_size_as_product_str (self):
    ret = ''
    for dd in self.sizes:
      if (not ret == ''):
        ret += ' * '
      ret += self.get_extent_as_str (dd)
    return ret

  ## Reference.get_tile_allocator_name ():
  def get_tile_allocator_name (self):
    allocator = ''
    if (len(self.dims) == 1):
      allocator = DIMAGE_TILE_ALLOCATOR_1D
    if (len(self.dims) == 2):
      allocator = DIMAGE_TILE_ALLOCATOR_2D
    if (len(self.dims) == 3):
      allocator = DIMAGE_TILE_ALLOCATOR_3D
    if (len(self.dims) == 4):
      allocator = DIMAGE_TILE_ALLOCATOR_4D
    return allocator

  def get_single_node_tile_coord (self):
    ret = ''
    for ii in range(len(self.dims)):
      if (ii > 0):
        ret += ', '
      ret += '1'
    return ret
    

  ## Reference.allocate_local_tile (): create a call to an N-dimensional
  ## tile allocator.
  ## This method is called only from Statement.generate_operator ().
  def allocate_local_tile (self, df, PP, is_generator, stmt, is_single_node = False):
    allocator = self.get_tile_allocator_name ()
    if (allocator == ''):
      print ('[ERROR]: Unsupported tile dimension (> 4D)')
      sys.exit ()
    #dimensions = self.get_dimension_size_as_str_list (None, PP, ALLOC_MODE_SLICE)
    stride_list = self.get_tile_extent_list ()
    proc_geom = PP.get_processor_geometry_list_from_map (stmt.get_mu_map (), False)
    if (is_generator and stmt != None):
      proc_geom = PP.get_processor_geometry_list_from_map (self.map, False)
    buffer_name = self.name
    if (not is_generator):
      buffer_name = self.get_name_for_check () 
    if (is_single_node):
      # Adjust the buffer name and processor geometry for use on a single node.
      buffer_name = self.get_sna_ref_name ()
      #proc_geom = PP.get_single_node_processor_geometry ()
      proc_geom = self.get_single_node_tile_coord ()
    line = '{} = {}({}, {});\n'.format (buffer_name, allocator, stride_list, proc_geom)
    df.write ('\n')
    self.indent (df)
    df.write (line)

  ## Reference.allocate_tile_map(): Allocate memory for a local tile.
  ## Meant to be used in generators only.
  def allocate_tile_map (self, df, PP, ext_buffer = None):
    varname = self.get_tile_map_name (ext_buffer)
    arrdim = len(self.dims)
    tile_shape = PP.get_processor_geometry_list_from_map (self.map, True)
    line = '{} = {}_{}D ({});\n'.format (varname, DIMAGE_TILE_MAP_ALLOCATOR, arrdim, tile_shape)
    df.write ('\n')
    self.indent (df)
    df.write (line)
    
    

  ## @Reference: store the generated tile for debug.
  def dump_generated_tile (self, df, PP):
    #if (PP.get_num_dim () != len(self.dims)):
    #  print ("Skipping 'dump_tile for {}'".format (self.name))
    #  return
    dimensions = self.get_dimension_size_as_str_list (None, PP, ALLOC_MODE_TILE)
    pclist = PP.get_processor_coordinate_str_list ()

    strides = self.get_tile_extent_list ()
    proc_geom = self.PP.get_processor_geometry_list_from_map (self.get_pi_map (), False)

    line = '{}_tile{}D ("{}", {}, {}, {}, {}); /* write-to-file arg */\n'.format (WRITE_TO_FILE_FUNC, self.ndim, self.name, DIMAGE_RANK_ARRAY, self.name, strides, proc_geom)
    
    df.write ('\n')
    self.indent (df)
    df.write ('#ifdef DIMAGE_DEBUG\n')
    self.indent (df)
    df.write (line)
    self.indent (df)
    df.write ('#endif\n')

  def return_allocated (self, df):
    line = 'return {};\n'.format (self.name)
    self.indent (df)
    df.write (line)

  ## @Reference
  def reference_get_local_volume (self, stmt):
    ret = ""
    header = ""
    for dd in self.sizes:
      if (not ret == ""):
        ret += " * " 
        header += " * "
      lexval = self.get_dimension_size_as_val (stmt, dd, PP)
      tiles = self.reference_get_num_mapped_tiles_at_dim (stmt, dd)
      #lexval = self.get_dimension_size_as_str_list (stmt, self.PP, ALLOC_MODE_TILE)
      ret += lexval
      header += str(tiles)
    ret = '{} + {} * {}'.format (ret, DIMAGE_TILE_HEADER_SIZE, header)
    return ret

  def get_full_volume (self, stmt):
    ret = ""
    for dd in self.sizes:
      if (ret != ""):
        ret += " * " 
      ret += self.sizes[dd]
    return ret

  def get_tile_vol (self, stmt):
    return self.reference_get_local_volume (stmt)


  ## @Reference.get_tile_extent_list (): Return a comma-separated list of
  ## tile extent dimensions.
  def get_tile_extent_list (self):
    ret = ""
    for dd in self.sizes:
      if (not ret == ""):
        ret += ", " 
      num = int(self.sizes[dd])
      pdim = self.map[dd]
      denum = self.PP.lcm ()
      extent = '{} (({}), {})'.format (DIMAGE_CEIL, num, denum)
      ret += extent
    return ret

  def get_slice_varname (self, as_in):
    varname = 'slice_{}'.format (self.name)
    if (as_in):
      varname = 'ra_{}'.format (varname)
    else: 
      varname = 'wa_{}'.format (varname)
    return varname

  def set_use_slice (self, val):
    self.uses_slice = val

  def get_use_slice (self):
    return self.uses_slice

  ## Reference.generate_local_slice_buffer(): Generate the code to declare and
  ## allocate a slice of data tiles.
  def generate_local_slice_buffer (self, df, slice_vol, as_in):
    slice_var = self.get_slice_varname (as_in)
    allocator = self.get_tile_allocator_name ()
    line = '{} * {} = {}({});\n'.format (DIMAGE_DT, slice_var, allocator, slice_vol)
    self.indent (df)
    df.write (line)
    return slice_var

  ## Reference.generate_incoming_communication ():
  ## Determine the set of references of a given operator (@stmt), that
  ## requires incoming communication.
  ## For each ref used in @stmt, we check its pi-mapping relative to
  ## the mu-mapping of @stmt.
  ## When communication is found to be necessary we generate (declare) the
  ## necessary buffers. 
  ## The function returns the variable name generated or None if no 
  ## communication is deemed required.
  def reference_generate_incoming_communication (self, df, stmt, comm_type, PP):
    self.indent (df)
    df.write ("// Info of array {}[]\n".format (self.name))
    send_size = self.reference_get_local_volume (stmt)
    recv_size = stmt.get_slice_vol_by_name (self, PP)
    local_vol = self.reference_get_local_volume (stmt)
    self.indent (df)
    df.write ("// Local volume (elements): {}\n".format (local_vol))
    self.indent (df)
    df.write ("// Recv volume  (elements): {}\n".format (recv_size))
    self.indent (df)
    df.write ("// Comm. type: {}\n".format (comm_type_str (comm_type)))
    communicator = self.get_array_communicator_at_statement (stmt.get_name ())
    self.indent (df)
    df.write ("// Array communicator @ {}: {}\n".format (stmt.get_name(), communicator))
    generated_slice = None
    if (comm_type == COMM_TYPE_LOCAL):
      self.indent (df)
      df.write ("// Nothing to do\n")
    if (comm_type == COMM_TYPE_GATHER_SLICE):
      ## NOTE: Only case of incoming all-gather.
      self.set_use_slice (True)
      stride_list = self.get_tile_extent_list ()
      vec01 = self.get_vector_used_dims (stmt)
      proc_geom = PP.get_processor_geometry_list_from_map (stmt.get_mu_map (), False, vec01)
      alloc_args = '{}, {} /* in-alloc-args */'.format (stride_list, proc_geom)
      slice_var = self.generate_local_slice_buffer (df, alloc_args, True)
      generated_slice = slice_var
      primitive = COLLECTIVE_ALLGATHER
      # current array is only a piece of the slice.
      #source_array = slice_var 
      #target_array = self.name
      source_array = self.name 
      target_array = slice_var
      collective = '{} ({}, {}, {}, {}, {}, {}, {});\n'.format (primitive, source_array, send_size, get_mpi_datatype(DIMAGE_DT), target_array, send_size, get_mpi_datatype(DIMAGE_DT), communicator)
      self.indent (df)
      df.write (collective)
      ## Since we do an all-gather, we need to build the *new* map of tile blocks.
      self.set_is_allgat_in_slice (True)
      intermediate = None
      if (self.get_use_slice ()):
        intermediate = self.get_slice_varname (True)
      self.indent (df)
      df.write ('// Rebuild tile map after all-gather on buffer {}\n'.format (intermediate))
      self.indent (df)
      tile_map_name = self.get_tile_map_name (intermediate)
      line = '{} * {};'.format (DIMAGE_INT, tile_map_name)
      df.write (line)
      self.indent (df)
      self.allocate_tile_map (df, PP, intermediate)
      self.generate_tile_map_creation_code (df, PP, intermediate, proc_geom)
      df.write ('\n')
      self.append_to_free_list (tile_map_name)
      self.append_to_free_list (intermediate)
    if (comm_type == COMM_TYPE_P2P):
      print ('[ERROR@reference_generate_incoming_communication]: Unexpected P2P comm.type')
      sys.exit (42)
      #self.indent (df)
      #df.write ('// Pending: P2P\n'.format (self.name))
    return generated_slice

  # @Reference.generate_outgoing_communication(): Invoke only for non-sink statements.
  def generate_outgoing_communication (self, df, stmt, comm_type, PP):
    send_size = self.reference_get_local_volume (stmt)
    recv_size = stmt.get_slice_vol_by_name (self, PP)
    is_gen = stmt.is_data_generator ()
    #if (send_size == recv_size):
    #  return
    local_vol = self.reference_get_local_volume (stmt)
    if (option_debug >= 2):
      print ("[INFO@generate_outgoing_communication] COMM.TYPE for array {} @ stmt {}: {}".format (self.name, stmt.get_name (), comm_type_str(comm_type)))
    #if (comm_type == COMM_TYPE_LOCAL_SLICE or comm_type == COMM_TYPE_GATHER_SLICE):
    self.indent (df)
    df.write ("// Local volume (elements): {}\n".format (local_vol))
    self.indent (df)
    df.write ("// Recv volume  (elements): {}\n".format (recv_size))
    self.indent (df)
    df.write ("// Comm. type: {}\n".format (comm_type_str (comm_type)))
    if (comm_type == COMM_TYPE_LOCAL_SLICE and stmt.is_true_communication (self)):
      self.indent (df)
      df.write ("// Computation is local, but local contribution must be reconciled across the whole slice. AllGather will follow\n")
      slice_var = self.get_slice_varname (False)
      recv_buff = self.name
      header_size = self.get_tile_header_size (False)
      if (not is_gen): # If @stmt is not a generator, we need an intermediate buffer for the allgather.
        recv_buff = stmt.generate_intermediate_allred_buffer (df, self)
      communicator = self.get_array_communicator_at_statement (stmt.get_name ())
      primitive = COLLECTIVE_ALLGATHER
      collective = '{} ({}, {}, {}, {}, {}, {}, {});\n'.format (primitive, slice_var, send_size, get_mpi_datatype(DIMAGE_DT), recv_buff, send_size, get_mpi_datatype(DIMAGE_DT), communicator)
      self.indent (df)
      df.write (collective)
    #if (comm_type == COMM_TYPE_GATHER_SLICE):
    if (comm_type == COMM_TYPE_ALLRED and stmt.is_true_communication (self)):
      self.indent (df)
      df.write ("// Reduction dimension was mapped. AllReduce will follow\n")
      read_slice_var = self.get_slice_varname (False)
      interm = stmt.generate_intermediate_allred_buffer (df, self)
      #write_slice_var = self.generate_local_slice_buffer (df, recv_size, False)
      header_size = self.get_tile_header_size (False)
      communicator = self.get_array_communicator_at_statement (stmt.get_name ())
      primitive = COLLECTIVE_ALLREDUCE
      #df.write ('MPI_Comm_size ({}, &{});\n'.format (communicator, COMM_SIZE_VAR))
      send_size = recv_size
      collective = '{} ({}, {}, {} + {}, {}, {}, {});\n'.format (primitive, read_slice_var, interm, send_size, header_size, get_mpi_datatype(DIMAGE_DT), REDUCE_OP_ADD, communicator)
      self.indent (df)
      df.write (collective)
      #self.indent (df)
      #df.write ('free ({});\n'.format (read_slice_var))

  ## Reference.generate_tile_map_creation_code ():
  def generate_tile_map_creation_code (self, df, PP, ext_buffer = None, ext_tile_list = None):
    tile_map = self.get_tile_map_name (ext_buffer)
    stride_list = self.get_tile_extent_list ()
    max_tile_list = PP.get_processor_geometry_list_from_map (self.map, True)
    ref_tile_list = PP.get_processor_geometry_list_from_map (self.map, False)
    arrdim = len(self.dims)
    buff_name = self.name
    if (ext_buffer != None):
      buff_name = ext_buffer
      ref_tile_list = ext_tile_list
    collect_call = '{}_{}D ({}, {}, {}, {}, {});\n'.format (DIMAGE_COLLECT_TILE_MAP_FUNC, arrdim, buff_name, tile_map, stride_list, max_tile_list, ref_tile_list)
    self.indent (df)
    df.write (collect_call)
    store_call = '{}_{}D (\"{}\", {}, {}, {});\n'.format (DIMAGE_STORE_TILE_MAP_FUNC, arrdim, tile_map, DIMAGE_RANK_ARRAY, tile_map, max_tile_list)
    self.indent (df)
    df.write ('#ifdef DIMAGE_DEBUG\n')
    self.indent (df)
    df.write (store_call)
    self.indent (df)
    df.write ('#endif\n')

  def append_to_free_list (self, varname):
    self.free_list += '  if ({} != NULL) free ({});\n'.format (varname, varname)

  def get_free_list (self):
    return self.free_list

  def has_unmapped_dim (self, stmt):
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (self.map[dd] < 0):
        return True
    return False

  # @Reference: Return the access type for a single array reference (the @self).
  # Cases to consider:
  # Iter.Space x Data Space:
  #   M x M : tiled
  #   M x U : Sliced
  #   U x M : Even possible?
  #   U x U : tiled (full matrix)
  # A single data space dimension with different mapping suffices
  # to qualify the entire reference.
  def get_access_type (self, stmt, PP, producers):
    if (not self.name in producers):
      print ("Producers: {}".format (producers))
    is_lexpos = producers[self.name].is_mu_lexico_positive (self, producers)
    was_allgat =  producers[self.name].was_allgathered (self, stmt)
    return ACC_TYPE_TILE
    print ('[INFO@get_access_type: statement {}, reference {}, access {}, dmap={}, imap={}, lexpos={}, was_allgat={}'.
      format (stmt.get_name (), self.name, self.dims, self.map, stmt.get_mu_map (), is_lexpos, was_allgat))
    if (not is_lexpos and was_allgat): 
      # non lexico-pos partition and reassembling with allgather => layout change
      return ACC_TYPE_SLICE
    if (self.all_pi_mu_match (stmt)): # processor only ever owned a single tile and perfectly matched
      return ACC_TYPE_TILE
    if (is_lexpos and was_allgat): # original, linearized, single array, never distributed, all redundant.
      return ACC_TYPE_LIN
    if (is_lexpos and not was_allgat): # original, linearized, single array, never distributed, all redundant.
      return ACC_TYPE_LIN
    if (not is_lexpos and not was_allgat): # original, linearized, single array, never distributed, all redundant.
      return ACC_TYPE_LIN
    print ('[ERROR@get_access_type: Shouldn\'t get here.')
    return ACC_TYPE_ERROR

  # @Reference: Produce a linearized iterator list in string format.
  # Linearized access is only used when the array is deemed to have a 
  # lexico non-negative layout, e.g. <0,*> or <*,*>. 
  # As a result, iterators will not need to be reordered as in the
  # SLICE_ND access case.
  def get_linearized_iterator_str_list (self, stmt, PP, producers, is_write, is_acum):
    ttc = stmt.collect_tile_trip_counts (producers)
    ret = ''
    for adim in self.dims:
      iter_name = self.dims[adim]
      stmt_iter_idx = stmt.get_dim_used_in_ref_dim (self, adim)
      # Next, determine if a tile iterator is 'local'. If that's the case,
      # the data space dimension is effectively distributed, and the tile 
      # index offset associated to the dimension becomes zero. We enforce
      # this by multiplying by a zero factor.
      mu_map = stmt.get_mu_map ()
      mu_pdim = mu_map[stmt_iter_idx]
      pi_pdim = self.map[adim]
      factor = ''
      match_dim = self.is_pi_map_dim_equal (iter_name, mu_pdim)
      comm_type = stmt.determine_communication_type (self, PP)
      if (match_dim):
        factor = ' * 0'
      elif (mu_pdim >= 0 and pi_pdim < 0 and comm_type == COMM_TYPE_LOCAL_SLICE):
        if (not is_acum and is_write):
          ## if is_acum = False means it's part of the main computation
          ## is_write tells us is the slice-buf
          factor = ' * 0 /* pi < 0, slice-buffer -> 0 (ct={})*/'.format (comm_type)       
        elif (is_acum):
          factor = ' * 1 /* PROB: was 0, {}*/'.format (comm_type)
        else:
          factor = ' * 1 /* CHECK ME */'.format (comm_type)
      if (ret != ''):
        ret += ', '
      tile_iter = stmt.get_tile_iterator (stmt_iter_idx) + factor
      point_iter = stmt.get_point_iterator (stmt_iter_idx)
      extent = self.get_extent_as_str (adim)
      num_proc_along_dim = '1'
      if (iter_name in ttc):
        num_proc_along_dim = ttc[iter_name]
      stmt_pmap = stmt.get_proc_dim_map (stmt_iter_idx)
      if (stmt_pmap >= 0):
        num_proc_along_dim = PP.get_dim_size (stmt_pmap)
      tile_size = '{}(({}),{})'.format (DIMAGE_CEIL, extent, num_proc_along_dim)
      expr = '({}) * ({}) + ({})'.format (tile_iter, tile_size, point_iter)
      ret += expr
    return ret

  def get_right_stride (self, adim):
    ret = ''
    for dd in range(adim+1,len(self.dims)):
      ret += ' * {}'.format (self.sizes[dd])
    return ret

  def gen_canonical_access (self):
    ret = ''
    ret += self.name
    ret += '['
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (dd > 0):
        ret += ' + '
      stride = self.get_right_stride (dd)
      ret += '{}{}'.format (iter_name, stride)
    ret += ']'
    return ret


## Start of Statement Class
class Statement:
  def __init__(self, form, PP, NP):
    self.name = ""
    self.PP = PP
    self.np = NP
    self.cof = form
    self.ndim = 0
    self.dims = {}
    self.nref = 0
    self.refs = {}
    self.accs = [] # same as refs but a list
    self.map = {}
    self.last_writer_map = None
    self.ntpd = []
    self.kernel_name = None

  def init_from_file (self, ff):
    line = ff.readline ()
    line = line.strip ()
    parts = line.split (':')
    self.name = parts[0]
    dimlist = parts[1].split (',')
    if (len(parts) == 3):
      self.kernel_name = parts[2]
    for dd,dname in enumerate(dimlist):
      self.dims[dd] = dname
      self.map[dd] = DIM_UNMAPPED
      self.ndim += 1
    line = ff.readline ()
    line = line.strip ()
    self.nref = int(line)
    for aa in range(self.nref):
      ref = Reference (self.cof, self.PP, self.np)
      ref.init_from_file (ff)
      self.refs[ref.get_name ()] = ref
      self.accs.append (ref)
    
  def estimate_memory_requirements (self):
    total = 0
    print ("Mem-req Statement : {}".format (self.name))
    for ref in self.accs:
      ref_req = ref.estimate_memory_requirements () 
      total += ref_req
      print ("{} --> {}".format (ref.get_name (), ref_req))
    return total

  def check_capacity_requirement (self, solset, pnc):
    volvar = self.get_volume_var ()
    for vv in solset:
      if (vv.find (volvar) >= 0):
        used = float(solset[vv])
        if (pnc > 0):
          used = used * 100 / pnc
        print ('Memory used by {} : {:.2f}% (of {})'.format (self.name, used, pnc))

  #def collect_arrays (self, collection):
  #  for aa in self.accs:
  #    print (aa)
  #    collection[aa.get_name ()] = aa

  ## Statement.writes_to (): return True if the current statement
  ## modified the given array. Return False otherwise.
  def writes_to (self, ref):
    if (self.is_data_sink ()):
      return False
    nref = len(self.accs)
    write_ref = self.accs[nref-1]
    return (write_ref.get_name () == ref.get_name ())


  def set_last_writer_map (self, arg_last_writer_map):
    self.last_writer_map = arg_last_writer_map
    
  def get_ref (self, refid):
    if (refid >= len(self.accs)):
      return None
    return self.accs[refid]

  def get_ref_by_name (self, ref_name):
    for ref in self.accs:
      if (ref.get_name () == ref_name):
        return ref
    return None

  def gen_matrix_data (self):
    if (self.is_data_generator ()):
      self.accs[0].gen_matrix_data ()

  def show_info(self):
    print ("Statement: {} (ndim={})".format (self.name, self.ndim))
    for dd in self.dims:
      print ("Dim {}: {}".format (dd, self.dims[dd]))
    for aa in self.refs:
      self.refs[aa].show_info ()

  def get_dims (self):
    return self.dims

  def get_mu_map (self):
    return self.map

  def get_mu_dim_map (self, idim):
    if (idim >= len(self.map)):
      print ("ERROR @ get_mu_map")
      sys.exit (42)
    return self.map[idim]

  def get_mu_dim_map_by_name (self, dim_name):
    for ii in self.dims:
      if (self.dims[ii] == dim_name):
        return self.map[ii]
    return 'ERROR'

  def show_maps (self):
    print ("Statement {} mappings".format (self.name))
    for dd in self.map:
      print ("{}[{}]: {}".format (self.name, dd, self.map[dd]))
    for ref in self.accs:
      ref.show_maps ()

  # Draw tikz graph for statement using its mapping
  def print_tikz_graph (self, fout, par_x, par_y):
    #print ("\tArray {} mappings".format (self.name))
    for dd in self.map:
      mu = self.map[dd]
      #print ("\t\t{}[{}]: {}".format (self.name, dd, self.map[dd]))
      #\foreach \j in {0,...,4}
      nodename='{}_i{}'.format (self.name, dd)
      dimname = self.dims[dd]
      nodelabel = '{\\large\\textbf{ ' + '{}({})'.format (self.name, dimname) + '}}'
      if (mu < 0):
        nodelabel = '{\\large\\textbf{' + '{}({})=*'.format (self.name, dimname) + '}}'
      x=par_x
      y=par_y - dd
      command = '\\node [shape=rectangle,draw=blue,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
      ## Print edges
      if (mu>=0):
        procdim = 'p{}'.format (mu)
        src = '({}.east)'.format (nodename)
        tgt =  '({})'.format (procdim)
        command = '\path [{}] {} edge node[left] {} {};'.format ('->,line width=1mm,blue',src,'{}',tgt)
        fout.write (command + '\n')
    return len(self.map)

  # Draw tikz graph for statement using its mapping
  def print_ref_tikz_graph (self, fout, par_x, par_y):
    #print ("\tArray {} mappings".format (self.name))
    ref = self.accs[0]
    for dd in self.dims:
      dim_name = self.dims[dd]
      pi = ref.get_pi_by_dim_name_if_used (dim_name)
      if (pi == DIM_NOT_USED):
        continue
      #print ("\t\t{}[{}]: {}".format (self.name, dd, self.map[dd]))
      #\foreach \j in {0,...,4}
      nodename='{}_i{}'.format (ref.get_name (), dd)
      #dimname=self.dims[dd]
      nodelabel = '{\\large\\textbf{ ' + '{}[{}]'.format (ref.get_name (), dim_name) + '}}'
      if (pi < 0):
        nodelabel = '{\\large\\textbf{' + '{}[{}]=*'.format (ref.get_name (), dim_name) + '}}'
      x=par_x
      y=par_y - dd
      command = '\\node[shape=rectangle,draw=red,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
      if (pi >= 0):
        procdim = 'p{}'.format (pi)
        src = '({}.west)'.format (nodename)
        tgt =  '({})'.format (procdim)
        command = '\path [{}] {} edge node[right] {} {};'.format ('->,line width=1mm,red',src,'{}',tgt)
        fout.write (command + '\n')
    return len(ref.get_dims ())

  ## @Statement: return True if the given dimension is mapped.
  def is_dimension_mapped (self, idim):
    if (self.map[idim] >= 0):
      return True
    return False

  # Return the processor map for the given iteration space dimension.
  def get_proc_dim_map(self, idim):
    return self.map[idim]

  # @Statement: Determine the lexico-positivity of the mu-mapping associated to the
  # current statement. This is used to later determine if the tiled layout
  # of an array has been changed, e.g. whether the layout [[A,B],[C,D]] morphed
  # into [[A,C],[B,D]] after an Allgather.
  def is_mu_lexico_positive (self, ref, producers = None):
    all_gat_follows = False
    if (producers != None and ref.get_name() in producers):
      all_gat_follows = producers[ref.get_name ()].was_allgathered (ref, self)
    temp = []
    NOMAP = 99
    for dd in range(self.ndim):
      pdim = self.map[dd]
      if (pdim >= 0):
        temp.append (pdim)
      else:
        temp.append (NOMAP)
    for dd in range(1,self.ndim):
      ## Must handle the degenerated case of some processor dimension
      ## having only 1 processor along it. E.g., 16x1.
      np_prev = 0;
      if (temp[dd-1] != NOMAP):
        np_prev = self.PP.get_dim_size (temp[dd-1])
      np_next = 0;
      if (temp[dd] != NOMAP):   
        np_next = self.PP.get_dim_size (temp[dd])
      ## Two cases:
      ## 1) Dimension is unmapped and next dimension is mapped and all gather follows
      ## 2) Back-to-back dimensions are mapped but the first one has size 1, and all gather follows
      if ((temp[dd-1] == NOMAP and temp[dd] != NOMAP and np_next > 1 and all_gat_follows) or
          (temp[dd-1] != NOMAP and temp[dd] != NOMAP and np_prev == 1 and np_next > 1 and all_gat_follows)):
        return False
    return True

  # @Statement: Operator must be the original producer of the array.
  # Operator must have been fully distributed.
  def was_allgathered (self, ref, stmt):
    if (len(self.accs) != 1):
      print ("[ERROR@was_allgathered]: was_allgathered() should only be used when the operator is the original producer of an array slice (1) - Operator is not a generator.")
      sys.exit (42)
    if (ref.get_name () != self.accs[0].get_name ()):
      print ("[ERROR@was_allgathered]: was_allgathered() should only be used when the operator is the original producer of an array slice (2) - Operator is *NOT* the producer of {}".format (ref.get_name ()))
      sys.exit (42)
    ## Allgather is necessary if the work is partitioned but the data
    ## must end up in a replicated fashion.
    ## This method is part of the generator, but it's being ultimately
    ## invoked from some other statement.
    for dd in range(self.get_num_dim()):
      idim_name = self.get_dim_name (dd)
      mu_dim = self.get_mu_dim_map (dd)
      ref_at_gen = self.accs[0]
      pi_dim = ref_at_gen.get_pi_by_dim_name_if_used (idim_name)
      proc_dim_size = 1
      if (mu_dim >= 0):
        proc_dim_size = self.PP.get_dim_size (mu_dim)
      if (option_debug >= 2):
        print ("Debug All-Gather from Statement {}.{}[{}], Generator={} - ref={} : pi={}, mu={}, proc-dim-size={}".format (stmt.get_name (), ref_at_gen.get_name (), idim_name, self.name, ref_at_gen.get_as_str(), pi_dim, mu_dim, proc_dim_size))
      if (pi_dim == DIM_UNMAPPED and mu_dim >= 0 and proc_dim_size > 1):
        return True
    return False

  ## Determine whether the layout of a matrix has changed. 
  ## For the layout to change, it must first be lexico-negative and it must
  ## have been all-gathered afterwards.
  def layout_changed (self, ref):
    sys.exit (9999)
    if (self.was_allgathered (ref)):
      return True
    if (self.is_mu_lexico_positive (ref)):
      return True
    return False
    

  # @Statement: Find the loop dimension corresponding to an iterator
  def get_dim_by_name (self, iter_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_name):
        return dd
    return -1

  def get_dim_name (self, idim):
    return self.dims[idim]

  def get_num_dim (self):
    return len(self.dims)

  # Return mapped processor dimension associated to an iteration space 
  # dimension identified by the name of the latter.
  def get_proc_map_by_dim_name (self, iter_dim_name):
    for dd in range(self.ndim):
      if (self.dims[dd] == iter_dim_name):
        return self.map[dd]
    return -1

  # Return a comma-separated list containing the number
  # of processors along each dimension. If the statement is
  # mapped at the current dimension, then we include '1',
  # otherwise we include the max number of dimensions.
  def get_processor_geometry_str_list (self, ref, PP):
    ret = ''
    for dd in self.map:
      iter_name = self.dims[dd]
      if (not ref.is_dim_used (iter_name)):
        continue
      print ("\t\t{}[{}]: {}".format (self.name, dd, self.map[dd]))
      #print ("dict[{}] = {}".format (kk,pdim))
      if (not ret == ''):
        ret += ', '
      pdim = self.map[dd]
      if (pdim >= 0):
        ref_pdim = ref.get_proc_map_by_dim_name (iter_name)
        if (pdim == ref_pdim): 
          # If it's a match, and pdim >= 0, then access should be for a tile. 
          # Hence, we don't need the number of processor along the dimension pdim.
          ret += '1'
        elif (ref_pdim < 0): 
          # Statement still mapped. 
          # Local processor stores full extent of array dimension. 
          # Hence, will need the number of processors.
          ret += str(PP.get_dim_size (pdim))
        elif (ref_pdim >= 0 and ref_pdim != pdim):
          ret += str(PP.get_dim_size (pdim))
        else:
          ret += 'ERROR' # WEIRD CASE
      else:
        # Dimension is unmapped, so return the max among all the 
        # processor dimensions.
        #ret += str(PP.get_max_dim_size ())
        ret += '1'
    return ret

  # Return statement expression representing the volume of a data tile
  def get_tile_vol (self, ref, ttc):
    ret = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (ref.is_dim_used (iter_name)):
        if (ret != ''):
          ret += ' * '
        extent = ref.get_extent_as_str_by_dim_name (iter_name)
        denum = '1'
        if (iter_name in ttc):
          denum = ttc[iter_name]
        expr = '{}(({}), {})'.format (DIMAGE_CEIL, extent, denum)
        if (int(denum) == 1):
          expr = extent
        ret += expr
    return ret

  def pretty_print_map (self, df):
    df.write ('<')
    for dd in self.map:
      if (dd > 0):
        df.write (', ')
      map_dim = self.map[dd]
      if (map_dim >= 0):
        df.write ('{}'.format (map_dim))
      else:
        df.write ('{}=*'.format (map_dim))
    df.write ('>')

  def get_name (self):
    return self.name

  ## Return the loop tripcount associated to the given dimension id.
  ## The argument dim_id must be between 0 and (depth-1).
  def get_loop_dim_tripcount (self, dim_id):
    dim_name = self.dims[dim_id]
    for ref in self.accs:
      if (ref.is_dim_used (dim_name)):
        return ref.get_dim_size_if_used (dim_name)
    return 0


  ## Statement.get_map_varname ():
  ## Return the mu (map) variable name. Not to confuse with
  ## method get_mu_varname defined within the Reference class.
  def get_map_varname (self, idim, pdim):
    varname = 'mu_{}_i{}_p{}'.format (self.name, idim, pdim)
    return varname

  def get_mu_sum_varname (self, dim_id):
    varname = 'sum_mu_{}_i{}_pX'.format (self.name, dim_id)
    return varname

  ## Statement.get_sum_reduction_mu_expr_along_dim ():
  ## Return the sum of mu variables that are used on a concrete processor
  ## dimension and that are also a reduction dimension.
  ## On matmul this amount to a sum of a single term, while for
  ## mttkrp this results in as many terms as reduction dimensions.
  def get_sum_reduction_mu_expr_along_dim (self, ref, pdim):
    #ff = open('bug.txt', 'a')
    ret = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (self.is_reduction_dim (ref, iter_name)):
        if (ret != ''):
          ret += ' + '
        ret += self.get_map_varname (dd, pdim)
    #ff.write (ret + '\n')
    #ff.close ()
    return ret

  def is_write_ref (self, ref):
    if (self.is_data_sink ()):
      return False
    #if (self.is_data_generator ()):
    #  return True
    # Always assume that the last reference is write-ref.
    nref = len(self.accs)
    write_ref = self.accs[nref-1]
    return (write_ref.get_name () == ref.get_name ())
      
    
  def writeln(self, mf, line):
    mf.write(line + "\n")

  def add_constraint (self, mf, cstr):
    self.writeln (mf, 'opt.add ({})'.format (cstr))
    self.cof.add_cstr (cstr)

  def set_lower_bound (self, mf, varname, lb):
    plain_cstr = '{} >= {}'.format (varname, lb)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_upper_bound (self, mf, varname, ub):
    plain_cstr = '{} <= {}'.format (varname, ub)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_bounds (self, mf, varname, lb, ub):
    plain_cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (plain_cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (plain_cstr)

  def set_bounds_boolean (self, mf, varname):
    lb = 0
    ub = 1
    #plain_cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    #cmd = 'opt.add ({})'.format (plain_cstr)
    #self.writeln (mf, cmd)
    #self.cof.add_cstr (plain_cstr)
    self.set_bounds (mf, varname, lb, ub)

  def declare_variable (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Int('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_boolean (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Bool('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_float (self, mf, varname, decl):
    if (not varname in decl):
      cmd = "{} = Real('{}')".format (varname, varname)
      self.cof.add_var (cmd)
      self.writeln (mf, cmd)
      decl[varname] = varname
    return decl

  def declare_map_vars (self, mf, decl):
    if (decl == None):
      print ("[ERROR] Error. Dictionary is None.")
      sys.exit (42)
    NP = self.np
    for dd in self.dims:
      for pp in range(NP):
      #print ("Dim {}: {}".format (dd, self.dims[dd]))
        varname = self.get_map_varname (dd,pp)
        decl = self.declare_variable (mf, varname, decl)
        self.set_bounds_boolean (mf, varname)
    return decl

  def set_sum_bound_along_dim (self, mf, mode, dim, ub, decl):
    nn = self.ndim
    if (mode == PER_DIM):
      nn = self.np
    cstr = ""
    for kk in range(nn):
      if (not cstr == ""):
        cstr += " + "
      varname = ""
      if (mode == PER_DIM):
        varname = self.get_map_varname (dim, kk)
      if (mode == PER_PROC):
        varname = self.get_map_varname (kk, dim)
      cstr += varname
    cstr += " <= {}".format (ub)
    cmd = "opt.add ({})".format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    return decl

  def set_dim_sum_bounds (self, mf, decl):
    for dd in range(self.ndim):
      self.writeln (mf, '## set_dim_sum_bounds')
      decl = self.set_sum_bound_along_dim (mf, PER_DIM, dd, 1, decl)
    return decl

  def set_proc_sum_bounds (self, mf, decl):
    for dd in range(self.np):
      self.writeln (mf, '## set_proc_sum_bounds; np={}'.format (self.np))
      decl = self.set_sum_bound_along_dim (mf, PER_PROC, dd, 1, decl)
    return decl

  def set_ref_sum_bounds (self, mf, decl):
    for rr in self.refs:
      ref = self.refs[rr]
      decl = ref.set_dim_sum_bounds (mf, decl)
      decl = ref.set_proc_sum_bounds (mf, decl)
    return decl

  # Declare mu mapping variables for the current statement.
  def declare_ref_vars (self, mf, decl):
    if (decl == None):
      print ("[ERROR] Error in dictionary.")
      sys.exit (42)
    for ref in self.accs:
      #ref = self.refs[rr]
      decl = ref.declare_map_vars (mf, decl)
    return decl
    
  ## Link mu and pi variables
  def link_dimensions (self, mf):
    for pp in range(self.np):
      for dd in self.dims:
        dim = self.dims[dd]
        for rr in self.refs:
          ref = self.refs[rr]
          muvar = self.get_map_varname (dd, pp)
          ref.link_dimensions (mf, pp, dd, dim, muvar)

  def get_comm_slice_variable (self, ref_name, dim_id):
    varname = 'K_{}_{}_{}'.format (self.name, ref_name, dim_id)
    return varname

  def get_comm_ref_variable (self, ref_name):
    varname = 'K_{}_{}'.format (self.name, ref_name)
    return varname

  def declare_comm_slice_variable (self, mf, ref, dim_id, decl):
    #varname = self.get_comm_slice_variable (ref_name, dim_id)
    varname = ref.get_local_ref_vol_var (self.name)
    decl = self.declare_variable (mf, varname, decl)
    return decl

  def declare_comm_ref_variable (self, mf, ref, decl):
    varname = ref.get_local_ref_vol_var (self.name)
    decl = self.declare_variable (mf, varname, decl)
    return decl

  def set_comm_slice_function (self, mf, ref_name, dim_id, slice_var, pbs):
    cstr_sum = ""
    cstr_prod = ""
    cstr = ''
    USE_OLD = False
    # Model Optimization (09/14/2022):
    # The loop below creates the expression: N / (sum Pi x pi_var  + prod (1 - pi))
    # Which is expanded and simplified into:
    # sum ( pi_var x N/Pi ) + N * (1 - sum pi)
    # The above works because sum of pi variables is guaranteed to be upper bounded by 1.
    if (USE_OLD):
      for pp in range(self.np):
        #proc_var = 'p{}'.format (pp)
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dim_id, pp)
        term1 = '{} * {}'.format (proc_var, mu_var)
        term2 = '(1 - {})'.format (mu_var)
        if (pp > 0):
          cstr_sum += " + "
          cstr_prod += " * "
        cstr_sum += term1
        cstr_prod += term2
      self.writeln (mf, '## Defined in set_comm_slice_function')
      cstr = '{} == {} / ({} + {})'.format (slice_var, pbs, cstr_sum, cstr_prod)
    else:
      portions = []
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dim_id, pp)
        if (pp > 0):
          cstr_sum += " + "
          cstr_prod += " + "
        Nportion = ''
        if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          Nportion = int(math.ceil(int(pbs) * 1.0/int(proc_var)))
        else:
          Nportion = '({} / {})'.format (pbs, proc_var)
        portions.append (Nportion)
        cstr_sum += '{} * {}'.format (Nportion, mu_var)
        cstr_prod += mu_var
        # lower bound constraints for slice_var
        if (not DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          cstr_lb = '{} >= {} * {}'.format (slice_var, Nportion, mu_var)
          cmd = 'opt.add ({}) # parametric lower bound'.format (cstr_lb)
          self.writeln (mf, cmd)
          self.cof.add_cstr (cstr_lb)
      cstr = '{} == {} + {} - {} * ({})'.format (slice_var, cstr_sum, pbs, pbs, cstr_prod)
    self.cof.add_cstr (cstr)
    cmd = 'opt.add ({})'.format (cstr) 
    self.writeln (mf, cmd)
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      cstr = '{} >= {}'.format (slice_var, min(portions))
      cmd = 'opt.add ({})'.format (cstr) 
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    if (USE_MODULO):
      for pp in range(self.np):
        proc_var = self.PP.get_proc_dim_symbol (pp)
        cstr = '{} % {} == 0'.format (pbs, proc_var)
        cmd = 'opt.add ({})'.format (cstr)
        self.writeln (mf, cmd)
        self.cof.add_cstr (cstr)

  ## Declare and define a variable "Local_<stmt>_<ref>".
  def define_comm_slice (self, mf, ref, dim_id, decl):
    ref_name = ref.get_name ()
    #slice_var = self.get_comm_slice_variable (ref_name, dim_id)
    #slice_var = ref.get_local_ref_vol_var (self.name)
    slice_var = ref.get_local_ref_dim_vol_var (stmt.name, dim_id)
    decl = self.declare_variable (mf, slice_var, decl)
    extent = ref.get_array_extent_by_dim_name (self.dims[dim_id])
    decl = self.declare_comm_slice_variable (mf, ref, dim_id, decl)
    #print ("Ref {} {} = {}".format (self.name, dd, self.sizes[dd]))
    self.set_upper_bound (mf, slice_var, extent)
    self.set_comm_slice_function (mf, ref_name, dim_id, slice_var, extent)
    return decl

  ## Create the constraint on variables Local_{<stmt>,<ref>}
  ## and on the corresponding dimension variables, Local_{<stmt>,<ref>,<dim>}.
  ## Also add lower bounds: the Local_{stmt,ref} >= Local_{stmt,ref,dim>
  def set_comm_slice_expressions (self, mf, decl):
    for ref in self.accs:
      decl = self.declare_comm_ref_variable (mf, ref, decl)
      #decl = ref.get_local_ref_vol_var (self.name)
    for ref in self.accs:
      local_comm = ref.get_local_ref_vol_var (self.name)
      #ref = self.refs[rr]
      ref_name = ref.get_name ()
      comm_var = self.get_comm_ref_variable (ref_name)
      expr = ""
      for dd in self.dims:
        dim_name = self.dims[dd]
        if (ref.is_dim_used (dim_name)):
          decl = self.define_comm_slice (mf, ref, dd, decl)
          if (not expr == ""):
            expr += " * "
          #expr += self.get_comm_slice_variable (ref_name, dd)
          #expr += ref.get_local_ref_vol_var (self.name)
          local_comm_var = ref.get_local_ref_dim_vol_var (stmt.name, dd)
          expr += local_comm_var
          cstr = '{} >= {}'.format (local_comm, local_comm_var)
          self.add_constraint (mf, cstr)
      # Alternate between '>=' and '=='
      # Prefer '>=' over '=='. We are computing an upper bound after all.
      # Individual slice contributions from each array will be exact.
      cmd = '{} == {}'.format (local_comm, expr) #comm_var, expr)
      #self.writeln (mf, cmd)
      self.add_constraint (mf, cmd)
    return decl

  def get_volume_var (self):
    varname = 'req_{}'.format (self.name)
    return varname

  ## Create capacity constraints per statement.
  ## The maximum capacity is given for the whole program.
  ## The memory needed by a statement results from the sum of all its parts.
  def set_statement_capacity_constraint (self, mf, decl, pnc, maxprocs):
    total_expr = ''
    self.writeln (mf, "## Introduced by stmt.set_statement_capacity_constraint")
    total_var = self.get_volume_var ()
    decl = self.declare_variable (mf, total_var, decl)
    nn = len(self.accs)
    rid = 1
    for ref in self.accs:
      if (not total_expr == ''):
        total_expr += " + "
      #ref = self.refs[rr]
      # Use the current volume var as a lower bound of the total volume
      volvar = ref.get_volume_var ()
      decl = ref.define_volume_var (mf, decl)
      # Set an 'easy' lower-bound for the req_{stmt} variables
      cstr = '{} * {} >= {}'.format (maxprocs, total_var, volvar)
      cmd = 'opt.add ({}) # See set_statement_capacity_constraint ()'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
      ## Compensate for potential 3-buffers and MPI internal storage.
      ## Extra space for buffers.
      if (rid < nn):
        total_expr += volvar
      else:
        total_expr += '{} * {}'.format (DIMAGE_CAP_FACTOR, volvar)
      rid += 1
    # Only insert the equality: req_ss = \sum_{aa} req_{ss,aa} if we have
    # two or more arrays used ss.
    if (len(self.accs) > 1):
      cstr = '{} == {}'.format (total_var, total_expr)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    #cstr = '{} >= 1'.format (total_var)
    #cmd = 'opt.add ({})'.format (cstr)
    #self.writeln (mf, cmd)
    #self.cof.add_cstr (cstr)
    if (pnc > 0):
      cstr = '{} <= {}'.format (total_var, pnc)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
    return decl

  def get_comm_var (self):
    varname = 'K_{}'.format (self.name)
    return varname

  # Return the name of a computation cost variable (W / omega)
  def get_comp_cost_variable (self):
    varname = 'W_{}'.format (self.name)
    return varname

  ## Return the Global Performance Objective variable name of the
  ## current statement.
  def get_gpo_varname (self):
    varname = 'G_{}'.format (self.name)
    return varname

  def get_sanity_varname (self):
    varname = 'Z_{}'.format (self.name)
    return varname

  def get_ratio_varname (self):
    varname = 'R_{}'.format (self.name)
    return varname

  ## To avoid spurious mappings and matchings.
  def get_parity_check_expression (self, ref):
    ret = ''
    for idim in self.dims:
      dim_name = self.dims[idim]
      if (ref.is_dim_used (dim_name)):
        term = ''
        adim = ref.get_dim_if_used (dim_name)
        sum_pi_var = ref.get_sum_pi_var_along_dim (adim, -1)
        for pdim in range(self.PP.get_num_dim ()):
          pi_var = ref.get_pi_var_by_dim_name (dim_name, pdim)
          mu_var = self.get_map_varname (idim, pdim)
          if (term != ''):
            term += ' + '
          term += '(({}+{})%2)'.format (mu_var, pi_var)
        if (ret != ''):
          ret += ' + '
        term = '({}) * {}'.format (term, sum_pi_var)
        ret += term
    return '(' + ret + ') * {}'.format (MEM2COMP_RATIO)


  ## Statement.set_comm_constraints():
  ## Introduce for each statement (operator), communication volume
  ## constraints tying lambdas (matching variables) with effective volumes,
  ## L^{S,A}. Outgoing arrays also use rhos (replication) variables.
  ## Build the communication constraint for a statement.
  ## K_ss = \sum_ref K_{ss,ref}
  ## We skip read constraints for generators and write communications 
  ## constraints for data sinks.
  ## Further, we also add constraints of the form
  ## \forall ref: K_ss >= K_{ss,ref}
  ## K-constraints
  def set_comm_constraints (self, mf, decl):
    #for ref in self.accs:
    #  umv = ref.get_match_variable (self.name)
    #  decl = ref.define_stmt_ref_comm_var (mf, self.name, decl)
    temp = []
    for ref in self.accs:
      temp.append (ref)
    last = len(self.accs)
    temp.append (self.accs[last-1])
    total_expr = ''
    total_var = self.get_comm_var ()
    decl = self.declare_variable (mf, total_var, decl)
    ## In the future, it might be of interest to fine-tune between 
    ## equalities and inequalities.
    OPERATOR = '=='
    rep_factor = ''
    for ii,ref in enumerate(temp):
      if (self.is_data_sink () and ii > 0):
        continue
      if (self.is_data_generator () and ii < last):
        continue
      if (not total_expr == ''):
        total_expr += " + "
      #ref = self.refs[rr]
      umv = ref.get_match_variable (self.name)
      commvar = ref.get_stmt_read_ref_comm_var (self.name) # commvar is L^{S,A}
      rho_var = ref.get_rho_varname ()
      if (ii == last):
        commvar = ref.get_stmt_write_ref_comm_var (self.name) 
      volvar = ref.get_volume_var ()
      decl = ref.define_stmt_ref_local_vol_var (mf, self.name, decl)
      local_comm = ref.get_local_ref_vol_var (self.name)
      local = ref.get_match_variable (self.name)
      penalty = ''
      if (DIMAGE_EXCLUDE_CROSSDIM_MAP_SOLUTIONS):
        penalty = '{} == 0'.format (self.get_parity_check_expression (ref))        
        self.add_constraint (mf, penalty)
        penalty = ''
      else:
        # Will allow cross-dimension mappings as a penalty to the solution. Will 
        # increase time-to-solution.
        penalty = ' + {}'.format (self.get_parity_check_expression (ref))
      decl = self.declare_variable (mf, commvar, decl)
      ## Alternate between '==' and '>='. Will use variable OPERATOR defined above.
      term = '{} {} {} * (1 - {}{})'.format (commvar, OPERATOR, local_comm, umv, penalty)
      # Use the current commvar as the lower bound of the totalvar
      cstr = '{} >= {}'.format (total_var, commvar)
      cmd = 'opt.add ({})'.format (cstr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (cstr)
      if (ii == last): 
        ## This equality means the reference is a write.
        ## Eventually, may alternate between '==' and '>='
        ## Changed the '(rho_var)' to '(1 - rho_var)'. 
        ## Replication translates to all-reduce.
        decl = ref.set_rho_var_dim_constraints (mf, decl, self)
        rep_factor = ref.get_replication_out_factor_expr (self.name)
        term = '{} == {} * ({})'.format (commvar, local_comm, rep_factor)
      self.add_constraint (mf, term)
      cstr = '{} >= 0'.format (commvar)
      self.add_constraint (mf, cstr)
      total_expr += commvar
    # Set: comm^s >= sum comm^{s,ref}
    cstr = '{} {} {}'.format (total_var, OPERATOR, total_expr)
    cmd = 'opt.add ({}) # set_comm_constraints'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)
    return decl

  def get_objective_name (self, otype):
    varname = 'obj_{}_{}'.format (otype, self.name)
    return varname

  def set_req_objective (self, mf, max_obj):
    varname = self.get_objective_name ('req')
    obj_mode = ''
    if (max_obj):
      obj_mode = 'maximize'
    if (not max_obj):
      obj_mode = 'minimize'
    obj_var = self.get_volume_var ()
    cmd = '{} = opt.{}({})'.format (varname, obj_mode, obj_var)
    self.writeln (mf, cmd)

  def set_comm_objective (self, mf, max_obj):
    varname = self.get_objective_name ('K')
    obj_mode = ''
    if (max_obj):
      obj_mode = 'maximize'
    if (not max_obj):
      obj_mode = 'minimize'
    obj_var = self.get_comm_var ()
    cmd = '{} = opt.{}({})'.format (varname, obj_mode, obj_var)
    self.writeln (mf, cmd)

  ## Add the constraints of the form:
  ## LM_ss in [0,1]
  ## LM_{ss,aa,idim} = (1 - \sum_{p} pi_{aa,adim,p} + \sum_{pp} pi_{aa,adim=idim,pp} x pi_{ss,adim=idim,pp}
  ## LM variables correspond to the \lambda variables used in the paper.
  def add_matching_constraints (self, mf, decl):
    for ref in self.accs:
      ref_match_var = ref.get_match_variable (self.name)
      decl = self.declare_variable (mf, ref_match_var, decl)
      self.set_bounds_boolean (mf, ref_match_var)
    for ref in self.accs:
      for dd in self.dims:
        dim_name = self.dims[dd]
        if (ref.is_dim_used (dim_name)):
          sum_mu_var = self.get_mu_sum_varname (dd)
          decl = ref.declare_matching_variables (mf, self.name, dd, sum_mu_var, dim_name, decl)
          #decl = ref.declare_matching_variables_with_phi (mf, self.name, dd, dim_name, decl)
    # local_match = (1 - sum pi) + sum pi_d x mu_d
    for ref in self.accs:
      ref_match_var = ref.get_match_variable (self.name)
      #decl = self.declare_variable (mf, ref_match_var, decl)
      #self.set_bounds_boolean (mf, ref_match_var)
      USE_PROD = False
      USE_PROD = True
      if (USE_PROD):
        match_expr = ""
        for dd in self.dims:
          dim_name = self.dims[dd]
          if (ref.is_dim_used (dim_name)):
            if (not match_expr == ""):
              match_expr += " * "
            ref_match_dim_var = ref.get_match_dim_variable (self.name, dd)
            match_expr += ref_match_dim_var 
        cstr = '{} == {}'.format (ref_match_var, match_expr)
        self.add_constraint (mf, cstr)
      else:
        for dd in self.dims:
          dim_name = self.dims[dd]
          if (ref.is_dim_used (dim_name)):
            #if (not match_expr == ""):
            #  match_expr += " * "
            ref_match_dim_var = ref.get_match_dim_variable (self.name, dd)
            #match_expr += ref_match_dim_var 
            cstr = '{} == {}'.format (ref_match_var, ref_match_dim_var) #match_expr)
            self.add_constraint (mf, cstr)
    return decl


  def declare_replication_variables (self, mf, decl):
    for ref in self.accs:
      decl = ref.declare_replication_variables (mf, decl)
    return decl

  def bound_replication_variables (self, mf):
    for ref in self.accs:
      ref.bound_replication_variables (mf)

  def add_replication_constraints (self, mf):
    for ref in self.accs:
      ref.link_rho_variables (mf)

  def set_array_dim_replication_expression (self, mf):
    for ref in self.accs:
      ref.link_replication_to_placement (mf)

  def declare_block_variables (self, mf, decl):
    for ref in self.accs:
      decl = ref.declare_block_variables (mf, decl)
    return decl


  ## Create a sum variable for each mu variable, i.e.:
  ## \forall i: sum_mu_i = \sum_{j} \mu_{i,j}
  def set_mu_dimension_sum (self, mf, decl):
    for dd in self.dims:
      mu_sum_var = self.get_mu_sum_varname (dd)
      decl = self.declare_variable (mf, mu_sum_var, decl)
      sum_expr = ''
      for pp in range(self.np):
        #proc_var = self.PP.get_varname (pp)
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dd, pp)
        if (sum_expr != ''):
          sum_expr += ' + '
        sum_expr += mu_var
      expr = '{} == {}'.format (mu_sum_var, sum_expr)
      cmd = 'opt.add ({})\n'.format (expr)
      self.set_bounds_boolean (mf, mu_sum_var)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    return decl

  ## Build the computation cost expression of the form:
  ## \forall i: (\sum_j N_i * \mu_{i,j} / P_j) - N_i * mu_sum_i + N_i
  ## where i is a loop dimension, j is a processor dimension
  ## and mu_sum_i = \sum_j \mu_{i,j}.
  def set_computation_cost_expression (self, mf, decl):
    varname = self.get_comp_cost_variable ()
    decl = self.declare_variable (mf, varname, decl)
    cost_expr = ''
    all_min = []
    for dd in self.dims:
      expr = ''
      tripcount = self.get_loop_dim_tripcount (dd)
      mu_sum_var = self.get_mu_sum_varname (dd)
      size_list = []
      for pp in range(self.np):
        #proc_var = self.PP.get_varname (pp)
        proc_var = self.PP.get_proc_dim_symbol (pp)
        mu_var = self.get_map_varname (dd, pp)
        Nportion = ''
        term = ''
        if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          Nportion = int(math.ceil (int(tripcount) * 1.0 / proc_var))
          size_list.append (Nportion)
        else:
          Nportion = '({} / {})'.format (tripcount, proc_var)
        term = '({} * {})'.format (Nportion, mu_var)
        #term = '({} * {} / {})'.format (tripcount, mu_var, proc_var)
        #term = '(({} / {}) * {})'.format (tripcount, proc_var, mu_var)
        if (not DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
          cstr_lb = '{} >= {} * {}'.format (varname, Nportion, mu_var)
          cmd = 'opt.add ({}) # check'.format (cstr_lb)
          self.writeln (mf, cmd)
          self.cof.add_cstr (cstr_lb)
        if (expr != ''):
          expr += ' + '
        expr += term
      if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
        all_min.append (min(size_list))
      no_map_term = '{} * (1 - {})'.format (tripcount, mu_sum_var)
      factor = '({}) + {}'.format (expr, no_map_term)
      if (cost_expr != ''):
        cost_expr += ' * '
      cost_expr += '({})'.format (factor)
    # Set a better lower bound for the var
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      expr = '{} >= {}'.format (varname, prod(all_min))
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    cost_expr = '{} == {}'.format (varname, cost_expr)
    cmd = 'opt.add ({})'.format (cost_expr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cost_expr)
    #expr = '{} >= 0'.format (varname)
    #cmd = 'opt.add ({})'.format (expr)
    #self.writeln (mf, cmd)
    #self.cof.add_cstr (expr)
    return decl

  ## Build a performance expression constraint for the current statement.
  ## The expression will be of the form: work_cost + aplha * comm_cost,
  ## where alpha is defined as a machine specific memory-to-compute ratio.
  ## For each compute-statement ss do:
  ## G_ss = K_ss + 40 x W_ss
  ## where G is the global cost, K is the communication cost and W is the 
  ## computation cost.
  def set_performance_expression_constraints (self, mf, decl, obj_mode):
    objvar = self.get_gpo_varname () ## gov = Global Performance Objective
    decl = self.declare_variable (mf, objvar, decl)
    k_comm_var = self.get_comm_var ()
    w_comp_var = self.get_comp_cost_variable ()
    ## Alternate, eventually between == or >=.
    ## Default COMM_ONLY objective mode
    expr = '{} >= {}'.format (objvar, k_comm_var)
    ## Below: performance objective for current statement.
    if (obj_mode == DIMAGE_OBJ_COMM_COMP):
      expr = '{} == {} + {} * {}'.format (objvar, w_comp_var, MEM2COMP_RATIO, k_comm_var)
      #expr = '{} == {} + {} * {}'.format (objvar, k_comm_var, MEM2COMP_RATIO, w_comp_var)
      #expr = '{} == {}'.format (objvar, w_comp_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    # Set lower bounds for performance for each component.
    if (obj_mode == DIMAGE_OBJ_COMM_COMP):
      ## Lower bound G with W
      expr = '{} >= {}'.format (objvar, w_comp_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
      ## Lower bound G with MEM2COMP_RATIO x K
      #expr = '{} >= {} * {}'.format (objvar, MEM2COMP_RATIO, k_comm_var)
      expr = '{} >= {} * {}'.format (objvar, 1, k_comm_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    else:
      ## Lower bound G with MEM2COMP_RATIO x K
      expr = '{} >= {}'.format (objvar, k_comm_var)
      cmd = 'opt.add ({})'.format (expr)
      self.writeln (mf, cmd)
      self.cof.add_cstr (expr)
    return decl

  def set_sanity_check_constraints (self, mf, decl):
    k_comm_var = self.get_comm_var ()
    w_comp_var = self.get_comp_cost_variable ()
    z_var = self.get_sanity_varname ()
    ratio_var = self.get_ratio_varname ()
    decl = self.declare_boolean (mf, z_var, decl)
    expr = '{} == ({} / {} <= {})'.format (z_var, k_comm_var, w_comp_var, MEM2COMP_RATIO * 2)
    self.writeln (mf, expr)
    self.cof.add_cstr (expr)
    return decl


  def extract_dims_from_mu_var (self, mu_var):
    parts = mu_var.split ("_")
    idim_str = re.sub ("i","",parts[2])
    idim = int(idim_str)
    pdim_str = re.sub ("p","",parts[3])
    pdim = int(pdim_str)
    return (idim,pdim)

  ## Statement.extract_mappings_from_solution_set(): 
  ## Extract the values of the mu variables from the solution set.
  def extract_mappings_from_solution_set (self, solset):
    for vv in solset:
      if (vv.find ("sum") >= 0):
        continue
      muprefix='mu_{}_'.format (self.name)
      #if (vv.find ("mu_") == 0 and vv.find (self.name) > 0):
      if (vv.find (muprefix) == 0):
        if (int(solset[vv]) == 1):
          idim, pdim = self.extract_dims_from_mu_var (vv)
          self.map[idim] = pdim
    for ref in self.accs:
      ref.extract_mappings_from_solution_set (solset)

  ## Traverse the list of accesses and add the array names to the
  ## @arrset dictionary. Return the updated dictionary.
  def collect_arrays (self, arrset):
    for ref in self.accs:
      if (ref.get_name () in arrset):
        continue
      arrset[ref.get_name ()] = ref
    return arrset

  def collect_communicators (self, comms):
    for ref in self.accs:
      comms = ref.collect_communicators_for_statement (self.name, comms)
    return comms

  # Produce a vector of iteration space dimensions which appear
  # in the @ref reference argument. The following holds for each
  # entry v_i in the vector:
  # v_i == -1: dimension i is not used in ref
  # v_i >= 0: dimension i is used in ref[v_i]
  # The vector is finalized with a -2.
  def generate_ref_udim_declarations (self, mf, ref):
    dimlist = ""
    for dd in self.dims:
      dim_name = self.dims[dd];
      entry = ref.get_dim_if_used (dim_name)
      dimlist += '{}'.format (entry)
      dimlist += ', '
    dimlist += '-2'
    varname = ref.get_udim_varname (self.name)
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def generate_udim_declarations (self, mf):
    for ref in self.accs:   
      self.generate_ref_udim_declarations (mf, ref)

  def get_imap_varname (self):
    varname = 'DIMAGE_IMAP_{}'.format (self.name)
    return varname

  def generate_stmt_imap_declarations (self, mf):
    dimlist = ""
    for dd in self.map:
      dimlist += '{}'.format (self.map[dd])
      dimlist += ', '
    dimlist += '-2'
    varname = self.get_imap_varname ()
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def generate_communicators (self, df):
    for ref in self.accs:
      ref.generate_communicators_at_statement (df, self)
      df.write ('\n')
    
  def declare_communicators (self, df):
    for ref in self.accs:
      ref.declare_communicator_at_statement (df, self)
  
  def get_operator_name (self):
    opname = 'dimage_operator_{}'.format (self.name)
    return opname

  def indent (self, df):
    df.write ('  ')

  ## Determine if the current statement is a data generator.
  ## Assume that this is obtained from the statement's name, which
  ## must be prefixed with the keyword 'gen'.
  def is_data_generator (self):
    if (self.name.find ('gen') == 0):
      return True
    if (self.name.find ('Gen') == 0):
      return True
    return False

  ## Determine if the current statement is a data sink.
  ## Assume that this is obtained from the statement's name, which
  ## must be prefixed with the keyword 'sink'.
  def is_data_sink (self):
    if (self.name.find ('sink') == 0):
      return True
    if (self.name.find ('Sink') == 0):
      return True
    return False

  def is_compute_statement (self):
    if (self.is_data_generator ()):
      return False
    if (self.is_data_sink ()):
      return False
    return True

  def get_iterator_variable (self, idim, is_point):
    varname = 'i{}'.format (idim)
    if (not is_point):
      varname = 'b{}'.format (idim)
    return varname

  def is_input_array (self, ref):
    if (self.is_data_generator ()):
      return False
    if (self.is_data_sink ()):
      return self.accs[0].get_name () == ref.get_name ()
    if (self.accs[0].get_name () == ref.get_name ()):
      return True
    if (self.accs[1].get_name () == ref.get_name ()):
      return True
    return False

  def is_output_array (self, ref):
    if (self.is_data_sink ()):
      return False
    n_acc = len(self.accs)
    return self.accs[n_acc-1].get_name () == ref.get_name ()

  ## Statement.is_reduction_dim ():
  ## Return True if the given reference is a write-reference and if the
  ## given iterator doesn't appear in it. Return False otherwise.
  def is_reduction_dim (self, ref, iter_name):
    if (not self.is_output_array (ref)):
      return False
    return (not ref.is_dim_used (iter_name))

  ## @Statement: Determine whether the reduction dimension is being mapped
  ## or not. If it's in fact mapped, then we will need an allgather or allreduce
  def get_mapped_reduction_dimension (self, ref, PP):
    for dd in self.dims:
      iter_name = self.dims[dd]
      true_reduction = self.is_reduction_dim (ref, iter_name) and self.map[dd] >= 0
      space_reduction = not self.is_reduction_dim (ref, iter_name) and self.map[dd] >= 0 and ref.get_pi_by_name (iter_name) == -1
      if (true_reduction): # or space_reduction):
        pdim = self.map[dd]
        if (space_reduction):
          pdim = ref.get_pi_by_name (iter_name)
          #print ("Will use p-dim {} for dimension {}".format (pdim, iter_name))
        if (option_debug >= 4):
          print ("\t\tOperator {} - Is reduction dim ({}:{}) : map[{}]={} - psize={}".format (self.name, dd, iter_name, dd, pdim, PP.get_dim_size (pdim)))
        if (PP.get_dim_size (pdim) > 1): ## > 1 or == 1?
          return dd
    return -1
    
  def has_mapped_reduction_dimension (self, ref, PP):
    has = self.get_mapped_reduction_dimension (ref, PP) >= 0
    return has

  def get_point_trip_count (self, idim, PP, extent, ttc):
    num_proc = '1'
    iter_name = self.dims[idim]
    if (iter_name in ttc):
      num_proc = ttc[iter_name]
    expr = '{}({}, {})'.format (DIMAGE_CEIL, extent, num_proc)
    return expr

  ## Statement: Construct the loop structure of the current statement.
  def build_loop_structure (self, idim, PP, is_point, producers, skip_red_dim, for_accum, only_ub = False):
    nref = len(self.accs)
    out_ref = self.accs[nref-1]
    #red_dim_found = self.get_mapped_reduction_dimension (out_ref, PP)
    iter_name = self.dims[idim]
    red_dim_found = self.is_reduction_dim (out_ref, iter_name)
    if (skip_red_dim and option_debug >= 3):
      print ("[INFO] Showing reduction info>> found {}, expected {}".format (red_dim_found, idim))
    if (skip_red_dim and red_dim_found):
      return ''
    #print ('Producers: {}'.format (producers))
    ttc = self.collect_tile_trip_counts (producers)
    if (option_debug >= 2):
      for aa in producers:
        print ("Array {}[] produced at {}()".format (aa, producers[aa].get_name ()))
      for tt in ttc:
        print ("Tile UB {} : {}".format (tt, ttc[tt]))
    trip_count = 0
    dim_name = self.dims[idim]
    pdim = self.map[idim]
    is_replicated = (pdim == -1)
    num_proc = 1
    comment = '/* '
    if (not is_replicated):
      num_proc = PP.get_dim_size (pdim)
    # Extract the trip count from the extent of the array dimension accessed.
    array_ref = None
    for ref in self.accs:
      if (ref.is_dim_used (dim_name)):
        array_ref = ref
        if (is_point):
          trip_count = ref.get_dim_size_if_used (dim_name)
        else:
          array_pdim = ref.get_proc_map_by_dim_name (dim_name)
          if (array_pdim >= 0):
            num_proc = PP.get_dim_size (array_pdim)
            trip_count = num_proc 
          else:
          #  print ('[ERROR@build_loop_structure]: Unable to determine number of tiles by processor map')
          #  sys.exit (42)
            trip_count = ref.get_dim_size_if_used (dim_name)
        # WARNING: Assuming a loop dimension only appears ONCE in an access function.
        if (trip_count <= 0):
          print ("[ERROR@build_loop_structure]: Invalid trip count (trip_count={}) found. Aborting ...".format (trip_count))
          sys.exit (42)
        break
    itervar = self.get_iterator_variable (idim, is_point)
    array_pdim = array_ref.get_proc_map_by_dim_name (dim_name)
    lexical_ub = ''
    if (dim_name in ttc):
      lexical_ub = ttc[dim_name]
    #if (not dim_name in ttc):
    #  print ("Current operator : {}".format (self.name))
    #  print ('Did not find key {}'.format (dim_name))
    #  print ("Current ITS: {}".format (ttc))
    #  sys.exit (42)
    lb = 'ERROR'
    ub = 'ERROR'
    denom = 1
    lcm = self.PP.lcm ()
    if (not for_accum):
      if (pdim == array_pdim and pdim >= 0):
        # Partitioned space, mapping dimensions aligned.
        if (is_point):
          lb = 0
          ub = '({}({},{})) - 1'.format (DIMAGE_CEIL, trip_count, lcm) #, num_proc) + ' - 1'
          #if (int(num_proc) == 1):
          #  ub = '{} - 1'.format (trip_count)
        else:
          lb = str(PP.get_processor_coordinate_variable (pdim))
          #lb = 0
          ub = lb
        comment += 'IS: M, DS: M'
      elif (pdim != array_pdim and pdim >= 0 and array_pdim >= 0):
        # Both distributed but un-matched. Will have converted to slice-mode by all-gather within communicator.
        # Thus, point loops access a single tile in the slice.
        # Tile iterators using processor coordinates should remain explicit since 
        # we have to find the right tile within the slice. CHECKED.
        if (is_point):
          lb = 0
          ub = '{}({},{}) - 1'.format(DIMAGE_CEIL, trip_count, lcm) # num_proc) + ' - 1'
        else:
          lb = str(PP.get_processor_coordinate_variable (pdim))
          ub = lb
        comment += 'IS: M, DS: M, IS != DS: AllGather, Coordinates explicit.'
      elif (pdim < 0 and array_pdim >= 0):
        # Un-Partitioned iteration space, partitioned data space. 
        # In other words: Need all tiles, but got one.
        # Will require an AllGather.
        print ('[WARNING@build_loop_structure]: Statement {} (IS:Unmapped, DS:Mapped) - AllGather Required.'.format (self.name))
        if (is_point):
          lb = 0
          #ub = str(trip_count) + ' - 1'
          ub = '{}({},{}) - 1'.format(DIMAGE_CEIL, trip_count, lcm)# + ' - 1'
        else:
          lb = 0
          ub = str(num_proc) + ' - 1'
        comment += 'IS: U, DS: M, AllGather. Coordinates not explicit.'
      elif (pdim >= 0 and array_pdim < 0):
        # Partitioned iteration space, un-partitioned data space. 
        # Use processor coordinate since we have the full data slice.
        if (is_point):
          lb = 0
          #ub = '{}({},{})'.format(DIMAGE_CEIL, trip_count, num_proc) + ' - 1'
          ub = '{}({},{}) - 1'.format(DIMAGE_CEIL, trip_count, lcm)# + ' - 1'
          #if (int(num_proc) == 1):
          #  ub = '{} - 1'.format (trip_count)
        else:
          nblock = self.get_block_stride (idim, producers)
          stride = self.get_number_tiles_along_dim (array_ref, idim, producers)
          lcm = self.PP.lcm ()
          scale = 1 #lcm / stride
          lb = str(PP.get_processor_coordinate_variable (pdim))
          ub = lb
          if (scale > 1):
            pcoord = str(PP.get_processor_coordinate_variable (pdim))
            lb = '{} * {}'.format (pcoord, scale)
            ub = '({} + 1) * {} - 1'.format (pcoord, scale)
        comment += 'IS: M, DS: U'
      elif (pdim == array_pdim and pdim < 0):
        # Full (un-partitioned) iteration space
        if (is_point):
          lb = 0
          ub = '{}({},{}) - 1'.format(DIMAGE_CEIL, trip_count, lcm) 
          #ub = get_dimension_size_as_str (self, stmt, dd, PP):
          #ub = self.get_point_trip_count (idim, PP, trip_count, ttc) + ' - 1'
        else:
          lb = 0
          ub = self.get_block_stride (idim, producers) - 1 
          #PP.lcm () - 1
        comment += 'IS: U, DS: U '
    else:
      ## NOTE: lexical_ub will be empty if it's corresponding dimension is not used
      ## by any array by its data_generator.
      array_pdim = out_ref.get_pi_by_dim_name_if_used (self.dims[idim])
      comment += 'accum-case : pi={}'.format (array_pdim)
      if (array_pdim < 0):
        if (is_point):
          lb = 0
          ub = '{}({},{}) - 1'.format (DIMAGE_CEIL,trip_count,lcm) 
        else:
          lb = 0
          #block_stride = self.get_block_stride_from_ref (out_ref, idim, producers)
          tiles = self.get_number_tiles_along_dim (array_ref, idim, producers)
          blocks = self.PP.lcm () / tiles
          ub = '{} - 1'.format (blocks) 
        comment += "; pi:unmapped; mu:don't care"
      else:
        # Partitioned space, mapping dimensions aligned.
        if (is_point):
          lb = 0
          ub = '({}({},{})) - 1'.format (DIMAGE_CEIL, trip_count, lcm) #num_proc) + ' - 1'
          #if (int(num_proc) == 1):
          #  ub = '{} - 1'.format (trip_count)
        else:
          if (pdim < 0):
            print ('Cannot have mu={} and pi={} for an accumulation loop'.format (pdim, array_pdim))
            sys.exit (42)
          lb = str(PP.get_processor_coordinate_variable (pdim))
          ub = lb
        comment += "pi : {}; mu : {}".format (array_pdim, pdim)
    comment += ' */'
    if (only_ub):
      return ub
    ret = 'for ({} = {}; {} <= {}; {}++) {}'.format (itervar, lb, itervar, ub, itervar, comment)
    return ret

  ## @Statement: return the number of blocks to be executed at a 'b' loop.
  def get_block_stride (self, idim, producers):
    ttc = self.collect_tile_trip_counts (producers)
    trip = None
    iter_name = self.dims[idim]
    ret = 1
    mu_dim = self.map[idim]
    if (iter_name in ttc and mu_dim >= 0):
      trip = ttc[iter_name]
      ret = self.PP.lcm () / int(trip)
    if (ret == 1 and mu_dim >= 0):
      return ret
    num_proc = 1
    if (mu_dim >= 0):
      num_proc = self.PP.get_dim_size (mu_dim)
    ret = ret / num_proc
    return ret

  ## Return the number of blocks that must be traversed for a given loop dimension
  ## and on a given array (ref).
  def get_block_stride_from_ref (self, ref, idim, producers):
    ttc = self.collect_tile_trip_counts (producers)
    trip = None
    iter_name = self.dims[idim]
    ret = 1
    if (iter_name in ttc):
      trip = ttc[iter_name]
      ret = self.PP.lcm () / int(trip)
    if (ret == 1):
      return ret
    num_proc = 1
    pi_dim = ref.get_proc_map_by_dim_name (iter_name)
    if (pi_dim >= 0):
      num_proc = self.PP.get_dim_size (pi_dim)
    ret = ret / num_proc
    return ret

  ## Return the number of blocks that must be traversed for a given loop dimension
  ## and on a given array (ref).
  def get_number_tiles_along_dim (self, ref, idim, producers):
    ttc = self.collect_tile_trip_counts (producers)
    trip = None
    iter_name = self.dims[idim]
    mu_dim = self.map[idim]
    pi_dim = ref.get_pi_by_dim_name_if_used (iter_name)
    ret = 1
    #if (iter_name in ttc):
    #  ret = ttc[iter_name]
    if (mu_dim == pi_dim  and mu_dim >= 0):
      ret = self.PP.lcm () / self.PP.get_dim_size (pi_dim)
    elif (mu_dim >= 0 and pi_dim < 0):
      ret = self.PP.lcm () / self.PP.get_dim_size (mu_dim)
    elif (pi_dim < 0):
      ret = self.PP.lcm ()
    elif (mu_dim < 0 and pi_dim >= 0):
      ret = self.PP.lcm () / self.PP.get_dim_size (pi_dim)
    else:
      ret = 9999999999
    return ret

  ## @Statement
  def build_l2_loop (self, level, trip, skip_red_dim, gen_mode, producers = None):
    nref = len(self.accs)
    out_ref = self.accs[nref-1]
    iter_name = self.dims[level]
    red_dim_found = self.is_reduction_dim (out_ref, iter_name)
    if (skip_red_dim and option_debug >= 3):
      print ("[INFO] Showing reduction info>> found {}, expected {}".format (red_dim_found, level))
    if (skip_red_dim and red_dim_found):
      return '' #iter = {},  srd = {} - rdf = {}'.format (iter_name, skip_red_dim,red_dim_found)
    ref = None
    for rr in range(len(self.accs)):
      temp = self.accs[rr]
      if (temp.is_dim_used (iter_name)):
        ref = temp
        break
    bi='b{}'.format (level)
    ti='t{}'.format (level)
    if (trip == None):
      lcm = self.PP.lcm ()
      tiles = 1
      mu_dim = self.map[level]
      if (mu_dim >= 0):
        trip = self.PP.get_dim_size (mu_dim)
      #trip = lcm / tiles
    if (trip == None):
      trip = 1
    #stride = self.PP.lcm () / int(trip)
    stride = self.get_number_tiles_along_dim (ref, level, producers)
    l2_loop_lb = '{}*{}'.format (bi,stride)
    l2_loop_ub = '({}+1) * {} - 1'.format (bi,stride)
    if (gen_mode == L2_LOOP_GENMODE_LB):
      return l2_loop_lb
    if (gen_mode == L2_LOOP_GENMODE_UB):
      return l2_loop_ub
    self.ntpd.append (int(stride))
    ret = ''
    ret += 'for ({} = {}; /* lcm={} */'.format (ti, l2_loop_lb, self.PP.lcm ())
    ret += '{} <= {}; '.format (ti,l2_loop_ub)
    ret += '{}++)'.format (ti)
    return ret

  def build_cannonical_loop (self, level):
    iter_name = 'i{}'.format(level)
    trip_count = self.get_loop_dim_tripcount (level)
    indent = '  ' * (level + 1)
    ret = '{}for ({} = 0; {} < {}; {}++)'.format (indent, iter_name, iter_name, trip_count, iter_name)
    return ret


  # Return the list i0, i1, ... as a string and separated by commas.
  def get_iterator_str_list (self):
    ret = ''
    for ii,dd in enumerate(self.dims):
      if (ii > 0):
        ret += ', '
      ret += 'i{}'.format (ii)
    return ret


  # Return the list i0, i1, ... as a string and separated by commas,
  # and which are used in @ref.
  def get_iterator_str_list_used_in_ref (self, ref):
    ret = ''
    used = 0
    for ii in range(ref.get_num_dim ()):
      idim = self.get_dim_used_in_ref_dim (ref, ii)
      if (idim >= 0):
        if (used > 0):
          ret += ', '
        ret += 'i{}'.format (idim)
        used += 1
    return ret

  def get_sliced_iterator_str_list_used_in_ref (self, ref):
    ret = ''
    for ii in range(ref.get_num_dim ()):
      idim = self.get_dim_used_in_ref_dim (ref, ii)
      mu_dim = self.get_mu_dim_map (idim)
      code_iter = 'i{}'.format (idim)
      if (ret != ''):
        ret += ', '
      if (mu_dim >= 0):
        np = self.PP.get_dim_size (mu_dim)
        ret += '(t{} * {} + i{})'.format (idim, np, idim)
      else:
        ret += code_iter
    return ret

  # Return the list of tile iterators as a comma-separated string list.
  # If the loop dimension is mapped, we return the processor coordinate
  # associated to it; Otherwise '0' is returned.
  def get_tile_iterator_str_list_complete (self, ref, PP):
    ret = ''
    used = 0
    for ii,dd in enumerate(self.dims):
      iter_name = self.dims[ii]
      if (ref.is_dim_used (iter_name)):
        if (used > 0):
          ret += ', '
        pdim = self.map[ii]
        if (pdim >= 0):
          # If the dimension is mapped, then the operator will only access
          # a tile within its slice. Return tile offset '0' w.r.t to
          # the slice.
          array_pdim = ref.get_proc_map_by_dim_name (iter_name)
          if (array_pdim >= 0):
            if (pdim != array_pdim):
              print ("[ERROR@get_tile_iterator_str_list]: Processor dimension mismatch between operator {}({}) and {}[{}]".format (self.name, iter_name, ref.get_name (), iter_name))
              sys.exit (42)
            ret += '0'
          else:
            # Operator is mapped, array dimension is replicated. 
            # Hence, we only access the slice corresponding to the processor
            # coordinate.
            ret += PP.get_processor_coordinate_variable (pdim)
        else:
          # The operator has a loop dimension unmapped. We will access the full
          # slice. Hence, return the tile iterator corresponding to the loop.
          ret += 't{}'.format (ii)
        used += 1
    return ret

  def assemble_list (self, tup):
    ret = ''
    for tt in tup:
      if (ret != ''):
        ret += ', '
      ret += tt
    return ret

  ## statement.permute_tile_access (): Must be used from the producer (generator)
  ## of an array.
  def permute_tile_access (self, ref, tiles, is_acum):
    ## Not used.
    if (is_acum):
      return tiles
    num_data_dim = ref.get_num_dim ()
    if (num_data_dim == 1):
      return tiles
    pi_map = ref.get_pi_map ()
    mu_map = self.map 
    num_proc_dim = self.PP.get_num_dim ()
    tup = tiles.split(', ')
    if (num_data_dim == 2):
      if (len (tup) != 2):
        print ("did not find 2 dimensions")
        sys.exit (42)
      if (len (pi_map) != 2):
        print ("pi not find 2 dimensions")
        sys.exit (42)
      if (len (mu_map) != 2):
        print ("mu not find 2 dimensions")
        sys.exit (42)
      print ("\t\t Input tile order: {} - Output tile order: {}, mu={}, pi={}, ntpd[0]={}".format (tiles, self.assemble_list ([tup[1],tup[0]]),  mu_map[1], pi_map[1], self.ntpd[0]))
      mu_val = mu_map[1]
      if (num_proc_dim == 2 and pi_map[1] == -1 and mu_val == 0 and self.PP.get_dim_size (mu_val) > 1):
        return self.assemble_list ([tup[1],tup[0]])
      if (num_proc_dim == 2 and pi_map[1] == -1 and mu_val == 0 and self.PP.get_dim_size (mu_val) > 1 and self.ntpd[0] > 1):
        return self.assemble_list ([tup[1],tup[0]])
      if (num_proc_dim == 1 and pi_map[1] == -1 and mu_val == 0 and self.PP.get_dim_size (mu_val) > 1):
        return self.assemble_list ([tup[1],tup[0]])
      mu_val = mu_map[1]
      if (num_proc_dim == 3 and pi_map[1] == -1 and mu_val >= 1 and self.PP.get_dim_size (mu_val) > 1 and self.ntpd[0] > 1):
        return self.assemble_list ([tup[1],tup[0]])
      return tiles
    if (num_data_dim == 3):
      ## TODO: Double check this.
      ## QUESTION: WHICH OF THE MUs TO USE?
      print ("\t\t Input tile order (3DA): {} :: {} - Output tile order: {}, mu={}, pi={}".format (tiles, ref.get_name (), self.assemble_list ([tup[1],tup[0],tup[2]]),  mu_map[1], pi_map[1]))
      mu_val = mu_map[1]
      if (num_proc_dim == 2 and pi_map[1] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[1],tup[0],tup[2]])
      if (num_proc_dim == 1 and pi_map[1] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[1],tup[0],tup[2]])
      mu_val = mu_map[2]
      print ("\t\t Input tile order (3DB): {} :: {} - Output tile order: {}, mu={}, pi={}".format (tiles,  ref.get_name (), self.assemble_list ([tup[2],tup[0],tup[1]]),  mu_map[2], pi_map[2]))
      if (num_proc_dim == 2 and pi_map[2] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[2],tup[0],tup[1]])
      if (num_proc_dim == 1 and pi_map[2] == -1 and mu_val >= 0 and self.PP.get_dim_size (mu_val) > 1): 
        return self.assemble_list ([tup[2],tup[0],tup[1]])
      return tiles
    if (num_data_dim == 4):
      ## TODO: Pending to implement cases (1d, 2d and 3d arrays)
      return 'ERROR'
    return 'ERROR'
      
        
      

  ## Return the iteration space dimension used in position adim of the given ref.
  def get_dim_used_in_ref_dim (self, ref, adim):
    iter_name = ref.get_iter_name (adim)
    for ii in self.dims:
      if (iter_name == self.dims[ii]):
        return ii
    return -1

  # Return the permutation vector of an array reference.
  # This function makes sense when the @self instance is the 
  # producer of the array being accessed by @ref.
  def get_permutation_vector_from_map (self, ref, PP):
    ## Permutation vectors removed in lieu of including tile header and using tile maps.
    ret = [-2] * ref.get_num_dim ()
    left_i = set()
    left_p = set()
    for dd in self.dims:
      #iter_name = self.dims[dd]
      #if (ref.is_dim_used (iter_name)):
      left_i.add (dd)
    for pp in range(PP.get_num_dim ()):
      left_p.add (pp)
    n_used = 0
    print ("PI-map of reference {}: {}".format (ref.get_name (), ref.get_pi_map ()))
    print ("MU-map of statement {}: {}".format (self.get_name (), self.get_mu_map ()))
    for dd in self.dims:
      iter_name = self.dims[dd]
      #if (ref.is_dim_used (iter_name)):
      stmt_pdim = self.map[dd]
      if (stmt_pdim >= 0):
        print ('PV at {}, setting entry {} to {}'.format (self.name, dd, stmt_pdim))
        ret[dd] = stmt_pdim
        n_used += 1
        left_i.remove (dd)
        left_p.remove (stmt_pdim)
    print ("Pending processor dimensions: {}".format (left_p))
    print ("Unmapped iteration-space dimensions: {}".format (left_i))
    if (len(left_p) == 1):
      pending_i = next(iter(left_i))
      pending_p = next(iter(left_p))
      # The below condition returns an empty perm-vec when the 
      # missing processor dimension is not the first one and not a degenerate
      # processor dimension (of length 1).
      if (pending_i > 0 and PP.get_dim_size (pending_p) == 1): 
        # FIXME: will only work on 2D grids. See top NOTE.
        return None
      if (pending_i >= 0 and pending_p >= 0):
        ret[pending_i] = pending_p
        left_i.remove (pending_i)
        left_p.remove (pending_p)
    print ('Partial PV at {}: {}'.format (self.name, ret))
    # Shouldn't have pending dimensions.
    if (len(left_p) > 0):
      return None
    return ret

  def get_tile_iterator (self, idim):
    varname = 't{}'.format (idim)
    return varname

  def get_block_iterator (self, idim):
    varname = 'b{}'.format (idim)
    return varname

  def get_point_iterator (self, idim):
    varname = 'i{}'.format (idim)
    return varname

  def get_used_tile_iterator_list (self, ref):
    ret = ''
    adims = ref.get_dims ()
    for dd in adims:
      iter_name = adims[dd]
      for ii in self.dims:
        if (iter_name == self.dims[ii]):
          if (ret != ''):
            ret += ', '
          ret += self.get_tile_iterator (ii)
    return ret

  ## statement.
  def get_constant_tile_iterator_list (self, ref, constant):
    ret = ''
    for dd in self.dims:
      iter_name = self.dims[dd]
      if (ref.is_dim_used (iter_name)):
        if (ret != ''):
          ret += ', '
        ret += str(constant)
    return ret

  # @Statement: Return a comma-separated list of used tile iterators.
  # NOTE: Function get_tile_iterator_str_list_complete() performs a more 
  # complex job, which is to inline the iterator valued into the expression.
  def get_tile_iterator_str_list (self, ref, PP, producers, is_acum):
    ret = ''
    used = 0
    if (self.is_data_generator () or producers == None):
      for ii in range(ref.get_num_dim ()):
        idim = self.get_dim_used_in_ref_dim (ref, ii)
        if (used > 0):
          ret += ', '
        ret += self.get_tile_iterator (idim)
        used += 1
      return ret
    #for ii,dd in enumerate(self.dims):
    ttc = self.collect_tile_trip_counts (producers)
    comm_type = self.determine_communication_type (ref, PP)
    is_outgoing = False
    nacc = len(self.accs)
    if (not self.is_data_generator () and not self.is_data_sink ()
        and ref.get_name () == self.accs[nacc-1].get_name ()):
      is_outgoing = True
    is_allgather = is_outgoing and comm_type == COMM_TYPE_LOCAL_SLICE
    for ii in range(ref.get_num_dim ()):
      idim = self.get_dim_used_in_ref_dim (ref, ii)
      if (idim >= 0):
        ## Next, determine if a tile iterator is 'local'. If that's the case,
        ## the data space dimension is effectively distributed, and the tile 
        ## index offset associated to the dimension becomes zero. We enforce
        ## this by shifting the tile bound by a block multiple
        mu_pdim = self.map[idim]
        iter_name = self.dims[idim]
        shift = ''
        match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
        #if (match_dim):
        #  shift = ' * 0'
        #elif (ref.get_pi_dim_map (ii) < 0 and mu_pdim >= 0):
        num_proc = 1
        if (mu_pdim >= 0):
          trip = None
          if (iter_name in ttc):
            trip = ttc[iter_name]
          loop_lb = self.build_l2_loop (idim, trip, False, L2_LOOP_GENMODE_LB, producers)
          shift = ' - {}'.format (loop_lb)
          num_proc = int(self.PP.get_dim_size (mu_pdim))
        print ("\n Shift {} at {}({}) = {} --- pi={}".format (ref.get_name (), self.name, iter_name, shift, ref.get_pi_dim_map (ii)))
        # If we only have one processor, shifting doesn't make sense.
        if (num_proc == 1):
          shift = ''
        if (ref.get_pi_dim_map (ii) < 0 and not is_allgather): # array is not partitioned along this dimension
          shift = ''
        if (used > 0):
          ret += ', '
        # If it's an accumulation loop and if the data-space dimension is unmapped, we proceed
        # to cancel shift as well, since we have to accumulate everything from the
        # temporary buffer.
        if (is_acum and ref.get_pi_dim_map (ii) < 0):
          shift = ''
        ret += self.get_tile_iterator (idim) + shift
        used += 1
    return ret

  # Return a reordered list of tile iterators.  This routine is normally
  # invoked when the lexico-positivity of an access is False. When the
  # iterators are not reordered we return None.
  def get_reordered_tile_iterator_str_list (self, ref, PP, producers, use_full_extent = False):
    if (not ref.get_name () in producers):
      print ('[ERROR@get_reordered_tile_iterator_str_list]: Found un-produced array {}'.format (ref.get_name ()))
      sys.exit (42)
    prod = producers[ref.get_name ()]
    permvec = prod.get_permutation_vector_from_map (ref, PP)
    if (permvec == None):
      print ('\t[INFO@get_reordered_tile_iterator_str_list]: returning empty perm-vec for ref {} @ stmt {}.'.format (ref.get_name (), self.name))
      return None
    temp = [None] * ref.get_num_dim ()
    ttc = self.collect_tile_trip_counts (producers)
    for dd in self.dims:
      iter_name = self.dims[dd]
      adim = ref.get_dim_if_used (iter_name)
      # Next, determine if a tile iterator is 'local'. If that's the case,
      # the data space dimension is effectively distributed, and the tile 
      # index offset associated to the dimension becomes zero. We enforce
      # this by multiplying by a zero factor.
      mu_pdim = self.map[dd]
      factor = ''
      match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
      is_subset_dim = ref.is_mu_map_dim_strict_subset_of_pi (iter_name, mu_pdim)
      prod_idims = prod.get_dims ()
      tile_size = 1
      if (adim >= 0):
        iter_name_in_prod = prod_idims[adim]
        if (iter_name_in_prod in ttc):
          tile_size = ttc[iter_name_in_prod]
      # Below: match_dim addresses the case of mu and pi maps matching in full.
      # When this occurs, computation becomes relative to this loop dimensions becomes
      # local, and we nullify these iterators by appending a '* 0' factor. The 
      # assumption here is that the corresponding buffer is just big enough
      # to fit the needed data.
      # The second case of the 'or' arises from a special case where implicit
      # tiling took place, likely from the generator, and the array still being
      # all-gathered along this dimension. This means we have the full slice along
      # such dimension, but it may have been reorganized with the all-gather.
      # Hence... TODO-COMPLETE-RATIONALE.
      if (match_dim or (not use_full_extent and is_subset_dim and int(tile_size) > 1)):
        factor = ' * 0'
      if (adim >= 0):
        new_place = permvec[adim]
        temp[new_place] = self.get_tile_iterator (dd) + factor
    print ("=====> Reodered tile iterator list for reference {}[{}] @ statement {}({}): {}".format (ref.get_name (), ref.get_dims (), self.name, self.dims, temp))
    if (temp == None):
      return None
    ret = ''
    for tt in temp:
      if (ret != ''):
        ret += ', '
      if (tt == None):
        print ("[ERROR]: None entry found : {}".format (temp))
        return None
      ret += tt
    return ret

  def get_tile_trip_count_str_list (self, ref, PP, producers):
    nprocdim = PP.get_num_dim ()
    if (not ref.get_name () in producers):
      print ('[ERROR@get_tile_trip_count_str_list]: Found un-produced array {}'.format (ref.get_name ()))
      sys.exit (42)
    ttc = self.collect_tile_trip_counts (producers)
    ret = ''
    prod = producers[ref.get_name ()]
    permvec = prod.get_permutation_vector_from_map (ref, PP)
    temp = None
    # NOTE for below: the True branch of the below code will work
    # for the case when the dimensionality of the processor grid
    # matches that of the data space.
    if (permvec != None and ref.get_num_dim () == nprocdim):
      temp = [None] * ref.get_num_dim ()
      for idim in self.dims:
        iter_name = self.dims[idim]
        adim = ref.get_dim_if_used (iter_name)
        mu_pdim = self.map[idim]
        factor = ''
        match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
        degenerate_proc_dim = mu_pdim >= 0 and PP.get_dim_size (mu_pdim) == 1
        if (match_dim and not degenerate_proc_dim):
          factor = ' * 0'
        str_val = '1'
        if (iter_name in ttc):
          str_val = ttc[iter_name]
        if (adim >= 0):
          new_place = permvec[adim]
          if (new_place != None):
            temp[new_place] = str_val + factor
          else:
            temp[adim] = str_val + factor
        #if (ref.is_dim_used (iter_name)):
        #  factor = ''
        #  str_val = '1'
        #  if (iter_name in ttc):
        #    str_val = ttc[iter_name]
        #  str_val = str_val + factor
        #  ret += str_val
    else:
      # The permutation vector is empty, so we just populate
      # the temporary vector with the collected trip counts (number of
      # tiles per dimension.
      temp = []
      for idim in self.dims:
        iter_name = self.dims[idim]
        adim = ref.get_dim_if_used (iter_name)
        if (adim < 0):
          continue
        mu_pdim = self.map[idim]
        factor = ''
        match_dim = ref.is_pi_map_dim_equal (iter_name, mu_pdim)
        if (match_dim):
          factor = ' * 0'
        str_val = '1'
        if (iter_name in ttc):
          str_val = ttc[iter_name]
        str_val = str_val + factor
        temp.append(str_val)
    for ii,tt in enumerate(temp):
      if (ret != ''):
        ret += ', '
      if (tt == None):
        print ("[INFO]  Access funcf for reference {}: {}".format (self.name, self.dims))
        print ("[INFO]  Perm vec: {}".format (permvec))
        print ("[INFO]  Semi-reordered access: {}".format (temp))
        print ("[ERROR] None entry found : {}".format (temp))
        print ("[INFO]  entry {} : {}".format (ii,tt))
        return None
      ret += tt
    return ret

  # Return a list of tile sizes, possibly obtained from dividing
  # the loop trip count by the number of processors along a mapped dimension.
  def get_tile_size_str_list (self, ref, PP, producers):
    ttc = self.collect_tile_trip_counts (producers)
    #print ("Tile trip counts @ GTSSL, stmt={}, ref={}[{}], ttc={}\n".format (self.name, ref.get_name (), ref.get_dims (), ttc))
    ret = ''
    arr_dims = ref.get_dims ()
    for dd in range(len(arr_dims)):
      iter_name = arr_dims[dd]
      idim = self.get_dim_by_name (iter_name)
      mu_dim = self.map[idim]
      pi_dim = ref.get_pi_by_dim_name_if_used (iter_name)
      num_proc_dim = 1
      if (mu_dim >= 0):
        num_proc_dim = self.PP.get_dim_size (mu_dim)
      if (dd > 0):
        ret += ', '
      extent = ref.get_extent_as_str (dd)
      num_proc_along_dim = 1
      num_proc_cand = []
      if (iter_name in ttc):
        num_proc_cand.append (ttc[iter_name])
      #print ("TTC at get_tile_size_str_list={}".format (ttc))
      num_proc_along_dim = PP.lcm () 
      tile_size = '{}(({}),{})'.format (DIMAGE_CEIL, extent, num_proc_along_dim)
      if (int(num_proc_along_dim) == 1):
        tile_size = '{}'.format (extent)
      ret += tile_size
    return ret

  ## @Statement:
  ## Return the list of tiles along each dimension.
  def get_tile_count_list (self, ref, PP, producers, is_allgat_out_slice):
    ttc = self.collect_tile_trip_counts (producers)
    ret = ''
    arr_dims = ref.get_dims ()
    for dd in range(len(arr_dims)):
      iter_name = arr_dims[dd]
      idim = self.get_dim_by_name (iter_name)
      mu_dim = self.map[idim]
      pi_dim = ref.get_pi_by_dim_name_if_used (iter_name)
      tc = self.get_number_tiles_along_dim (ref, idim, producers)
      if (pi_dim < 0 and not is_allgat_out_slice):
      #if (pi_dim < 0):
        # The above function is also used in build_l2_loop to determine the 
        # number of tiles that must be traversed by a loop. That traversal is
        # 'array independent (only depends on the mu mappings) and so in some
        # cases we really care about exclusively of the data mapping. This
        # condition fixes the case we really need the number of tiles along
        # a particular data-space dimension caused by a pi-mapping < 0 (unpartitioned).
        # We do this here since get_number_tiles_along_dim is used in other
        # places as well, while get_tile_count_list () is used exclusively for
        # generating the accesses to TILES.
        tc = self.PP.lcm ()
      ## Below we handle the case where a local computation is performed
      ## on a data slice that will then be all-gathered. In addition,
      ## the array is replicated along a particular dimension, and the 
      ## computation is mapped. Hence, the all-gather. In this case, 
      ## a temporary array has been allocated for the slice, and hence the
      ## the extents will be offset to the tiles locally computed.
      ## As a result, the number of effective tiles along the allocated slice
      ## has to be adjusted.
      if (mu_dim >= 0 and pi_dim < 0 and is_allgat_out_slice):
        denum = self.PP.get_dim_size (mu_dim)
        tc = '({}/{})'.format (PP.lcm (), denum)
      if (dd > 0):
        ret += ', '
      ret += '{} /* idim={} */'.format (tc, idim)
    return ret


  ## @Statement: Return the size (number of nodes) along a processor dimension.
  def get_num_proc_for_slice (self, ref, PP, producers, ttc):
    ## NOTE: ttc must have been populated.
    arr_dims = ref.get_dims ()
    ret = []
    prod = producers[ref.get_name ()]
    for dd in range(len(arr_dims)):
      iter_name = prod.get_dim_name (dd)
      #idim = prod.get_dim_by_name (iter_name)
      mu_dim = prod.get_mu_dim_map (dd)
      pi_dim = ref.get_pi_dim_map (dd)
      num_proc_dim = 1
      print ("[INFO] =============> ttc={}, it={}, mu={}, pi={}".format (ttc, iter_name, mu_dim, pi_dim))
      if (mu_dim >= 0 and pi_dim < 0): #and iter_name in ttc):
       num_proc_dim = self.PP.get_dim_size (mu_dim)
      ##  num_proc_dim = int(ttc[iter_name])
      ret.append (num_proc_dim)
    if (len(ret) == 1):
      return '{} /* {} */'.format (ret[0], ret)
    print ("[INFO] =============> Collected processor dimensions: {}".format (ret))
    ret = sorted(ret)
    first = ret[0]
    all_equal = True
    non_degen = 0
    for ds in ret:
      if (ds > 1):
        first = ds
        non_degen += 1
    if (non_degen == 0):
      return '1'
    if (non_degen == 1):
      return first
    print ("Found two or more non-degenerate (>1) grid space-dimensions. Need max.common divisor")
    sys.exit (42)
    #return ret[0]
    return '{} /* {} */'.format (ret[0], ret)

  # Statement.generate_access():
  # Generate a single (linearized) access reference.
  # Access type is determined by the number of tiles accessed in the 
  # current rank: TILE (single tile) and SLICE (multi-tile)
  # The SLICE case can account for multi-dimensional tiles.
  # Determining the access type considers the 4 (statement x array) scenarios:
  # case 1) mapped + mapped: single tile
  # case 2) unmapped + unmapped: single tile but which represents the full array
  # case 3) mapped + unmapped: all ranks have space for the full array, but 
  #   work only on their corresponding tile. Leads to reduce or allreduce 
  #   communication.
  # case 4) unmapped + mapped: all ranks attempt to work on the full spread 
  #   or slice of the array, hence require an all-gather first.
  def generate_access (self, ref, PP, producers, is_write = False, is_acum = False, use_full_extent = False, is_allgat_out_slice = False):
    acc_type = ref.get_access_type (self, PP, producers)
    ttc = self.collect_tile_trip_counts (producers)
    local_linear = (acc_type == ACC_TYPE_SLICE and is_write)
    macro = ''
    if (acc_type == ACC_TYPE_TILE):
      macro = DIMAGE_ACC_TILE
    elif (acc_type == ACC_TYPE_LIN or local_linear):
      macro = DIMAGE_ACC_LIN
    elif (acc_type == ACC_TYPE_SLICE):
      macro = DIMAGE_ACC_SLICE
    else:
      print ('[ERROR@generate_access]: Unexpected access type at statement {}, reference {}: {}'.format (self.name, ref.get_name (), acc_type))
      sys.exit (42)
    prod = producers[ref.get_name ()]
    ref_at_prod = prod.get_ref(0)
    is_lexpos = prod.is_mu_lexico_positive (ref, producers)
    special_lin = False 
    if (option_debug >= 10):
      print ("\n=====> SPECIAL LIN ({}) for ref={} at {}: rep={}, acc-type={}, lex_pos={}\n".format (special_lin, ref.get_name (), self.name, ref_at_prod.is_fully_replicated (), acc_type == ACC_TYPE_SLICE, is_lexpos))
    if (special_lin):
      macro = DIMAGE_ACC_LIN
    acc = '{}{}D'.format (macro, ref.get_ref_dim())
    acc += '('
    # Produce tile iterators: 0, t? or p?
    if (acc_type == ACC_TYPE_LIN or (local_linear) or special_lin):
      acc += ref.get_linearized_iterator_str_list (self, PP, producers, is_write, is_acum)
      acc += ', '
      #acc += ref.get_array_extents_as_str_list ()
      acc += ref.get_mapped_array_extents_as_str_list (self, False, is_write)
    elif (acc_type == ACC_TYPE_TILE):
      # Will include only tile iterators used in the access function of the 
      # current reference.
      acc += self.get_constant_tile_iterator_list (ref, 0)
      acc += ', '
      acc += self.get_iterator_str_list_used_in_ref (ref)
      acc += ', '
      acc += self.get_tile_size_str_list (ref, PP, producers)
      #acc += ref.get_array_extents_as_str_list ()
      #acc += ref.get_mapped_array_extents_as_str_list (None, True, is_write)
      #acc += ', '
      acc += ', '
      ## NOTE: statement.get_tile_count_list () deals with several tile-extent cases,
      ## including all-gather-outgoing-slices.
      acc += self.get_constant_tile_iterator_list (ref, 1)
    elif (acc_type == ACC_TYPE_SLICE and not special_lin):
      reordered_tile_iter = self.get_reordered_tile_iterator_str_list (ref, PP, producers, use_full_extent)
      acc += self.get_tile_iterator_str_list (ref, PP, producers, is_acum)
      acc += ', '
      #acc += self.get_iterator_str_list_used_in_ref (ref)
      acc += self.get_sliced_iterator_str_list_used_in_ref (ref)
      acc += ', '
      acc += ref.get_mapped_array_extents_as_str_list (None, False, is_write)
      acc += ', '
      acc += '{}'.format (self.get_num_proc_for_slice (ref, PP, producers, ttc))
      prod = None
      if (ref.get_name () in producers):
        prod = producers[ref.get_name ()]
      #acc += self.get_tile_vol (ref, ttc)
    else:
      print ("[ERROR] Unexpected access type")
      sys.exit (42)
    acc += ')'
    return acc

  ## Statement.generate_tile_fetch_code(): Insert RT call to find the data 
  ## tile corresponding to the passed tile iterators.
  ## 
  def generate_tile_fetch_code (self, df, ref, producers, is_acum, intermediate = None):
    tile_name = ref.get_tile_name (intermediate)
    tile_map = ref.get_tile_map_name (intermediate)
    tile_iters = self.get_used_tile_iterator_list (ref)  #self.get_tile_iterator_str_list (ref, PP, producers, is_acum)
    extent_list = ref.get_tile_extent_list ()
    source_array = ref.get_name ()
    is_out = self.is_outgoing_slice (ref)
    is_incom = self.is_incoming_slice (ref)
    if (option_debug >= 3):
      print ('[INFO] generate_tile_fetch_code (): stmt={}, ref={}, is_acum={}, interm={}, is_incom={}, is_outgoing={}'.format (self.name, ref.get_name (), is_acum, intermediate, is_incom, is_out))
    if (is_out and not is_acum and intermediate != None):
      tile_map = ref.get_tile_map_name () + '/* tile map same as target buffer */'
    if (is_out and not is_acum and self.is_true_communication (ref)):
      ref.set_use_slice (True)
      if (option_debug >= 5):
        print ("\t Slice name before : {}".format (source_array))
      source_array = ref.get_slice_varname (False)
      if (option_debug >= 5):
        print ("\t Slice name after: {}".format (source_array))
    if (is_incom and not is_acum):
      source_array = ref.get_slice_varname (True) + ' /* gathered slice */ '
    if (intermediate != None):
      source_array = intermediate 
    proc_size_list = self.PP.get_processor_geometry_list_from_map (ref.get_pi_map (), True)
    call = '{} *{} = {}_{}D ({}, {}, {}, {}, {});\n'.format (DIMAGE_DT, tile_name, DIMAGE_FETCH_TILE_FUNC, ref.get_num_dim (), source_array, tile_map, extent_list, tile_iters, proc_size_list)
    df.write (call)


  def generate_set_tile_coordinate (self, df, ref):
    source_array = ref.get_tile_name ()
    tile_iters = self.get_used_tile_iterator_list (ref)
    call = '{}_{}D({}, {});\n'.format (DIMAGE_SET_TILE_COORD_FUNC, ref.get_num_dim (), source_array, tile_iters)
    self.indent (df)
    df.write (call)

  ## @Statement.get_tile_pointer ():
  ## This function is called solely by statement.generate_dimage_operator_call ().
  def get_tile_pointer (self, ref, producers, is_allgat_slice, is_read = False):
    ret = ' &'
    refname = ref.get_tile_name ()
    if (is_allgat_slice and is_read):
      refname = ref.get_tile_name (ref.get_slice_varname (True))
      #if (ref.get_use_slice ()):
      #  refname = ref.get_slice_varname (False)
    ret += refname
    ret += '['
    ret += '{}{}D ('.format (DIMAGE_TILE_POINTER, ref.get_num_dim ())

    prod = producers[ref.get_name ()]
    ref_at_prod = prod.get_ref(0)

    ret += self.get_constant_tile_iterator_list (ref, 0)
    ret += ', '
    ret += self.get_tile_size_str_list (ref, self.PP, producers)
    ret += ', '
    ret += self.get_constant_tile_iterator_list (ref, 1)
    ret += ')]'
    return ret



  def get_allred_intermediate (self, ref):
    return 'interm_{}_at_{}'.format (ref.get_name (), self.name )

  ## @Statement:
  def generate_write_to_file_arguments (self, ref, PP, is_operator = False):
    acc = ''
    if (is_operator):
      acc += DIMAGE_RANK_ARRAY #PP.get_processor_coordinate_str_list ()
    else:
      acc += DIMAGE_RANK_ARRAY
    acc += ', '
    acc += ref.get_name ()
    acc += ', '
    #acc += ref.get_dimension_size_as_str_list (self, PP, ALLOC_MODE_SLICE)
    acc += ref.get_tile_extent_list ()
    acc += ', '
    acc += self.PP.get_processor_geometry_list_from_map (ref.get_pi_map (), False)
    return acc

  ## @Statement.generate_single_node_write_to_file_arguments ():
  ## Collect arguments for call to function that dumps an entire 
  ## -- single node -- matrix to a file.
  def generate_single_node_write_to_file_arguments (self, ref, PP):
    acc = ''
    acc += 'sna_' + ref.get_name ()
    acc += ', '
    acc += ref.get_dimension_size_as_str_list (self, PP, ALLOC_MODE_FULL)
    #acc += ref.get_tile_extent_list ()
    #acc += ', '
    #acc += self.PP.get_processor_geometry_list_from_map (ref.get_pi_map (), False)
    return acc

  ## Statement: Generate the arguments for reading one or more tiles into a buffer.
  def generate_read_from_file_arguments (self, ref, PP, check_mode = DIMAGE_CHECK_NO_CHECK):
    #send_size = ref.reference_get_local_volume (self)
    #recv_size = self.get_slice_vol_by_name (ref, PP)
    acc = ''
    array_name = ''
    #comment = ' /* send={} vs recv={} */ '.format (send_size, recv_size)
    comment = ''
    comm_type = self.determine_communication_type (ref, PP)
    if (check_mode == DIMAGE_CHECK_READ_REF_ARRAY):
      array_name = ref.get_name_for_check ()
      comment = ' /* DIMAGE_CHECK_READ_REF_ARRAY */ '
    elif (check_mode == DIMAGE_CHECK_CALL_CHECK):
      ## Array name for correctness check.
      array_name = ref.get_tile_name ()
      comment = ' /* DIMAGE_CHECK_CALL_CHECK - get_tile_name () */ '
      if (ref.get_use_slice ()):
        array_name = ref.get_tile_name (ref.get_slice_varname (True)) + '/* fetched slice */'
        comment = ' /* DIMAGE_CHECK_CALL_CHECK - get_slice_varname (T) */ '
    elif (comm_type == COMM_TYPE_LOCAL):
      array_name = ref.get_name ()
      comment = ' /* COMM_TYPE_LOCAL */ '
    elif (self.is_true_communication (ref)):
      array_name = ref.get_slice_varname (False)
      comment = ' /* ITC */ '
    else:
      array_name = ref.get_name ()
      comment = ' /* GRFFA - default */ '
    acc += array_name
    acc += ', '
    if (check_mode == DIMAGE_CHECK_CALL_CHECK):
      #acc += ref.get_name_for_check ()
      acc += ref.get_sna_ref_name ()
      acc += ', '
    acc += ref.get_array_extents_as_str_list ()
    acc += ', '
    acc += ref.get_dimension_size_as_str_list (self, PP, ALLOC_MODE_TILE)
    acc += ', '
    acc += self.get_tile_iterator_str_list (ref, PP, None, False)
    acc += comment
    return acc
    

  # Statement: Generate the right-hand expression of the initialization statement.
  # Take into account if an array dimension has been distributed
  # or if its fully local.
  def create_dimension_expression (self, ref, PP, idim, pc):
    expr = ''
    if (pc >= 0):
      proc_coord = PP.get_processor_coordinate_variable (pc)
      tile_size = ref.get_dimension_size_as_str (self, idim, PP)
      tile_var = 't{}'.format (idim)
      iter_var = 'i{}'.format (idim)
      expr = '({} * {} + {})'.format (tile_var,tile_size,iter_var)
    else:
      iter_var = 'i{}'.format (idim)
      expr = '({})'.format (iter_var)
    return expr
    
  # Statement: Generate a linearized global expression for a data generator.
  def generate_init_expression (self, PP):
    ref = self.accs[0]
    #macro = DIMAGE_ACC_TILE
    macro = DIMAGE_INIT_DIAG
    acc = '{}_{}D'.format (macro, ref.get_ref_dim())
    expr = '{}('.format (acc)
    for ii,pc in enumerate(self.map):
      dim_expr = self.create_dimension_expression (ref, PP, ii, pc)
      expr += dim_expr
      expr += ', '
    expr += ref.get_array_extents_as_str_list ()
    expr += ')'
    return expr

  ## Statement: 
  def compute_matmul (self, ref_out, ref_in, ref_ker):
    N0 = int (ref_out.get_extent_as_str (0))
    N1 = int (ref_out.get_extent_as_str (1))
    N2 = int (ref_ker.get_extent_as_str (0))
    mat_out = ref_out.get_data () 
    mat_in = ref_in.get_data () 
    mat_ker = ref_ker.get_data () 
    for ii in range(N0):
      for jj in range(N1):
        for kk in range (N2):
          mat_out[ii * N1 + jj] += mat_in[ii * N2 + kk] * mat_ker[kk * N1 + jj]
    self.write_matrix_to_file (mat_out, ref_out.get_name (), N0, N1)

  def get_operator_c_filename (self):
    operator_filename = '{}.dimage-op.c'.format (self.name)
    return operator_filename 

  def get_operator_bin_filename (self):
    operator_filename = self.get_operator_c_filename ()
    bin_filename = re.sub ("\.c",".exe", operator_filename)
    return bin_filename

  # Generate a baseline C-implementation.
  def compute_operator (self, op_refs, init_val = 1.0):
    operator_filename = self.get_operator_c_filename ()
    rcf = open(operator_filename, 'w')
    indent = '  '
    rcf.write ('#include "dimage-rt.h"\n')
    rcf.write ('int {}[] = {};\n'.format (DIMAGE_GRID_DIMS, '{1,1,1,1,}'))
    rcf.write ('int main () {\n')
    trips = {}
    for dd in self.dims:
      iter_name = self.dims[dd]
      for ref in op_refs:
        if (not iter_name in trips and ref.is_dim_used (iter_name)):
          ub = ref.get_array_extent_by_dim_name (iter_name)
          trips[iter_name] = ub
    if (not self.is_data_generator ()):
      for ref in op_refs:
        data_source = ref.get_matrix_filename ()
        #extents = ref.get_array_extents_as_str_list ()
        array_size = ref.get_array_size_as_product_str ()
        rcf.write ('{}{} * {} = read_matrix_from_file (\"{}\", {});\n'.format(indent, DIMAGE_DT, ref.get_name (), data_source, array_size))
    for it in self.dims:
      iter_name = self.dims[it]
      rcf.write ('{}int {};\n'.format (indent, iter_name))
    depth = 1
    if (not self.is_data_sink () and not self.is_data_generator ()):
      for it in self.dims:
        iter_name = self.dims[it]
        ub = trips[iter_name]
        rcf.write ('{}for ({} = 0; {} < {}; {}++) {}\n'.format (indent * depth, iter_name, iter_name, ub, iter_name, '{'))
        depth += 1
      stmt_body = ''
      nref = len(op_refs)
      for ii,ref in enumerate(op_refs):
        if (ii == 0):
          continue
        access = ref.gen_canonical_access ()
        if (ii > 1):
          stmt_body += ' * '
        stmt_body += access
      write_access = op_refs[0].gen_canonical_access ()
      stmt_body = '{}{} += {};\n'.format (indent * depth, write_access, stmt_body)
      rcf.write (stmt_body)
      for it in self.dims:
        depth -= 1
        rcf.write ('{}{}\n'.format (indent * depth, '}'))
    out_size = op_refs[0].get_array_size_as_product_str ()
    mat_dim = op_refs[0].get_num_dim ()
    extents = op_refs[0].get_array_extents_as_str_list ()
    ## Generate code for logging, and free the arrays
    if (not self.is_data_sink () and not self.is_data_generator ()):
      data_sink = op_refs[0].get_matrix_filename (self.name)
      rcf.write ('{}{}_{}D (\"{}\", {}, {});\n'.format(indent, WRITE_MATRIX_TO_FILE, mat_dim, data_sink, op_refs[0].get_name (), extents))
      for ii,ref in enumerate(op_refs):
        rcf.write ('{}free ({});\n'.format (indent, ref.get_name ()))
    else:
      data_sink = op_refs[0].get_matrix_filename ()
      rcf.write ('{}generate_datafile_{}D (\"{}\", {}, {});\n'.format(indent, mat_dim, data_sink, extents, init_val))
    rcf.write ('{}return 0;\n'.format (indent))
    rcf.write ('}')
    rcf.close ()

  def write_matrix_to_file (self, mat, name, N0, N1):
    fmat = open ('{}-ref.mat'.format (name), 'w') 
    for ii in range(N0):
      for jj in range(N1):
        fmat.write ('{:.6f} '.format (mat[ii * N1 + jj]))
      fmat.write ('\n')
    fmat.close ()  

  ## statement.is_outgoing_slice ():
  def is_outgoing_slice (self, ref):
    is_write = self.is_write_ref (ref)
    if (not is_write):
      return False
    out_comm_type = self.determine_communication_type (ref, self.PP)
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (ref)
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    cmnt = '/*{} or {} or {} and {}*/'.format (out_comm_type == COMM_TYPE_GATHER_SLICE, out_comm_type == COMM_TYPE_ALLRED, out_comm_type == COMM_TYPE_LOCAL_SLICE, self.is_true_communication (ref))
    return outgoing_slice_comm_type 

  ## Statement: Determine whether a reference at the current operator requires 
  ## incoming communication.
  def is_incoming_slice (self, ref):
    is_write = self.is_write_ref (ref)
    if (is_write):
      return False
    in_comm_type = self.determine_communication_type (ref, PP)
    is_true_comm = self.is_true_communication (ref)
    return in_comm_type == COMM_TYPE_GATHER_SLICE and is_true_comm
    ## The Below code is never reachable.
    out_comm_type = self.determine_communication_type (ref, self.PP)
    is_outgoing_gather = out_comm_type == COMM_TYPE_GATHER_SLICE
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (ref)
    is_outgoing_allreduce = out_comm_type == COMM_TYPE_ALLRED
    is_outgoing_local_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    cmnt = '/*{} or {} or {} and {}*/'.format (is_outgoing_gather, is_outgoing_allreduce, is_outgoing_local_slice, is_true_comm)

  ## Statement: Generate the statement body for the three types of statements 
  ## i.e., regular, generator or sink.
  def generate_statement (self, df, PP, producers, mrap, indent, check_mode = DIMAGE_CHECK_NO_CHECK):
    if (self.is_data_generator ()):
      array = self.accs[0]
      if (DO_REF and False):
        array.gen_matrix_data ()
      mrap[array.get_name ()] = array
      rff_args = self.generate_read_from_file_arguments (array, PP)
      rff_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      buffer_name = array.get_matrix_filename ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (rff_func, num_array_dim, buffer_name, rff_args, DIMAGE_BLOCK_COUNT)
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_READ_REF_ARRAY):
      ## Generate the call to read_from_file_tile to load the reference tile.
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      #buffer_name = self.last_writer_map [array.get_name ()].get_debug_filename (array) + '.mat'
      buffer_name = array.get_sna_reference_filename ()
      #return '{}_tile{}D(\"data_{}\", {});\n'.format (acf_func, num_array_dim, array.get_name (), acf_args)
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, buffer_name, acf_args, DIMAGE_BLOCK_COUNT)
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_CALL_CHECK):
      ## Call to function for correctness check.
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = ARRAY_CHECK_FUNC
      num_array_dim = array.get_num_dim ()
      check_filename = '{}_at_{}'.format (array.get_name (), self.get_name ())
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, check_filename, acf_args, DIMAGE_REFBLOCK_COUNT)
    if (self.is_data_sink ()):
      array = self.accs[0]
      sink_varname = array.get_name ()
      if (array.get_use_slice ()):
        sink_varname = array.get_slice_varname (True)
      wtf_args = self.generate_write_to_file_arguments (array, PP)
      wtf_func = WRITE_TO_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      #self.indent (df)
      return '//{}_tile{}D(\"data_{}\", {});\n'.format (wtf_func, num_array_dim, sink_varname, wtf_args)
    num_accs = len(self.accs)
    if (num_accs != 3):
      print ('[ERROR@generate_statement]: Expected only 3 array references. Found {} instead.'.format (num_accs))
      sys.exit (42)
    in_ref = self.accs[0]
    ker_ref = self.accs[1]
    out_ref = self.accs[2]
    if (not in_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (in_ref.get_name ()))
      sys.exit (42)
    if (not ker_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (ker_ref.get_name ()))
      sys.exit (42)
    if (not out_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (out_ref.get_name ()))
      sys.exit (42)
    lp_in_ref = producers[in_ref.get_name ()].is_mu_lexico_positive (in_ref, producers)
    lp_ker_ref = producers[ker_ref.get_name ()].is_mu_lexico_positive (ker_ref, producers)
    lp_out_ref = producers[out_ref.get_name ()].is_mu_lexico_positive (out_ref, producers)
    # Compute: fetch most recently produced arrays.
    mrap_in_ref = None
    mrap_ker_ref = None
    mrap_out_ref = None
    if (in_ref.get_name () in mrap):
      mrap_in_ref = mrap[in_ref.get_name ()]
    if (ker_ref.get_name () in mrap):
      mrap_ker_ref = mrap[ker_ref.get_name ()]
    if (out_ref.get_name () in mrap):
      mrap_out_ref = mrap[out_ref.get_name ()]
    #if (in_ref.get_data () == None):
    #  in_ref.gen_matrix_data ()
    #if (ker_ref.get_data () == None):
    #  ker_ref.gen_matrix_data ()
    #if (out_ref.get_data () == None):
    #  out_ref.gen_matrix_data ()
    #self.compute_matmul (out_ref, in_ref, ker_ref)
    if (DO_REF and False):
      mrap_in_ref.show_data ()
      mrap_ker_ref.show_data ()
      mrap_out_ref.show_data ()
    if (DO_REF and False):
      self.compute_matmul (mrap_out_ref, mrap_in_ref, mrap_ker_ref)
      print ("Array {} after matmul...".format (mrap_out_ref.get_name ()))
    if (DO_REF and False):
      mrap_out_ref.show_data ()
    if (option_debug >= 4):
      print ("[INFO] Replacing old array {} in mrap ...".format (mrap_out_ref.get_name ()))
    mrap[mrap_out_ref.get_name ()] = mrap_out_ref

    in_comm_type = self.determine_communication_type (in_ref, PP)
    ker_comm_type = self.determine_communication_type (ker_ref, PP)
    out_comm_type = self.determine_communication_type (out_ref, PP)
    
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (out_ref)
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    cmnt = '/*{} or {} or {} and {}*/'.format (out_comm_type == COMM_TYPE_GATHER_SLICE, out_comm_type == COMM_TYPE_ALLRED, out_comm_type == COMM_TYPE_LOCAL_SLICE, self.is_true_communication (out_ref))

    in_acc = self.generate_access (in_ref, PP, producers)
    ker_acc = self.generate_access (ker_ref, PP, producers)
    out_acc = self.generate_access (out_ref, PP, producers, True, False, False, is_allgat_out_slice)
    ## Ugly fix: We store this here to avoid recomputing a bunch of information
    ## later needed in generating the pointer-access for external kernels.
    out_ref.set_is_allgat_out_slice (is_allgat_out_slice)
    out_ref.set_precollective_buffer_access (out_acc)

    in_ref_was_allgat =  in_ref.get_is_allgat_in_slice ()  #producers[in_ref.get_name ()].was_allgathered (in_ref, self)
    ker_ref_was_allgat = ker_ref.get_is_allgat_in_slice () #producers[ker_ref.get_name ()].was_allgathered (ker_ref, self)
    out_ref_was_allgat = producers[out_ref.get_name ()].was_allgathered (out_ref, self)

    comment_in  = '' # /* Lexico+ : {} - ag = {} */'.format (lp_in_ref, in_ref_was_allgat)
    comment_ker = '' # /* Lexico+ : {} - ag = {} - */'.format (lp_ker_ref, ker_ref_was_allgat)
    comment_out = '' #/* Lexico+ : {} - ag = {} - */'.format (lp_out_ref, out_ref_was_allgat)

    ret = ''
    ##if (outgoing_slice_comm_type):
    ##  out_ref.set_use_slice (True)
    ##  ret += '{}[{}] += {} {}\n'.format (out_ref.get_slice_varname (False), out_acc, comment_out, cmnt)
    ##else:
    ##  ret += '{}[{}] += {} \n'.format (out_ref.get_name (), out_acc, comment_out)
    ##ret += indent + '  '
    ##if (in_comm_type == COMM_TYPE_GATHER_SLICE and self.is_true_communication ()):
    ##  ret += '{}[{}] * {} \n'.format (in_ref.get_slice_varname (True), in_acc, comment_in)
    ##else:
    ##  ret += '{}[{}] * {} \n'.format (in_ref.get_name (), in_acc, comment_in)
    ##ret += indent + '  '
    ##if (ker_comm_type == COMM_TYPE_GATHER_SLICE and self.is_true_communication ()):
    ##  ret += '{}[{}]; {} \n'.format (ker_ref.get_slice_varname (True), ker_acc, comment_ker)
    ##else:
    ##  ret += '{}[{}]; {} \n'.format (ker_ref.get_name (), ker_acc, comment_ker)
    ret += '{}[{}] += {} \n'.format (out_ref.get_tile_name (), out_acc, comment_out)
    ret += indent + '  '
    left_buf = in_ref.get_tile_name ()
    if (in_ref_was_allgat):
      left_buf = in_ref.get_tile_name (in_ref.get_slice_varname (True))
    ret += '{}[{}] * {} \n'.format (left_buf, in_acc, comment_in)
    ret += indent + '  '
    right_buf = ker_ref.get_tile_name ()
    if (ker_ref_was_allgat):
      right_buf = ker_ref.get_tile_name (ker_ref.get_slice_varname (True))
    ## TODO: Do the same for generate_statement_generic ()
    ret += '{}[{}]; {} \n'.format (right_buf, ker_acc, comment_ker)
    return ret


  def generate_statement_generic (self, df, PP, producers, mrap, indent, check_mode = DIMAGE_CHECK_NO_CHECK):
    if (self.is_data_generator ()):   
      array = self.accs[0]
      if (DO_REF and False):
        array.gen_matrix_data ()
      mrap[array.get_name ()] = array
      rff_args = self.generate_read_from_file_arguments (array, PP)
      rff_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (rff_func, num_array_dim, array.get_matrix_filename (), rff_args, DIMAGE_BLOCK_COUNT)
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_READ_REF_ARRAY):
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = READ_FROM_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      buffer_name = array.get_sna_reference_filename ()
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, buffer_name, acf_args, DIMAGE_BLOCK_COUNT )
    if (self.is_data_sink () and check_mode == DIMAGE_CHECK_CALL_CHECK):
      array = self.accs[0]
      acf_args = self.generate_read_from_file_arguments (array, PP, check_mode)
      acf_func = ARRAY_CHECK_FUNC
      num_array_dim = array.get_num_dim ()
      check_filename = '{}_at_{}'.format (array.get_name (), self.get_name ())
      return '{}_tile{}D(\"{}\", {}, &{});\n'.format (acf_func, num_array_dim, check_filename, acf_args, DIMAGE_REFBLOCK_COUNT)
    if (self.is_data_sink ()):   
      array = self.accs[0]
      wtf_args = self.generate_write_to_file_arguments (array, PP)
      wtf_func = WRITE_TO_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      return '//{}_tile{}D(\"data_{}\", {});\n'.format (wtf_func, num_array_dim, array.get_name (), wtf_args)
    operator_ref = []
    num_accs = len(self.accs)
    out_ref = self.accs[num_accs-1]
    if (not out_ref.get_name () in producers):
      print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (out_ref.get_name ()))
      sys.exit (42)
    lp_out_ref = producers[out_ref.get_name ()].is_mu_lexico_positive (out_ref, producers)
    mrap_out_ref = None
    if (out_ref.get_name () in mrap):
      mrap_out_ref = mrap[out_ref.get_name ()]
    operator_ref.append (out_ref)
    ret = ''
    out_comm_type = self.determine_communication_type (out_ref, PP)
    is_allgat_out_slice = out_comm_type == COMM_TYPE_LOCAL_SLICE and self.is_true_communication (out_ref)
    outgoing_slice_comm_type = (out_comm_type == COMM_TYPE_GATHER_SLICE or out_comm_type == COMM_TYPE_ALLRED or is_allgat_out_slice)
    out_acc = self.generate_access (out_ref, PP, producers, True, False, False, is_allgat_out_slice)
    out_ref.set_is_allgat_out_slice (is_allgat_out_slice)
    out_ref.set_precollective_buffer_access (out_acc)
    was_allgat =  producers[out_ref.get_name ()].was_allgathered (out_ref, self)
    comment_out = '' # /* Lexico+ : {} - ag = {} - */'.format (lp_out_ref, was_allgat)
    ret += '{}[{}] += {} \n'.format (out_ref.get_tile_name (), out_acc, comment_out)
    print ("Array {} before operator ...".format (mrap_out_ref.get_name ()))
    mrap_out_ref.show_data ()
    for refid in range(num_accs-1):
      in_ref = self.accs[refid]
      if (not in_ref.get_name () in producers):
        print ('[ERROR@generate_statement]: Producer operator not found for array {}.'.format (in_ref.get_name ()))
        sys.exit (42)
      lp_in_ref = producers[in_ref.get_name ()].is_mu_lexico_positive (in_ref, producers)
      was_allgat = in_ref.get_is_allgat_in_slice () #producers[in_ref.get_name ()].was_allgathered (in_ref, self)
      # Compute: fetch most recently produced arrays.
      mrap_in_ref = None
      if (in_ref.get_name () in mrap):
        mrap_in_ref = mrap[in_ref.get_name ()]
      operator_ref.append (in_ref)
      mrap_in_ref.show_data ()
      in_comm_type = self.determine_communication_type (in_ref, PP)
      in_acc = self.generate_access (in_ref, PP, producers)
      comment_in = '' #'/* Lexico+ : {} - ag = {} - */'.format (lp_in_ref, was_allgat)
      ret += indent + '  '
      mid_op = '*'
      if (refid == num_accs - 2):
        mid_op = ';'
      in_buf = in_ref.get_tile_name ()
      if (was_allgat):
        in_buf = in_ref.get_tile_name (in_ref.get_slice_varname (True))
      ret += '{}[{}] {} {} \n'.format (in_buf, in_acc, mid_op, comment_in)
    if (DO_REF):
      self.compute_operator (operator_ref)
    if (self.is_data_generator ()):
      ref = self.accs[0]
      ref.dump_generated_tile (df, PP)
      ref.return_allocated (df)
    #print ("Array {} after matmul...".format (mrap_out_ref.get_name ()))
    mrap_out_ref.show_data ()
    if (option_debug >= 5):
      print ("Replacing old array {} in mrap ...".format (mrap_out_ref.get_name ()))
    mrap[mrap_out_ref.get_name ()] = mrap_out_ref
    return ret

  ## @Statement: Generate the call to an external DIMAGE operator
  def generate_dimage_operator_call (self, df, PP, producers, mrap, indent):
    num_accs = len(self.accs)
    out_ref = self.accs[num_accs-1]
    is_allgat_out_slice = out_ref.get_is_allgat_out_slice ()
    loop_depth = len(self.dims)
    call ='dimage_{} ('.format (self.name)
    if (self.kernel_name != None):
      call ='dimage_{} ('.format (self.kernel_name)
    call += self.get_tile_pointer (out_ref, producers, is_allgat_out_slice, False)
    for refid in range(num_accs-1):
      ref = self.accs[refid]
      is_in_slice = ref.get_is_allgat_in_slice ()
      call += ', \n'
      call += '{} '.format (indent)
      ## Always pass False as the third argument (is_allgat_out_slice), since
      ## it will be a read-array.
      call += self.get_tile_pointer (ref, producers, is_in_slice, True) #False)
    for idim in range(len(self.dims)):
      call += ', '
      ub = self.build_loop_structure (idim, PP, True, producers, False, False, True)
      call += '{} + 1'.format (ub)
    call += ', 1.0'
    call += ', 1.0'
    call += ');'
    return call
    

  def gencode_matrix_data_generator (self, init_val):
    operator_ref = []
    num_accs = len(self.accs)
    out_ref = self.accs[num_accs-1]
    operator_ref.append (out_ref)
    for refid in range(num_accs-1):
      in_ref = self.accs[refid]
      operator_ref.append (in_ref)
    self.compute_operator (operator_ref, init_val)

  # @Statement: 
  def generate_simple_increment_statement (self, df, ref, PP, producers, mrap, indent):
    out_acc = self.generate_access (ref, PP, producers, False, True, True)
    buffer_acc = ref.get_precollective_buffer_access ()
    interm = 'tile_' + self.get_allred_intermediate (ref)
    ret = ''
    ret += '{}[{}] += \n{}{}{}[{}];'.format (ref.get_tile_name (), out_acc, indent, indent, interm, out_acc) #buffer_acc)
    return ret

  # @Statement: 
  def insert_omp_pragmas (self, df, indent):
    if (self.is_data_generator () or self.is_data_sink ()):
      return
    df.write (indent)
    iter_list = self.get_iterator_str_list ()
    df.write ('#pragma omp parallel for private({})\n'.format (iter_list))

  # @Statement: 
  # Construct the loop body, tile and point loops, as well as the statement
  # body for the current operator. For generators and data sinks, will use
  # the dimage-rt API to load and write data tiles in the correct order.
  # For computational operators, build_operator_body will decide different
  # access types for each reference. This is done by querying the @producers
  # dictionary to find out the statement that produced, and hence defined,
  # the layout of a data array.
  def build_operator_body (self, df, PP, producers, mrap):
    indent = BASE_INDENT
    depth = 1
    self.start_computation_timer (df, indent)
    self.ntpd = [] # number of tiles per dimension
    # Generate tile loops
    for idim in self.dims:
      loop = self.build_loop_structure (idim, PP, False, producers, False, False)
      line = '{} {}\n'.format (loop, '{')
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      depth += 1
    ttc = self.collect_tile_trip_counts (producers)
    level = 0
    for idim in self.dims:
      dim_name = self.dims[idim]
      trip = None
      if (dim_name in ttc):
        trip = ttc[dim_name]
      loop = self.build_l2_loop (level, trip, False, L2_LOOP_GENMODE_FULL, producers)
      line = '{} {} /* {} - {} */\n'.format (loop, '{', dim_name, trip)
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      level += 1
      depth += 1
    for tt in ttc:
      line = '{}// dim {} : {}\n'.format (indent, tt, ttc[tt])
      df.write (line)
    ## Insert data tile fetching code
    if (not self.is_data_generator ()):
      for ref in self.accs:
        df.write (indent)
        ## Need to determine if/when intermediates are used. These are the 
        ## result of incoming all-gathers.
        intermediate = None
        if (ref.get_use_slice ()):
          intermediate = ref.get_slice_varname (True)
        self.generate_tile_fetch_code (df, ref, producers, False, intermediate)
        if (self.is_output_array (ref)):
          df.write (indent)
          self.generate_set_tile_coordinate (df, ref)
    if (not self.is_data_sink () and not self.is_data_generator ()):
      df.write (indent)
      df.write ('#ifdef DIMAGE_KERNEL_LOOP\n')
    self.insert_omp_pragmas (df, indent)
    # Generate point loops
    if (not self.is_data_sink () and not self.is_data_generator ()):
      for idim in self.dims:
        loop = self.build_loop_structure (idim, PP, True, producers, False, False)
        line = '{} {}\n'.format (loop, '{')
        df.write (indent)
        df.write (line)
        indent += BASE_INDENT
        depth += 1
    stmt_body = ''
    if (len(self.accs) == 3):
      stmt_body = self.generate_statement (df, PP, producers, mrap, indent)
    else:
      stmt_body = self.generate_statement_generic (df, PP, producers, mrap, indent)
    ## Reset the block count for data sinks: only one block is used when
    ## performing a check between the distributed tensor and the reference one.
    if (self.is_data_sink ()):
      ## Data sink check.
      ref = self.accs[0]
      #self.generate_tile_fetch_code (df, ref, producers, False)
      ## Compute address of tile_ref: Expect a single block in the SNA 
      ## (single-node access) reference: 
      ## tile pointer results from the shift on the SNA reference.
      line = '{}{} *{} = ({} + {});\n'.format (indent, DIMAGE_DT, ref.get_sna_ref_name (), ref.get_name_for_check (), DIMAGE_TILE_HEADER_MACRO)
      df.write (line)
      df.write ('{}{} = 0;\n'.format (indent, DIMAGE_BLOCK_COUNT))
      if (self.is_data_sink ()):
        df.write ('{}{} = 0;\n'.format (indent, DIMAGE_REFBLOCK_COUNT))
    df.write (indent)
    df.write (stmt_body)
    if (DIMAGE_OPTION_DO_CHECK and self.is_data_sink ()):
      if (len(self.accs) == 1):
        df.write (indent)
        df.write ('/* DIMAGE_OPTION_DO_CHECK */\n')
        stmt_body = self.generate_statement (df, PP, producers, mrap, indent, DIMAGE_CHECK_READ_REF_ARRAY) 
        df.write (indent)
        df.write (stmt_body)
        stmt_body = self.generate_statement (df, PP, producers, mrap, indent, DIMAGE_CHECK_CALL_CHECK)
        df.write (indent)
        df.write (stmt_body)
      else:
        stmt_body = self.generate_statement_generic (df, PP, producers, mrap, indent, DIMAGE_CHECK_READ_REF_ARRAY)
        df.write (indent)
        df.write (stmt_body)
        stmt_body = self.generate_statement_generic (df, PP, producers, mrap, indent, DIMAGE_CHECK_CALL_CHECK)
        df.write (indent)
        df.write (stmt_body)
    for lev in range(depth-1):
      if (not self.is_data_sink () and not self.is_data_generator () and lev == len(self.dims)):
        ## Make call to external DIMAGE operator         
        df.write (indent)
        df.write ('#else\n')
        df.write (indent)
        df.write ('// External DIMAGE operator\n')
        dimage_call = self.generate_dimage_operator_call (df, PP, producers, mrap, indent)
        df.write ('{}{}{}\n'.format (indent, indent, dimage_call))
        df.write (indent)
        df.write ('#endif\n')
      indent = BASE_INDENT * (depth-1)
      df.write (indent)
      df.write ('}\n')
      depth = depth - 1
    self.stop_computation_timer (df, indent)

  def declare_local_iterators (self, df):
    for dd in self.dims:
      line = 'int b{};\n'.format (dd)
      self.indent (df)
      df.write (line)
      line = 'int t{};\n'.format (dd)
      self.indent (df)
      df.write (line)
      line = 'int i{};\n'.format (dd)
      self.indent (df)
      df.write (line)

  ## @Statement:
  ## Do map(mu) <- map(pi) but only for generators that would normally require
  ## an all-gather.
  def statement_equalize_mu_to_pi_map (self, ref):
    if (self.statement_can_allgather_incoming (ref)):
      return
    for idim in self.dims:     
      dim_name = self.dims[idim]
      pival = ref.get_pi_by_dim_name_if_used (dim_name)
      muval = self.map[idim] 
      if (option_debug >= 4):
        print ('[INFO:mu<-pi] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
      # Normally, we would need to check that pival is not DIM_NOT_USED.
      # However, since we are in a generator, the number of iteration and
      # data space dimensions should be the same. In other words, all 
      # iteration space dimensions should always be used.
      self.map[idim] = pival

  def statement_can_allgather_incoming (self, ref):
    scenario = self.statement_is_unsupported_incoming_allgather (ref)
    return scenario == 0

  # @Statement:
  def statement_is_unsupported_incoming_allgather (self, ref):
    # Statement must be a generator
    if (self.is_compute_statement ()):
      return 0
    # Traverse the iteration and data dimensions to find an iteration 
    # space dimension i and data space dimension d that are used in the 
    # reference and s.t. mu_i is unmapped and pi_d is mapped.
    # When mu_i is unmapped (< 0) and pi_d is mapped, we would normally need
    # an all-gather to make local all data needed by the generator.
    # However, generators cannot do all-gathers since, by definition, they
    # only produce data.
    for idim in self.dims:     
      dim_name = self.dims[idim]
      # BACK
      pival = ref.get_pi_by_dim_name_if_used (dim_name)
      muval = self.map[idim]
      if (pival >= 0 and muval < 0):
        return 1
      if (pival >= 0 and muval >= 0 and pival != muval):
        return 2
    return 0

  # @Statement:
  # Repair mapping for the following scenarios:
  # 1) dim(grid) > dim(tensor), e.g., a 3D grid on a 2D tensor. There has to be replication.
  # 2) mu unmapped (-1) and pi mapped on generators. A regular operator would have the ability
  #    to `fetch' the data from wherever it is, but generator can't do this since they are the
  #    ones setting the initial placement of the data.
  def statement_align_mu_mappings (self, PP):
    dim_grid = PP.get_num_dim ()
    dim_comp = len(self.dims) 
    if (dim_grid <= dim_comp and self.is_compute_statement ()):
      if (option_debug >= 3):
        print ('@ statement_align_mu_mappings(): Nothing to do for computation {}'.format (self.name))
      return
    grid_case = dim_grid > dim_comp
    generator_case = not self.is_compute_statement ()
    if (option_debug >= 3):
      if (grid_case):
        print ('[INFO:R1] dim(grid) = {} > {} = dim({})'.format (dim_grid, dim_comp, self.name))
      if (generator_case):
        print ('[INFO:R2] Generator / Sink case on {}', self.name)
    if (grid_case):
      for ii,aa in enumerate(self.refs):
        ref = self.refs[aa]
        for idim in self.dims:     
          dim_name = self.dims[idim]
          pival = ref.get_pi_by_dim_name_if_used (dim_name)
          muval = self.map[idim]
          if (muval >= 0 and pival >= 0 and muval != pival):
            if (option_debug >= 4):
              print ('[INFO:R1] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
            self.map[idim] = pival
          elif (muval < 0 and pival >= 0):
            if (option_debug >= 4):
              print ('[INFO:R1] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
            self.map[idim] = pival
          elif (muval >= 0 and pival < 0):
            if (option_debug >= 4):
              print ('[INFO:R1] Updating mu({})[{}] from {} to {}'.format (self.name, idim, muval, pival))
            self.map[idim] = pival
          #else:
          #  print ('[INFO] Unexpected case. Please report.')
          #  sys.exit (42)
    ## Repair Case 2: mu unmapped (work serialized), pi mapped (partitioned).
    ## Generators don't have the advantage of doing an all-gather at the 
    ## beginning of their execution.
    if (generator_case):
      for ii,aa in enumerate(self.refs):
        ref = self.refs[aa]
        if (not self.statement_can_allgather_incoming (ref)):
          scenario = self.statement_is_unsupported_incoming_allgather (ref)
          if (option_debug >= 4):
            print ('[INFO]: Aligning mappings of generators: mu^{} with pi^{} - Scenario {}'.format (self.name, ref.get_name (), scenario))
          self.statement_equalize_mu_to_pi_map (ref)


  # @Statement: 
  def report_mappings (self, AA, PP):
    line='{}:'.format (self.name)
    for idim in self.dims:
      mu = self.map[idim]
      if (mu < 0):
        mu = '*'
      if (idim > 0):
        line += ','
      line += str(mu)
    print (line)
    nref = len(self.refs)
    for ii,aa in enumerate(self.refs):
      ref = self.refs[aa]
      line = '  {}:'.format (ref.get_name ())
      pilist = ''
      for idim in self.dims:     
        dim_name = self.dims[idim]
        pival = ref.get_pi_by_dim_name_if_used (dim_name)
        if (pival == DIM_NOT_USED):
          continue
        if (pilist != ''):
          pilist += ','
        if (pival >= 0):
          pilist += str(pival)
        else:
          pilist += '*'
      print (line+pilist)
      dim_comp = len(self.dims)
      dim_grid = PP.get_num_dim ()
      for idim in self.dims:     
        dim_name = self.dims[idim]
        pival = ref.get_pi_by_dim_name_if_used (dim_name)
        if (pival == DIM_NOT_USED):
          continue
        muval = self.map[idim]
        tag = 'ERR'
        if (muval == pival and pival == -1):
          tag = '(case 1) Data replicated, work replicated '
        elif (muval == pival and pival != -1):
          tag = '(case 2) Data distributed, work distributed (and matched)'
        elif (muval >= 0 and pival < 0):
          tag = '(case 3) Data replicated, work distributed'
        elif (muval < 0 and pival >= 0):
          collective_type = ''
          if (self.is_data_sink ()):
            collective_type = 'allgather [read]'
          elif (self.is_data_generator ()):
            collective_type = 'allbroadcast [write]'
          elif (ii == nref - 1):
            collective_type = 'allbroadcast [write]'
          else:
            collective_type = 'allgather [read]'
          tag = '(case 4) Data distributed, work replicated (need {})'.format (collective_type)
        elif (muval >= 0 and pival >= 0 and muval != pival and dim_grid > dim_comp):
          tag = '(case 5) Grid dimensionality ({}) exceeds dimensionality of computation ({})'.format (dim_grid, dim_comp)
        else:
          tag = 'Unexpected case (mu={},pi={})'.format (muval,pival)
        print ('    {}: {}'.format (dim_name, tag))
        
  def describe_communication_vector (self, vec):
    line = ''
    for ct in vec:
      if (line != ''):
        line += ', '
      if (ct == COMM_TYPE_ALLRED):
        line += 'ALL_REDUCE'
      if (ct == COMM_TYPE_LOCAL):
        line += 'LOCAL_COMM'
      if (ct == COMM_TYPE_LOCAL_SLICE):
        line += 'LOCAL_SLICE'
      if (ct == COMM_TYPE_GATHER_SLICE):
        line += 'GATHER_SLICE'
    line = '\t[INFO] Communication vector: [' + line + ']'
    print (line)

  ## Statement.determine_communication_type():
  def determine_communication_type (self, ref, PP):
    ctypes = []
    grid_dim = PP.get_num_dim ()
    for idim in self.dims:
      dim_name = self.dims[idim]
      red_dim = self.get_mapped_reduction_dimension (ref, PP)

      pp_red_dim_size = 1  ## Size of 1 means really no reduction.
      if (red_dim >= 0):
        pp_red_dim = self.map[red_dim]
        pp_red_dim_size = PP.get_dim_size (pp_red_dim)
      if (red_dim >= 0 and red_dim == idim and pp_red_dim_size > 1): ## NOTE Adding PP > 1 condition
        print ("{}({}={}) is reduction dimension".format (self.name, idim, dim_name))
        ctypes.append (COMM_TYPE_ALLRED)
        continue
      if (ref.is_dim_used (dim_name)):
        stmt_pdim = self.map[idim]
        ref_pdim = ref.get_proc_map_by_dim_name (dim_name)
        if (option_debug >= 3):
          print ("Scenario: {}({}={}->{}), {}[{}->{}]".format (self.name, idim, dim_name, stmt_pdim, ref.get_name (), dim_name, ref_pdim))
        if (stmt_pdim == ref_pdim): # perfect match, local comm.
          if (option_debug >= 3):
            print ("\t --> LOCAL COMM")
          ctypes.append (COMM_TYPE_LOCAL)
          continue
        if (ref_pdim == -1 and stmt_pdim >= 0): # ref is replicated, so we have what we need (access only a piece of it, though), all-reduce!
          ## Make the distinction here because data generators can only do all-gathers.
          if (self.is_data_generator ()):
            if (option_debug >= 3):
              print ("\t [INFO] Generator {} mapped (mu >= 0), data replicated (pi < 0). Will all-gather".format (self.name))
            ctypes.append (COMM_TYPE_LOCAL_SLICE)
            continue
          else: #if (PP.get_dim_size (stmt_pdim) > 1):
            if (option_debug >= 2):
              print ("\t ALL-REDUCE (mu-mapped, pi-replicated) each procs computes different. All-reduce needed (psize={}). (COMM_TYPE_ALLRED)".format (pp_red_dim_size))
            ctypes.append (COMM_TYPE_ALLRED)
            continue
        if (ref_pdim >= 0 and stmt_pdim == -1): # ref is distributed, but all statements access everything. Need AllGather.
          if (option_debug >= 3):
            print ("\t REPLICATED WORK, PARTITIONED DATA (COMM_TYPE_GATHER_SLICE)")
            print ("\t[WARNING@determine_communication_type:1]: Accessing non-local data with mappings (stmt={},ref={},dim={}). Will require AllGather.".format (self.name, ref.get_name (), dim_name))
          #sys.exit (42)
          ctypes.append (COMM_TYPE_GATHER_SLICE)
          continue
        if (stmt_pdim != ref_pdim and ref_pdim >= 0 and stmt_pdim >= 0): 
          if (option_debug >= 3):
            print ("\tDIMENSIONS MAPPED AND CROSSED")
          # Stmt and reference are mapped, but they don't match. Hence, we require AllGather comm. 
          #ctypes.append (COMM_TYPE_P2P)
          print ("\t[ERROR@determine_communication_type:2]: Accessing non-local data with mappings (stmt={},ref={},dim={}). Dimensions crossed.".format (self.name, ref.get_name (), dim_name))
          ctypes.append (COMM_TYPE_GATHER_SLICE)
          continue
        if (ref_pdim == -1 and stmt_pdim == -1):
          if (option_debug >= 3):
            print ("\t ALL-REPLICATED / No comm")
          ctypes.append (COMM_TYPE_LOCAL)
          continue
        print ("\t[ERROR@determine_communication_type:3]: Unexpected combination of mappings")
        sys.exit (42)
        continue
    all_local = True
    n_local_slices = 0
    n_gather_slices = 0
    n_allred = 0
    n_p2p = 0
    for ct in ctypes:
      all_local = all_local and ct == COMM_TYPE_LOCAL
      if (ct == COMM_TYPE_ALLRED):
        n_allred += 1
      if (ct == COMM_TYPE_LOCAL_SLICE):
        n_local_slices += 1
      if (ct == COMM_TYPE_GATHER_SLICE):
        n_gather_slices += 1
      if (ct == COMM_TYPE_P2P):
        n_p2p += 1
    if (all_local):
      return COMM_TYPE_LOCAL
    if (n_allred >= 1): ## We can expect two or more reduction dimensions (e.g. mttkrp)
      return COMM_TYPE_ALLRED
    if (option_debug >= 2):
      self.describe_communication_vector (ctypes)
    #if (n_allred > 1):
    #  print ("\t[ERROR:4]: Didn't expect more than 1 allred dimensions.")
    #  sys.exit (42)
    if (n_p2p > 0):
      print ("\t[WARNING][determine_communication_type:5] Found p2p comm.type. Unsupported. Aborting ...")
      #sys.exit (42)
      return COMM_TYPE_LOCAL
    if (n_local_slices > 0 and n_gather_slices > 0):
      print ("\t[ERROR][determine_communication_type:6]: Cannot require local ({}) and gather slices ({}) along different dimensions. Aborting ...".format (n_local_slices,n_gather_slices))
      sys.exit (42)
    if (n_local_slices > 0 and n_gather_slices == 0):
      return COMM_TYPE_LOCAL_SLICE
    if (n_local_slices == 0 and n_gather_slices > 0):
      return COMM_TYPE_GATHER_SLICE
    # Shouldn't get to this point.
    print ("\t[ERROR][determine_communication_type:7] Found p2p comm.type. Unsupported. Aborting ...")
    sys.exit (42)
    return COMM_TYPE_LOCAL_SLICE

  ## @Statement: print the mapping properties of a statement and a reference
  def generate_accessor_summary (self, df, ref):
    self.indent (df)
    df.write ('// Array {}[]: '.format (ref.get_name ()))
    df.write ('\n')
    self.indent (df)
    df.write ('// mu-vector: ')
    self.pretty_print_map (df)
    df.write ('\n')
    self.indent (df)
    df.write ('// pi-vector: ')
    ref.pretty_print_map (df)
    df.write ('\n')

  ## Statement: 
  def statement_generate_incoming_communication (self, df, PP):
    if (self.is_data_generator ()):   
      ref = self.accs[0]
      self.indent (df)
      df.write ('// Generators don\'t require incoming communication\n')
      self.indent (df)
      df.write ('// Array {}[]: '.format (ref.get_name ()))
      df.write ('\n')
      self.indent (df)
      df.write ('// mu-vector: ')
      self.pretty_print_map (df)
      df.write ('\n')
      self.indent (df)
      df.write ('// pi-vector: ')
      ref.pretty_print_map (df)
      df.write ('\n')
      return
    acc_pos = 0
    slices = []
    for ref in self.accs:
      self.indent (df)
      df.write ("// **************************************************** \n")
      #if (acc_pos == len(self.accs) - 1):
      #  self.indent (df)
      #  df.write ('// Assuming output array {}[] doesn\'t require incoming communication.\n'.format ( ref.get_name()))
      #  continue
      self.generate_accessor_summary (df, ref)
      comm_type = self.determine_communication_type (ref, PP)
      ref_slice = ref.reference_generate_incoming_communication (df, self, comm_type, PP)
      if (ref_slice != None):
        slices.append (ref_slice)
      acc_pos += 1
    return slices

  ## Statement.is_true_communication(): Determine whether the current statement
  ## will require communication under the give array mapping.
  ## True communication happens along a given dim in two cases:
  ## 1) k is used mu_k is mapped and pi_k is not mapped: that means that computation
  ##    is local, but that data is replicated. Hence, an all-reduce will be necessary.
  ## 2) k is not used in the reference, and mu_k is mapped.
  ## Equivalently, if k is used, mu_k and pi_k are mapped, they match.
  def is_true_communication (self, ref):
    for idim in self.dims:
      mu_dim = self.map[idim]
      proc_size = 1
      if (mu_dim >= 0):
        proc_size = self.PP.get_dim_size (mu_dim)
      if (proc_size == 1):
        ## Cannot ever have communication if we have only one processor
        ## or rank along the current dimension, even if the pi is mapped.
        continue
      dim_name = self.dims[idim]
      if (ref.is_dim_used (dim_name)):
        pi_dim = ref.get_pi_by_name (dim_name)
        ## Data-space dimension is mapped and dimensions don't match. 
        ## So we have communication along some space dimension.
        if (mu_dim >= 0 and pi_dim >= 0 and mu_dim != pi_dim):
          return True
        ## Data-space dimension is not mapped (replicated) but only
        ## one processor is mapped. So the result will have to be broadcast.
        if (mu_dim >= 0 and pi_dim == -1):
          return True
      elif (mu_dim >= 0):
        ## The iteration space dimension is not used to access the array.
        ## Hence, it will become a reduction. We have already checked that
        ## the number of processors along the iteration space dimension is greater
        ## than 1 (and hence each processor has at least one other partner to
        ## communicate with).
        return True 
    return False


  ## @Statement: Generate outgoing communication for current statement.
  def generate_outgoing_communication (self, df, PP, producers, mrap):
    #if (self.is_data_generator ()):
    #  return
    if (self.is_data_sink ()):
      return
    num_acc = len(self.accs)
    ref = self.accs[num_acc-1]
    red_dim = self.get_mapped_reduction_dimension (ref, PP)
    comm_type = self.determine_communication_type (ref, PP)
    #self.generate_accessor_summary (df, ref)
    self.indent (df)
    df.write ('// Communication for outgoing-data\n')
    self.indent (df)
    df.write ('// Array {}[], comm.type = {}'.format (ref.get_name (), comm_type_str (comm_type)))
    df.write ('\n')
    self.indent (df)
    df.write ('// mu-vector: ')
    self.pretty_print_map (df)
    df.write ('\n')
    self.indent (df)
    df.write ('// pi-vector: ')
    ref.pretty_print_map (df)
    df.write ('\n')
    self.indent (df)
    df.write ('// reduction dimension: {}\n'.format (red_dim))
    df.write ('\n')
    if (self.is_true_communication (ref)):
      self.indent (df)
      ref.generate_outgoing_communication (df, self, comm_type, PP)
      local_indent='  '
      self.stop_communication_timer (df, local_indent)
      mrd = self.get_mapped_reduction_dimension (ref, PP)
      if (option_debug >= 5):
        print ("======================================> MRD = {}".format (mrd))
      has_map_red_dim = self.has_mapped_reduction_dimension (ref, PP)
      if (not self.is_data_generator () and 
        (has_map_red_dim or comm_type == COMM_TYPE_LOCAL_SLICE)): ## Was 'or'
        #interm = self.generate_intermediate_allred_buffer (df, ref)
        interm = self.get_allred_intermediate (ref)
        self.restart_computation_timer (df, local_indent)
        self.add_to_global_output (df, PP, producers, mrap)
        self.stop_computation_timer (df, local_indent)
      if (not self.is_data_generator () and not has_map_red_dim and comm_type == COMM_TYPE_ALLRED):
        interm = self.get_allred_intermediate (ref)
        self.restart_computation_timer (df, local_indent)
        self.add_to_global_output (df, PP, producers, mrap)
        self.stop_computation_timer (df, local_indent)
    self.debug_store_computation_result (df, PP)


  def get_debug_filename (self, ref):
    return '{}_at_{}'.format (ref.get_name (), self.name)

  ## @Statement; store the matrix (or slices of it) that have been computed 
  ## at a compute-statement.
  def debug_store_computation_result (self, df, PP):
    num_acc = len(self.accs)
    if (not self.is_data_sink () and not self.is_data_generator ()):
      array = self.accs[num_acc-1]
      wtf_args = self.generate_write_to_file_arguments (array, PP, True)
      wtf_func = WRITE_TO_FILE_FUNC
      num_array_dim = array.get_num_dim ()
      self.writeln (df, '')
      self.indent (df)
      df.write ('#ifdef DIMAGE_DEBUG\n')
      self.indent (df)
      df.write ('// Storing generated tile for debug.\n')
      call = '{}_tile{}D(\"{}\", {});'.format (wtf_func, num_array_dim, self.get_debug_filename (array), wtf_args)
      self.indent (df)
      self.writeln (df, call)
      self.indent (df)
      self.writeln (df, '#endif\n')

  ## Statement.declare_outgoing_slices(): Declare a temporary variable to be 
  ## used for an output all-gather or all-reduce array.
  def declare_outgoing_slices (self, df, PP):
    if (self.is_data_generator ()):
      ref = self.accs[0]
      # All (local) data-slices will be allocated as a list of tiles, where 
      # each tile extent is the array extent divided by the lcm of the 
      # processor grid. The number of tiles along each dimension depends on 
      # the array pi-map.
      # send_size = ref.reference_get_local_volume (self)
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      if (option_debug >= 5):
        print ("[INFO@declare_outgoing_slices] Calling get_processor_geometry_list_from_map (ref={}): map={}, False, vec01={}".format (ref.get_name (), self.get_mu_map (), vec01))
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01)
      if (proc_geom == ''):
        print ("stmt {} - Proc-geom is empty".format (self.name))
        sys.exit (42)
      alloc_args = '{}, {}'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    if (self.is_data_sink ()):
      return
    n_acc = len(self.accs)
    #if (n_acc != 3):
    #  print ("[ERROR@declare_outgoing_slices]: Expected 3 references in operator.")
    #  sys.exit (42)
    ref = self.accs[n_acc-1]
    comm_type = self.determine_communication_type (ref, PP)
    if (comm_type == COMM_TYPE_GATHER_SLICE):
      #send_size = ref.reference_get_local_volume (self)
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01)
      alloc_args = '{}, {} /* out-slice */'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    if (comm_type == COMM_TYPE_LOCAL_SLICE):
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01)
      alloc_args = '{}, {} /* ctls */'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    if (comm_type == COMM_TYPE_ALLRED):
      #send_size = ref.reference_get_local_volume (self)
      #send_size = self.get_slice_vol_by_name (ref, PP)
      stride_list = ref.get_tile_extent_list ()
      vec01 = ref.get_vector_used_dims (self)
      #print ("\n\nVector 01: {}".format (vec01))
      proc_geom = PP.get_processor_geometry_list_from_map (self.get_mu_map (), False, vec01, ref, self.dims) ##ref.get_pi_map (), False, vec01)
      alloc_args = '{}, {} /* ctar */'.format (stride_list, proc_geom)
      slice_var = ref.generate_local_slice_buffer (df, alloc_args, False)
      return slice_var
    return None

  def generate_intermediate_allred_buffer (self, df, ref):
    interm = self.get_allred_intermediate (ref)
    #send_size = ref.reference_get_local_volume (self)
    recv_size = self.get_slice_vol_by_name (ref, PP)
    allocator = DIMAGE_BUFFER_ALLOCATOR
    header_payload = ref.get_aggregated_tile_header_space (None)
    line = '{} * {} = {}({} + {});\n'.format (DIMAGE_DT, interm, allocator, recv_size, header_payload)
    self.indent (df)
    df.write (line)
    return interm

  ## Statement: Deallocate (all-gathered) incoming slice buffers.
  def free_in_slices (self, df, in_slices):
    if (in_slices == None):
      return
    for ref in in_slices:
      self.indent (df)
      self.writeln (df, 'free ({});'.format (ref))

  def free_local_pointers (self, df):
    df.write ('\n')
    for ref in self.accs:
      line = ref.get_free_list ()
      df.write (line)

  # Collect all the induced tile trip counts in a dictionary. 
  # The produced dictionary must be previously initialized to {}.
  # Further, the dictionary must be queried in the lexical order
  # of the iterators as defined by the statement.
  def collect_tile_trip_counts_from_ref (self, ref, tss, producers):
    for dd in self.dims:
      iter_name = self.dims[dd]
      # Get array dimension where iteration is used
      adim = ref.get_dim_if_used (iter_name)
      if (adim >= 0 and not iter_name in tss):
        prod = None
        # Fetch the original producer of the array.
        if (ref.get_name () in producers):
          prod = producers[ref.get_name ()]
        if (prod != None):
          # Fetch the original array in the producer
          prod_ref = prod.get_ref_by_name (ref.get_name ())
          tile_size_str = prod_ref.get_num_proc_along_dim (prod, adim, PP)
          # Uncomment to debug tile-sizes used for each loop @ operator.
          if (DEBUG_BLOCK_SIZE_OP_TEN_DIM):
            print ("At operator {}, reference {}, dimension {}: tile size is {}".format (self.name, ref.get_name (), iter_name, tile_size_str))
          if (tile_size_str != '1'):
            tss[iter_name] = tile_size_str
        # Pending
    return tss

  def collect_tile_trip_counts (self, producers): 
    ret = {}
    for ref in self.accs:
      ret = self.collect_tile_trip_counts_from_ref (ref, ret, producers)
    return ret

  ## Statement.add_to_global_output(): Generate a loop nest to accumulate the 
  ## all-gather result, an intermediate, into the result array.
  def add_to_global_output (self, df, PP, producers, mrap):
    nref = len(self.accs)
    out_ref = self.accs[nref-1]
    df.write ('\n\n')
    self.indent (df)
    df.write ('// Adding local contribution to array {}.\n'.format (out_ref.get_name ()))
    indent = BASE_INDENT
    depth = 1
    for idim in self.dims:
      loop = self.build_loop_structure (idim, PP, False, producers, True, True)
      if (loop == ''):
        continue
      line = '{} {}\n'.format (loop, '{')
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      depth += 1
    ttc = self.collect_tile_trip_counts (producers)
    level = 0
    for idim in self.dims:
      dim_name = self.dims[idim]
      trip = None
      if (dim_name in ttc):
        trip = ttc[dim_name]
      loop = self.build_l2_loop (level, trip, True, L2_LOOP_GENMODE_FULL, producers)
      df.write (indent)
      df.write ('//idim = {} - trip = {}\n'.format (dim_name, trip))
      line = '{} {} /* {} - {} */\n'.format (loop, '{', dim_name, trip)
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      level += 1
      depth += 1
    # Generate tile fetch code.
    df.write (indent)
    # Fetch tile pointer of outgoing buffer.
    self.generate_tile_fetch_code (df, out_ref, producers, True)
    interm = self.get_allred_intermediate (out_ref)
    df.write (indent)
    # Fetch tile pointer of all-reduced buffer
    self.generate_tile_fetch_code (df, out_ref, producers, False, interm)
    self.insert_omp_pragmas (df, indent)
    # Generate point loops
    for idim in self.dims:
      loop = self.build_loop_structure (idim, PP, True, producers, True, True)
      if (loop == ''):
        continue
      line = '{} {}\n'.format (loop, '{')
      df.write (indent)
      df.write (line)
      indent += BASE_INDENT
      depth += 1
    stmt_body = self.generate_simple_increment_statement (df, out_ref, PP, producers, mrap, indent)
    df.write (indent)
    df.write (stmt_body)
    df.write ('\n')
    for lev in range(depth-1):
      indent = BASE_INDENT * (depth-1)
      df.write (indent)
      df.write ('}\n')
      depth = depth - 1
    df.write (indent)
    interm = self.get_allred_intermediate (out_ref)
    df.write ('free ({});\n'.format (interm))
    df.write ('\n')

    
  # Statement.generate_operator(): Generate the code associated to a full 
  # distributed operator.
  # Data generators are identified by having the prefix 'gen' to their names.
  # Similarly, data sinks are identified by the prefix 'sink'.
  # For data generators, allocate their local tile.
  def generate_operator (self, df, PP, producers, mrap):
    local_indent = '  '
    if (self.is_data_generator ()):
      df.write ('{} *\n'.format (DIMAGE_DT))
    else:
      df.write ('void\n')
    df.write ('{} () {}\n'.format (self.get_operator_name (), '{'))
    self.indent (df)
    df.write ('// Body of operator {}\n'.format (self.name))
    if (self.is_data_generator ()):
      ref = self.accs[0]
      ref.allocate_local_tile (df, PP, True, self)
      ref.allocate_tile_map (df, PP)
    if (self.is_data_sink ()):
      self.indent (df)
      self.writeln (df, 'MPI_Barrier (MPI_COMM_WORLD);\n')
    if (self.is_data_sink () and DIMAGE_OPTION_DO_CHECK):
      ref = self.accs[0]
      self.indent (df)
      df.write ('{} * {};\n'.format (DIMAGE_DT, ref.get_name_for_check ()))
      self.indent (df)
      ref.allocate_local_tile (df, PP, False, self)
    df.write ('\n')
    self.indent (df)
    df.write ('int {};\n'.format (COMM_SIZE_VAR))
    self.indent (df)
    df.write ('int {} = 0;\n'.format (DIMAGE_BLOCK_COUNT))
    if (self.is_data_sink ()):
      self.indent (df)
      df.write ('int {} = 0;\n'.format (DIMAGE_REFBLOCK_COUNT))
    self.declare_local_iterators (df)
    self.start_communication_timer (df, local_indent)
    in_slices = self.statement_generate_incoming_communication (df, PP)
    self.stop_communication_timer (df, local_indent)
    out_slice = self.declare_outgoing_slices (df, PP)
    self.build_operator_body (df, PP, producers, mrap)
    df.write ('\n')
    # In general, there's no outgoing communication, since we adopt a 
    # push-model mechanism. Nonetheless, in some cases we may need 
    # to perform a collective communication on the global array to 
    # merge partial results and contributions, such as in summa.
    # We include that case here.
    self.start_communication_timer (df, local_indent)
    self.generate_outgoing_communication (df, PP, producers, mrap)
    ## Generate code to initialize tile-map.
    if (self.is_data_generator ()):
      ref = self.accs[0]
      ref.generate_tile_map_creation_code (df, PP)
    #self.free_in_slices (df, in_slices)
    if (out_slice != None):
      self.indent (df)
      self.writeln (df, 'free ({});'.format (out_slice))
    if (self.is_data_sink () and DIMAGE_OPTION_DO_CHECK):
      self.indent (df)
      ref = self.accs[0]
      self.writeln (df, 'free ({});'.format (ref.get_name_for_check ()))
    if (self.is_data_generator ()):
      self.indent (df)
      self.writeln (df, 'MPI_Barrier (MPI_COMM_WORLD);\n')
      ref = self.accs[0]
      ref.dump_generated_tile (df, PP)
      ref.return_allocated (df)
    self.free_local_pointers (df)
    df.write ('}')
    df.write ('\n')
    # Return the name of the generated array when the operator
    # produces one, i.e., return 'None' when the operator is a sink.
    if (self.is_data_generator ()):
      return self.accs[0].get_name ()
    if (self.is_data_sink ()):
      return None
    return self.accs[len(self.accs)-1].get_name ()

  def generate_cannonical_loop_top (self, df, producers):
    level = 0
    for idim in self.dims:
      dim_name = self.dims[idim]
      trip = None
      skip_red_dim = False
      #loop = self.build_l2_loop (level, trip, skip_red_dim, L2_LOOP_GENMODE_FULL, producers)
      loop = self.build_cannonical_loop (level)
      line = '{} {} /* {} - {} */\n'.format (loop, '{', dim_name, trip)
      df.write (line)
      level += 1

  def generate_cannonical_loop_bottom (self, df):
    level = len(self.dims)
    for idim in self.dims:
      dim_name = self.dims[idim]
      line = '  ' * level
      df.write (line + '}\n')
      level -= 1


  def generate_single_node_access (self, ref, PP, producers):
    ret = 'sna_' + ref.get_name ()
    ret += '['
    ret += '{}{}D ('.format (DIMAGE_TILE_POINTER, ref.get_num_dim ())
    ret += self.get_tile_size_str_list (ref, self.PP, producers)
    ret += ', '
    ret += self.get_constant_tile_iterator_list (ref, 1)
    ret += ')]'
    return ret

  def generate_single_node_statement (self, PP, producers):
    ret = ''
    is_write = True
    is_acum = False
    use_full_extent = True
    is_allgat_out_slice = False
    operator = ' += '
    num_acc = len(self.accs)
    PP.set_single_node ()
    gen_accs = []
    for ii,ref in enumerate(self.accs):
      acc = self.generate_access (ref, PP, producers, is_write, is_acum, use_full_extent, is_allgat_out_slice)
      gen_accs.append (acc)
    for ii,ref in enumerate(self.accs):
      acc = None
      acc_name = ''
      if (ii == 0):
        idx = len(self.accs)-1
        acc = gen_accs[idx]
        acc_name = self.accs[idx].get_tile_name ()
      else:
        idx = ii-1
        acc = gen_accs[ii-1]
        acc_name = self.accs[idx].get_tile_name ()
      ret += '{}[{}]'.format (acc_name, acc)
      if (ii < num_acc - 1):
        ret += operator
      operator = ' * '
      is_write = False
    ret += ';\n'
    PP.unset_single_node ()
    return ret


  ## Statement.generate_tile_fetch_for_single_node_access ():
  ## Generate the code necessary to fetch the single tile of a non-distributed array.
  ## This function is meant to be used exclusively for generating reference results.
  def generate_tile_fetch_for_single_node_access (self, df, producers):
    df.write ('  // Fetching SNA-references.\n')
    for ref in self.accs:
      #self.generate_tile_fetch_code (df, ref, producers, False)
      line = '  {} *{} = ({} + {});\n'.format (DIMAGE_DT, ref.get_tile_name (), ref.get_sna_ref_name (), DIMAGE_TILE_HEADER_MACRO)
      df.write (line)

  ## Statement.generate_single_node_operator ():
  ## Generate the corresponding reference code for a single operator of the DAG.
  ## This generator will differentiate between data generators, data sinks and 
  ## pure computational operators.
  def generate_single_node_operator (self, df, PP, producers, mrap):
    local_indent = '  '
    indent = local_indent * (len(self.dims) + 1)
    PP.set_single_node ()
    #if (len(self.accs) == 3):
    #  stmt_body = self.generate_statement (df, PP, producers, mrap, local_indent)
    #else:
    #  stmt_body = self.generate_statement_generic (df, PP, producers, mrap, local_indent)
    stmt_body = self.generate_single_node_statement (PP, producers)
    if (not self.is_compute_statement ()):
      PP.set_single_node ()
      stmt_body = self.generate_statement_generic (df, PP, producers, mrap, local_indent)
      PP.unset_single_node ()
      indent = local_indent
    if (self.is_data_generator ()):
      ## Read initial data for tensors of generators.
      PP.set_single_node ()
      ref = self.accs[0]
      arr_name = ref.get_sna_ref_name () #'sna_{}'.format (ref.get_name ())
      decl = '  {} *{};'.format (DIMAGE_DT, arr_name)
      df.write (decl)
      bcount_init = '  {} = 0;'.format (DIMAGE_BLOCK_COUNT)
      df.write (bcount_init)
      ref.allocate_local_tile (df, PP, True, self, True)
      PP.unset_single_node ()
    PP.set_single_node ()
    dimage_call = self.generate_dimage_operator_call (df, PP, producers, mrap, local_indent)
    PP.unset_single_node ()
    if (self.is_compute_statement ()):
      df.write ('{}// {} \n'.format (local_indent, self.name))
      df.write (local_indent + '#ifdef DIMAGE_KERNEL_LOOP\n')
      self.generate_tile_fetch_for_single_node_access (df, producers)
      self.generate_cannonical_loop_top (df, producers)
    if (not self.is_compute_statement ()):
      arr_name = self.get_ref (0).get_name ()
      needle = ', {}'.format (arr_name)
      nail = ', sna_{}'.format (arr_name)
      stmt_body = re.sub (needle, nail, stmt_body)
      if (self.is_data_sink ()):
        out_ref = self.get_ref (0)
        wtf_args = self.generate_single_node_write_to_file_arguments (out_ref, PP)
        num_array_dim = out_ref.get_num_dim ()
        stmt_body = '{}_{}D(\"{}\", {});\n'.format (WRITE_MATRIX_TO_FILE, num_array_dim, out_ref.get_sna_reference_filename (), wtf_args)
    df.write (indent + stmt_body)
    if (self.is_compute_statement ()):
      self.generate_cannonical_loop_bottom (df)
      df.write (local_indent + "#else\n")
      dimage_call = re.sub ('tile_', 'sna_', dimage_call)
      df.write (local_indent + dimage_call + '\n')
      df.write (local_indent + "#endif")
      df.write ('\n')
    PP.unset_single_node ()
    if (self.is_data_generator ()):
      return self.accs[0].get_name ()
    if (self.is_data_sink ()):
      return None
    return self.accs[len(self.accs)-1].get_name ()

  def generated_array_name (self):
    varname = re.sub ('^gen','',self.name)
    return varname

  def insert_operator_call (self, df):
    line = '{}();\n'.format (self.get_operator_name())
    if (self.is_data_generator ()):
      line = self.generated_array_name () + ' = ' + line
    df.write (line)

  # Compute and return the volume of an array slice of @ref
  # at the current statement. The slice is computed from
  # dividing an array extent by the number of processors
  # along the mapped dimension.
  def get_slice_vol_by_name (self, ref, PP):
    vol = ""
    for idim in self.dims:
      dim_name = self.dims[idim]
      ispace_pdim = self.map[idim]
      if (ref.is_dim_used (dim_name)):
        dim_size = ref.get_dim_size_if_used (dim_name)
        denom = 1
        array_pdim = ref.get_proc_map_by_dim_name (dim_name)
        # If the iteration space is unmapped, the current statement requires
        # the full slice. Hence, we don't divide the extent.
        if (array_pdim >= 0 and ispace_pdim >= 0):
          denom = PP.get_dim_size (ispace_pdim)
        if (not vol == ""):
          vol += " * "
        term = '{} ({}, {})'.format (DIMAGE_CEIL, dim_size, denom)
        vol = vol + term
    return vol 
    
  def get_local_communication_timer (self):
    return 'timer_KOMM_local_{}'.format (self.name)

  def get_local_computation_timer (self):
    return 'timer_comp_local_{}'.format (self.name)

  def start_computation_timer (self, df, indent):
    df.write (indent)
    df.write ('{} = -{};\n'.format(self.get_local_computation_timer (), DIMAGE_CLOCK))
    df.write ('\n')

  def restart_computation_timer (self, df, indent):
    df.write (indent)
    df.write ('{} += -{};\n'.format(self.get_local_computation_timer (), DIMAGE_CLOCK))
    df.write ('\n')

  def stop_computation_timer (self, df, indent):
    df.write (indent)
    timer_var = self.get_local_computation_timer ()
    df.write ('{} = {} + {};\n'.format(timer_var, DIMAGE_CLOCK, timer_var))
    df.write ('\n')


  def start_communication_timer (self, df, indent):
    df.write (indent)
    df.write ('{} = -{};\n'.format(DIMAGE_START_TIMER, DIMAGE_CLOCK))
    df.write ('\n')

  def stop_communication_timer (self, df, indent):
    df.write (indent)
    timer_var = self.get_local_communication_timer ()
    df.write ('{} += {} + {};\n'.format(timer_var, DIMAGE_CLOCK, DIMAGE_START_TIMER))
    df.write ('\n')

  def declare_timer (self, df): 
    df.write ('double {} = 0.0;\n'.format(self.get_local_computation_timer ()))
    df.write ('double {} = 0.0;\n'.format(self.get_local_communication_timer ()))
    



## Processor Space class. Store processor geometry.
class Processor:
  def __init__(self, num_dim, max_procs, proc_vector, form):
    self.np = num_dim
    self.dims = [max_procs] * num_dim
    self.max_procs = max_procs
    self.pvec = proc_vector
    self.single_node = False
    if (proc_vector == None):
      self.pvec = {}
      for pp in range(self.np):
        pname = self.get_varname (pp)
        self.pvec[pp] = pname
    self.sizes = {}
    self.cof = form

  def get_num_dim (self):
    return self.np

  def get_sizes (self):
    return self.sizes

  def set_single_node (self):
    self.single_node = True

  def unset_single_node (self):
    self.single_node = False

  def gcd(self):
    if (self.np == 2):
      return gcd(self.sizes[0],self.sizes[1])
    if (self.np == 3):
      return gcd(gcd(self.sizes[0],self.sizes[1]),gcd(self.sizes[1],self.sizes[2]))
    if (self.np > 3):
      sys.exit (42)
    return self.sizes[0]

  def product(self,vec):
    ret=1
    for vv in vec:
      ret = ret * vec[vv]
    return ret

  def lcm(self):
    if (self.single_node):
      return 1
    num = self.product (self.sizes)
    nzs = []
    for vv in self.sizes:
      if (self.sizes[vv] > 1):
        nzs.append (self.sizes[vv])
    mygcd = 1
    if (len(nzs) == 1):
      mygcd = nzs[0]
    if (len(nzs) == 2):
      mygcd = gcd(nzs[0],nzs[1])
    if (len(nzs) == 3):
      mygcd = gcd(gcd(nzs[0],nzs[1]),gcd(nzs[1],nzs[2]))
    den = mygcd ** (len(nzs) - 1)
    return num/den

  def get_max_procs (self):
    return self.max_procs

  def get_varname (self, pdim):
    varname = 'H_p{}'.format (pdim)
    return varname

  def get_value (self, pdim):
    if (pdim > len(self.pvec)):
      return 0
    return self.pvec[pdim]

  def get_proc_dim_symbol (self, pdim):
    if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
      return self.get_value (pdim)
    else:
      return self.get_varname (pdim)

  def get_dim_lb_str (self, pdim, lb):
    varname = self.get_proc_dim_symbol (pdim)
    ret = '{} >= {}'.format (varname, lb)
    return ret

  def get_dim_ub_str (self, pdim, ub):
    varname = self.get_proc_dim_symbol (pdim)
    ret = '{} <= {}'.format (varname, ub)
    return ret

  ## Return a constraint string of the form: \prod p{i} == max_procs
  def get_product_constraint_str (self):
    expr = ''
    for pp in range(self.np):
      varname = self.get_proc_dim_symbol (pp)
      if (pp > 0):
        expr += ' * '
      expr += str(varname)
    cstr = '{} == {}'.format (expr, self.max_procs)
    return cstr

  ## Declare one processor dimension variable in both the
  ## model and shadow files.
  def declare_dimension (self, pp, mf):
    pvar = self.get_varname (pp)
    cmd = "{} = Int('{}')\n".format (pvar, pvar)
    self.cof.add_var (cmd)
    mf.write (cmd)

  ## Processor.add_processor_space_constraints ():
  ## Declare the grid dimension variables 'pX', bound them
  ## between 1 and the max_procs, and add the constraint
  ## that their product should be equal to max_procs.
  def add_processor_space_constraints (self, mf):
    for pp in range(self.np):
      self.declare_dimension (pp, mf)
    for pp in range(self.np):
      ## Force at least 2 ranks per per process-space dimension.
      ## This will avoid degenerated cases of having 1 processor along 
      ## any one processor space.
      cstr = self.get_dim_lb_str (pp, 2)
      self.cof.add_cstr (cstr)
      cmd = 'opt.add ({})\n'.format (cstr)
      mf.write (cmd)
      cstr = self.get_dim_ub_str (pp, self.max_procs)
      self.cof.add_cstr (cstr)
      cmd = 'opt.add ({})\n'.format (cstr)
      mf.write (cmd)
    ## Introduce p_{i} >= p_{i+1} constraints. Accelerates convergence.
    for pp in range(self.np-1):
      var_left = self.get_proc_dim_symbol (pp)
      var_right = self.get_proc_dim_symbol (pp+1)
      cstr = '{} >= {}'.format (var_left, var_right)
      self.cof.add_cstr (cstr)
      cmd = 'opt.add ({})\n'.format (cstr)
      mf.write (cmd)
    cstr = self.get_product_constraint_str ()
    self.cof.add_cstr (cstr)
    cmd = 'opt.add ({})\n'.format (cstr)
    mf.write (cmd)

  ## Return a comma-separated list of '1' with as many 1s as dimensions in the grid.
  def get_single_node_processor_geometry (self):
    ret = ''
    for ii in range(len(self.sizes)):
      if (ii > 0):
        ret += ', '
      ret += '1'
    return ret

    
  ## Processor.get_processor_geometry_list_from_map (): return a list of comma
  ## separated processor dimensions given a computation or data mapping.
  ## @amap can store the mu-mapping of an operator or the pi-mapping of 
  ## tensor @ref.
  ## We iterate through @amap dimensions and add the number of PEs found along
  ## the mapped dimension.
  ## If @ref == None we will use only the mu-mappings.
  ## If so, a dimension we check whether the current dimension is unmapped 
  ## (pp < 0) or if @use_full is True, we add LCM to the list of PE dimensions.
  ## @use_full is only used when @ref == None.
  ## If @ref != None then we require @iters != None.
  ## Finally, if @ref != None, we can fetch the pi values and compare them.
  ## CBH
  def get_processor_geometry_list_from_map (self, amap, use_full, vec01 = None, ref = None, iters = None):
    ret = ''
    for dd in amap:
      ## Determine the number of tiles along a given processor grid dimension.
      if ((vec01 != None and len(amap) == len(vec01) and vec01[dd] == 1) or vec01 == None):
        if (ret != ''):
          ret += ', '
        pp = amap[dd]
        ## print ("p-sizes {}, dim {}, map {}, len.amap {}".format (self.sizes, dd, pp, len(amap)))
        ## If we don't receive the ref object, then we only use @amap.
        if (ref == None):
          if (pp < 0 or use_full):
            ret += str(self.lcm ())
          else:
            ret += str(self.lcm () / self.sizes[pp])
        else:
          ## We received a ref object and the iters (array of iterator names)
          ## We need to compare mu and pi mappings.
          iter_name = iters[dd]
          pp_arr_dim = ref.get_pi_by_name (iter_name)
          if (pp_arr_dim < 0 and pp >= 0):
            ret += str(self.lcm ())
          elif (pp_arr_dim == pp and pp >= 0):
            ret += str(self.lcm () / self.sizes[pp])           
          elif (pp_arr_dim == pp and pp < 0):
            ret += str(self.lcm ())
          else: # pp_arr_dim >= 0 and pp < 0 (Cannot need all but have only one piece)
            ret += 'ERROR'
    return ret

  def get_dim_macro_name (self, pdim):
    macroname = 'DIMAGE_P{}'.format (pdim)
    return macroname

  def get_processor_coordinate_str_list (self):
    ret = ''
    for ii,pp in enumerate(self.dims):
      varname = self.get_processor_coordinate_variable (ii)
      if (ii > 0):
        ret += ', '
      ret += varname
    return ret

  def get_array_of_pointer_processor_coordinates (self):
    ret = ''
    for ii,pp in enumerate(self.dims):
      varname = self.get_processor_coordinate_variable (ii)
      if (ii > 0):
        ret += ', '
      ret += '&{}'.format(varname)
    return ret

  def get_processor_coordinate_variable (self, pp):
    varname = 'dimage_p{}'.format (pp)
    return varname

  ## Declare the variables used to keep the logical coordinates of the
  ## processor grid within each rank.
  ## Also declare an array where the pointers to these variables are stored
  ## and then passed together to debugging functions.
  def declare_processor_coordinate_variables (self, df):
    for ii,pp in enumerate(self.dims):
      proc_var = self.get_processor_coordinate_variable (ii)
      df.write ('int {};\n'.format (proc_var))
    pclist = self.get_array_of_pointer_processor_coordinates ()
    df.write ('int *{}[] = {}{}, NULL{};\n'.format (DIMAGE_RANK_ARRAY, '{', pclist, '}'))


  ## @Processor.init_processor_coordinates : 
  ## Call DIMAGE_PROC_COORD_FUNC to convert the process rank to grid coordinates.
  ## The array of grid dimensions must have been populated in the program.
  def init_processor_coordinates (self, df):
    #rank_to_coords (int nd, int rank, int * dims, int * cc);
    nd = len(self.dims)
    rank = DIMAGE_PROC_RANK
    grid_dims = DIMAGE_GRID_DIMS
    proc_coords = DIMAGE_PROC_COORDS
    df.write ('  ')
    df.write ('{} ({}, {}, {}, {});\n'.format (DIMAGE_PROC_COORD_FUNC, nd, rank, grid_dims, proc_coords))
    ## Create 'dimage_pX' variables. These will be used to 
    ## access loops and data slices.
    for ii,pp in enumerate(self.dims):
      proc_var = self.get_processor_coordinate_variable (ii)
      dimage_proc_func = DIMAGE_PROC_COORD_FUNC
      df.write ('  ')
      #df.write ('{} = {}_{}D({}, {});\n'.format (proc_var, dimage_proc_func, len(self.dims), DIMAGE_PROC_RANK, ii))
      df.write ('{} = {}[{}];\n'.format (proc_var, DIMAGE_PROC_COORDS, ii))

  ## @Processor: Return the number of processor (ranks) along the given dimension.
  def get_dim_size (self, pdim):
    if (pdim < 0 or pdim > len(self.sizes)):
      print ("ERROR: Invalid dimension ({}) used to access Processor-grid object.".format (pdim))
      sys.exit (42)
    if (self.single_node):
      return 1
    return self.sizes[pdim]

  def get_max_dim_size (self):
    return max(self.sizes)
    
  def writeln(self, mf, line):
    mf.write(line + "\n")

  # Declare Z3 variable.
  def declare_variable (self, mf, varname):
    cmd = "{} = Int('{}')".format (varname, varname)
    self.writeln (mf, cmd)
    self.cof.add_var (cmd)

  def set_bounds (self, mf, varname, lb, ub):
    cstr = '{} >= {}, {} <= {}'.format (varname, lb, varname, ub)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def get_product (self):
    ptotal = 1
    for pp in range(self.np):
      ptotal *= self.dims[pp]
    return ptotal

  def maximize_parallelism (self, mf):
    objvar = 'O_par'
    cmd = "{} = Int('{}')".format (objvar, objvar) # Instead of FP also Int
    self.cof.add_var (cmd)
    self.writeln (mf, cmd)
    expr = ""
    for pp in range(len(self.dims)):
      if (not expr == ""):
        expr += " * "
      expr += 'p{}'.format (pp)
    cstr = '{} <= {}'.format (objvar, expr)
    cmd = 'opt.add ({})'.format (cstr)
    self.cof.add_cstr (cstr)
    self.writeln (mf, cmd)
    cmd = 'P_obj = opt.maximize ({})'.format (objvar)
    self.writeln (mf, cmd)

  def maximize_parallelism_old_ (self, mf):
    objvar = 'O_par'
    cmd = "{} = Int('{}')".format (objvar, objvar)
    self.writeln (mf, cmd)
    self.cof.add_var (cmd)
    expr = ""
    for pp in range(len(self.dims)):
      if (not expr == ""):
        expr += " + "
      expr += 'p{}'.format (pp)
    cmd = "opt.add ({} >= {})".format (objvar, expr)
    self.writeln (mf, cmd)
    cmd = 'P_obj = opt.minimize ({})'.format (objvar)
    self.writeln (mf, cmd)

  def declare_dimensions (self, mf):
    ptotal = self.dims[0] #self.get_product ()
    prod_str = ""
    for pp in range(self.np):
      if (pp > 0):
        prod_str += " * "
      pname = self.get_varname (pp)
      self.declare_variable (mf, pname)
      self.set_bounds (mf, pname, 1, ptotal)
      prod_str += pname
    cstr = '{} >= {}'.format (ptotal, prod_str)
    cmd = 'opt.add ({})'.format (cstr)
    self.writeln (mf, cmd)
    self.cof.add_cstr (cstr)

  def show_geometry (self):
    print ("P = {}".format (self.sizes))

  def read_processor_geometry (self, PP, solset):
    for pp in range(self.np):
      name = PP.get_varname (pp) #'p{}'.format (pp)
      val = solset[name]
      self.sizes[pp] = int(val)

  ## Store the sizes of the processor grid received as input.
  ## This function is meant to be used *only* with fixed-shape grids.
  def set_dim_sizes_from_fixed_grid (self, procvec):
    for ii,pp in enumerate(procvec):
      self.sizes[ii] = int(pp)

  def get_dimage_grid_varname (self):
    varname = DIMAGE_GRID_DIMS
    return varname

  def generate_processor_space_declarations (self, mf):
    dimlist = ""
    for pp in self.sizes:
      dimlist += '{}'.format (self.sizes[pp])
      dimlist += ', '
    dimlist += '0'
    varname = self.get_dimage_grid_varname ()
    decl = 'int {}[] = {}{}{};\n'.format (varname, '{', dimlist, '}')
    mf.write (decl)

  def print_tikz_graph (self, fout, par_x, par_y):
    for pp,dd in enumerate(self.pvec):
      nodename='p{}'.format (pp)
      nodelabel = '{\\large ' + nodename + '}'
      x=par_x
      y=par_y - 3 * pp
      command = '\\node [shape=rectangle,draw=green,line width=1mm] ({}) at ({},{}) {};'.format (nodename,x,y,nodelabel)
      fout.write (command + '\n')
    return len(self.pvec)

  def is_dimension_mapped (self, idim):
    if (self.map[idim] >= 0):
      return True
    return False


## Global routines used to write the script and read the solution.

## Old version of read_solution used exclusively with explore_processor_space.
def read_solution (filename):
  ff = open (filename, 'r')
  line = ff.readline ()
  line = line.strip ()
  if (line == ""):
    return 'unsat'
  if (line == "unsat"):
    return line
  line = re.sub ('\(','', line)
  line = re.sub ('\)','', line)
  parts = line.split (',')
  ff.close ()
  return int (parts[1])
  
def test_model_fixed_proc (modelfile, p1, p2):
  testmodel = 'test.model.py'
  cmd = 'cp {} {}'.format (modelfile, testmodel)
  os.system (cmd)
  ff = open (testmodel, 'a')
  ff.write ('opt.add (p0 == {})\n'.format (p1))
  ff.write ('opt.add (p1 == {})\n'.format (p2))
  ff.write ('print(opt.check())\n')
  ff.write ('sol = opt.model ()\n')
  ff.write ('for vv in sol:\n')
  ff.write ('  print(vv, sol[vv])\n')
  ff.close ()
  cmd = 'python ' + testmodel + " | grep req_MM > a.sol"
  os.system (cmd)
  return read_solution('a.sol')

def explore_processor_space (modelfile, PP):
  totalp = PP.get_product ()
  if (PP.get_num_dim () == 2):
    for pp in range(totalp):
      if (totalp % (pp+1) == 0):
        p1 = pp + 1
        p2 = totalp / p1
        #print ("Testing {} x {}".format (p1,p2))
        comm = test_model_fixed_proc (modelfile, p1, p2)
        print ("Comm {} x {} = {}".format (p1,p2,comm))
  if (PP.get_num_dim () == 3):
    for pp in range(totalp):
      if (totalp % (pp+1) == 0):
        p1 = pp + 1
        p2 = totalp / p1
        #print ("Testing {} x {}".format (p1,p2))
        comm = test_model_fixed_proc (modelfile, p1, p2)
        print ("Comm {} x {} = {}".format (p1,p2,comm))
  
## Extract the per-node capacity and convert it to number of elements
## depending on the data type.
def read_per_node_capacity (param_arg):
  factor = 1
  arg = re.sub ('-memcap=','', param_arg) 
  if (arg.find ("k") >= 0 or arg.find("K") >= 0):
    arg = re.sub ("[Kk]","",arg)
    factor = 1024
  if (arg.find ("M") >= 0 or arg.find("m") >= 0):
    arg = re.sub ("[Mm]","",arg)
    factor = 1024*1024
  if (arg.find ("G") >= 0 or arg.find("g") >= 0):
    arg = re.sub ("[Gg]","",arg)
    factor = 1024**3
  elem_size = 8
  if (DIMAGE_DT == 'float'):
    elem_size = 4
  ret = int(arg) * factor / elem_size
  return ret

def read_max_processors (arg):
  if (arg.find ("procs=") < 0):
    print ("Invalid processor geometry")
    sys.exit (42)
  arg = re.sub ('-procs=','',arg)
  parts = arg.split(",")
  npdim = int(re.sub ('[Dd]','',parts[0]))
  pg = int(re.sub ('[Pp]','',parts[1]))
  return (npdim,pg)

def read_grid_shape (arg):
  if (arg.find ("procs=") < 0):
    print ("Invalid processor geometry")
    sys.exit (42)
  arg = re.sub ('-procs=','',arg)
  parts = arg.split(",")
  ret = []
  prod = 1
  for pp in parts:
    procs = int(pp)
    ret.append (procs)
    prod = prod * procs
  print ('Proc. vec : {}'.format (ret))
  return (ret,prod)

def declare_variable (mf, varname, decl, cof):
  decl[varname] = varname
  decl_cmd = '{} = Int (\'{}\') #MK\n'.format (varname, varname)
  mf.write (decl_cmd)
  cof.add_var (decl_cmd)
  return decl

def declare_float (mf, varname, decl, cof):
  decl[varname] = varname
  decl_cmd = '{} = Real (\'{}\') #MK\n'.format (varname, varname)
  mf.write (decl_cmd)
  cof.add_var (decl_cmd)
  return decl


## Write the main communication objective:
##   K_prog = \sum K_i, where K_i is a computation of the program.
def set_main_comm_objective (SS, mf, decl, cof):
  comm_expr = ""
  for sid in SS:
    stmt = SS[sid]
    if (not comm_expr == ""):
      comm_expr += " + "
    comm_expr += stmt.get_comm_var ()
  obj_var = 'K_prog'
  decl = declare_variable (mf, obj_var, decl, cof)
  comm_expr = '{} >= {}'.format (obj_var, comm_expr)
  cmd = 'opt.add ({})\n'.format (comm_expr)
  cof.add_cstr (comm_expr)
  mf.write (cmd)
  cstr = '{} >= 0'.format (obj_var)
  cmd = 'opt.add ({})\n'.format (cstr)
  cof.add_cstr (cstr)
  mf.write (cmd)
  # NOTE: the minimize command is invoked directly in the COF object.
  cmd = 'K_obj = opt.minimize ({})\n'.format (obj_var)
  mf.write (cmd)
  return decl


def set_global_performance_objectve (SS, mf, decl, cof):
  obj_var = 'G_prog'
  #decl = declare_float (mf, obj_var, decl, cof)
  decl = declare_variable (mf, obj_var, decl, cof)
  expr = ''
  for sid in SS:
    stmt = SS[sid]
    if (not stmt.is_compute_statement ()):
      continue
    s_gov = stmt.get_gpo_varname () 
    cstr = '{} >= {}'.format (obj_var, s_gov)
    cmd = 'opt.add ({})\n'.format (cstr)
    mf.write (cmd)
    cof.add_cstr (cstr)
    if (expr != ''):
      expr += ' + '
    expr += s_gov
  ## Alternate ops.
  cstr = '{} == {}'.format (obj_var, expr)
  cmd = 'opt.add ({})\n'.format (cstr)
  mf.write (cmd)
  cof.add_cstr (cstr)
  #cstr = '{} >= 0'.format (obj_var) 
  #cmd = 'opt.add ({})\n'.format (cstr)
  #mf.write (cmd)
  #cof.add_cstr (cstr)
  cmd = '{} = opt.minimize ({})\n'.format (obj_var, obj_var)
  mf.write (cmd)
  return decl

## Read model solution from solution file. 
## Solution file has the same name as the input file, but with the '.rels'
## extension replaced by '.sol'.
def read_solution_from_file (solfile):
  ff = open (solfile, 'r')
  stf = open (solfile + '.stats', 'w')
  ret = {}
  for line in ff.readlines ():
    #if (line[0] != '('):
    #  stf.write (line + '\n')
    #  continue
    line = line.strip ()
    line = re.sub (" ","",line)
    line = re.sub ("\(","",line)
    line = re.sub ("\)","",line)
    if (option_debug >= 3):
      print (line)
    if (line.find ("unsat") >= 0):
      ff.close ()
      stf.close ()
      return None
    if (line == "sat"):
      continue
    #if (line.find ("div") >= 0):
    #  continue
    #if (line.find ("mod") >= 0):
    #  continue
    if (line.find ("->") >= 0):
      continue
    parts = line.split (",")
    # If line components are not separated by ',', then search for ':'
    if (len(parts) <= 1):
      parts = line.split (":")
    ret[parts[0]] = parts[1] 
  ff.close ()
  stf.close ()
  return ret


def show_solution_from_table (solset):
  for kk in sorted(solset):
    print ("{} : {}".format (kk, solset[kk]))

def compare_costs (solset):
  k_cost = 0
  w_cost = 0
  for kk in sorted(solset):
    if (kk.find ("K_") == 0):
      k_cost += float(solset[kk])
    if (kk.find ("W_") == 0):
      w_cost += float(solset[kk])
  k_cost = k_cost / 10**9
  w_cost = w_cost / 10**9
  print ("Comm. cost (K): {}".format (k_cost))
  print ("Comp. cost (W): {}".format (w_cost))


def compare_costs_stmt (SS, solset):
  for ii in SS:
    stmt = SS[ii]
    k_var = stmt.get_comm_var ()
    w_var = stmt.get_comp_cost_variable ()
    k_cost = int(solset[k_var])
    w_cost = int(solset[w_var])
    ratio = 'inf'
    if (k_cost != 0):
      ratio = w_cost * 1.0/ k_cost
    print ("Statement {} : K={}, W={}, ratio(w/k)={}".format (stmt.get_name (), k_cost, w_cost, ratio))

def store_solution_to_file (solset, solfile):
  ff = open (solfile, "w")
  for kk in sorted(solset):
    ff.write ("{}:{}\n".format (kk, solset[kk]))
  ff.close ()


def print_program_tikz_graph (tikzfilename, PP, SS, AA):
  ff = open (tikzfilename, 'w')
  ff.write ('\\documentclass[tikz]{standalone}\n')
  ff.write ('\\begin{document}\n')
  ff.write ('\\begin{tikzpicture}\n')
  ff.write ('\\tikzstyle{arrow} = [thick,->,>=stealth]\n')
  rows = 0
  for ss in SS:
    stmt = SS[ss]
    rows += len(stmt.get_dims())
  ## Print processors dims
  x = 5
  y = - int(rows/2)
  PP.print_tikz_graph (ff, x, y)

  ## Print iteration-space dims
  x = 0
  y = 0
  for ss in sorted(SS):
    stmt = SS[ss]
    y -= stmt.print_tikz_graph (ff, x, y)

  ## Print array-space dims
  x = 10
  y = 0
  for aa in sorted(AA):
    ref = AA[aa]
    y -= ref.print_tikz_graph (ff, x, y)
  #for ss in sorted(SS):
  #  stmt = SS[ss]
  #  if (stmt.is_data_sink () or stmt.is_data_generator ()):
  #    y -= stmt.print_ref_tikz_graph (ff, x, y)

  ff.write ('\\end{tikzpicture}\n')
  ff.write ('\\end{document}')
  ff.close ()
  os.system ('pdflatex {} > /dev/null'.format (tikzfilename))


def show_help ():
  names = []
  desc = []
  otype = []
  defval = []

  names.append('-procs')
  desc.append('Process geometry tuple (e.g., "2D,4p")')
  otype.append('str')
  defval.append('None')

  names.append('-solve')
  desc.append('Invoke Z3 solver')
  otype.append('int')
  defval.append('[1 (Default), 0 (Reuse last solution)]')

  names.append('-obj')
  desc.append('Objective function used')
  otype.append('str')
  defval.append('["comm+comp" (Default), "comm-only", "calc-node-req"]')

  names.append('-memcap')
  desc.append('Memory cap to use')
  otype.append('str')
  defval.append('["0" (Default, no cap)]')

  names.append('-check')
  desc.append('Every operator compares results against a pre-computed reference.')
  otype.append('bool')
  defval.append('[False (Default, no check)]')

  names.append('-graph')
  desc.append('Generate a graph (in PDF) describing the found mapping.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-debug')
  desc.append('Enable debugging level. Internal use.')
  otype.append('int')
  defval.append('[0 (Default, no debug)]')

  names.append('-verbose')
  desc.append('Show additional mapping information.')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-reference')
  desc.append('Generate single node reference (only matmul-like).')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-help')
  desc.append('Show summary of options (this help).')
  otype.append('bool')
  defval.append('[False (Default)]')

  names.append('-used')
  desc.append('Show summary of used options.')
  otype.append('bool')
  defval.append('[False (Default)]')

  print ('***************************************************')
  print ("Showing help:")
  idx=1
  for nn,tt,vv,dd in zip(names,otype,defval,desc):
    print ('[{}] : {} ({}) {} : {}'.format (idx, nn, tt, vv, dd))
    idx += 1
  print ('***************************************************')


##################################################################################
##
##        Main driver starts here.
##
##################################################################################




## Extract options passed to the script.
option_grid = None
option_solve = '-solve=1'
option_objective = '-obj=comm+comp'
option_memcap = "-memcap=0"
option_infile = None
option_check = False
option_graph = False
option_debug = 0
option_verbose = False
option_reference = False
option_help = False
option_used = False
option_include_all = False
for arg_id,dimage_option in enumerate(sys.argv):
  if (arg_id < 1):
    continue
  if (dimage_option.find ('-procs=') == 0):
    option_grid = dimage_option
  if (dimage_option.find ('-solve=') == 0):
    option_solve = dimage_option
  if (dimage_option.find ('-obj=') == 0):
    option_objective = dimage_option
  if (dimage_option.find ('-memcap=') == 0):
    option_memcap = dimage_option
  if (dimage_option.find ('-check') == 0):
    option_check = True
  if (dimage_option.find ('-graph') == 0):
    option_graph = True
  if (dimage_option.find ('-debug=') == 0):
    option_debug = int(re.sub('-debug=','',dimage_option))
  if (dimage_option.find ('-verbose') == 0):
    option_verbose = True
  if (dimage_option.find ('-reference') == 0):
    option_reference = True
  if (dimage_option.find ('-help') == 0):
    option_help = True
  if (dimage_option.find ('-used') == 0):
    option_used = True
  if (len(re.findall ('\.rels$', dimage_option)) == 1):
    option_infile = dimage_option


if (option_help):
  show_help ()
  sys.exit (0)

if (len(sys.argv) < 3):
  print ("Usage: python dimage.py input.rels -procs=#D,#p [-memcap=C[M|K]] [-solve=0:No|1:Yes] [-obj=comm-only|comm+comp|calc-node-req]")
  print ("Legend:")
  print ("Processor space: -procs=#D,#p")
  print ("Per-node capacity (Optional, Default 0): -memcap=C[MB|KB]")
  print ("Request solve (Optional, Default 1): -solve=0|1; 0=No-solve (Reuse previous solution),1=solve")
  print ("Objective-mode (Optional, Default 'comm+comp'): -obj=comm-only|comm+comp|calc-node-req")
  print ("Example: time python {} 2mm-960.rels -memcap=1024K -procs=2D,8p -solve=1 -obj=comm-only".format (DIMAGE_PY_SCRIPT))
  sys.exit(42)

print ("Summary of received options")
print ("Input File: {}".format (option_infile))
print ("Option memory cap: {}".format (option_memcap))
print ("Option grid: {}".format (option_grid))
print ("Option check: {}".format (option_check))
print ("Option solve: {}".format (option_solve))
print ("Option objective: {}".format (option_objective))
print ("Option verbose: {}".format (option_verbose))
print ("Option debug: {}".format (option_debug))
print ("Option graph: {}".format (option_graph))
print ("Option help: {}".format (option_help))
print ("Option used: {}".format (option_used))

option_estimate_per_node_requirement = False

infile  = option_infile
pnc = read_per_node_capacity (option_memcap)

npdim=0
maxprocs=0
procvec = None

# Extract grid information
if (DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
  procvec, maxprocs = read_grid_shape (option_grid)
  npdim = len(procvec)
else:
  npdim, maxprocs = read_max_processors (option_grid)


per_node_cap_str_arg = option_grid

# Determine whether to call solver or not.
call_solver = True
if (option_solve.find ("=0") >= 0):
  call_solver = False
if (option_solve.find ("=1") >= 0):
  call_solver = True

## Make comm+comp the default objective mode.
obj_mode = DIMAGE_OBJ_COMM_COMP
if (option_objective.find ("=comm+comp") >= 0):
  obj_mode = DIMAGE_OBJ_COMM_COMP
elif (option_objective.find ("=comm-only") >= 0):
  obj_mode = DIMAGE_OBJ_COMM_ONLY
elif (option_objective.find ("=calc-node-req") >= 0):
 option_estimate_per_node_requirement = True 
elif (option_objective != None):
  print ("[ERROR] Unknown objective selected. Expected '=comm-only', '=comm+comp' or '=calc-node-req'")
  sys.exit (42)

ff = open (infile, "r")

modelfile = re.sub ('\.rels','.model.py', infile)
solfile = re.sub ('\.rels','.sol', infile)
tikzfile =  re.sub ('\.rels','.tex', infile)
cfilename = re.sub ('\.rels','.dimage.c', infile)

mf = open (modelfile + '.shadow', "w")
mf.write('from z3 import *\n\n')
mf.write("opt = Then('simplify','ufnia','qfnra').solver ()\n\n")

cmd = '## Per node max. capacity :{}\n'.format (pnc)
mf.write (cmd)

mf.write ('## Num. processor dimension: {}\n'.format (npdim))
mf.write ('## Max. total processors: {}\n'.format (maxprocs))

# Formulation object
form = Comm_Opt_Form (modelfile, procvec)

# Processor space object
PP = Processor (npdim, maxprocs, procvec, form)
PP.add_processor_space_constraints (mf)

NP = PP.get_num_dim ()

nstmt = int (ff.readline())
SS = {}
CG = [] # control graph
AA = {}
for ss in range(nstmt):
  stmt = Statement (form, PP, NP)
  stmt.init_from_file (ff)
  AA = stmt.collect_arrays (AA)
  ## Enable statement below to show info for each statement.
  SS[stmt.get_name()] = stmt
  CG.append (stmt)
 
# Gather all the arrays in a separate collection.
for name in AA:
  aa = AA[name]
  if (option_debug >= 4):
    print ('Array {} - :{}:'.format (name, aa.get_name ()))

ff.close ()

if (option_estimate_per_node_requirement):
  estimate_per_node_requirement (SS, PP, procvec)
  sys.exit (0)

decl = {}

mf.write ("\n")
mf.write ("## Define mu-variables for iteration spaces\n")
for ss in SS:
  stmt = SS[ss]
  decl = stmt.declare_map_vars (mf, decl)

if (decl == None):
  print ("decl is None at 4182")
  sys.exit (42)

mf.write ("\n")
mf.write ("## Bound IS-dims and P-dims to not allow multiple mappings of any dimension\n")
for ss in SS:
  stmt = SS[ss]
  ## NOTE: Sum of mu variables across a fixed iteration-space
  ## dimension must be done through the 'sum_mu_*' variables and constraints.
  mf.write ("\n")
  mf.write ('## {} - set_proc_sum_bounds\n'.format (stmt.get_name ()))
  decl = stmt.set_proc_sum_bounds (mf, decl)
  mf.write ("\n")
  mf.write ('## {} - declare_ref_vars \n'.format (stmt.get_name ()))
  decl = stmt.declare_ref_vars (mf, decl)
  mf.write ("\n")
  mf.write ('## {} - set_ref_sum_bounds \n'.format (stmt.get_name ()))
  decl = stmt.set_ref_sum_bounds (mf, decl)

for arrname in AA:
  continue
  mf.write ("\n")
  mf.write ('## {} - declare_ref_vars \n'.format (arrname))
  aa = AA[arrname]
  mf.write ("\n")
  mf.write ('## {} - set_dim_sum_bounds \n'.format (arrname))
  decl = aa.set_dim_sum_bounds (mf, decl)
  mf.write ("\n")
  mf.write ('## {} - set_proc_sum_bounds \n'.format (arrname))
  decl = aa.set_proc_sum_bounds (mf, decl)

mf.write ("\n")
mf.write ("## Define capacity expressions")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  decl = stmt.declare_block_variables (mf, decl)


mf.write ("\n")
mf.write ("## Compute communication slice-expressions")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  decl = stmt.set_comm_slice_expressions (mf, decl)

if (include_capacity_constraints (per_node_cap_str_arg)):
  mf.write ("\n")
  mf.write ("## Set capacity constraints (original K_*?)\n")
  for ss in SS:
    stmt = SS[ss]
    mf.write ("\n")
    decl = stmt.set_statement_capacity_constraint (mf, decl, pnc, maxprocs)


mf.write ("\n")
mf.write ("## Local-Mapping (lambda / \lambda) constraints: link mu and pi\n")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  decl = stmt.add_matching_constraints (mf, decl)


mf.write ("\n")
mf.write ("## Declaring replication variables for each array\n")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  decl = stmt.declare_replication_variables (mf, decl)


mf.write ("\n")
mf.write ("## Bounding replication variables of each array\n")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  stmt.bound_replication_variables (mf)

mf.write ("\n")
mf.write ("## Replication constraints: linking rho variables of each array\n")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  stmt.add_replication_constraints (mf)

mf.write ("\n")
mf.write ("## Replication constraints: setting rho = f(pi_k), for each dim k of A \n")
for ss in SS:
  stmt = SS[ss]
  mf.write ("\n")
  stmt.set_array_dim_replication_expression (mf)


mf.write ("\n")
mf.write ("## Communication (K) constraints\n")
for ss in SS:
  stmt = SS[ss]
  decl = stmt.set_comm_constraints (mf, decl)

mf.write ("\n")
mf.write ("## Computation (W) cost expressions \n")
for ss in SS:
  stmt = SS[ss]
  if (obj_mode == DIMAGE_OBJ_COMM_COMP):
    decl = stmt.set_mu_dimension_sum (mf, decl)
    decl = stmt.set_computation_cost_expression (mf, decl)
  decl = stmt.set_performance_expression_constraints (mf, decl, obj_mode)

mf.write ("\n")
mf.write ("## Global Performance Objective (GPO): \sum_S (W_S + alpha * K_S)\n")
decl = set_global_performance_objectve (SS, mf, decl, form)

mf.close ()

opt_val = 0
solset = None
GLOBAL_SOL_VAR='G_prog'
it = 1
max_tries = DIMAGE_MAX_TRIES
n_fails = 0
g_sols = []
start_time = timer ()
while (call_solver and n_fails < max_tries):
  form.write_formulation (opt_val, n_fails)
  curr_sol = None
  if (call_solver):
    if (option_debug >= 2):
      print ("Using Z3-solver")
      print ("Model file: {}".format (modelfile))
    os.system ('rm -f {}'.format (solfile))
    cmd = 'python {} | sort > {}'.format (modelfile, solfile)
    os.system (cmd)
    # NOTE: G_prog is has been temporarily removed.
    cmd = 'grep G_prog {} > a.sol'.format (solfile)
    os.system (cmd)
    curr_sol = read_solution_from_file (solfile) 
    if (curr_sol == None or not GLOBAL_SOL_VAR in curr_sol):
      n_fails += 1
      continue
      #break
    #print (curr_sol)
    if (curr_sol != None and GLOBAL_SOL_VAR in curr_sol):
      new_opt_val = int(curr_sol[GLOBAL_SOL_VAR])
      if (opt_val == 0):
        solset = curr_sol
        opt_val = new_opt_val
        g_sols.append (opt_val)
        print ("Found first solution : {}".format (opt_val))
      elif (new_opt_val < opt_val):
        solset = curr_sol
        opt_val = new_opt_val
        g_sols.append (opt_val)
        #opt_val = read_solution ('a.sol')
        if (option_debug >= 2):
          print ("Found improved solution : {}".format (opt_val))
      else:
        print ("No new solution found, reducing step to: {} G_prog vs {} {}".format (n_fails + 2, n_fails + 1, opt_val))
        n_fails += 1
      print ("Iteration #{} - Solution found: {}".format (it, opt_val))
      it += 1
      print ("------------------------------------------------------------------")
stop_time = timer ()

avg_time = (stop_time - start_time) / (it + n_fails)
print ("[INFO] No. solutions found: {}".format (it))
print ("[INFO] No. of attempted retries: {}".format (n_fails))
print ("[INFO] Total solver calls: {}".format (it + n_fails))
print ("[INFO] Average time per solver call: {}sec".format (avg_time))

if (solset == None and call_solver):
  print ("No solution found.")
  sys.exit (1)

if (call_solver):
  if (option_debug >= 2):
    print (solset)
  store_solution_to_file (solset, solfile)

  if (option_verbose):
    # Show all intermediate results
    for ii,gg in enumerate(g_sols):
      print ("Sol.{} : {}".format (ii, gg))

## If the solver was not invoked, read the latest solution.
if (not call_solver):
  print ("[INFO] Reading previous solution from {}".format (solfile))
  solset = read_solution_from_file (solfile) 
  print (solset)

for ss in SS:
  stmt = SS[ss]
  stmt.extract_mappings_from_solution_set (solset)

for ss in SS:
  stmt = SS[ss]
  if (option_verbose):
    stmt.show_maps ()

if (option_graph):
  print_program_tikz_graph (tikzfile, PP, SS, AA)

if (option_debug >= 5):
  for ss in SS:
    stmt = SS[ss]
    stmt.check_capacity_requirement (solset, pnc)

if (option_reference):
  print ("[INFO] Generating serial references ...")
  for ii,ss in enumerate(SS):
    stmt = SS[ss]
    stmt.gencode_matrix_data_generator (ii+1)

if (option_verbose):
  print ("********************************************************************")
  print ("Showing computed mappings:")
  for ss in SS:
    stmt = SS[ss]
    stmt.report_mappings (AA, PP)
  print ("********************************************************************")

# Cases to align: 
# 1) dim(grid) > dim(comp); 
# 2) replication work (mu unmapped) but pi mapped on generators. 
for ss in SS:
  stmt = SS[ss]
  stmt.statement_align_mu_mappings (PP)

if (option_verbose):
  show_solution_from_table (solset)

## See note at the top of the script.
if (not DIMAGE_OPTION_USE_FIXED_PROC_GEOMETRY):
  PP.read_processor_geometry (PP, solset)
  if (option_verbose or option_debug >= 2):
    PP.show_geometry ()
else:
  PP.set_dim_sizes_from_fixed_grid (procvec)

dist = Dist (PP,SS,CG,cfilename)
dist.codegen (sys.argv, avg_time, it, n_fails, solset)
dist.gen_makefile ()
sys.exit (0)
