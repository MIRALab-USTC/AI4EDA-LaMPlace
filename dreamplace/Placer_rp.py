##
# @file   Placer.py
# @author Yibo Lin
# @date   Apr 2018
# @brief  Main file to run the entire placement flow.
#
import logging
import os
import sys
import time
from typing import Tuple
import argparse
import matplotlib
import numpy as np

# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import NonLinearPlace  # noqa: E402
import Params  # noqa: E402
import PlaceDB  # noqa: E402
import Timer  # noqa: E402
import pickle
import dreamplace.configure as configure  # noqa: E402

matplotlib.use("Agg")
pl_file = " "

def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False





def read_pl_file(
    placedb: PlaceDB.PlaceDB, pl_file: str, params: Params.Params
):
    t1 = time.time()
    num = 0
    # macro_names = None
    # if os.path.exists(f'benchmarks/{params.design_name()}/macro_names.pkl'):
    #     with open(f'benchmarks/{params.design_name()}/macro_names.pkl','rb') as f:
    #         macro_names = pickle.load(f)
    with open(pl_file, encoding="utf8") as f:
        scale = False
        lines = f.readlines()
        #ine = lines[0].strip().split()
        #if line[0] == "True":
        #    scale = True
        #    shift_factor = params.shift_factor
        #    scale_factor = params.scale_factor
        #scale = False
        for line in lines:
            #判断是否需要scale
            line = line.strip().split()
            if len(line) >= 4:
                node_name = line[0]
                if node_name in placedb.node_name2id_map:
                    bottom_left_x, bottom_left_y = float(line[1]), float(line[2])
                    print(f"[INFO   ] read macro: {node_name}; (x,y): ({bottom_left_x},{bottom_left_y})")
                    idx = placedb.node_name2id_map[node_name]
                    if not scale:
                        placedb.node_x[idx] = bottom_left_x
                        placedb.node_y[idx] = bottom_left_y
                
    print(f'[INFO   ] read pl_file takes {time.time()-t1}s')
    return placedb


def read_dreamplace_pl_file(
    placedb: PlaceDB.PlaceDB, pl_file: str, shift_factor: Tuple[float, float]
):
    with open(pl_file, encoding="utf8") as f:
        for line in f:
            if line.startswith("o"):
                line = line.strip().split()
                node_name = line[0]
                bottom_left_x, bottom_left_y = int(line[1]), int(line[2])
                if node_name in placedb.fixed_node_name:
                    idx = placedb.node_name2id_map[node_name]
                    placedb.node_x[idx] = bottom_left_x - shift_factor[0]
                    placedb.node_y[idx] = bottom_left_y - shift_factor[1]
    return placedb


def place(params):
    """
    @brief Top API to run the entire placement flow.
    @param params parameters
    """
    assert (not params.gpu) or configure.compile_configurations[
        "CUDA_FOUND"
    ] == "TRUE", "CANNOT enable GPU without CUDA compiled"

    np.random.seed(params.random_seed)
    # read database
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    read_pl_file(placedb, pl_file, params)

    logging.info("reading database takes %.2f seconds" % (time.time() - tt))

    # Read timing constraints provided in the benchmarks into out timing analysis
    # engine and then pass the timer into the placement core.
    timer = None
    if params.timing_opt_flag:
        tt = time.time()
        timer = Timer.Timer()
        timer(params, placedb)
        # This must be done to explicitly execute the parser builders.
        # The parsers in OpenTimer are all in lazy mode.
        timer.update_timing()
        logging.info("reading timer takes %.2f seconds" % (time.time() - tt))

        # Dump example here. Some dump functions are defined.
        # Check instance methods defined in Timer.py for debugging.
        # timer.dump_pin_cap("pin_caps.txt")
        # timer.dump_graph("timing_graph.txt")

    # solve placement
    tt = time.time()
    placer = NonLinearPlace.NonLinearPlace(params, placedb, timer)
    logging.info(
        "non-linear placement initialization takes %.2f seconds" % (time.time() - tt)
    )
    metrics = placer(params, placedb)
    logging.info("non-linear placement takes %.2f seconds" % (time.time() - tt))

    # write placement solution
    path = "%s/%s" % (params.result_dir, params.design_name())
    if not os.path.exists(path):
        os.system("mkdir -p %s" % (path))
    #gp_out_file = os.path.join(
    #    path, "%s.gp.%s" % (params.design_name(), params.solution_file_suffix())
    #)
    gp_out_file = pl_file+".def"
    placedb.write(params, gp_out_file)

    # call external detailed placement
    # TODO: support more external placers, currently only support
    # 1. NTUplace3/NTUplace4h with Bookshelf format
    # 2. NTUplace_4dr with LEF/DEF format
    if params.detailed_place_engine and os.path.exists(params.detailed_place_engine):
        logging.info(
            "Use external detailed placement engine %s" % (params.detailed_place_engine)
        )
        if params.solution_file_suffix() == "pl" and any(
            dp_engine in params.detailed_place_engine
            for dp_engine in ["ntuplace3", "ntuplace4h"]
        ):
            dp_out_file = gp_out_file.replace(".gp.pl", "")
            # add target density constraint if provided
            target_density_cmd = ""
            if params.target_density < 1.0 and not params.routability_opt_flag:
                target_density_cmd = " -util %f" % (params.target_density)
            cmd = "%s -aux %s -loadpl %s %s -out %s -noglobal %s" % (
                params.detailed_place_engine,
                params.aux_input,
                gp_out_file,
                target_density_cmd,
                dp_out_file,
                params.detailed_place_command,
            )
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info(
                "External detailed placement takes %.2f seconds" % (time.time() - tt)
            )

            if params.plot_flag:
                # read solution and evaluate
                placedb.read_pl(params, dp_out_file + ".ntup.pl")
                iteration = len(metrics)
                pos = placer.init_pos
                pos[0 : placedb.num_physical_nodes] = placedb.node_x
                pos[
                    placedb.num_nodes : placedb.num_nodes + placedb.num_physical_nodes
                ] = placedb.node_y
                hpwl, density_overflow, max_density = placer.validate(
                    placedb, pos, iteration
                )
                logging.info(
                    "iteration %4d, HPWL %.3E, overflow %.3E, max density %.3E"
                    % (iteration, hpwl, density_overflow, max_density)
                )
                placer.plot(params, placedb, iteration, pos)
        elif "ntuplace_4dr" in params.detailed_place_engine:
            dp_out_file = gp_out_file.replace(".gp.def", "")
            cmd = "%s" % (params.detailed_place_engine)
            for lef in params.lef_input:
                if "tech.lef" in lef:
                    cmd += " -tech_lef %s" % (lef)
                else:
                    cmd += " -cell_lef %s" % (lef)
                benchmark_dir = os.path.dirname(lef)
            cmd += " -floorplan_def %s" % (gp_out_file)
            if params.verilog_input:
                cmd += " -verilog %s" % (params.verilog_input)
            cmd += " -out ntuplace_4dr_out"
            cmd += " -placement_constraints %s/placement.constraints" % (
                # os.path.dirname(params.verilog_input))
                benchmark_dir
            )
            cmd += " -noglobal %s ; " % (params.detailed_place_command)
            # cmd += " %s ; " % (params.detailed_place_command) ## test whole flow
            cmd += "mv ntuplace_4dr_out.fence.plt %s.fence.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.init.plt %s.init.plt ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out %s.ntup.def ; " % (dp_out_file)
            cmd += "mv ntuplace_4dr_out.ntup.overflow.plt %s.ntup.overflow.plt ; " % (
                dp_out_file
            )
            cmd += "mv ntuplace_4dr_out.ntup.plt %s.ntup.plt ; " % (dp_out_file)
            if os.path.exists("%s/dat" % (os.path.dirname(dp_out_file))):
                cmd += "rm -r %s/dat ; " % (os.path.dirname(dp_out_file))
            cmd += "mv dat %s/ ; " % (os.path.dirname(dp_out_file))
            logging.info("%s" % (cmd))
            tt = time.time()
            os.system(cmd)
            logging.info(
                "External detailed placement takes %.2f seconds" % (time.time() - tt)
            )
        else:
            logging.warning(
                "External detailed placement only supports NTUplace3/NTUplace4dr API"
            )
    elif params.detailed_place_engine:
        logging.warning(
            "External detailed placement engine %s or aux file NOT found"
            % (params.detailed_place_engine)
        )
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default='test/iccad2015.ot/superblue1.json')
    parser.add_argument("--pl",default='graph_data/dataset/placement/superblue1/result67.pl')
    parser.add_argument("--res_dir",default="try.txt")
    args = parser.parse_args()
    """
    @brief main function to invoke the entire placement flow.
    """
    logging.root.name = "DREAMPlace"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)-7s] %(name)s - %(message)s",
        stream=sys.stdout,
    )
    params = Params.Params()
    params.printWelcome()
    # load parameters
    # params.load(sys.argv[1])
    params.load(args.config)
    logging.info("parameters = %s" % (params))
    # control numpy multithreading
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    pl_file = args.pl
    # run placement
    tt = time.time()
    metrics = place(params)
    metric = metrics[-1]
    if args.res_dir:
        with open(args.res_dir, "a") as f:
            f.write(f"{metric.hpwl.data/1e9} {metric.overflow.item()}  {metric.tns} {metric.wns}  --{metric.iteration} --{args.pl}\n")
    logging.info("placement takes %.3f seconds" % (time.time() - tt))