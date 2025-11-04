import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import numpy as np
import logging
# for consistency between python2 and python3
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)
import dreamplace.configure as configure
import Params
import PlaceDB
import Timer
import NonLinearPlace
import pdb
from typing import Tuple
import pickle



def write_nodes(
    nodes_file: str, placedb: PlaceDB.PlaceDB
):
    # is_fixed 表示是否固定 macro
    fwrite = open(nodes_file, "w", encoding="utf8")
    fwrite.write(
        """\
UCLA nodes 1.0
# Created	:	Jan  6 2005
# User   	:	Gi-Joon Nam & Mehmet Yildiz at IBM Austin Research({gnam, mcan}@us.ibm.com)\n
"""
    )
    node_size_y_mean = placedb.node_size_y.mean()
    #macro_id = np.where(placedb.node_size_y > 10*node_size_y_mean)[0]#按大小分macro
    macro_id = placedb.macro_ids
    macro_names = {}
    num_macros = len(macro_id)
    fwrite.write(f"NumNodes : {placedb.num_physical_nodes}\n")
    fwrite.write(f"NumTerminals : {num_macros}\n")
    #for i in range(placedb.num_physical_nodes):
    #    if i not in macro_id:
    #        width = int(placedb.node_size_x[i])
    #        height = int(placedb.node_size_y[i])
    #        #fwrite.write(f"\t{str(placedb.node_names[i])}\t{int(width)}\t{int(height)}\n")
    #        fwrite.write(f"\t{str(placedb.node_names[i])}\t{int(width)}\t{int(height)}\n")
    for i in macro_id:
        width = int(placedb.node_size_x[i])
        height = int(placedb.node_size_y[i])
        fwrite.write(f"\t{placedb.node_names[i]}\t{width}\t{height}\tterminal\n")
        macro_names[placedb.node_names[i]] = i
    fwrite.close()
    return macro_names


def write_pl( placedb: PlaceDB.PlaceDB, pl_file):
        """
        @brief write .pl file
        @param pl_file .pl file
        """
        content = "UCLA pl 1.0\n\n"
        node_x = placedb.node_x
        node_y = placedb.node_y
        for node_id in placedb.macro_ids:
            content += f"{placedb.node_names[node_id]}\t{float(node_x[node_id])}\t{float(node_y[node_id])}\t: N /FIXED\n"
        with open(pl_file, "w") as fwrite:
            fwrite.write(content)

def write_nets(net_file, placedb: PlaceDB.PlaceDB):
    num_nets = len(placedb.net2pin_map)
    num_pins = len(placedb.pin2net_map)
    content = """\
UCLA nets 1.0
# Created	:	Dec 27 2004
# User   	:	Gi-Joon Nam & Mehmet Yildiz at IBM Austin Research({gnam, mcan}@us.ibm.com)
"""
    content += "\n"
    content += f"""\
NumNets : {num_nets}
NumPins : {num_pins}
"""
    for id,net_name in enumerate(placedb.net_names):
        net_pin_num = len(placedb.net2pin_map[id])
        content += f"NetDegree : {net_pin_num} {net_name}\n"
        net_pin = placedb.net2pin_map[id]#pin id
        net_macro = placedb.pin2node_map[net_pin]#macro id
        for i,idmacro in enumerate(net_macro):
            pin = net_pin[i]#pin_id
            if placedb.pin_direct[pin] == 'OUTPUT':
                direct = 'O'
            else:
                direct = 'I'
            
            #content += f"\t{str(placedb.rawdb.nodeName(idmacro))} {direct} : {placedb.pin_offset_x[pin]:.6f} {placedb.pin_offset_y[pin]:.6f}\n"
            content += f"\t{placedb.node_names[idmacro]} {direct} : {placedb.pin_offset_x[pin]:.6f} {placedb.pin_offset_y[pin]:.6f}\n"
    with open(net_file, "w") as fwrite:
        fwrite.write(content)




def writebookshelf(placedb: PlaceDB.PlaceDB, params: Params.Params):
    benchmark = params.design_name()
    benchdir = f'./benchmarks/{benchmark}'
    if not os.path.exists(benchdir):
        os.makedirs(benchdir)
    pl_file = os.path.join(
        benchdir,
        f'{benchmark}.pl'
    )
    nodes_file = os.path.join(
        benchdir,
        f'{benchmark}.nodes'
    )
    nets_file = os.path.join(
        benchdir,
        f'{benchmark}.nets'
    )
    aux_file = os.path.join(
        benchdir,
        f'{benchmark}.aux'
    )
    _ = write_nodes(nodes_file=nodes_file, placedb=placedb)
    print('finish nodes writing')
    with open(f'benchmarks/{params.design_name()}/macro_names.pkl','wb') as f:
        pickle.dump(_,f)
    write_nets(net_file=nets_file,placedb=placedb)
    print('finish net writing')
    write_pl(placedb=placedb,pl_file=pl_file)
    print('finish pl writing')
    with open(aux_file,'w') as fwrite:
        fwrite.write(f'RowBasedPlacement :  {benchmark}.nodes  {benchmark}.nets  {benchmark}.pl')



if __name__ == "__main__":
    logging.root.name = 'DREAMPlace'
    logging.basicConfig(level=logging.INFO,
                        format='[%(levelname)-7s] %(name)s - %(message)s',
                        stream=sys.stdout)
    params = Params.Params()
    params.printWelcome()
    if len(sys.argv) == 1 or '-h' in sys.argv[1:] or '--help' in sys.argv[1:]:
        params.load('test/iccad2015.ot/swerv_wrapper_nt.json')
    elif len(sys.argv) != 2:
        logging.error("One input parameters in json format in required")
        params.printHelp()
        exit()
    else:
        # load parameters
        params.load(sys.argv[1])
    logging.info("parameters = %s" % (params))
    os.environ["OMP_NUM_THREADS"] = "%d" % (params.num_threads)
    tt = time.time()
    placedb = PlaceDB.PlaceDB()
    placedb(params)
    logging.info("reading database takes %.2f seconds" % (time.time() - tt))
    writebookshelf(placedb, params)
    
   