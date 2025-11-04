import argparse
import os
import random
import time
import numpy as np
from common import grid_setting, my_inf                                                                                                                                                
from place_db import PlaceDB
from utils import (
    Record,
    random_guiding,
    write_final_placement,
    write_pl_for_gp,
    draw_macros,
    get_l_flow,
    l_mask_placer,
    rank_macros,
    draw_macro_placement,
)

def set_seed(seed: int):
    np.random.seed(seed)
    random.seed(seed)


def swap(node1: Record, node2: Record, grid_size: int):
    node1.bottom_left_x, node2.bottom_left_x = node2.bottom_left_x, node1.bottom_left_x
    node1.bottom_left_y, node2.bottom_left_y = node2.bottom_left_y, node1.bottom_left_y

    node1.grid_x, node2.grid_x = node2.grid_x, node1.grid_x
    node1.grid_y, node2.grid_y = node2.grid_y, node1.grid_y

    node1.refresh(grid_size)
    node2.refresh(grid_size)



def place(
    init_round,
    stop_round,
    placedb: PlaceDB,
    grid_num,
    grid_size,
    placement_file,
    result_dir,
    fig_dir,
):
    
    place_records = []
    l_flow,lf = get_l_flow(placedb)
    m2m_flow = None
    node_id_ls = rank_macros(placedb,l_flow)
        
    best_value = my_inf
    for cnt in range(init_round):
        print(f"init {cnt}")
        place_record = random_guiding(node_id_ls, placedb, grid_size,grid_num)
        placed_macros, value = l_mask_placer(
            node_id_ls, placedb, grid_num, grid_size, place_record,l_flow,lf
        )
        if not value in [r[1] for r in place_records]:
            placement_file = os.path.join(result_dir, f"placement_{value:.3f}.csv")
            write_final_placement(placed_macros, value, placement_file)
            place_records.append((placed_macros,value))
        if value < best_value:
            best_value = value
            best_placed_macro = placed_macros
    print("begin ea")
    candidates = sorted(placedb.macro_name)
    if best_value != my_inf:
        place_record = best_placed_macro
        for cnt in range(stop_round):
            print(cnt)
            node_a = random.sample(rank_macros(placedb,l_flow)[:5],1)[0]
            node_b = random.sample(candidates, 1)[0]
            swap(place_record[node_a], place_record[node_b], grid_size)
            placed_macros, value = l_mask_placer(
            node_id_ls, placedb, grid_num, grid_size, place_record,l_flow,lf
        )
            # save frame for this EA iteration (attempt)
            try:
                frame_path = os.path.join(fig_dir, f"ea_{cnt:04d}_{value:.3f}.png")
                draw_macro_placement(placed_macros, frame_path, placedb, draw_id=False)
            except Exception as e:
                print(f"warn: failed to draw ea frame {cnt}: {e}")
            if value >= best_value:
                swap(place_record[node_a], place_record[node_b], grid_size)
            else:
                best_value= value
                best_placed_macro = place_record = placed_macros
            if not value in [r[1] for r in place_records]:
                placement_file = os.path.join(result_dir, f"placement_{value:.3f}.csv")
                write_final_placement(placed_macros, value, placement_file)
                place_records.append((placed_macros,value))
    return place_records



def main():
    parser = argparse.ArgumentParser(description="argparse testing")
    parser.add_argument("--benchmark", default= 'superblue1')
    parser.add_argument("--seed", default= 2024)
    parser.add_argument("--init_round", default=50)
    parser.add_argument("--stop_round", default=20)
    args = parser.parse_args()
    benchmark = args.benchmark
    seed1 = args.seed
    stop_round = int(args.stop_round)
    init_round = int(args.init_round)
    set_seed(int(seed1))
    grid_num = grid_setting[benchmark]["grid_num"]
    grid_size = grid_setting[benchmark]["grid_size"]
    placedb = PlaceDB(benchmark, grid_size)
    result_dir = os.path.join("placement_data/lamplace", f"{benchmark}")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    if not os.path.exists(os.path.join(result_dir, f"pl_{seed1}")):
        os.makedirs(os.path.join(result_dir, f"pl_{seed1}"))
    if not os.path.exists(os.path.join(result_dir, f"fig_{seed1}")):
        os.makedirs(os.path.join(result_dir, f"fig_{seed1}"))
    placement_file = os.path.join(result_dir, f"placement_seed_{seed1}.csv")
    fig_dir = os.path.join(result_dir, f"fig_{seed1}")
    start = time.time()
    place_records = place(
        init_round, stop_round, placedb, grid_num, grid_size, placement_file,result_dir,fig_dir
    )
    end = time.time()
    print(f"time: {end-start}s")
    place_records = sorted(place_records, key=lambda x: x[1])
    #record best 5 placements
    for i in range(5):
        place_record,value = place_records[i]
        pic_file = os.path.join(result_dir, f"fig_{seed1}/{benchmark}_{value:.3f}.png")
        placement_file = os.path.join(result_dir, f"placement_{value:.3f}.csv")
        draw_macros(placedb, placement_file, grid_size, pic_file)
        pl_file = os.path.join(result_dir, f"pl_{seed1}/{benchmark}_{value:.3f}.gp.pl")
        write_pl_for_gp(place_record, placedb, pl_file)


if __name__ == "__main__":
    main()
