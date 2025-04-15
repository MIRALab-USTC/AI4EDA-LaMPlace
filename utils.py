import math
from typing import Dict, List, Tuple
import numpy as np
from scipy.spatial import distance
from common import my_inf
from place_db import PlaceDB
import csv
import time
import matplotlib.patches as patches
import matplotlib.pyplot as plt

class Record:
    def __init__(
        self,
        name: str,
        _width: int,
        _height: int,
        grid_x: int,
        grid_y: int,
        bottom_left_x: int,
        bottom_left_y: int,
        grid_size: (tuple),
    ) -> None:
        self.name = name
        self.width: int = _width
        self.height: int = _height
        self.grid_x: int = grid_x
        self.grid_y: int = grid_y
        self.bottom_left_x: int = bottom_left_x
        self.bottom_left_y: int = bottom_left_y
        self.scaled_width: int = math.ceil(
            (_width + bottom_left_x - grid_size[0] * grid_x) / grid_size[0]
        )
        self.scaled_height: int = math.ceil(
            (_height + bottom_left_y - grid_size[1] * grid_y) / grid_size[1]
        )
        self.center_x: float = bottom_left_x + 0.5 * _width
        self.center_y: float = bottom_left_y + 0.5 * _height

    def refresh(self, grid_size: tuple, grid: bool = False):
        if grid:
            self.grid_x = int(round(self.bottom_left_x/grid_size[0]))
            self.grid_y = int(round(self.bottom_left_y/grid_size[1]))
        self.scaled_width: int = math.ceil(
            (self.width + self.bottom_left_x - grid_size[0] * self.grid_x) / grid_size[0]
        )
        self.scaled_height: int = math.ceil(
            (self.height + self.bottom_left_y - grid_size[1] * self.grid_y) / grid_size[1]
        )
        self.center_x: float = self.bottom_left_x + 0.5 * self.width
        self.center_y: float = self.bottom_left_y + 0.5 * self.height


L_Flow = Dict[str, Dict[str, float]]
PlaceRecord = Dict[str, Record]



def random_guiding(
    node_name_list: List[str], placedb: PlaceDB, grid_size: tuple,grid_num: int
) -> PlaceRecord:
    place_record: PlaceRecord = {}
    for node_name in node_name_list:
        width = placedb.node_info[node_name].width
        height = placedb.node_info[node_name].height
        if placedb.node_info[node_name].is_port:
            loc_x = placedb.node_info[node_name].bottom_left_x // grid_size[0]
            loc_y = placedb.node_info[node_name].bottom_left_y // grid_size[1]
            bottom_left_x = placedb.node_info[node_name].bottom_left_x
            bottom_left_y = placedb.node_info[node_name].bottom_left_y
        else:
            loc_x = np.random.randint(0, grid_num)
            loc_y = np.random.randint(0, grid_num)
            bottom_left_x = loc_x * grid_size[0]
            bottom_left_y = loc_y * grid_size[1]
        place_record[node_name] = Record(
            node_name,
            width,
            height,
            loc_x,
            loc_y,
            bottom_left_x,
            bottom_left_y,
            grid_size,
        )
    return place_record



def rank_macros(placedb: PlaceDB,l_flow) -> List[str]:
    rank_df = {}
    for node_name in placedb.macro_name:
        df = np.sum(np.abs(np.array(list(l_flow[node_name].values()))))
        rank_df[node_name] = df*placedb.node_info[node_name].area
    node_name_ls = sorted(placedb.port_name) + sorted(
        placedb.macro_name, key=lambda x: (rank_df[x], x), reverse=True
    )
    return node_name_ls

def get_l_flow(placedb: PlaceDB) -> L_Flow:
    new_df_file = f'graph_data/l_flow/{placedb.benchmark}.npy'
    lf = np.load(new_df_file)
    lf_1 = lf[:,:,0]
    lf_1 = (lf_1+lf_1.T)/2
    lf_2 = lf[:,:,1]
    lf_2 = (lf_2+lf_2.T)/2
    lf_3 = lf[:,:,2]
    lf_3 = (lf_3+lf_3.T)/2
    l_flow = {}
    min_dis = {}
    node_ls = list(placedb.node_info.keys())
    num_macro = len(node_ls)
    for i,mi in enumerate(node_ls):
        l_flow[mi] = {}
        min_dis[mi] = {}
    for i in range(num_macro):
        for j in range(i+1,num_macro):
            mj = node_ls[j]
            mi = node_ls[i]
            l_flow[mi][mj] = l_flow[mj][mi] =  [lf_1[i][j],lf_2[i][j],lf_3[i][j]]
    return l_flow,lf

def cal_lflow(place_record: PlaceRecord, placedb: PlaceDB, l_flow: L_Flow):
    lflow_total = 0
    for node_name1 in placedb.macro_name:
        for node_name2 in l_flow[node_name1]:
            if node_name2 in place_record:
                l_total += (
                    abs(
                        place_record[node_name1].center_x
                        - place_record[node_name2].center_x
                    )
                    + abs(
                        place_record[node_name1].center_y
                        - place_record[node_name2].center_y
                    )
                ) * l_flow[node_name1][node_name2]
    return lflow_total






def df_mul(d, df):
    return d * df








def cal_positionmask(
    node_name1: str,
    placedb: PlaceDB,
    place_record: PlaceRecord,
    grid_num, 
):
    scaled_width = placedb.node_info[node_name1].scaled_width
    scaled_height = placedb.node_info[node_name1].scaled_height

    position_mask = np.zeros((grid_num, grid_num), dtype=bool)
    position_mask[: grid_num - scaled_width, : grid_num - scaled_height] = True

    for node_name2 in place_record.keys():
        bottom_left_x = max(0, place_record[node_name2].grid_x - scaled_width + 1)
        bottom_left_y = max(0, place_record[node_name2].grid_y - scaled_height + 1)
        top_right_x = min(
            grid_num - 1,
            place_record[node_name2].grid_x + place_record[node_name2].scaled_width,
        )
        top_right_y = min(
            grid_num - 1,
            place_record[node_name2].grid_y + place_record[node_name2].scaled_height,
        )

        position_mask[bottom_left_x:top_right_x, bottom_left_y:top_right_y] = False
    return position_mask


def chose_position(
    node_name,
    l_mask: np.ndarray,
    position_mask: np.ndarray,
    place_record: PlaceRecord,
) -> Tuple[int, int]:
    min_ele = np.nanmin(l_mask[position_mask])
    chosen_loc_x, chosen_loc_y = np.where(l_mask == min_ele)
    distance_ls = []
    pos_ls = []
    for grid_xi, grid_yi in zip(chosen_loc_x, chosen_loc_y):
        if position_mask[grid_xi, grid_yi]:
            pos_ls.append((grid_xi, grid_yi))
            distance_ls.append(
                distance.euclidean(
                    (grid_xi, grid_yi),
                    (place_record[node_name].grid_x, place_record[node_name].grid_y),
                )
            )
    idx = np.argmin(distance_ls)
    chosen_loc_x, chosen_loc_y = pos_ls[idx]
    return chosen_loc_x, chosen_loc_y





def cal_lmask(
    node_name1: str,
    place_record: PlaceRecord,
    grid_num,  
    grid_size: tuple,
    l_flow
):

    l_mask = np.zeros((grid_num, grid_num))

    for node_name in place_record.keys():
        lf1, lf2, lf3 = l_flow[node_name1][node_name]
        x =  place_record[node_name].grid_x
        y =  place_record[node_name].grid_y
        grid_y, grid_x = np.meshgrid(np.arange(grid_num), np.arange(grid_num))
        grid_x -= x
        grid_y -= y
        grid_x = abs(grid_x)
        grid_y = abs(grid_y)
        grid_r = np.sqrt((grid_x*grid_size[0])**2 + (grid_y*grid_size[1])**2)/2e4
        grid_r[x][y] = np.inf
        grid_r_1 = 1.0 / grid_r
        grid_r[x][y] = 0
        potential_mask = lf1 * grid_r_1 + lf2 * (grid_r_1**2) + lf3
        l_mask += potential_mask
    return l_mask







def l_mask_placer(
    node_name_ls: List[str],
    placedb: PlaceDB,
    grid_num,
    grid_size,
    place_record: PlaceRecord,
    l_flow,
    lf
):
    shuffle = 0
    new_place_record: PlaceRecord = {}
    N2_time = 0
    count = 0
    for node_name in node_name_ls:
            position_mask = cal_positionmask(
                node_name, placedb, new_place_record, grid_num
            )
            if not np.any(position_mask == 1):
                print("no_legal_place\n\n")
                return {}, my_inf
            l_mask = cal_lmask(
                node_name, new_place_record, grid_num, grid_size, l_flow
            )
            chosen_loc_x, chosen_loc_y = chose_position(
                node_name, l_mask, position_mask, place_record
            )
            bottom_left_x = grid_size[0] * chosen_loc_x
            bottom_left_y = grid_size[1] * chosen_loc_y
            new_place_record[node_name] = Record(
                node_name,
                placedb.node_info[node_name].width,
                placedb.node_info[node_name].height,
                chosen_loc_x,
                chosen_loc_y,
                bottom_left_x,
                bottom_left_y,
                grid_size,
            )
            count += 1
    value = cal_predict_value(new_place_record,lf,grid_size)
    print("N2_time:", N2_time)
    print("value:", value)
    print("shuffle or not: ", shuffle)
    print("\n")
    return new_place_record, value



def cal_predict_value(new_place_record,lf,grid_size):
    new_place_record = {k: new_place_record[k] for k in sorted(new_place_record)}
    pos_x = np.array([rec.grid_x for rec in new_place_record.values()])
    pos_y = np.array([rec.grid_y for rec in new_place_record.values()])
    num = len(pos_x)
    dx = pos_x[:, np.newaxis] - pos_x[np.newaxis, :]
    dy = pos_y[:, np.newaxis] - pos_y[np.newaxis, :]
    dr = np.sqrt((dx*grid_size[0])**2+(dy*grid_size[1])**2)
    dr[range(num),range(num)] = np.inf
    dr_1 = ((1.0/dr)*2e4)
    df1 = lf[:,:,0]
    df2 = lf[:,:,1]
    df3 = lf[:,:,2]
    return (df2*(dr_1**2)+df1*dr_1+df3).mean()


def write_final_placement(best_placed_macro: PlaceRecord, best_hpwl, dir):
    csv_file2 = open(dir, "a+")
    csv_writer2 = csv.writer(csv_file2)
    csv_writer2.writerow([best_hpwl, time.time()])
    for node_id in list(best_placed_macro.keys()):
        csv_writer2.writerow(
            [
                node_id,
                best_placed_macro[node_id].bottom_left_x,
                best_placed_macro[node_id].bottom_left_y,
            ]
        )
    csv_writer2.writerow([])
    csv_file2.close()
    
    
def write_pl_for_gp(place_record: PlaceRecord, placedb: PlaceDB, pl_file: str):
    with open(pl_file, "w", encoding="utf8") as f:
        f.write("UCLA pl 1.0\n\n")
        for node, record in place_record.items():
            f.write(f"{str(node)}\t{record.bottom_left_x}\t{record.bottom_left_y} : N")
            if placedb.node_info[node].is_port:
                f.write(" /FIXED")
            f.write("\n")
            
def draw_macros(placedb: PlaceDB, pl_file, grid_size, pic_path,draw_id = False):
    t1 = time.time()
    place_record = read_placement(placedb, grid_size, pl_file)
    draw_macro_placement(place_record, pic_path, placedb,draw_id=draw_id)
    print(f'draw pictures takes {time.time()-t1}s')
    
def is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
    
def read_placement(
    placedb: PlaceDB, grid_size, file_path
) -> PlaceRecord:  # 将所有macro随机放置
    place_record: PlaceRecord = {}
    f = open(file_path, encoding="utf8")
    # 将 f 移动到最后一个记录的位置
    pos = 0
    line = f.readline()
    while line != "":
        line = line.strip().split(",")
        if is_float(line[0]):
            pos = f.tell()
        line = f.readline()
    f.seek(pos)

    line = f.readline()
    while line != "" and line != "\n":
        node_name, bottom_left_x, bottom_left_y = line.split(",")
        bottom_left_x, bottom_left_y = int(bottom_left_x), int(bottom_left_y)
        chosen_loc_x = bottom_left_x // grid_size[0]
        chosen_loc_y = bottom_left_y // grid_size[1]
        place_record[node_name] = Record(
            node_name,
            placedb.node_info[node_name].width,
            placedb.node_info[node_name].height,
            chosen_loc_x,
            chosen_loc_y,
            bottom_left_x,
            bottom_left_y,
            grid_size,
        )
        line = f.readline()
    return place_record


def draw_macro_placement(
    place_record: PlaceRecord,
    file_path,
    placedb: PlaceDB,
    draw_id: bool = True,
):
    fig = plt.figure(figsize=(1,0.3))
    ax = fig.add_axes([0, 0, 1, 1], frameon=True, aspect='auto')
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.margins(x=0, y=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    max_y = 0
    for node_name in place_record:
        width, height = place_record[node_name].width, place_record[node_name].height
        x, y = (
            place_record[node_name].bottom_left_x,
            place_record[node_name].bottom_left_y,
        )
        if y+height > max_y:
            max_y = y+height
        color = '#FF0000'
        ax.add_patch(
            patches.Rectangle(
                (x / placedb.max_width, y / placedb.max_height),  # (x,y)
                float(width / placedb.max_width),  # width
                float(height / placedb.max_height),
                linewidth=0.1,
                edgecolor="k",
                alpha=0.5,
                facecolor=color,
            )
        )
        cx = (x + width / 2) / placedb.max_width
        cy = (y + height / 2) / placedb.max_height
        if (not placedb.node_info[node_name].is_port) and draw_id:
            ax.annotate(
                placedb.node_info[node_name].name,
                (x / placedb.max_width, y / placedb.max_height),
                (cx, cy),
                color="k",
                weight="bold",
                fontsize=6,
                ha="center",
                va="center",
            )                    
    plt.xlim(0, 1)
    plt.ylim(0, max_y/placedb.max_height)
    plt.savefig(file_path, dpi=2000, bbox_inches="tight")
    plt.close()