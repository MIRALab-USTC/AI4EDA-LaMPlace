import re

def parse_line(line):
    
    pattern = r'(\S+)\s+(\d+)\s+(\d+)\s*:\s*(\w)'
    match = re.match(pattern, line)
    if match:
        name = match.group(1)
        x_coord = match.group(2)
        y_coord = match.group(3)
        direction = match.group(4)
        return [name, x_coord, y_coord, ":", direction]
    else:
        return None

def read_pl(pl_file):
    pl_dict={}
    with open(pl_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = parse_line(line)
            #print(line)
            if len(line) >= 4:
                # x, y, orientation
                #print(line[0])
                pl_dict[line[0]] = [float(line[1]), float(line[2]), line[4]]

    return pl_dict

def unscale_xy(node_x,node_y, shift_factor, scale_factor):
    unscale_factor = 1.0 / scale_factor
    
    if shift_factor[0] == 0 and shift_factor[1] == 0 and unscale_factor == 1.0:
            x = node_x
            y = node_y
    else:
        x = node_x * unscale_factor + shift_factor[0]
        y = node_y * unscale_factor + shift_factor[1]

    return x,y

def write2def(pl_dict,def_file,target_filename):
    def_lines = []
    
    with open(def_file, "r") as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if line.startswith("COMPONENTS"):
                #print(line)
                start = True
            if "END COMPONENTS" in line:
                #print(line)
                start = False
            if start and line.lstrip().startswith("-"):
                #print(line)
                split = line.lstrip().split(" ")
                #print(split)
                instance = (
                    split[1].replace("\[", "[").replace("\]", "]").replace("\/", "/")
                )
                #print(instance)
                if instance in pl_dict and len(split)> 5 :

                    x, y, o = pl_dict[instance]
                    x=round(x / 10) * 10
                    y=round((y-70)/280)*280+70

                    old_pos = line[line.find("(") : line.find(")") + 1]
                    line = line.replace(old_pos, "( %d %d )" % (x, y))
                    
                    line = re.sub(
                        r"(.*\) )(N|S|W|E|FN|FS|FW|FE)($| )", r"\1%s\3" % o, line
                    )
            def_lines.append(line)

    with open(target_filename, "w") as f:
        f.writelines(def_lines)
       

'''
If the node coordinates in the pl file have been scaled by DreamPlace,
you can specify shift_factor and scale_factor in the function to unscale them.
These are generally the params.shift_factor and params.scale_factor in DreamPlace.
'''
def pl2def(pl_input,def_input,final_def,shift_factor=[0,0],scale_factor=1):

    pl_dict=read_pl(pl_input)
    
    for key in pl_dict:
        pl_dict[key][0],pl_dict[key][1]=unscale_xy(pl_dict[key][0],pl_dict[key][1],shift_factor, scale_factor)

    write2def(pl_dict,def_input,final_def)





    
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('--pl', default='placement_data/lamplace/swerv_wrapper/pl_2024/swerv_wrapper_-14938827.898.gp.pl')
    parser.add_argument('--inputdef', default='benchmarks/swerv_wrapper/swerv_wrapper.def')
    parser.add_argument('--outputdef', default='swerv_wrapper.def')
    args = parser.parse_args()
    shift_factor={
        "ariane133": [20140.0, 25200.0],
        "ariane136": [20140.0, 25200.0],
        "bp": [20140.0, 22400.0],
        "bp_be": [20140.0, 22400.0],
        "bp_fe": [20140.0, 22400.0],
        "bp_multi": [20140.0, 19600.0],
        "swerv_wrapper": [20140.0, 22400.0]
    }
    scale_factor={
        "ariane133": 0.002631578947368421,
        "ariane136": 0.002631578947368421,
        "bp": 0.002631578947368421,
        "bp_be": 0.002631578947368421,
        "bp_fe": 0.002631578947368421,
        "bp_multi": 0.002631578947368421,
        "swerv_wrapper": 0.002631578947368421
    }
    dataset = args.outputdef[:-4]
    pl2def(args.pl, args.inputdef, args.outputdef,shift_factor[dataset],scale_factor[dataset])




