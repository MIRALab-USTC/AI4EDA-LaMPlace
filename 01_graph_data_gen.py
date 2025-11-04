import torch
import os
import numpy as np
import pickle


train_ids=[1,3,4,5,7,10]

test_ids = [16,18]
scale = 2e4
def read_pl_file(
    pl_file: str,macro_id = None
):
    pos = []
    if not macro_id:
        with open(pl_file, encoding="utf8") as f:
            for line in f:
                line = line.strip().split()
                if len(line) <=4:
                    continue
                bottom_left_x, bottom_left_y = int(line[1])/scale, int(line[2])/scale
                #assert(int(line[1]) != 29250 or int(line[2]) != -1680)
                pos.append([bottom_left_x,bottom_left_y])
        return pos
    else:
        with open(pl_file, encoding="utf8") as f:
            for line in f:
                    num += 1
                    if num > 2:
                        line = line.strip().split()
                        node_name = line[0] 
                        if node_name in macro_id:
                            bottom_left_x, bottom_left_y = float(line[1])/scale, float(line[2])/scale
                            #assert(int(line[1]) != 29250 or int(line[2]) != -1680)
                            pos.append([bottom_left_x,bottom_left_y])
        return pos



def getListOfFiles(dirName):
    listOfFile = os.listdir(dirName)
    listOfFile.sort()
    allFiles = list()
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


def gen_dataset(train: True):
    #load gp
    HPWL = []
    CON = []
    TNS = []
    POS = []
    WNS = []
    BENCH = []
    if train:
        bench_ids = train_ids
    else:
        bench_ids = test_ids
    for bench_id in bench_ids:
        results = np.loadtxt(f'./graph_data/dataset/gp_result/superblue{bench_id}.txt',usecols=range(4))
        hpwl = results[:,0]
        n = min(len(hpwl),150)
        hpwl = hpwl[:n]
        con = results[:,1][:n]
        tns = results[:,2][:n]
        wns = results[:,3][:n]
        pl_dir = f'./graph_data/dataset/placement/superblue{bench_id}/pl';
        #mp_file = getListOfFiles(pl_dir)
        #assert n == len(mp_file)
        pos = []
        for i in range(n):
            mp_file = f"./graph_data/dataset/placement/superblue{bench_id}/result{i}.pl"
            print(f"read {mp_file}")
            p = read_pl_file(mp_file)
            assert(len(p) != 0)
            pos.append(torch.tensor(p))
        bench = np.repeat(np.array([bench_id]),n)
        HPWL.append(hpwl)
        CON.append(con)
        POS.extend(pos)
        TNS.append(tns)
        WNS.append(wns)
        BENCH.append(bench)
    hpwl = torch.tensor(np.concatenate(HPWL))
    pos = POS
    tns = torch.tensor(np.concatenate(TNS))
    wns = torch.tensor(np.concatenate(WNS))
    con = torch.tensor(np.concatenate(CON))
    bench = torch.tensor(np.concatenate(BENCH))
    return {'con': con, 'pos': pos,'hpwl': hpwl,'tns': tns,'wns': wns,'bench': bench}

test_set = gen_dataset(train=True)
print("train dataset generated")
with open(f"graph_data/dataset/dataset.pkl","wb") as f:
    pickle.dump(test_set,f)