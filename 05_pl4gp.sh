
benchmark="$1"
seed="$2"
folder="placement_data/lamplace/$benchmark/pl_$seed"
if [ -d "$folder" ]; then
    for file in "$folder"/*; do
        echo "$(basename "$file")"
        python dreamplace/Placer_rp.py --pl "$file" --config "test/iccad2015.ot/$benchmark.json" --res_dir "placement_data/global_placement_results/$benchmark-$seed.txt"
    done
else
    echo "文件夹 $folder 不存在"
fi