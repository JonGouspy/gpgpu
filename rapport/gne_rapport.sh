#!/usr/bin/env bash
# Usage : ./rapport.sh <video>           rapport de la version actuelle
#         ./rapport.sh compare <video>   rapport de chaque tag v1.1.*
#
#   Avec <video> = chemin absolu
#
# Une fois fini, faire 
# cd all_rapport
#
# Pour voir le rapport faire la commande
#
# nsys stats rapport_<version>.nsys-rep --report cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum
#

profile() {
    cmake -S ../ -B ../build -DCMAKE_BUILD_TYPE=Release
    cmake --build ../build -j
    nsys profile -o "gpu_report/report_$1" \
    ../build/stream --mode=gpu $VIDEO --low=10 --high=50

    # Mesure du FPS (sans nsys, qui ralentit l'exécution)
    start=$(date +%s.%N)
    ../build/stream --mode=gpu $VIDEO --low=10 --high=50
    end=$(date +%s.%N)
    echo "$1, $(echo "$FRAMES / ($end - $start)" | bc -l)" >> gpu_report/fps.csv
}

if [ "$1" = compare ]; then VIDEO="$2"; else VIDEO="$1"; fi
[ -f "$VIDEO" ] || { echo "Usage : ./rapport.sh [compare] <video>"; exit 1; }

FRAMES=$(ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames -of csv=p=0 "$VIDEO")
[ -n "$FRAMES" ] || { echo "ffprobe n'a pas pu compter les frames de $VIDEO"; exit 1; }
echo "$VIDEO : $FRAMES frames"

mkdir -p gpu_report

if [ "$1" = compare ]; then
    git diff --quiet HEAD || { echo "Commitez ou stashez vos modifications avant"; exit 1; }
    start=$(git rev-parse --abbrev-ref HEAD)
    trap 'git checkout -q "$start"' EXIT     # revient sur votre branche à la fin
    for tag in $(git tag -l 'v1.1.*' --sort=version:refname --merged main-rebased); do
        git checkout -q "$tag"
        profile "$tag"
    done
else
    profile "$(echo "here")"
fi
