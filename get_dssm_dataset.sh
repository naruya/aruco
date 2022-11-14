if [ $# != 3 ]; then
  echo invalid args!
  exit 1
fi

# TS=$(date "+%Y%m%d-%H%M%S")
echo timestamp: $3  # $TS

# ./nerfstudio_train.sh $1 $TS


for i in `seq -f '%03g' 0 $2`
do
  python get_nerf_dataset.py --vid /root/workspace/data/record3d/tonpy-$1/raw/main_$i.mp4 --skip 3
  ./nerfstudio_render.sh $1 $i $3
done

