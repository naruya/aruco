if [ $# != 2 ]; then
  echo invalid args!
  exit 2
fi

for i in `seq -f '%03g' 0 $1`
do
  python get_nerf_dataset.py --vid ~/workspace/data/record3d/tonpy-v9/raw/main_$i.mp4 --skip 3
  ~/workspace/nerfstudio/render.sh v9 $i $2
done

