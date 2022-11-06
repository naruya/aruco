if [ $# != 2 ]; then
  echo invalid args!
  exit 2
fi

python get_nerf_dataset.py --vid ~/workspace/data/record3d/tonpy-$1/raw/back.mp4
mv charuco ~/workspace/data/record3d/tonpy-$1
~/workspace/nerfstudio/train.sh $1

