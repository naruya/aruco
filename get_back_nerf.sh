if [ $# != 2 ]; then
  echo invalid args!
  exit 2
fi

python get_nerf_dataset.py --vid /root/workspace/data/record3d/tonpy-$1/raw/back.mp4
mv charuco /root/workspace/data/record3d/tonpy-$1
./nerfstudio_train.sh $1

