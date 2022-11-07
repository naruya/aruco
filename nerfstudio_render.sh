if [ $# != 3 ]; then
  echo invalid args!
  exit 1
fi

ns-render --load-config /root/workspace/logs/nerfstudio/tonpy-$1/nerfacto/$3/config.yml --traj filename --camera-path-filename temp/transforms_$2.json --output-path temp/main_pred_$2.mp4 --downscale-factor 2
