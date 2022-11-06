if [ $# != 3 ]; then
  echo invalid args!
  exit 1
fi

ns-render --load-config /root/workspace/logs/nerfstudio/tonpy-$1/nerfacto/$3/config.yml --traj filename --camera-path-filename transforms_$2.json --output-path main_pred_$2.mp4 --downscale-factor 2
