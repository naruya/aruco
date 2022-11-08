if [ $# != 2 ]; then
  echo invalid args!
  exit 1
fi

ns-train nerfacto --vis viewer --viewer.websocket-port 6006 --data /root/workspace/data/record3d/tonpy-$1/charuco/ --output-dir /root/workspace/logs/nerfstudio --experiment-name tonpy-$1 --timestamp $2 --trainer.max-num-iterations 10000
