for env in halfcheetah hopper walker2d
do 
    pkl=data/${env}/${env}*/params.pkl
    target_dir=../D4RL/snapshots/${env}
    mkdir ${target_dir}
    cp ${pkl} ${target_dir}/params.pkl 
done