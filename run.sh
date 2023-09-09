for task in halfcheetah
do
python examples/sac.py --task ${task} &
done
wait