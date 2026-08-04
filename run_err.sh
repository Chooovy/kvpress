JOBID=$(sbatch slurm.sh | awk '{print $4}')
echo "Job ID: ${JOBID}"
sleep 3

tail -f logs/slurm-${JOBID}.err
