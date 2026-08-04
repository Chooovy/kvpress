JOBID=$(sbatch slurm_rebuttal.sh | awk '{print $4}')
echo "Job ID: ${JOBID}"
sleep 3

tail -f logs/rebuttal_slurm-${JOBID}.out
