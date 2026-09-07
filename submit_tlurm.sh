timestamp=$(date +%y%m%d-%H_%M_%S)
cluster="ARCH_GZ_H20_gy"
exp_name="AngelPTM"
trun _tlurm/_${cluster}.yaml --auto-commit tlurm/guhao \
    name="${exp_name}_${cluster}_${timestamp}" \
    start_cmd="source taiji/init_shim.sh && source taiji/setup_env.sh && bash taiji/job.sh" \
    host_gpu_num=8 \
    host_num=1