pip install nvitop
pip install swanlab
wget -O /etc/yum.repos.d/ceph_el7_1.repo http://gaia.repo.oa.com/ceph_el7.repo
yum install -y ceph-fuse
wget http://jizhi.oa.com/taiji_client_golang/taiji_client -O /usr/bin/taiji_client && chmod +x /usr/bin/taiji_client

taiji_client mount -tk FTpEVOZQpuEH8YmOeqfnpA -bf TaiJi_HYAide_AILab_Cog_SH_A100H -l tj
sleep 65536d