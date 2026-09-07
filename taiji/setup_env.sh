#!/bin/bash

set_ip_env() {
    # 检查环境变量是否存在
    if [ -n "$NODE_IP_LIST" ]; then
        # 以逗号分割字符串为数组
        IFS=',' read -r -a ips_array <<< "$NODE_IP_LIST"
        
        # 使用代码块将输出重定向追加到 .bashrc
        {
            echo ""
            # 遍历数组及索引
            for i in "${!ips_array[@]}"; do
                # 1. ${ips_array[$i]%%:*}  -> 移除冒号及其后的内容 (split(":")[0])
                # 2. $(echo ...)           -> 利用 echo 去除首尾空格 (strip())
                current_ip=$(echo "${ips_array[$i]%%:*}")

                # 立即导出到当前 shell，并持久化到后续交互 shell
                export "NODE_IP_$i=$current_ip"
                echo "export NODE_IP_$i=\"$current_ip\""
            done
            echo ""
        } >> /root/.bashrc
    fi
}

add_proxy(){
    echo '# add proxy' >> ~/.bashrc
    echo 'export http_proxy="http://star-proxy.oa.com:3128"' >> ~/.bashrc
    echo 'export https_proxy="http://star-proxy.oa.com:3128"' >> ~/.bashrc
    echo 'export ftp_proxy="http://star-proxy.oa.com:3128"' >> ~/.bashrc
    echo 'export no_proxy=".woa.com,mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,tlinux-mirrorlist.tencent-cloud.com,localhost,127.0.0.1,mirrors-tlinux.tencentyun.com,.oa.com,.local,.3gqq.com,.7700.org,.ad.com,.ada_sixjoy.com,.addev.com,.app.local,.apps.local,.aurora.com,.autotest123.com,.bocaiwawa.com,.boss.com,.cdc.com,.cdn.com,.cds.com,.cf.com,.cjgc.local,.cm.com,.code.com,.datamine.com,.dvas.com,.dyndns.tv,.ecc.com,.expochart.cn,.expovideo.cn,.fms.com,.great.com,.hadoop.sec,.heme.com,.home.com,.hotbar.com,.ibg.com,.ied.com,.ieg.local,.ierd.com,.imd.com,.imoss.com,.isd.com,.isoso.com,.itil.com,.kao5.com,.kf.com,.kitty.com,.lpptp.com,.m.com,.matrix.cloud,.matrix.net,.mickey.com,.mig.local,.mqq.com,.oiweb.com,.okbuy.isddev.com,.oss.com,.otaworld.com,.paipaioa.com,.qqbrowser.local,.qqinternal.com,.qqwork.com,.rtpre.com,.sc.oa.com,.sec.com,.server.com,.service.com,.sjkxinternal.com,.sllwrnm5.cn,.sng.local,.soc.com,.t.km,.tcna.com,.teg.local,.tencentvoip.com,.tenpayoa.com,.test.air.tenpay.com,.tr.com,.tr_autotest123.com,.vpn.com,.wb.local,.webdev.com,.webdev2.com,.wizard.com,.wqq.com,.wsd.com,.sng.com,.music.lan,.mnet2.com,.tencentb2.com,.tmeoa.com,.pcg.com,www.wip3.adobe.com,www-mm.wip3.adobe.com,mirrors.tencent.com,csighub.tencentyun.com"' >> ~/.bashrc
    echo '' >> ~/.bashrc
}

set_prompt(){
    echo "# set prompt" >> ~/.bashrc
    echo 'eval "$(starship init bash)"' >> ~/.bashrc
    # echo 'export PS1="\[\e]0;H20-${INDEX}@\h: \w\a\]\[\033[01;32m\]H20-${INDEX}@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]$ "' >> ~/.bashrc
}

set_permission(){
    echo "# resolve file permission issue between root and normal user" >> ~/.bashrc
    echo "umask 002" >> ~/.bashrc
}

set_pixi_cache(){
    # Persist pixi's package/wheel cache on ceph. The default (~/.cache) lives on
    # the container overlay, which is wiped on restart -> the flash-attn source build
    # (no torch-2.9 wheel exists) would recompile every time. Pinning the cache to
    # ceph means it compiles ONCE and every later pod reuses the built wheel.
    # PIXI_DISABLE_NETFS_REDIRECT=1 is REQUIRED: ceph is a FUSE netfs, and pixi would
    # otherwise redirect the cache to a local (ephemeral) disk, defeating persistence.
    # See scripts/taiji/init_shim.sh. $CEPH_ROOT is expanded here so the actual
    # path is written into .bashrc.
    echo "# persist pixi cache on ceph" >> ~/.bashrc
    echo "export PIXI_CACHE_DIR=\"$CEPH_ROOT/envs/pixi-cache\"" >> ~/.bashrc
    echo 'export PIXI_DISABLE_NETFS_REDIRECT=1' >> ~/.bashrc
}
# wget http://jizhi.oa.com/taiji_client_golang/taiji_client -O /usr/bin/taiji_client && chmod +x /usr/bin/taiji_client && taiji_client update

# 挂载 private ceph 存储
# taiji_client mount -tk [token] -l cq
#taiji_client mount -bf TaiJi_HYAide_AILab_MM_SH_A100H -tk [token]

# wandb login [token]

fix_bashenv(){
    echo "# fix the stupid bash env set by taiji, it breaks the pixi env setup and make the "pixi run" use system env" >> ~/.bashrc
    echo "unset BASH_ENV" >> ~/.bashrc
}


# Parse script arguments
USE_PROXY=true  # default: enable proxy
while [[ $# -gt 0 ]]; do
    case "$1" in
        --proxy)
            USE_PROXY=true
            shift
            ;;
        --no-proxy)
            USE_PROXY=false
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# python3 set_env.py
set_ip_env
if [ "$USE_PROXY" = true ]; then
    add_proxy
fi
set_prompt
set_permission
if [ -n "$CEPH_ROOT" ]; then
    set_pixi_cache
else
    echo "WARNING: CEPH_ROOT is not set, skipping pixi cache setup." >&2
fi
fix_bashenv

