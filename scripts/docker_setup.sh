#! /bin/bash

# setup SSH service
apt update && apt install openssh-server -y && \
apt clean && rm -rf /var/lib/apt/lists/*

echo "PubkeyAuthentication yes
PermitRootLogin yes" >> /etc/ssh/sshd_config

/etc/init.d/ssh start

# setup ROOT password
echo "root:666666" | chpasswd

# recover PATH variables
echo "export PATH=${PATH}" >> /root/.bashrc

# enable terminal color prompt
sed -i '39s/^#//' /root/.bashrc

source /root/.bashrc
