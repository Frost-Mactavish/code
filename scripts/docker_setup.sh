#!/bin/bash

# setup SSH service
apt update && apt install openssh-server -y && \
apt clean && rm -rf /var/lib/apt/lists/*

# enable root login and password authen
echo "PasswordAuthentication yes
PermitRootLogin yes" >> /etc/ssh/sshd_config

# refresh SSH config
/etc/init.d/ssh restart

# setup ROOT password
echo "root:666666" | chpasswd

# recover PATH variables
echo "export PATH=${PATH}" >> /root/.bashrc

# enable terminal color prompt
sed -i '39s/^#//' /root/.bashrc

source /root/.bashrc
