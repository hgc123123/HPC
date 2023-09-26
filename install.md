## How to install Docker
### Adding the external repository

```
$ sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo
```
We can verify that the repository has been enabled, by looking at the output of the following command:


```
$ sudo dnf repolist -v
```

### Installing docker-ce
```
dnf list docker-ce --showduplicates | sort -r
dnf install docker-ce-3:18.09.1-3.el7
```

### Start and enable the docker daemon
```
systemctl enable --now docker
systemctl is-active docker
```

- [link](https://linuxconfig.org/how-to-install-docker-in-rhel-8)
