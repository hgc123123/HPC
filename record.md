## Issues
### Failed to get D-Bus connection: Operation not permitted

ref: https://blog.csdn.net/shi_hong_fei_hei/article/details/115337684

```
docker run -d --name centos7 --privileged=true cento:7 /usr/sbin/init
docker exec -it centos7 /bin/bash
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
