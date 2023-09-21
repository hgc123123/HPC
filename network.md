## Partition QOS 
### Checkout port is opened

```
[root@login ~]# netstat -an | grep 9668 | grep  -i listen
tcp        0      0 10.xx.xx.xx:9668     0.0.0.0:*               LISTEN

or

netstat -ntlp | grep 1234
```

### How to open a port
```
[root@login ~]# nc -l 9668
Ncat: bind to 0.0.0.0:9668: Address already in use. QUITTING.
```

Use sudo iptables -I INPUT -p tcp -m tcp --dport 22 -j ACCEPT to open a port. 
In this example, we're opening incoming connections to port 22, but you can replace 22 with the port you want to open.
If you're opening an outbound port, replace INPUT with OUTPUT.
If opening a UDP port, replace tcp with udp.
To only open the port to a particular IP address or subnet, use sudo iptables -I INPUT -s xxx.xxx.xxx.xxx -p tcp -m tcp --dport 22 -j ACCEPT

### How to open o linux port
```
https://www.cnblogs.com/linuxpro/p/17197147.html
```

- [QOS](https://slurm.schedmd.com/qos.html)
