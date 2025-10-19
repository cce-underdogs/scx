# Note

## build the network

```bash
sudo podman network create "scx_net"
```

```bash
sudo podman network rm scx_net
```



## build the image

```bash
sudo podman build -t scx_rustland .
```

## run Server

```bash
## run server
sudo podman run \
    --name my-server \
    --privileged \
    --pid host \
    --network scx_net \
    -it \
    --rm \
    localhost/scx_rustland
```

```bash
## open server
iperf3 -s
```

## run Client

```bash
## run client
sudo podman run \
    --name my-client \
    --privileged \
    --pid host \
    --network scx_net \
    --rm \
    -it \
    localhost/scx_rustland
```

```bash
## Add network delay
tc qdisc add dev eth0 root netem loss 20%

## send traffic
iperf3 -c my-server -u -b 1000G

## clean up
tc qdisc del dev eth0 root
```
