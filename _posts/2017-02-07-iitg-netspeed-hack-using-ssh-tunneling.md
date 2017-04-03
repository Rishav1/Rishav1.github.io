---
layout: post
title:  "How to establish SSH behind a firewall, and use it to bypass port restrictions (Better than OpenVPN)."
date:   2017-02-07
desc: "How to establish SSH behind a firewall, and use it to bypass port restrictions (Better than OpenVPN)."
keywords: "SSH, SSH tunneling, bypass firewall, corkscrew, Psiphon, squid"
categories: [Linux]
tags: [SSH, SSH tunneling, bypass firewall, corkscrew, Psiphon, squid]
icon: icon-html
---

Virtual Private Network(VPN) has become a necessity in today's time, but most of the VPN services are paid once and quite slow in use. If you have a private server outside the firewall, you can use SSH tunneling to redirect all your traffic via the private server(which acts as a simple VPN). If you don't have a private server outside the firewall, you can use a free open-source VPN [Psiphon](https://psiphon.ca/index.html){:target="_blank"} which is based on SSH tunneling instead of common OpenVPN VPNs. Here's a way to setup a proxy server inside the firewall which tunnels all the connection via Psiphon VPN. Keep in mind that this is different than the Psiphon android Application, that uses some fixed limited VPN servers(out of 100s of free servers) under heavy load.
{: .text-justify}

First lets see how to bypass firewall using ssh tunneling. Let's say that the firewall blocks all destination ports except port 80 and 443, which are used for HTTP and HTTPS connections. By default SSH works behind port 22, and so one cannot establish a SSH connection through firewall directly. In order to use port 443 for SSH connection, we need a tool called Corkscrew, which is used to tunnel TCP connections via HTTP proxies. Here's how you download and use it.
{: .text-justify}

```bash
sudo apt-get install corkscrew
sudo apt-get install connect-proxy
```
After installation, update the SSH config file(create if not present already).

```bash
gedit ~/.ssh/config
```
Add the following lines(replace 202.141.80.22 with your proxy server address and 3128 with your proxy server listening port).
```bash
# Inside the firewall, without HTTPS proxy
Host 172.*
  ProxyCommand connect %h %p
Host 10.*
   ProxyCommand connect %h %p

# Outside of the firewall, with HTTPS proxy
Host *
   ProxyCommand corkscrew 202.141.80.22 3128 %h %p ~/.ssh/auth
```

Make sure that the SSH config file created has correct ownership. If the proxy server has no authentication, no need to add ~/.ssh/auth in the last line above. Otherwise, create a file auth and store user-name and password for the HTTPS proxy in there as shown.

```bash
# Set correct ownership
chmod 600 ~/.ssh/config

# Create auth file 
touch ~/.ssh/auth
gedit ~/.ssh/auth
```

Add the line replacing **user** and **pass** with your own username and password for accessing the HTTPS proxy.

```
user:pass
```

This should allow you to SSH to any server outside that listens on port 443. To ssh to a server outside the firewall, run the following.

```bash
ssh your_username@your_private_server -p 443
```

We are able to bypass firewall to do ssh to servers outside, but we still need to tunnel our connections via SSH. If you plan to use your own private server outside firewall for this purpose, you can do this by creating a Psiphon server on your it. If you plan to use already existing free Pshipon servers, all you need to do is setup a Psiphon python client on your local machine inside firewall. The client machine acts as a SOCKS proxy server and all the computers in the network can use the Psiphon VPN via this SOCKS proxy.
{: .text-justify}

It seems complicated, but it is very simple to setup. Run the following commands on a bash terminal.

```bash
#Clone the open sourced psiphon-circumvention-system Mercurial repository
cd ~/
hg clone https://bitbucket.org/psiphon/psiphon-circumvention-system

#Generate the custom ssh executable.
cd psiphon-circumvention-system/Server/3rdParty/openssh-5.9p1/
./configure
make

#Copy the ssh executable to pyclient folder
cd ~/psiphon-circumvention-system
cp Server/3rdParty/openssh-5.9p1/ssh pyclient/

#Python scrpit to download the list of free psiphon servers
cd pyclient/
touch update.py
gedit update.py
```
Add the following lines to update.py file.

```python
import os, json

# Delete 'server_list' if exists
if os.path.exists("server_list"):
    os.remove("server_list")

# Download 'server_list' and convert server_list to psi_client.dat 
url ="https://psiphon3.com/server_list"
os.system('wget ' + url)

dat = {}
dat["propagation_channel_id"] = "FFFFFFFFFFFFFFFF"
dat["sponsor_id"] = "FFFFFFFFFFFFFFFF"
dat["servers"] = json.load(open('server_list'))['data'].split()
json.dump(dat, open('psi_client.dat', 'w'))
```

Now download the dependencies and run the update script.

```bash
#Download dependencies before running the client
sudo apt-get install python-socksipy

# Update the server list
python update.py

# Remove conflicting environment variables
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY

# Finally Start the Psiphon client
python psi_client.py -e
```

After failing to connect to a few servers that refuse port 443, it should be able to connect to some Psiphon server(there are many) on port 443. You should get an output like the following.

```
$ python psi_client.py -e

Your SOCKS proxy is now running at 0.0.0.0:1080

Please visit our sponsor's homepage:
http://www.psiphontoday.com/index_desktop.html?client_region=IN

Press Ctrl-C to terminate.
```

If you get the above output, your free VPN is working. To use this VPN connection, you need to change system proxy settings to use your SOCKS proxy on the localhost instead of the firewall proxy. The following video shows how to set system proxy in Ubuntu.
{: .text-justify}

<div align="center">
<iframe src="//giphy.com/embed/3oKIPa8ww6VYfAIPIc" width="480" height="262" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/3oKIPa8ww6VYfAIPIc">Setting SOCKS system proxy in Ubuntu.</a></p>
</div>

Other systems on the network can use this proxy server too by setting their socks proxy server as the IP of the local machine running the SSH tunnel, and the port as 1080(default port). Keep in mind that for sharing this connection with other users on the network, you need to add "-e" option while running psi_client.py as this option exposes your socks proxy to the network.

A VPN via SSH tunneling has been seen to perform much better than a VPN via OpenVPN, which is employed by most paid VPN services. This [servervault page](http://serverfault.com/questions/653211/ssh-tunneling-is-faster-than-openvpn-could-it-be){:target="_blank"} explains why SSH tunneling is faster than OpenVPN.
{: .text-justify}

From inside my institute IIT Guwahati, I was able to observe massive difference between the normal proxy connection and the SSH tunneled connection. Here's a recording of the same.
{: .text-justify}

<div align="center">
<iframe width="640" height="480" src="https://www.youtube.com/embed/TWBWphI9S8w" frameborder="0"></iframe>
<p>Ookla speedtest comparison between net speed via squid proxy firewall and free SSH tunneled VPN.</p>
</div>
