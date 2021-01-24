---
layout: post
title:  "IITG proxy-checker and autoconfiguration package for ubuntu."
date:   2018-01-25
desc: "IITG proxy-checker and autoconfiguration package for ubuntu."
keywords: "Http/s proxy"
categories: [Linux]
tags: [Http/s proxy]
icon: icon-html
---

IIT Guwahati, like many indian institutes have time regulated internet access in hostels, while 24/7 access at academic departments and computer centers. Some students with networking knowledege or googling abilities eventually figure out the joy of 24/7 net access via proxy servers. For others who don't know or don't want to set up a proxy server, but use the facility nonetheless, here's a utility for you.
{: .text-justify}


A [proxy-search](https://github.com/Rishav1/iitg-acproxy) program that discovers open proxy servers over popular ports in your local subnet. For ubuntu users, it is really easy to install. Just follow the following steps.
{: .text-justify}

```bash
sudo apt-add-repository ppa:rishav1/iitg-rishav1
sudo apt-get update
sudo apt-get install iitg-acproxy
```

Those who don't have linux, but have bash in windows, you can run the script by cloning from the github repository.
{: .text-justify}

For IIT Guwahati students, I have already preconfigured the default in the script for our campus subnet. All you have to do is run the following command and it will create a file "proxylist.txt" in the home directory.
{: .text-justify}

```bash
proxy-search
```

I have also added a script to autoconfigure the internet settings throughout the system. The script "iitg-acproxy" uses the discovered open proxies to set up a [redsocks](https://github.com/darkk/redsocks) transparent proxy redirection. In short, no system proxy settings or environment proxy settings would be needed if you use this. After proxy-script discovers all open proxies, just run the following and your internet should work without any system proxy settings.
{: .text-justify}

```bash
iitg-acproxy start
```

Cheers!
