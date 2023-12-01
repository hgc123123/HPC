# Grafana+Prometheus

## Description

This repository contains a simple template for building
[Pandoc](http://pandoc.org/) documents; Pandoc is a suite of tools to compile
markdown files into readable files (PDF, EPUB, HTML...).

## Grafana

### Installing

```
wget https://dl.grafana.com/enterprise/release/grafana-enterprise-10.2.2-1.x86_64.rpm
sudo rpm -Uvh grafana-enterprise-10.2.2-1.x86_64.rpm

*remove*
rpm -qa grafa | xargs rpm -e
```

### Start


```
systemctl enable grafana-server.service
systemctl start grafana-server.service
```

### Check port

```
netstat -anp | grafa
```

### Anonymout Access

Changing both files:

/etc/grafana/grafana.ini
/usr/share/grafana/conf/defaults.ini

to 

```
[auth.anonymous]
enabled = true
org_role = Admin
org_name = Main Org.

[auth]
disable_login_form = true
disable_login = true

[auth.basic]
enabled = false
```
Refer Link: https://talk.plesk.com/threads/grafana-anonymous-auth-doesnt-work.367584/


- [Grafana](https://grafana.com/grafana/download)

## Prometheus

### Installing

Download the Prometheus package

Go to the official Prometheus downloads page, and copy the URL of the Linux “tar” file.
- [prometheus](https://prometheus.io/download/)

Create Prometheus User and modify the permission
```
useradd --no-create-home --shell /bin/false prometheus
mkdir /etc/prometheus /var/lib/prometheus
chown prometheus:prometheus /etc/prometheus
chown prometheus:prometheus /var/lib/prometheus
tar -xvzf prometheus-2.42.0.linux-amd64.tar.gz
mv prometheus-2.42.0.linux-amd64 prometheus
cp prometheus/prometheus /usr/local/bin/
cp prometheus/promtool /usr/local/bin/
chown prometheus:prometheus /usr/local/bin/prometheus
chown prometheus:prometheus /usr/local/bin/promtool
cp -r prometheus/consoles /etc/prometheus
cp -r prometheus/console_libraries /etc/prometheus
chown -R prometheus:prometheus /etc/prometheus/consoles
chown -R prometheus:prometheus /etc/prometheus/console_libraries
```

Add the following configurations to file */etc/prometheus/prometheus.yml*
```
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: 'prometheus_master'
    scrape_interval: 5s
    static_configs:
      - targets: ['localhost:9090']
```

```
chown prometheus:prometheus /etc/prometheus/prometheus.yml
```

Configure Prometheus Systemd service

Add the following content to the file */etc/systemd/system/prometheus.service*

```
[Unit]
Description=Prometheus
Wants=network-online.target
After=network-online.target

[Service]
User=prometheus
Group=prometheus
Type=simple
ExecStart=/usr/local/bin/prometheus \
--config.file /etc/prometheus/prometheus.yml \
--storage.tsdb.path /var/lib/prometheus/ \
--web.console.templates=/etc/prometheus/consoles \
--web.console.libraries=/etc/prometheus/console_libraries

[Install]
WantedBy=multi-user.target
```

```
systemctl daemon-reload
systemctl enable --now prometheus
```



- [Prometheus](https://github.com/hgc123123/HPC/blob/prometheus/Install.md)

Edit the *metadata.yml* file to set configuration data:


You can find the list of all available keys on [this
page](http://pandoc.org/MANUAL.html#extension-yaml_metadata_block).

### Creating chapters

Creating a new chapter is as simple as creating a new markdown file in the
*src/* folder; you'll end up with something like this:


This is the second subsection.
```

Each title (*#*) will represent a chapter, while each subtitle (*##*) will
represent a chapter's section. You can use as many levels of sections as
markdown supports.

#### Links between chapters

Anchor links can be used to link chapters within the document:

```md
// src/01-introduction.md
# Introduction

For more information, check the [Usage] chapter.

// src/02-installation.md
# Usage

...
```

If you want to rename the reference, use this syntax:

```md
For more information, check [this](#usage) chapter.
```

Anchor names should be downcased, and spaces, colons, semicolons... should be
replaced with hyphens. Instead of `Chapter title: A new era`, you have:
`#chapter-title-a-new-era`.

#### Links between sections

It's the same as anchor links:

```md
# Introduction

## First

For more information, check the [Second] section.

## Second

...
```

Or, with al alternative name:

```md
For more information, check [this](#second) section.
```

### Inserting objects

Text. That's cool. What about images and tables?

#### Insert an image

Use Markdown syntax to insert an image with a caption:

```md
![A cool seagull.](images/seagull.png)
```

Pandoc will automatically convert the image into a figure (image + caption).

If you want to resize the image, you may use this syntax, available in Pandoc
1.16:

```md
![A cool seagull.](images/seagull.png){ width=50% height=50% }
```

Also, to reference an image, use LaTeX labels:

```md
Please, admire the gloriousnes of Figure \ref{seagull_image}.

![A cool seagull.\label{seagull_image}](images/seagull.png)
```

#### Insert a table

Use markdown table, and use the `Table: <Your table description>` syntax to add
a caption:

```md
| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table.
```

If you want to reference a table, use LaTeX labels:

```md
Please, check Table /ref{example_table}.

| Index | Name |
| ----- | ---- |
| 0     | AAA  |
| 1     | BBB  |
| ...   | ...  |

Table: This is an example table.\label{example_table}
```

#### Insert an equation

Wrap a LaTeX math equation between `$` delimiters for inline (tiny) formulas:

```md
This, $\mu = \sum_{i=0}^{N} \frac{x_i}{N}$, the mean equation, ...
```

Pandoc will transform them automatically into images using online services.

If you want to center the equation instead of inlining it, use double `$$`
delimiters:

```md
$$\mu = \sum_{i=0}^{N} \frac{x_i}{N}$$
```

[Here](https://www.codecogs.com/latex/eqneditor.php)'s an online equation
editor.

### Output

This template uses *Makefile* to automatize the building process. Instead of
using the *pandoc cli util*, we're going to use some *make* commands.

#### Export to PDF

Use this command:

```sh
make pdf
```

The generated file will be placed in *build/pdf*.

Please, note that PDF file generation requires some extra dependencies (~ 800
MB):

```sh
sudo apt-get install texlive-latex-base texlive-fonts-recommended texlive-latex-extra 
```

#### Export to EPUB

Use this command:

```sh
make epub
```

The generated file will be placed in *build/epub*.

#### Export to HTML

Use this command:

```sh
make html
```

The generated file(s) will be placed in *build/html*.

## References

- [Pandoc](http://pandoc.org/)
- [Pandoc Manual](http://pandoc.org/MANUAL.html)
- [Wikipedia: Markdown](http://wikipedia.org/wiki/Markdown)
