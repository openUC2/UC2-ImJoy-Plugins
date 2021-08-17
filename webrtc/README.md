
https://www.linux-projects.org/uv4l/installation/
https://www.linux-projects.org/uv4l/tutorials/custom-webapp-with-face-detection/
https://telebit.cloud/
sudo service uv4l_raspicam restart
~/telebit http 3000


https://github.com/mpromonet/webrtc-streamer
~/telebit http 8000
ssh uc2@192.168.43.90
wget https://github.com/mpromonet/webrtc-streamer/releases/download/v0.2.6/webrtc-streamer-v0.2.6-Linux-armv7l-Release.tar.gz
tar -xvzf webrtc-streamer-v0.2.6-Linux-armv7l-Release.tar.gz
cd ~/Downloads/webrtc-streamer-v0.2.6-Linux-armv7l-Release/
./webrtc-streamer 
./webrtc-streamer -H 0.0.0.0:8000 -sstun4.l.google.com:19302


Own TURN server (cotton)
youseetoo.ddns.net:5439


./webrtc-streamer -S -s$(curl ifconfig.me -s):3478
http://192.168.43.90:8000/

Bridge SSH from Pi to MAC using VNC
 ssh -L 5901:localhost:5901 uc2@youseetoo.ddns.net


http://www.raspberrypi-tutorials.de/software/dyndns-mit-no-ip-fuer-den-raspberry-pi-einrichten.html
wget http://www.no-ip.com/client/linux/noip-duc-linux.tar.gz
sudo tar xf noip-duc-linux.tar.gz
cd noip-2.1.9-1/	
sudo make install
sudo noip2
crontab -e -> @reboot cd /home/pi/noip && sudo noip2

For larger screens 
http://youseetoo.ddns.net:8000/webrtcstreamer.html?video=mmal%20service%2016.1&options=rtptransport%3Dtcp%26timeout%3D60%26width%3D1280%26height%3D720



# JANUS GATEWAY on Raspi

### Install GStreamer
```
sudo nano /etc/apt/sources.list
```

and add the following to the end of the file:

```
deb http://vontaene.de/raspbian-updates/ . main 
```

Press CTRL+X to save and exit
Now run an update (which will make use of the line just added):




First du some package updates

```
sudo apt-get update  --fix-missing

sudo apt-get install libmicrohttpd-dev libjansson-dev libnice-dev libssl-dev libsrtp-dev libsofia-sip-ua-dev libglib2.0-dev libopus-dev libogg-dev libini-config-dev libcollection-dev pkg-config gengetopt libtool automake dh-autoreconf  libconfig-dev libsrtp2-dev gstreamer1.0  gstreamer1.0-tools libcurl4-openssl-dev -y
```

### update libnice

```
sudo apt-get purge -y libnice-dev
```

Install build tools:
```
sudo apt-get install gcc autoconf automake libtool pkg-config gtk-doc-tools gettext python3 gengetopt
```

Build libnice from sources:

```
git clone https://gitlab.freedesktop.org/libnice/libnice /tmp/libnice 
cd /tmp/libnice
git checkout 0.1.16
sed -i -e 's/NICE_ADD_FLAG(\[-Wcast-align\])/# NICE_ADD_FLAG(\[-Wcast-align\])/g' ./configure.ac
sed -i -e 's/NICE_ADD_FLAG(\[-Wno-cast-function-type\])/# NICE_ADD_FLAG(\[-Wno-cast-function-type\])/g' ./configure.ac
./autogen.sh --prefix=/usr --disable-gtk-doc
make
sudo make install
```


Then build janus server on the pi, see also [here](https://www.raspberrypi.org/forums/viewtopic.php?t=99283) and [here](https://dustinoprea.com/2014/05/21/lightweight-live-video-in-a-webpage-with-gstreamer-and-webrtc/)

```
cd ~/Downloads
git clone https://github.com/meetecho/janus-gateway.git
cd janus-gateway
sh autogen.sh

./configure --prefix=/opt/janus --disable-websockets --disable-data-channels --disable-rabbitmq --disable-docs --disable-aes-gcm
# Alternative? RS Electronics ./configure --disable-websockets --disable-data-channels --disable-rabbitmq --disable-docs --prefix=/opt/janus --disable-aes-gcm
```

Add configuration:


In

```
sudo nano /opt/janus/etc/janus/janus.jcfg
```

replace ```general: {```:

```
configs_folder = "/opt/janus/etc/janus"
plugins_folder = "/opt/janus/lib/janus/plugins"
```


In

```
sudo nano /opt/janus/etc/janus/janus.plugin.streaming.jcfg
```

add ```gst-raspicam```

```
gst-raspicam: {
        type = "rtp"
        id = 1
        description = "RPWC H264 test streaming"
        audio = false
        video = true
        videoport = 8004
        videopt = 96
        videortpmap = "H264/90000"
        videofmtp = profile-level-id=42e028\;packetization-mode=1
}
```

```
gst-rpwc: {
        type = "rtp"
        id = 1
        description = "RPWC H264 test streaming"
        audio = false
        video = true
        videoport = 8004
        videopt = 96
        videortpmap = "H264/90000"
        videofmtp = profile-level-id=42e028\;packetization-mode=1
}
```



Make the project and install it 

```
make clean
make
sudo make install
sudo make configs
```



Copy the file to the ngnix server:

```
sudo cp -r /opt/janus/share/janus/demos/ /var/www/html/ #/usr/share/nginx/www/
```


Execute this line to start the videoserver:

```
raspivid --verbose --nopreview -hf -vf --width 640 --height 480 --framerate 15 --bitrate 1000000 --profile baseline --timeout 0 -o - | gst-launch-1.0 -v fdsrc ! h264parse ! rtph264pay config-interval=1 pt=96 ! udpsink host=127.0.0.1 port=8004
```


```
sudo /etc/init.d/nginx reload
```

### Run Janus

```
/opt/janus/bin/janus -F /opt/janus/etc/janus/
```
or

```
./janus -F /opt/janus/etc/janus/
```