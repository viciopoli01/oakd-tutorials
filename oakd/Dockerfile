FROM duckietown/challenge-aido_lf-baseline-duckietown:daffy-amd64

RUN pip3 install -U "pip>=20.2"
COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN  pip3 install --use-feature=2020-resolver -r .requirements.txt

RUN pip list


##########################
#
#       DepthAI dep
#
##########################
RUN apt-get update && apt-get install -y \
    bc \
    bzip2 \
    can-utils \
    freeglut3-dev \
    gstreamer1.0-alsa \
    gstreamer1.0-libav \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-tools \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    curl \
    udev \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev \
    usbutils \
    wget \
    llvm

RUN apt-get update && apt-get install -y wget build-essential cmake pkg-config libjpeg-dev libtiff5-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk2.0-dev libgtk-3-dev libatlas-base-dev gfortran
RUN wget http://docs.luxonis.com/_static/install_dependencies.sh && chmod +x install_dependencies.sh && ./install_dependencies.sh
RUN wget https://github.com/libusb/libusb/releases/download/v1.0.24/libusb-1.0.24.tar.bz2
RUN tar xf libusb-1.0.24.tar.bz2
RUN cd libusb-1.0.24 && \
    ./configure --disable-udev && \
    make -j && make install

RUN echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | tee /etc/udev/rules.d/80-movidius.rules
RUN pip3 install depthai==2.5.0


RUN mkdir /code/solution

COPY solution /code/solution
COPY launchers /code

CMD ["bash", "/code/submit.sh"]