docker image rm cpp-scons-image

docker build -t cpp-scons-image .

xhost +local:docker

docker run -it --rm \
    -v "$(pwd)":/app \
    -v "$HOME/.ssh:/root/.ssh" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    cpp-scons-image