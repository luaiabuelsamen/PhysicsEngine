docker image rm cpp-scons-image

docker build -t cpp-scons-image .

docker run -it --rm -v "$(pwd)":/app -v "$HOME/.ssh:/root/.ssh" cpp-scons-image