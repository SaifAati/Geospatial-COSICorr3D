#!/bin/bash

INSTALL_CONDA=false
INSTALL_DOCKER=false
OVERWRITE=false
UPDATE_ENV=false

DOCKERFILE_PATH="Dockerfile"
BASE_IMAGE_NAME="ghcr.io/saifaati/geospatial-cosicorr3d/base_cosicorr3d_image"
BASE_IMAGE_TAG="base.1.1"

show_help() {
    echo "Usage: $0 [OPTION]"
    echo "Options:"
    echo "  --conda       Install Miniconda and set up the geoCosiCorr3D environment."
    echo "  --docker      Install Docker."
    echo "  --overwrite   Overwrite the existing geoCosiCorr3D environment if it exists (only valid with --conda)."
    echo "  --update      Update the existing geoCosiCorr3D environment if it exists (only valid with --conda)."
    echo "  -h, --help    Show this help message and exit."
    exit 0
}

while (( "$#" )); do
    case "$1" in
        --conda)
            INSTALL_CONDA=true
            shift
            ;;
        --docker)
            INSTALL_DOCKER=true
            shift
            ;;
        --overwrite)
            OVERWRITE=true
            shift
            ;;
        --update)
            UPDATE_ENV=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Error: Unknown option '$1'"
            show_help
            ;;
    esac
done

# Check if no operation was selected
if [ "$INSTALL_CONDA" = false ] && [ "$INSTALL_DOCKER" = false ]; then
    echo "Error: No operation selected. You must specify either --conda or --docker."
    show_help
fi

install_conda() {
    export LD_LIBRARY_PATH=$(pwd)/lib/:$LD_LIBRARY_PATH
    echo $LD_LIBRARY_PATH
    # Check if Miniconda is installed by looking for the ~/miniconda3 directory
    if ! [ -d ~/miniconda3 ]; then
        echo "Miniconda is not installed. Installing now..."
        mkdir -p cosicorr_tmp_install
        cd cosicorr_tmp_install
        wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
        bash Miniconda3-latest-Linux-x86_64.sh
        cd ..
        rm -rf cosicorr_tmp_install
        . ~/miniconda3/etc/profile.d/conda.sh
    else
        echo "Miniconda is already installed."
        . ~/miniconda3/etc/profile.d/conda.sh
    fi

    # Check if the geoCosiCorr3D environment already exists
    if conda env list | grep -q 'geoCosiCorr3D'; then
        if [ "$OVERWRITE" = true ]; then
            echo "The geoCosiCorr3D environment exists but will be deleted and recreated as per --overwrite option."
            conda remove --name geoCosiCorr3D --all
            conda env update --file geoCosiCorr3D.yml --prune
        elif [ "$UPDATE_ENV" = true ]; then
            echo "Updating the existing geoCosiCorr3D environment as per --update option."
            conda env update --file geoCosiCorr3D.yml --prune
        else
            echo "The geoCosiCorr3D environment is already installed. To activate it, enter the cmd: conda activate geoCosiCorr3D"
        fi
    else
        echo "The geoCosiCorr3D environment does not exist. Creating it..."
        conda env update --file geoCosiCorr3D.yml --prune
        . ~/miniconda3/etc/profile.d/conda.sh
    fi
}

get_version() {
    VERSION=$(python3 setup.py --version)
    if [ $? -ne 0 ]; then
        echo "Failed to get the package version."
        exit 1
    fi
    echo "COSI-CORR-3D::Package version: $VERSION"
}

build_package() {
    echo "Building the geoCosiCorr3D package version $VERSION..."
    python3 setup.py sdist bdist_wheel
    if [ $? -ne 0 ]; then
        echo "Failed to build the package."
        exit 1
    fi
    echo "Package built successfully."
}

install_package() {
    echo "Installing the geoCosiCorr3D package..."
    PACKAGE_FILE=$(ls dist/geoCosiCorr3D-${VERSION}-py3-none-any.whl | head -n 1)
    if [ -z "$PACKAGE_FILE" ]; then
        echo "Package file not found."
        exit 1
    fi
    conda run -n geoCosiCorr3D pip install "$PACKAGE_FILE" --force-reinstall
    rm -rf build dist

    echo "geoCosiCorr3D package installed successfully."
}

init_docker() {
    echo "Installing Docker..."
    sudo apt-get update
    sudo apt-get install -y docker.io docker-compose
    sudo systemctl start docker
    sudo systemctl enable docker
    echo "Docker and Docker compose  installed and started successfully."
}

start_docker() {
    echo "Starting Docker service..."
    sudo systemctl start docker
    echo "Docker service started."
}

check_dockerfile_exists() {
    if [ ! -f "$DOCKERFILE_PATH" ]; then
        echo "Error: Dockerfile '$DOCKERFILE_PATH' does not exist. Please check the file path."
        exit 1
    else
        echo "Dockerfile '$DOCKERFILE_PATH' exists."
    fi
}

pulling_base_image() {
    # Check if Docker is installed
    if ! command -v docker &> /dev/null
    then
        echo "Docker is not installed."
        init_docker
    else
        echo "Docker is installed."
        # Check if Docker service is running
        DOCKER_STATUS=$(systemctl is-active docker)
        if [ "${DOCKER_STATUS}" = "active" ]; then
            echo "Docker service is active and running."
        else
            echo "Docker service is not running."
            start_docker
        fi
    fi

    if ! docker image ls | grep -q "${BASE_IMAGE_NAME}.*${BASE_IMAGE_TAG}"; then
        echo "Base Image does not exist locally. Attempting to pull..."
        # Pull the Docker image from GCR
        docker pull ${BASE_IMAGE_NAME}:${BASE_IMAGE_TAG}
        if [ $? -eq 0 ]; then
            echo "Docker image pulled successfully."
        else
            echo "Failed to pull the Base Docker image."
        fi
    else
        echo "Base Docker image already exists locally."
    fi
}

install_docker() {
    echo "Installing Docker version ..."
    echo 'Building cosicorr3D base image '
    check_dockerfile_exists
    pulling_base_image

    docker-compose -f docker-compose.yml build geocosicorr3d
}

if [ "$INSTALL_CONDA" = true ]; then
    install_conda
#    get_version
#    build_package
#    install_package
fi

if [ "$INSTALL_DOCKER" = true ]; then
    install_docker
fi