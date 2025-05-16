FROM ubuntu:latest

# Set noninteractive frontend for apt-get to avoid prompts
ARG DEBIAN_FRONTEND=noninteractive

# Install dependencies: build-essential for make/C, python3, pip, gawk for scripts, git for LKH if needed, bc for bc command
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    python3 \
    python3-pip \
    gawk \
    git \
    bc \
    && apt-get install -y python3-natsort \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
# RUN pip3 install natsort # Replaced by apt package

WORKDIR /app

# Setup for model_build.sh
# model_build.sh is placed in /app/build_context/ so that its HOME_DIR resolves to /app,
# ensuring DATA_DIR becomes /app/data/ without script modification.
# It expects LKH-AMZ and scripts relative to its own location (BASE_DIR = /app/build_context).
COPY LKH-AMZ /app/build_context/LKH-AMZ/
COPY scripts /app/build_context/scripts/
COPY model_build.sh /app/build_context/model_build.sh

# Setup for model_apply.sh
# model_apply.sh expects to be run from /app (HOME_DIR=pwd)
# and looks for its scripts in /app/src/scripts/ (as BASE_DIR is defined as ${HOME_DIR}/src).
COPY scripts /app/src/scripts/
COPY model_apply.sh /app/model_apply.sh

# Make scripts executable
RUN chmod +x /app/build_context/model_build.sh /app/model_apply.sh && \
    find /app/build_context/scripts -type f \( -iname "*.py" -o -iname "*.sh" \) -exec chmod +x {} \; && \
    find /app/src/scripts -type f \( -iname "*.py" -o -iname "*.sh" \) -exec chmod +x {} \; && \
    find /app/build_context/LKH-AMZ -type f -name "*.sh" -exec chmod +x {} \; # Make any .sh scripts in LKH-AMZ executable

# Create data directories expected by the scripts.
# These directories will reside under /app/data/.
# model_build.sh writes to /app/data/model_build_outputs
# model_apply.sh reads from /app/data/model_build_outputs and writes to /app/data/model_apply_outputs
# Input directories (/app/data/model_build_inputs, /app/data/model_apply_inputs)
# are expected to be mounted or populated at runtime.
RUN mkdir -p /app/data/model_build_inputs && \
    mkdir -p /app/data/model_build_outputs && \
    mkdir -p /app/data/model_apply_inputs && \
    mkdir -p /app/data/model_apply_outputs && \
    chmod -R 777 /app/data

# Copy model build inputs required by model_build.sh for the image build process
COPY data/model_build_inputs /app/data/model_build_inputs/

# Run the build script
# This compiles LKH in /app/build_context/LKH-AMZ,
# copies compiled binaries to /app/data/model_build_outputs/bin,
# and creates model.json in /app/data/model_build_outputs/model.json.
RUN /app/build_context/model_build.sh

# Copy test_data directory
COPY data /app/data/

# Set the default command to run the apply script with test_data
CMD ["/app/model_apply.sh", "/app/test_data"] 