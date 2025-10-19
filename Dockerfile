FROM rust:1.87-slim

ADD https://github.com/sched-ext/scx.git /scx

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    libseccomp-dev \
    libc6-dev \
    libncurses5-dev \
    libz-dev \
    libelf1 \
    libelf-dev \
    libz1 \
    pkg-config \
    curl \
    gnupg2 \
    lsb-release \
    wget \
    iperf3 \
    libiperf0 \
    iproute2 \
    bash \
    software-properties-common \
    && curl -fsSL https://apt.llvm.org/llvm.sh | bash -s -- 20 \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-20 100 \
    && update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-20 100

WORKDIR /scx

RUN rustup component add rustfmt

RUN cargo build --release -p scx_rustland

CMD ["/bin/bash"]