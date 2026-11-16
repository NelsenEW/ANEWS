FROM rust:1.94-bookworm AS build
WORKDIR /src
COPY anews-rs anews-rs
RUN cargo build --release --locked --manifest-path anews-rs/Cargo.toml -p anews

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libvulkan1 \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/anews-rs/target/release/anews /usr/local/bin/anews
COPY venues/demo_cafe /anews/venues/demo_cafe
COPY config/sensors /anews/config/sensors
COPY demo/dinner_rush.yaml /anews/demo/dinner_rush.yaml
WORKDIR /anews/demo
ENTRYPOINT ["anews"]
