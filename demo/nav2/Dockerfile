FROM ros:kilted-ros-base-noble
RUN apt-get update && apt-get install -y --no-install-recommends ros-kilted-nav2-bringup ros-kilted-rmw-zenoh-cpp && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION=rmw_zenoh_cpp
COPY nav2_params.yaml run.sh /anews-nav2/
ENTRYPOINT ["bash", "/anews-nav2/run.sh"]
