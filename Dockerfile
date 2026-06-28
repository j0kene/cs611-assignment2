# Apache Airflow base image (matches the version used in Lab 5)
FROM apache/airflow:2.6.1

# Switch to root to install OS-level packages
USER root

ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless) for PySpark, plus procps + bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash curl && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

# Install Python dependencies as the airflow user
COPY requirements.txt /requirements.txt
USER airflow
RUN pip install --no-cache-dir -r /requirements.txt
