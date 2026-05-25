#!/bin/bash
cd ~
curl -L -o OpenJDK17U-jre_x64_linux.tar.gz "https://github.com/adoptium/temurin17-binaries/releases/download/jdk-17.0.11%2B9/OpenJDK17U-jre_x64_linux_hotspot_17.0.11_9.tar.gz"
tar -xzf OpenJDK17U-jre_x64_linux.tar.gz
mv jdk-17.0.11+9-jre ~/java-jre
export PATH="$HOME/java-jre/bin:$PATH"
java -version