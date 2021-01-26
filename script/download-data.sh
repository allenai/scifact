#!/bin/bash

# Check if the data directory already exists
if [ -e "data" ]
then
    echo "Data directory already exists. Skip download."
    exit 0
fi

# Download and unpack data from AI2 S3 bucket.
name="https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"
wget $name

tar -xvf data.tar.gz
rm data.tar.gz
