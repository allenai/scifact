# Download and unpack data from AI2 S3 bucket.

name="https://ai2-s2-scifact.s3-us-west-2.amazonaws.com/release/2020-05-01/data.tar.gz"
wget $name

tar -xvf data.tar.gz
rm data.tar.gz
