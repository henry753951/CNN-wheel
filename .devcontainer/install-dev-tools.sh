apt-get update
apt-get install unzip wget curl ssh gdb pip nodejs -y
apt-get install build-essential ninja-build -y --no-install-recommends 
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt