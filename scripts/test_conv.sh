export PYTHONPATH=$(pwd)/dist:$PYTHONPATH
python3 -m pytest -v ./scripts/tests/test_conv2d.py --disable-warnings --tb=short