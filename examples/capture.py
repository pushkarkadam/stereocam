import sys 

sys.path.append('../')
from stereocam import *


def main():
    capture_stereo(width=1920*2, height=1080)

if __name__ == '__main__':
    main()