from PIL import Image
import sys

def usage():
    print("usage python3 trans.py a.ppm b.jpg")

if __name__ == "__main__":
    # usage python3 trans.py a.ppm b.jpg
    if len(sys.argv) != 3:
        usage()

    img = Image.open(sys.argv[1])
    img.save(sys.argv[2])