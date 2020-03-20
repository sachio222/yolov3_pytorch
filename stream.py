import urllib.request
import cv2 as cv
import numpy as np

url = 'http://192.168.68.123'

with urllib.request.urlopen(url) as stream:

    bytes = bytearray()

    while True:
        bytes += stream.read(1024)

        # Identify start and end of jpegs
        jpg_head = bytes.find(b'\xff\xd8')
        jpg_end = bytes.find(b'\xff\xd9')

        if jpg_head != -1 and jpg_end != -1:

            jpg = bytes[jpg_head:jpg_end + 2]
            bytes = bytes[jpg_end + 2:]

            img = cv.imdecode(np.frombuffer(jpg, dtype=np.uint8),
                              cv.IMREAD_COLOR)
            img = cv.resize(img, (560, 420))
            cv.imshow('Stream', img)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

cv.destroyAllWindows()
