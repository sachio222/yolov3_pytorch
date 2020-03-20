import cv2
import urllib.request

url = 'http://192.168.68.123'
# url = urllib.request.urlopen(url)

print("Before URL")
cap = cv2.VideoCapture(url)
print(cap.isOpened())
print("After URL")

while True:

    #print('About to start the Read command')
    ret, frame = cap.read()
    #print('About to show frame of Video.')
    cv2.imshow("Capturing", frame)
    #print('Running..')

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
