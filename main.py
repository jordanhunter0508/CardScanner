import cv2
from camera.usb_camera import UsbCamera

usbCamera = UsbCamera()

# When a camera is connected
# try:

#     camera = usbCamera.connect_carmera(0)
#     print("Camera Connected")

#     while(True):
#         raw_image = usbCamera.capture_image(camera)
#         prepared_image = usbCamera.prepare_image(raw_image)
#         points = usbCamera.find_card(prepared_image)

#         if points is not None:
#             points = usbCamera.order_points(points)
#             card = usbCamera.get_card(raw_image, points)
#             card = cv2.resize(card, (366,512))
#             cv2.imshow("Card", card)
        
#         cv2.imshow("Live Stream",raw_image)

#         if cv2.waitKey(1) == ord('q'):
#             break
    
#     print("Card")

#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# except Exception as e:
#     print("An error occured: ", e)




# If no camera is connected 
try:
    raw_image = cv2.imread("sample_card.jpg")
    prepared_image = usbCamera.prepare_image(raw_image)
    points = usbCamera.find_card(prepared_image)

    if points is not None:
        points = usbCamera.order_points(points)
        card = usbCamera.get_card(raw_image, points)
        card = cv2.resize(card, (366,512))
        cv2.imshow("Card", card)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("The card could not be found.")
    
except Exception as e:
    print("An error occured: ", e)
    