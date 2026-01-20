from gwel.network import Classifier 
import sys 
import cv2

try:
    from qreader import QReader

except ImportError:
    sys.exit("qreader not found. Install with 'pip install qreader pyzbar'. You may need to check system requirements at https://pypi.org/project/qreader/.")



class QRreader(Classifier):
    def __init__(self, max_size: int = 1800, merge: bool = True):
        self.qreader=QReader()
        self.max_size = max_size
        self.merge = merge


    def set_device(self):
        pass

    def load_weights(self):
        pass

    def inference(self, image):
        resized = self.resize_image(image)
        QR_data = self.qreader.detect_and_decode(resized, return_detections=True)
        caption = [QR for QR in QR_data[0] if QR is not None]

        caption = [qr for qr in QR_data[0] if qr]

        if self.merge and caption:
            caption = longest_common_substring(caption)
            caption = caption.removesuffix("_")

        return caption

    def resize_image(self,image):
        (h, w) = image.shape[:2]
        ratio = min(self.max_size / w, self.max_size / h, 1)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        resized = cv2.resize(
            image,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA
        )

        return resized


def longest_common_substring(strs):
    """Return the longest common substring among a list of strings."""
    if not strs:
        return ""

    # start with the shortest string to limit work
    shortest = min(strs, key=len)

    longest = ""
    for i in range(len(shortest)):
        for j in range(i + 1, len(shortest) + 1):
            candidate = shortest[i:j]
            if all(candidate in other for other in strs):
                if len(candidate) > len(longest):
                    longest = candidate
    return longest

   
