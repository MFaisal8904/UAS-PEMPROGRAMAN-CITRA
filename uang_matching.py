import glob
import cv2 
import numpy as np
import imutils 


def uang_matching():
    template_data = []
    template_files = glob.glob('template/*.jpg', recursive=True)
    print("Templates loaded:", template_files)
    
    for template_file in template_files:
        tmp = cv2.imread(template_file)
        if tmp is None:
            print(f"Error loading template {template_file}")
            continue

        tmp = imutils.resize(tmp, width=int(tmp.shape[1] * 0.5))  
        tmp = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)  
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        tmp = cv2.filter2D(tmp, -1, kernel) 
        tmp = cv2.blur(tmp, (3, 3)) 
        tmp = cv2.Canny(tmp, 50, 200)  
        nominal = template_file.replace('template/', '').replace('.jpg', '')
        template_data.append({"glob": tmp, "nominal": nominal})
     
    for image_glob in glob.glob('test/*.jpg'):
        image_test = cv2.imread(image_glob)
        if image_test is None:
            print(f"Error loading test image {image_glob}")
            continue

        image_test_gray = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
        image_test_edges = cv2.Canny(image_test_gray, 50, 200)

        for template in template_data:
            (tmp_height, tmp_width) = template['glob'].shape[:2]
            found = None
            threshold = 0.4

            for scale in np.linspace(0.2, 1.0, 20)[::-1]: 
                resized = imutils.resize(image_test_edges, width=int(image_test_edges.shape[1] * scale))
                r = image_test_edges.shape[1] / float(resized.shape[1])

                if resized.shape[0] < tmp_height or resized.shape[1] < tmp_width:
                    break

                result = cv2.matchTemplate(resized, template['glob'], cv2.TM_CCOEFF_NORMED)
                (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

                if found is None or maxVal > found[0]:
                    found = (maxVal, maxLoc, r)

                if maxVal >= threshold:
                    print(f"Money: {template['nominal']} detected in {image_glob}")
                    break  

            if found is not None and found[0] >= threshold:
                (maxVal, maxLoc, r) = found
                (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
                (endX, endY) = (int((maxLoc[0] + tmp_width) * r), int((maxLoc[1] + tmp_height) * r))

                cv2.rectangle(image_test, (startX, startY), (endX, endY), (0, 0, 255), 2)
        
    
        cv2.imshow("Result", image_test)
        cv2.waitKey(0)  


if __name__ == "__main__": 
    uang_matching()
