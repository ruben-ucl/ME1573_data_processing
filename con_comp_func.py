import numpy as np
import cv2

def con_comp(im, connectivity):
    
    # Get connected components with stats
    (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(im, connectivity, cv2.CV_32S)
    con_comp_dict = {'numLabels': numLabels,
                     'labels': labels,
                     'stats': stats,
                     'centroid': centroids
                     }
    
    # Initialise mask to store component locations and RGB image to store bounding boxes
    mask = np.zeros_like(im)
    output_im = cv2.cvtColor(np.zeros_like(im), cv2.COLOR_GRAY2RGB)
    
    # Loop through all components
    for i in range(1, numLabels):
        # Get stats for component i
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        (cX, cY) = centroids[i]
        
        # Set selection criteria for components
        keepWidth = w > 3 and w < 50
        keepHeight = h > 3 and h < 50
        keepArea = area > 30 and area < 1000
        keepY = y > 320 and y < 450
        
        # For components that satisfy conditions: print details, add to mask, and add bounding box to output image
        if all((keepWidth, keepHeight, keepArea, keepY)):
            print(f'Keeping component {i+1}/{numLabels}\nx: {x}, y: {y}, w: {w}, h: {h}, area: {area}, centroid: {(cX, cY)}')
            componentMask = (labels == i).astype("uint8") * 255
            mask += componentMask
            output_im = cv2.rectangle(output_im, (x, y), (x + w, y + h), (0, 255, 0), 1)
            bounding_box_mask = output_im != 0
            
    return bounding_box_mask, con_comp_dict