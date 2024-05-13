#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Bounding box structure
typedef struct
{
    float x1, y1, x2, y2;
} Box;

// Function to calculate intersection over union (IoU)
float calculate_iou(Box box1, Box box2)
{
    // Calculate intersection area
    float intersection = fmax(0, fmin(box1.x2, box2.x2) - fmax(box1.x1, box2.x1)) *
                         fmax(0, fmin(box1.y2, box2.y2) - fmax(box1.y1, box2.y1));

    // Calculate union area
    float area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1);
    float area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1);
    float union_area = area1 + area2 - intersection;

    // Calculate IoU
    float iou = intersection / union_area;
    return iou;
}

// Function to perform Non-Maximum Suppression (NMS)
int *nms(Box *boxes, float *scores, int num_boxes, float iou_threshold, int *selected_count)
{
    int *selected_indices = (int *)malloc(num_boxes * sizeof(int));
    *selected_count = 0;

    // Sort boxes based on scores
    for (int i = 0; i < num_boxes - 1; i++)
    {
        for (int j = i + 1; j < num_boxes; j++)
        {
            if (scores[j] > scores[i])
            {
                // Swap scores
                float temp_score = scores[i];
                scores[i] = scores[j];
                scores[j] = temp_score;

                // Swap boxes
                Box temp_box = boxes[i];
                boxes[i] = boxes[j];
                boxes[j] = temp_box;
            }
        }
    }

    // Perform NMS
    for (int i = 0; i < num_boxes; i++)
    {
        // Check if the box is selected
        int is_selected = 1;
        for (int j = 0; j < *selected_count; j++)
        {
            if (calculate_iou(boxes[i], boxes[selected_indices[j]]) > iou_threshold)
            {
                is_selected = 0;
                break;
            }
        }
        // If the box is selected, add its index to the selected indices
        if (is_selected)
        {
            selected_indices[(*selected_count)++] = i;
        }
    }

    // Return selected indices
    return selected_indices;
}
