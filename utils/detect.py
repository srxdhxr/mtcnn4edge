import math
import cv2
import torch
import time
from functools import wraps
import utils.functional as func

def _no_grad(func):

    def wrapper(*args, **kwargs):
        with torch.no_grad():
            ret = func(*args, **kwargs)
        return ret

    return wrapper


def timit(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        # Only execute the profiling logic if self.prof is True
        if self.prof:  
            if hasattr(self, 'profDict'):
                if func.__name__ not in self.profDict:
                    self.profDict[func.__name__] = [execution_time]  # Assign the list to the profDict
                else:
                    self.profDict[func.__name__].append(execution_time)  # Append the execution time to the existing list
        return result
    return wrapper



class FaceDetector(object):

    def __init__(self, pnet, rnet,onet, device='cpu',prof = False,use3stage = False,use_jit = True,use_c = False,use32 = False):
        self.device = torch.device(device)
        self.use_c = use_c
        self.pnet = pnet.to(self.device)
        self.rnet = rnet.to(self.device)
        self.onet = onet.to(self.device)
        self.use32 = use32

        if prof:
            self.prof = prof
            self.profDict = {}
            self.profDict['nms_stage_one'] = []
            self.profDict['nms_stage_two'] = []
            
       
        self.use_jit = use_jit

        self.use3stage = use3stage

        if self.use3stage:
            self.profDict['nms_stage_three'] = []

    def to_script(self):
        if isinstance(self.pnet, torch.nn.Module):
            self.pnet.to_script()
        
        if isinstance(self.rnet, torch.nn.Module):
            self.rnet.to_script()
            
        return self

    @timit
    def _preprocess(self, img):
        
        if isinstance(img, str):
            img = cv2.imread(img)

        # Convert image from rgb to bgr for Compatible with original caffe model.
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = img.transpose(2, 0, 1)
        img = torch.FloatTensor(img).to(self.device)
        img = func.imnormalize(img)
        img = img.unsqueeze(0)
        
        return img

    @timit
    def detect(self, img, threshold=[0.95 ,0.75, 0.85], factor=0.7, minsize=12, nms_threshold=[0.7, 0.7, 0.6]):
        
        img = self._preprocess(img)
        if not self.use3stage:
            threshold = [0.95,0.95]
            nms_threshold=[0.3, 0.1]
            
        stage_one_boxes = self.stage_one(img, threshold[0], factor, minsize, nms_threshold[0])
        stage_two_boxes = self.stage_two(img, stage_one_boxes, threshold[1], nms_threshold[1])
       
        if self.use3stage:
            stage_three_boxes = self.stage_three(img, stage_two_boxes, threshold[2], nms_threshold[2])
            return stage_three_boxes
    
        
        return stage_two_boxes
    
    @timit
    def _generate_bboxes(self, probs, offsets, scale, threshold):

    
        """Generate bounding boxes at places
        where there is probably a face.

        Arguments:
            probs: a FloatTensor of shape [1, 2, n, m].
            offsets: a FloatTensor array of shape [1, 4, n, m].
            scale: a float number,
                width and height of the image were scaled by this number.
            threshold: a float number.

        Returns:
            boxes: LongTensor with shape [x, 4].
            score: FloatTensor with shape [x].
        """

        # applying P-Net is equivalent, in some sense, to
        # moving 12x12 window with stride 2
        stride = 2
        cell_size = 12

        # extract positive probability and resize it as [n, m] dim tensor.
        probs = probs[0, 1, :, :]

        # indices of boxes where there is probably a face
        inds = (probs > threshold).nonzero()

        if inds.shape[0] == 0:
            return torch.empty((0, 4), dtype=torch.int32, device=self.device), torch.empty(0, dtype=torch.float32, device=self.device), torch.empty((0, 4), dtype=torch.float32, device=self.device)

        # transformations of bounding boxes
        tx1, ty1, tx2, ty2 = [offsets[0, i, inds[:, 0], inds[:, 1]]
                              for i in range(4)]
        # they are defined as:
        # w = x2 - x1 + 1
        # h = y2 - y1 + 1
        # x1_true = x1 + tx1*w
        # x2_true = x2 + tx2*w
        # y1_true = y1 + ty1*h
        # y2_true = y2 + ty2*h

        offsets = torch.stack([tx1, ty1, tx2, ty2], 1)
        score = probs[inds[:, 0], inds[:, 1]]

        # P-Net is applied to scaled images
        # so we need to rescale bounding boxes back
        bounding_boxes = torch.stack([
            stride*inds[:, 1] + 1.0,
            stride*inds[:, 0] + 1.0,
            stride*inds[:, 1] + 1.0 + cell_size,
            (stride*inds[:, 0] + 1.0 + cell_size),
        ], 0).transpose(0, 1).float()

        bounding_boxes = torch.round(bounding_boxes / scale).int()

        
        return bounding_boxes, score, offsets

    @timit
    def _calibrate_box(self, bboxes, offsets):
        
        """Transform bounding boxes to be more like true bounding boxes.
        'offsets' is one of the outputs of the nets.

        Arguments:
            bboxes: a IntTensor of shape [n, 4].
            offsets: a IntTensor of shape [n, 4].

        Returns:
            a IntTensor of shape [n, 4].
        """
        x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
        w = x2 - x1 + 1.0
        h = y2 - y1 + 1.0
        w = torch.unsqueeze(w, 1)
        h = torch.unsqueeze(h, 1)

        # this is what happening here:
        # tx1, ty1, tx2, ty2 = [offsets[:, i] for i in range(4)]
        # x1_true = x1 + tx1*w
        # y1_true = y1 + ty1*h
        # x2_true = x2 + tx2*w
        # y2_true = y2 + ty2*h
        # below is just more compact form of this

        # are offsets always such that
        # x1 < x2 and y1 < y2 ?

        translation = torch.cat([w, h, w, h], 1).float() * offsets
        bboxes += torch.round(translation).int()

        
        return bboxes

    @timit
    def _convert_to_square(self, bboxes):
        
        """Convert bounding boxes to a square form.

        Arguments:
            bboxes: a IntTensor of shape [n, 4].

        Returns:
            a IntTensor of shape [n, 4],
                squared bounding boxes.
        """

        square_bboxes = torch.zeros_like(bboxes, device=self.device, dtype=torch.float32)
        x1, y1, x2, y2 = [bboxes[:, i].float() for i in range(4)]
        h = y2 - y1 + 1.0
        w = x2 - x1 + 1.0
        max_side = torch.max(h, w)
        square_bboxes[:, 0] = x1 + w*0.5 - max_side*0.5
        square_bboxes[:, 1] = y1 + h*0.5 - max_side*0.5
        square_bboxes[:, 2] = square_bboxes[:, 0] + max_side - 1.0
        square_bboxes[:, 3] = square_bboxes[:, 1] + max_side - 1.0

        square_bboxes = torch.ceil(square_bboxes + 1).int()

        
        return square_bboxes

    @timit
    def _refine_boxes(self, bboxes, w, h):
        
        bboxes = torch.max(torch.zeros_like(bboxes, device=self.device), bboxes)
        sizes = torch.IntTensor([[h, w, h, w]] * bboxes.shape[0]).to(self.device)
        bboxes = torch.min(bboxes, sizes)
        
        return bboxes

    @_no_grad
    @timit
    def stage_one(self, img, threshold, factor, minsize, nms_threshold):
       
        width = img.shape[2]
        height = img.shape[3]

        # Compute valid scales
        scales = []
        cur_width = width
        cur_height = height
        cur_factor = 1
        while cur_width >= 12 and cur_height >= 12:
            if 12 / cur_factor >= minsize:  # Ignore boxes that smaller than minsize
                w = cur_width
                h = cur_height
                scales.append((w, h, cur_factor))

            cur_factor *= factor
            cur_width = math.ceil(cur_width * factor)
            cur_height = math.ceil(cur_height * factor)

        # Get candidate boxesi ph
        candidate_boxes = torch.empty((0, 4), dtype=torch.int32, device=self.device)
        candidate_scores = torch.empty((0), device=self.device)
        candidate_offsets = torch.empty((0, 4), dtype=torch.float32, device=self.device)
        for w, h, f in scales:
            resize_img = torch.nn.functional.interpolate(
                img, size=(w, h), mode='bilinear')

            
            p_distribution, box_regs = self.pnet(resize_img)


            candidate, scores, offsets = self._generate_bboxes(
                p_distribution, box_regs, f, threshold)

            candidate_boxes = torch.cat([candidate_boxes, candidate])
            candidate_scores = torch.cat([candidate_scores, scores])
            candidate_offsets = torch.cat([candidate_offsets, offsets])

        # nms

        
        if candidate_boxes.shape[0] != 0:
            candidate_boxes = self._calibrate_box(candidate_boxes, candidate_offsets)
            
            if self.prof:
                
                st = time.time()
            if self.use_c:
                keep = func.nms_c(candidate_boxes.cpu().numpy(), candidate_scores.cpu().numpy(), nms_threshold)
    
            keep = func.nms(candidate_boxes.cpu().numpy(), candidate_scores.cpu().numpy(), nms_threshold, use_jit = self.use_jit)

            if self.prof:
                self.profDict['nms_stage_one'].append(time.time()-st)

            
            return candidate_boxes[keep]
        else:
            
            return candidate_boxes
    
    
    @_no_grad
    @timit
    def stage_two(self, img, boxes, threshold, nms_threshold):
        if self.use32:
            in_size = 32
        else:
            in_size = 24
        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes

        width = img.shape[2]
        height = img.shape[3]

        boxes = self._convert_to_square(boxes)
        boxes = self._refine_boxes(boxes, width, height)

        # get candidate faces
        candidate_faces = list()


        for box in boxes:
            im = img[:, :, box[1]: box[3], box[0]: box[2]]
            im = torch.nn.functional.interpolate(
                im, size=(in_size, in_size), mode='bilinear')
            candidate_faces.append(im)
        
        candidate_faces = torch.cat(candidate_faces, 0)

        # rnet forward pass
    
        p_distribution, box_regs = self.rnet(candidate_faces)
        


        # filter negative boxes
        scores = p_distribution[:, 1]
        mask = (scores >= threshold)
        boxes = boxes[mask]
        box_regs = box_regs[mask]
        scores = scores[mask]

        if boxes.shape[0] > 0:
            boxes = self._calibrate_box(boxes, box_regs)
            # nms
           
            st = time.time()
            if self.use_c:
                keep = func.nms_c(boxes.cpu().numpy(), scores.cpu().numpy(), nms_threshold)

            keep = func.nms(boxes.cpu().numpy(), scores.cpu().numpy(),nms_threshold, use_jit = self.use_jit)
            if self.prof:
                self.profDict['nms_stage_two'].append(time.time()-st)
            

        
            boxes = boxes[keep]

        if not self.use3stage:
            boxes = self._convert_to_square(boxes)
            boxes = self._refine_boxes(boxes, width, height)
        return boxes
    
    
    @_no_grad
    @timit
    def stage_three(self, img, boxes, threshold, nms_threshold):
        
        # no candidate face found.
        if boxes.shape[0] == 0:
            return boxes, torch.empty(0, device=self.device, dtype=torch.int32)

        width = img.shape[2]
        height = img.shape[3]

        boxes = self._convert_to_square(boxes)
        boxes = self._refine_boxes(boxes, width, height)

        # get candidate faces
        candidate_faces = list()

        for box in boxes:
            im = img[:, :, box[1]: box[3], box[0]: box[2]]
            im = torch.nn.functional.interpolate(
                im, size=(48, 48), mode='bilinear')
            candidate_faces.append(im)
        
        candidate_faces = torch.cat(candidate_faces, 0)
        
        p_distribution, box_regs = self.onet(candidate_faces)

        

        # filter negative boxes
        scores = p_distribution[:, 1]
        mask = (scores >= threshold)
        boxes = boxes[mask]
        box_regs = box_regs[mask]
        scores = scores[mask]

        if boxes.shape[0] > 0:

            
            boxes = self._calibrate_box(boxes, box_regs)
            boxes = self._refine_boxes(boxes, width, height)
            
            # nms
            
            
            st = time.time()
            if self.use_c:
                keep = func.nms_c(boxes.cpu().numpy(), scores.cpu().numpy(), nms_threshold)

            keep = func.nms(boxes.cpu().numpy(), scores.cpu().numpy(), nms_threshold, use_jit = self.use_jit)
            if self.prof:
                self.profDict['nms_stage_three'].append(time.time()-st)   
            
           
            boxes = boxes[keep]
        
        return boxes

    def get_prof(self):
        return self.profDict