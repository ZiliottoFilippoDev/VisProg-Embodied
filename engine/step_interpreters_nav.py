# library imports
import cv2
import os
import torch
import openai
import functools
import numpy as np
import io, tokenize
from PIL import Image

# model imports
from transformers import (OwlViTProcessor,
                          OwlViTForObjectDetection)
# FAKE model imports
#  from PointNavFolder import pointnav_model

# utils import
from .nms import nms
from vis_utils import html_embed_image, html_colored_span, vis_masks

def parse_step(step_str,partial=False):
    tokens = list(tokenize.generate_tokens(io.StringIO(step_str).readline))
    output_var = tokens[0].string
    step_name = tokens[2].string
    parsed_result = dict(
        output_var=output_var,
        step_name=step_name)
    if partial:
        return parsed_result

    arg_tokens = [token for token in tokens[4:-3] if token.string not in [',','=']]
    num_tokens = len(arg_tokens) // 2
    args = dict()
    for i in range(num_tokens):
        args[arg_tokens[2*i].string] = arg_tokens[2*i+1].string
    parsed_result['args'] = args
    return parsed_result

class DetectInterpreter():
    step_name = 'DETECT'

    def __init__(self,thresh=0.1,nms_thresh=0.5):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.processor = OwlViTProcessor.from_pretrained(
            "google/owlvit-large-patch14")
        self.model = OwlViTForObjectDetection.from_pretrained(
            "google/owlvit-large-patch14").to(self.device)
        self.model.eval()
        self.thresh = thresh
        self.nms_thresh = nms_thresh

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        obj_name = eval(parse_result['args']['object'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return img_var,obj_name,output_var

    def normalize_coord(self,bbox,img_size):
        w,h = img_size
        x1,y1,x2,y2 = [int(v) for v in bbox]
        x1 = max(0,x1)
        y1 = max(0,y1)
        x2 = min(x2,w-1)
        y2 = min(y2,h-1)
        return [x1,y1,x2,y2]

    def predict(self,img,obj_name):
        encoding = self.processor(
            text=[[f'a photo of {obj_name}']], 
            images=img,
            return_tensors='pt')
        encoding = {k:v.to(self.device) for k,v in encoding.items()}
        with torch.no_grad():
            outputs = self.model(**encoding)
            for k,v in outputs.items():
                if v is not None:
                    outputs[k] = v.to('cpu') if isinstance(v, torch.Tensor) else v
        
        target_sizes = torch.Tensor([img.size[::-1]])
        results = self.processor.post_process_object_detection(outputs=outputs,threshold=self.thresh,target_sizes=target_sizes)
        boxes, scores = results[0]["boxes"], results[0]["scores"]
        boxes = boxes.cpu().detach().numpy().tolist()
        scores = scores.cpu().detach().numpy().tolist()
        if len(boxes)==0:
            return []

        boxes, scores = zip(*sorted(zip(boxes,scores),key=lambda x: x[1],reverse=True))
        selected_boxes = []
        selected_scores = []
        for i in range(len(scores)):
            if scores[i] > self.thresh:
                coord = self.normalize_coord(boxes[i],img.size)
                selected_boxes.append(coord)
                selected_scores.append(scores[i])

        selected_boxes, selected_scores = nms(
            selected_boxes,selected_scores,self.nms_thresh)
        return selected_boxes

    def top_box(self,img):
        w,h = img.size        
        return [0,0,w-1,int(h/2)]

    def bottom_box(self,img):
        w,h = img.size
        return [0,int(h/2),w-1,h-1]

    def left_box(self,img):
        w,h = img.size
        return [0,0,int(w/2),h-1]

    def right_box(self,img):
        w,h = img.size
        return [int(w/2),0,w-1,h-1]

    # def box_image(self,img,boxes,highlight_best=True):
    #     img1 = img.copy()
    #     draw = ImageDraw.Draw(img1)
    #     for i,box in enumerate(boxes):
    #         if i==0 and highlight_best:
    #             color = 'red'
    #         else:
    #             color = 'blue'

    #         draw.rectangle(box,outline=color,width=5)

    #     return img1

    # def html(self,img,box_img,output_var,obj_name):
    #     step_name=html_step_name(self.step_name)
    #     obj_arg=html_arg_name('object')
    #     img_arg=html_arg_name('image')
    #     output_var=html_var_name(output_var)
    #     img=html_embed_image(img)
    #     box_img=html_embed_image(box_img,300)
    #     return f"<div>{output_var}={step_name}({img_arg}={img}, {obj_arg}='{obj_name}')={box_img}</div>"


    def execute(self,prog_step,inspect=False):
        img_var,obj_name,output_var = self.parse(prog_step)
        img = prog_step.state[img_var]
        if obj_name=='TOP':
            bboxes = [self.top_box(img)]
        elif obj_name=='BOTTOM':
            bboxes = [self.bottom_box(img)]
        elif obj_name=='LEFT':
            bboxes = [self.left_box(img)]
        elif obj_name=='RIGHT':
            bboxes = [self.right_box(img)]
        else:
            bboxes = self.predict(img,obj_name)

        box_img = self.box_image(img, bboxes)
        prog_step.state[output_var] = bboxes
        prog_step.state[output_var+'_IMAGE'] = box_img
        if inspect:
            html_str = self.html(img, box_img, output_var, obj_name)
            return bboxes, html_str

        return bboxes

class NavigateInterpreter():
    # module to tell the agent to keep nagivating the env
    step_name= 'NAVIGATE'

    def __init__(self):
        print(f'Registering {self.step_name} step')
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        # self.model = pointnav_model().to(self.device)

    def parse(self, prog_step):    
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        img_var = parse_result['args']['image']
        output_var = parse_result['output_var']
        step_input = parse_result['args']['goal']
        assert(step_name==self.step_name)
        return step_input, img_var, output_var
    
    def navigate(self, prog_step):
        step_input, img_var, output_var = self.parse(prog_step)

        # fake code navigation 
        # return distance from target at stop action of agent
        fake_dist = np.random.uniform(0., 1.)
        print(fake_dist)
        prog_step.state[output_var] = fake_dist

        return fake_dist

    def execute(self,prog_step,inspect=False):    
        
        return self.navigate(prog_step)

class LocateInterpreter():
    # module to locate (x,y) pos in the map
    step_name= 'LOCATE'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):    
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        step_input = parse_result['args']['pos']
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def locate(self, prog_step, th_ = 0.5):
        # fake code for locating the target
        # if agent within target range return True else False
        step_input, output_var = self.parse(prog_step)

        dist = prog_step.state[step_input] <= th_
        prog_step.state[output_var] = dist 
        return dist

    def execute(self,prog_step,inspect=False):    
        return self.locate(prog_step)
    
class EvalInterpreter():
    step_name = 'EVAL'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        step_input = eval(parse_result['args']['expr'])
        output_var = parse_result['output_var']
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def eval_(self, prog_step):
        # step_input the variable of the step_name
        # output_var is the name of the same variable
        step_input, output_var = self.parse(prog_step)

        prog_state = dict()
        for var_name,var_value in prog_step.state.items():
            prog_state[var_name] = var_value

        # .format(**dict) method to insert values from a dict.
        # 'hi my name is {VAR}' --> 'hi my name is John'
        step_input = step_input.format(**prog_state)
        step_output = eval(step_input)

        prog_step.state[output_var] = step_output
        return step_output

    def execute(self, prog_step,inspect=False):
        return self.eval_(prog_step)

class StopInterpreter():
    # module to stop navigation and episode
    step_name= 'STOP'

    def __init__(self):
        print(f'Registering {self.step_name} step')

    def parse(self,prog_step):  
        parse_result = parse_step(prog_step.prog_str)
        step_name = parse_result['step_name']
        output_var = parse_result['output_var']
        step_input = parse_result['args']['var']
        assert(step_name==self.step_name)
        return step_input, output_var
    
    def stop(self, prog_step):
        # takes the EVAL input and decides if episode is terminated
        step_input, output_var = self.parse(prog_step)

        stop = prog_step.state[step_input] == 'STOP'
        prog_step.state[output_var] = stop

        return stop

    def execute(self,prog_step,inspect=True):    
        return self.stop(prog_step)
    
    
def register_step_interpreters(dataset='pointnav'):
    if dataset == 'pointnav':
        return dict(
            NAVIGATE=NavigateInterpreter(), # navigate
            # DETECT=DetectInterpreter(), # detect an object
            STOP=StopInterpreter(), # stop navigation
            LOCATE=LocateInterpreter(), # locate agent in env
            EVAL=EvalInterpreter(), # eval if position is within range
        )